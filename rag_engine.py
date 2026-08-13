#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2025-2026  CEA
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA

"""
RAGBackend — adapts the `raglib` documentation chatbot (git submodule, a
standalone RAG/Agentic project) to the API salomeAssistant.py's PyQt5 GUI
expects: a single merged config.json driving both retrieval and the
extraction pipeline, connect/ask/build-index actions, and an editable config.

`raglib` is not an installable package (no pyproject.toml, no __init__.py in
chatbot/) — it is designed to be run with `raglib/chatbot/` on sys.path so
that `import core` resolves to `raglib/chatbot/core/`. We replicate that here
explicitly instead of relying on cwd.
"""

import os
import sys
import copy
import json
import shutil
import tempfile
import subprocess
from dataclasses import fields as _dc_fields

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RAGLIB_CHATBOT_DIR = os.path.join(_THIS_DIR, "raglib", "chatbot")
_RAGLIB_EXTRACTION_DIR = os.path.join(_THIS_DIR, "raglib", "extraction")
if _RAGLIB_CHATBOT_DIR not in sys.path:
    sys.path.insert(0, _RAGLIB_CHATBOT_DIR)

from core import (  # noqa: E402
    ChatbotConfig, LLMConfig, EmbeddingConfig, RerankerConfig, AgenticConfig,
    DocumentationChatbot, AgenticChatbot,
)

# Exposed for salomeAssistant.py's response-style selector.
RESPONSE_STYLES = dict(ChatbotConfig.RESPONSE_STYLES)

# Top-level scalar keys ChatbotConfig understands. Everything else in the
# merged config.json (output_dir, use_token_chunking, modules, chunking,
# quality, and any "_comment*" key) is extraction-only or documentation and
# must be filtered out before building a ChatbotConfig — raglib's own loader
# (ChatbotConfig._from_json) raises on unknown top-level keys.
_CHATBOT_SCALAR_KEYS = {
    "project_name", "chromadb_path", "k_standard", "k_deep_dive", "k_retrieve",
    "deep_dive_batch_size", "top_n_after_rerank", "expansion_char_budget",
    "temperature", "max_tokens", "bm25_enabled", "title_boost_enabled",
    "hyde_enabled",
}

# Paths in the merged config that the "relative to this config file's
# directory" rule (documented in config.example.json) applies to. raglib's
# own code does NOT implement that rule itself — e.g. DocumentationChatbot
# resolves a relative chromadb_path against raglib/chatbot/, and
# process_docs.py resolves module dev_path/user_path against the process's
# cwd — so every such path is made absolute here before being handed off.
_TOP_PATH_KEYS = ("chromadb_path", "output_dir")
_MODULE_PATH_KEYS = ("dev_path", "user_path", "methodology_path")


def _strip_comments(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _dataclass_kwargs(cls, d):
    """Keep only the keys `cls` (a dataclass) actually declares — silently
    drops comment keys and any legacy/unknown field instead of raising, since
    this config is hand-edited and shared with the extraction pipeline."""
    valid = {f.name for f in _dc_fields(cls)}
    return {k: v for k, v in _strip_comments(d).items() if k in valid}


def _resolve_paths(raw, base_dir):
    """Return a deep copy of `raw` with every documented relative path
    resolved against `base_dir` (the config file's own directory).

    Also expands shell-style environment variables (`$SMESH_ROOT_DIR`,
    `${DOCUMENTATION_ROOT_DIR}`, ...) — SALOME exports one `<MODULE>_ROOT_DIR`
    per module (see env_launch.sh), so module paths can be written relative to
    those instead of a filesystem location that moves between machines/builds.
    Expansion runs before the relative/absolute check, since an expanded
    SALOME root is always absolute. A variable left unset is not an error here
    — expandvars leaves it as literal text, which then fails the normal
    "path not found" check downstream with a clear enough message."""
    data = copy.deepcopy(raw)

    def abspath(p):
        if not p:
            return p
        p = os.path.expandvars(p)
        if os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(base_dir, p))

    for key in _TOP_PATH_KEYS:
        if data.get(key):
            data[key] = abspath(data[key])

    agentic = data.get("agentic")
    if isinstance(agentic, dict) and agentic.get("page_index_path"):
        agentic["page_index_path"] = abspath(agentic["page_index_path"])

    for module in (data.get("modules") or {}).values():
        if not isinstance(module, dict):
            continue
        for key in _MODULE_PATH_KEYS:
            if module.get(key):
                module[key] = abspath(module[key])

    llm = data.get("llm")
    if isinstance(llm, dict) and llm.get("ssl_cert_file"):
        llm["ssl_cert_file"] = abspath(llm["ssl_cert_file"])

    return data


def _build_chatbot_config(resolved):
    """Build a ChatbotConfig from a merged (extraction+chatbot) config dict
    that has already had its paths resolved to absolute."""
    kwargs = {k: v for k, v in resolved.items() if k in _CHATBOT_SCALAR_KEYS}

    llm = resolved.get("llm")
    if isinstance(llm, dict):
        kwargs["llm"] = LLMConfig(**_dataclass_kwargs(LLMConfig, llm))

    embedding = resolved.get("embedding")
    if isinstance(embedding, dict):
        kwargs["embedding"] = EmbeddingConfig(**_dataclass_kwargs(EmbeddingConfig, embedding))

    if "reranker" in resolved:
        reranker = resolved["reranker"]
        kwargs["reranker"] = (
            RerankerConfig(**_dataclass_kwargs(RerankerConfig, reranker)) if reranker else None
        )

    if "agentic" in resolved:
        agentic = resolved["agentic"]
        kwargs["agentic"] = (
            AgenticConfig(**_dataclass_kwargs(AgenticConfig, agentic)) if agentic else None
        )

    return ChatbotConfig(**kwargs)


class RAGBackend:
    """
    Owns the active merged config, the raglib chatbots built from it, and the
    extraction subprocess. One instance per MainWindow.
    """

    default_save_path = os.path.join(
        os.path.expanduser("~"), ".config", "salome", "chatbot.config.json")

    # Starter config shipped next to this module (config.example.json lives at
    # the repo root in dev, and is copied next to the installed modules by the
    # sarag build script) — used to pre-populate the GUI on first run, before
    # the user has ever saved a config of their own.
    _bundled_example_config = os.path.join(_THIS_DIR, "config.example.json")

    def __init__(self):
        self.config_path = None
        self._raw_config = None       # as loaded/edited, relative paths intact
        self._resolved_config = None  # same, with paths made absolute
        self.config = ChatbotConfig()

        self._rag_chatbot = None
        self._agentic_chatbot = None
        self.available_modules = []
        self.has_reranker = False
        self.has_agentic = False
        self.initialized = False

        for path in (self.default_save_path, self._bundled_example_config):
            if os.path.isfile(path):
                try:
                    self.reload_config(path)
                    break
                except Exception:
                    continue  # try the next candidate; user can also Browse for a config

    # ------------------------------------------------------------ config I/O
    def reload_config(self, path):
        """Load a merged config.json from `path` as the active config."""
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        raw = _strip_comments(raw)

        base_dir = os.path.dirname(os.path.abspath(path))
        resolved = _resolve_paths(raw, base_dir)

        self.config = _build_chatbot_config(resolved)
        self._raw_config = raw
        self._resolved_config = resolved
        self.config_path = os.path.abspath(path)
        self._drop_chatbots()

    def save_and_reload(self, path, data):
        """Write the merged config dict `data` to `path` and make it active."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        self.reload_config(path)

    @property
    def extraction_modules(self):
        raw = self._raw_config or {}
        modules = raw.get("modules")
        return modules if isinstance(modules, dict) else {}

    def _drop_chatbots(self):
        """A config change invalidates any chatbot already built from the
        previous one — the user must Connect again."""
        self._rag_chatbot = None
        self._agentic_chatbot = None
        self.available_modules = []
        self.has_reranker = False
        self.has_agentic = False
        self.initialized = False

    # ------------------------------------------------------------ connect
    def set_llm_endpoint(self, base_url=None, model=None, api_key=None):
        if base_url:
            self.config.llm.base_url = base_url
        if model:
            self.config.llm.model = model
        if api_key:
            self.config.llm.api_key = api_key

    def initialize(self):
        """Build the RAG chatbot (always) and the Agentic chatbot (only if
        config.agentic is set and smolagents is available)."""
        self._rag_chatbot = DocumentationChatbot(self.config)
        self.available_modules = self._rag_chatbot.available_modules
        self.has_reranker = self._rag_chatbot.reranker is not None

        agentic_error = None
        self._agentic_chatbot = None
        if self.config.agentic is not None:
            try:
                self._agentic_chatbot = AgenticChatbot(self.config)
            except Exception as e:
                agentic_error = str(e)
        self.has_agentic = self._agentic_chatbot is not None

        self.initialized = True

        msg = (
            f"Connected. Project '{self.config.project_name}', "
            f"{len(self.available_modules)} module(s): "
            f"{', '.join(self.available_modules) or '(none)'}. "
            f"Reranker: {'on' if self.has_reranker else 'off'}. "
            f"Agentic: {'available' if self.has_agentic else 'unavailable'}."
        )
        if agentic_error:
            msg += f"\nAgentic mode disabled: {agentic_error}"
        return msg

    # ------------------------------------------------------------ query
    def ask(self, query, mode="rag", **params):
        if not self.initialized:
            return {"answer": None, "sources": [], "filters": {},
                     "error": "Not connected. Click Connect first."}

        max_tokens = params.get("max_tokens")

        if mode == "agentic":
            if not self._agentic_chatbot:
                return {
                    "answer": None, "sources": [], "filters": {},
                    "error": "Agentic mode is not available (no 'agentic' block in "
                             "the config, or smolagents is not installed).",
                }
            return self._agentic_chatbot.ask(
                query,
                max_steps=params.get("max_steps"),
                max_tokens=max_tokens,
                temperature=params.get("temperature"),
            )

        return self._rag_chatbot.ask(
            query,
            module=params.get("module"),
            doc_type=params.get("doc_type"),
            deep_dive=bool(params.get("deep_dive")),
            k=params.get("k"),
            temperature=params.get("temperature"),
            max_tokens=max_tokens,
            reranker_enabled=bool(params.get("reranker_enabled", True)),
            top_n=params.get("top_n"),
        )

    # ------------------------------------------------------------ extraction
    def run_extraction(self):
        """Run raglib's extraction pipeline against the active config,
        yielding output lines as they're produced, then reload the config so
        the freshly-built database is picked up on the next Connect."""
        if not self.config_path:
            yield "No config.json loaded."
            return

        tmp_dir = tempfile.mkdtemp(prefix="salome-assistant-extract-")
        try:
            # process_docs.py resolves module/output paths against its own
            # cwd, not the config file's directory — feed it the already
            # path-resolved config instead of the one on disk.
            tmp_config = os.path.join(tmp_dir, "config.resolved.json")
            with open(tmp_config, "w", encoding="utf-8") as f:
                json.dump(self._resolved_config, f)

            script = os.path.join(_RAGLIB_EXTRACTION_DIR, "process_docs.py")
            cmd = [sys.executable, "-u", script, "--config", tmp_config]
            proc = subprocess.Popen(
                cmd, cwd=_RAGLIB_EXTRACTION_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            try:
                for line in proc.stdout:
                    yield line.rstrip("\n")
            finally:
                proc.wait()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if proc.returncode != 0:
            yield f"Extraction exited with code {proc.returncode}"
            return

        try:
            self.reload_config(self.config_path)
        except Exception as e:
            yield f"Warning: failed to reload config after extraction: {e}"
