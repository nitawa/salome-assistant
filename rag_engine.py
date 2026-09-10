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
expects: connect/ask/build-index actions, and an editable configuration.

raglib's own README distinguishes two independent stages, each with its own
native config.json: extraction (`extraction/config.json`, consumed by
`process_docs.py`) and the chatbot (`chatbot/config.json`, consumed by
`chatbot.py` / `core.ChatbotConfig`). RAGBackend keeps a single config.json
on disk for SALOME Assistant, split into two top-level sections —
`{"chatbot": {...}, "extraction": {...}}` — one holding exactly raglib's
chatbot config.json schema, the other exactly its extraction config.json
schema. Connect and Ask go through exactly the calls documented in raglib's
README ("Python API" section):

    from core import ChatbotConfig, DocumentationChatbot, AgenticChatbot
    config = ChatbotConfig.load("config.json")
    chatbot = DocumentationChatbot(config)
    result = chatbot.ask("...")

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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RAGLIB_CHATBOT_DIR = os.path.join(_THIS_DIR, "raglib", "chatbot")
_RAGLIB_EXTRACTION_DIR = os.path.join(_THIS_DIR, "raglib", "extraction")
if _RAGLIB_CHATBOT_DIR not in sys.path:
    sys.path.insert(0, _RAGLIB_CHATBOT_DIR)

from core import ChatbotConfig, DocumentationChatbot, AgenticChatbot  # noqa: E402

# Exposed for salomeAssistant.py's response-style selector.
RESPONSE_STYLES = dict(ChatbotConfig.RESPONSE_STYLES)

# Paths that raglib documents as "may be relative" but resolves itself
# against the process's cwd rather than the config file's own directory
# (see CLAUDE.md's "Relative-path base mismatch" note) — expanded to
# absolute here before either config is handed to raglib, so a saved
# config.json stays portable (and $ENV_VAR-relative) regardless of where
# SALOME Assistant is launched from.
_CHATBOT_PATH_KEYS = ("chromadb_path",)
_MODULE_PATH_KEYS = ("dev_path", "user_path", "methodology_path")


def _strip_comments(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _abspath(p, base_dir):
    """Expand shell-style environment variables (`$SMESH_ROOT_DIR`,
    `${DOCUMENTATION_ROOT_DIR}`, ...) — SALOME exports one `<MODULE>_ROOT_DIR`
    per module (see env_launch.sh) — then resolve against `base_dir` if the
    result isn't already absolute. A variable left unset is not an error
    here — expandvars leaves it as literal text, which then fails the
    normal "path not found" check downstream with a clear enough message."""
    if not p:
        return p
    p = os.path.expandvars(p)
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def _resolve_chatbot_paths(raw, base_dir):
    """Deep copy of a chatbot config.json dict with chromadb_path,
    agentic.page_index_path and llm.ssl_cert_file resolved to absolute."""
    data = copy.deepcopy(raw)

    for key in _CHATBOT_PATH_KEYS:
        if data.get(key):
            data[key] = _abspath(data[key], base_dir)

    agentic = data.get("agentic")
    if isinstance(agentic, dict) and agentic.get("page_index_path"):
        agentic["page_index_path"] = _abspath(agentic["page_index_path"], base_dir)

    llm = data.get("llm")
    if isinstance(llm, dict) and llm.get("ssl_cert_file"):
        llm["ssl_cert_file"] = _abspath(llm["ssl_cert_file"], base_dir)

    return data


def _resolve_extraction_paths(raw, base_dir):
    """Deep copy of an extraction config.json dict with output_dir and every
    module dev_path/user_path/methodology_path resolved to absolute."""
    data = copy.deepcopy(raw)

    if data.get("output_dir"):
        data["output_dir"] = _abspath(data["output_dir"], base_dir)

    for module in (data.get("modules") or {}).values():
        if not isinstance(module, dict):
            continue
        for key in _MODULE_PATH_KEYS:
            if module.get(key):
                module[key] = _abspath(module[key], base_dir)

    return data


class RAGBackend:
    """
    Owns the active config (single file, `{"chatbot": {...}, "extraction":
    {...}}`), the raglib chatbots built from it, and the extraction
    subprocess. One instance per MainWindow.
    """

    default_save_path = os.path.join(
        os.path.expanduser("~"), ".config", "salome", "chatbot.config.json")

    # Starter config shipped next to this module (also at the repo root in a
    # dev checkout; copied next to the installed modules by the sarag build
    # script) — used to pre-populate the GUI on first run, before the user
    # has ever saved a config of their own.
    _bundled_example_config = os.path.join(_THIS_DIR, "chatbot.config.example.json")

    def __init__(self):
        self.config_path = None
        self._raw_chatbot_config = None       # as loaded/edited, relative paths intact
        self._raw_extraction_config = None    # same, for the "extraction" section
        self.config = ChatbotConfig()

        self._rag_chatbot = None
        self._agentic_chatbot = None
        self.available_modules = []
        self.has_reranker = False
        self.has_agentic = False
        self.agentic_error = None    # why has_agentic is False after initialize(), if it is
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
        """Load `{"chatbot": {...}, "extraction": {...}}` from `path` as the
        active config. The "chatbot" section is handed to raglib's own
        documented loader (`ChatbotConfig.load`, see README's "Python API"
        section); the "extraction" section feeds `run_extraction()`."""
        with open(path, encoding="utf-8") as f:
            raw = _strip_comments(json.load(f))

        raw_chatbot = raw.get("chatbot") or {}
        raw_extraction = raw.get("extraction") or {}

        base_dir = os.path.dirname(os.path.abspath(path))
        resolved_chatbot = _resolve_chatbot_paths(raw_chatbot, base_dir)

        # ChatbotConfig.load() is raglib's own loader (see README's "Python
        # API" section): it validates the file's keys itself and raises on
        # anything not part of ChatbotConfig's schema. It only accepts a
        # path on disk, so the already-path-resolved dict is written to a
        # throwaway temp file rather than reconstructed field-by-field here.
        fd, tmp_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(resolved_chatbot, f)
            self.config = ChatbotConfig.load(tmp_path)
        finally:
            os.unlink(tmp_path)

        self._raw_chatbot_config = raw_chatbot
        self._raw_extraction_config = raw_extraction
        self.config_path = os.path.abspath(path)
        self._drop_chatbots()

    def save_and_reload(self, path, data):
        """Write `{"chatbot": {...}, "extraction": {...}}` to `path` and make
        it the active config."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        self.reload_config(path)

    @property
    def extraction_modules(self):
        raw = self._raw_extraction_config or {}
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
        self.agentic_error = None
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
        config.agentic is set and smolagents is available) — the same
        DocumentationChatbot(config) / AgenticChatbot(config, ...) calls
        raglib's README shows under "Python API"."""
        self._rag_chatbot = DocumentationChatbot(self.config)
        self.available_modules = self._rag_chatbot.available_modules
        self.has_reranker = self._rag_chatbot.reranker is not None

        agentic_error = None
        self._agentic_chatbot = None
        if self.config.agentic is None:
            agentic_error = (
                "No 'agentic' block in config.json. Enable it via 'Edit "
                "configuration...' > Agentic tab (needs page_index.json, built "
                "by running extraction after 'Enable agentic mode' is checked)."
            )
        else:
            try:
                # Mirrors raglib's own chatbot.py wiring (see its --mode agentic
                # setup): without reranker_score_fn/section_select_fn,
                # AgenticChatbot's search_sections_tool silently falls back to
                # its page-level branch instead of section-level retrieval —
                # agentic.section_top_n/section_char_budget then have no
                # effect, and results go unreranked.
                reranker_score_fn = (
                    self._rag_chatbot.score_against_query
                    if self._rag_chatbot.reranker is not None else None
                )
                self._agentic_chatbot = AgenticChatbot(
                    self.config,
                    vectorstore=self._rag_chatbot.vectorstore,
                    bm25_index=self._rag_chatbot.bm25_index,
                    reranker_score_fn=reranker_score_fn,
                    section_select_fn=self._rag_chatbot.expand_sections,
                )
            except Exception as e:
                agentic_error = str(e)
        self.has_agentic = self._agentic_chatbot is not None
        self.agentic_error = agentic_error

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
        """Ask the already-initialized chatbot — chatbot.ask(...) /
        agentic_chatbot.ask(...), exactly as shown in raglib's README. The
        returned dict's "answer" key is the final answer handed back to the
        user; "error" is set instead if the call failed."""
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
            # raglib's agentic mode can have the LLM return a blank final
            # message with no exception raised (commonly once the accumulated
            # search/read tool history grows large) — check_grounding() finds
            # no code blocks to flag in an empty string, so it isn't caught as
            # a degraded/ungrounded answer either, and "error" comes back
            # None. This is usually transient, so retry the same query a
            # couple of times before surfacing it as an error.
            MAX_EMPTY_ANSWER_RETRIES = 2
            agentic_params = dict(
                max_steps=params.get("max_steps"),
                max_tokens=max_tokens,
                temperature=params.get("temperature"),
                max_chars_per_page=params.get("max_chars_per_page"),
            )
            result = self._agentic_chatbot.ask(query, **agentic_params)
            attempts = 1
            while (not result.get("error") and not (result.get("answer") or "").strip()
                   and attempts <= MAX_EMPTY_ANSWER_RETRIES):
                result = self._agentic_chatbot.ask(query, **agentic_params)
                attempts += 1
            if not result.get("error") and not (result.get("answer") or "").strip():
                result = dict(result)
                result["error"] = (
                    f"The assistant returned an empty answer after {attempts} attempt(s). "
                    "This usually means the conversation the agent built up (search/read "
                    "tool results) grew too large for the model's context window. Try "
                    "lowering agentic.max_steps or max_chars_per_page in the config, or "
                    "rephrase the question."
                )
            return result

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
        """Run raglib's extraction pipeline (`process_docs.py`) against the
        active config's "extraction" section, yielding output lines as
        they're produced, then reload the config so the freshly-built
        database is picked up on the next Connect."""
        if not self.config_path:
            yield "No chatbot.config.json loaded."
            return

        tmp_dir = tempfile.mkdtemp(prefix="salome-assistant-extract-")
        try:
            # process_docs.py resolves module/output paths against its own
            # cwd, not the config file's directory — feed it the already
            # path-resolved config instead of the one on disk.
            base_dir = os.path.dirname(self.config_path)
            resolved_extraction = _resolve_extraction_paths(
                self._raw_extraction_config or {}, base_dir)
            tmp_config = os.path.join(tmp_dir, "config.resolved.json")
            with open(tmp_config, "w", encoding="utf-8") as f:
                json.dump(resolved_extraction, f)

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
