#!/bin/env python3
"""SALOME Assistant — PyQt5 GUI over the raglib documentation chatbot.

The GUI loads a raglib config.json (LLM endpoint, ChromaDB path, embeddings),
lets the user tweak the LLM endpoint and per-query parameters, and queries the
documentation in either RAG or Agentic mode. It can also (re)build the vector
database by running the extraction pipeline.
"""
import sys
import os
import re
import time
import copy
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLineEdit, QPushButton,
                             QLabel, QMessageBox, QComboBox, QGroupBox, QCheckBox,
                             QSpinBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QDialog, QDialogButtonBox, QTabWidget, QTableWidget,
                             QTableWidgetItem, QHeaderView, QTabBar)

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings, QWebEnginePage
from rag_engine import RAGBackend, RESPONSE_STYLES
import salome_mcp_client
import markdown


class HistoryLineEdit(QLineEdit):
    """QLineEdit with shell-style Up/Down prompt history recall."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history = []
        self._index = 0        # position in _history; len(_history) == "editing a new line"
        self._draft = ""       # text being typed before Up was first pressed

    def add_history(self, text):
        """Record a submitted prompt and reset recall to "new line"."""
        if not text:
            return
        if not self._history or self._history[-1] != text:
            self._history.append(text)
        self._index = len(self._history)
        self._draft = ""

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            if self._history and self._index > 0:
                if self._index == len(self._history):
                    self._draft = self.text()
                self._index -= 1
                self.setText(self._history[self._index])
            return
        if event.key() == Qt.Key_Down:
            if self._history and self._index < len(self._history):
                self._index += 1
                if self._index == len(self._history):
                    self.setText(self._draft)
                else:
                    self.setText(self._history[self._index])
            return
        super().keyPressEvent(event)


class ChatWebEnginePage(QWebEnginePage):
    """Chat/browser page whose clicked links are handed to a callback
    instead of navigating the current tab away from its content (source
    links open in a new tab; "Run in SALOME" links trigger code execution
    — see MainWindow._handle_chat_link)."""

    def __init__(self, link_handler, parent=None):
        super().__init__(parent)
        self._link_handler = link_handler

    def acceptNavigationRequest(self, url, nav_type, is_main_frame):
        if nav_type == QWebEnginePage.NavigationTypeLinkClicked:
            self._link_handler(url)
            return False
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)


class Worker(QThread):
    """Background thread for blocking backend calls (connect / query)."""
    finished = pyqtSignal(object)
    status_update = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, backend, task_type, params=None):
        super().__init__()
        self.backend = backend
        self.task_type = task_type
        self.params = params or {}

    def run(self):
        try:
            if self.task_type == 'init':
                self.status_update.emit("Connecting & loading vector database...")
                self.backend.set_llm_endpoint(
                    base_url=self.params.get('base_url'),
                    model=self.params.get('model'),
                    api_key=self.params.get('api_key'),
                )
                msg = self.backend.initialize()
                self.finished.emit(msg)

            elif self.task_type == 'query':
                self.status_update.emit("Thinking...")
                result = self.backend.ask(self.params.pop('query'), **self.params)
                self.finished.emit(result)

        except Exception as e:
            self.failed.emit(str(e))


class SalomeRunWorker(QThread):
    """Runs one "Run in SALOME" code block via salome_mcp_client, off the
    GUI thread (it's a blocking TCP round-trip to the SALOME process)."""
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, code):
        super().__init__()
        self.code = code

    def run(self):
        try:
            result = salome_mcp_client.run_python(self.code)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class ExtractionWorker(QThread):
    """Runs the extraction pipeline subprocess, streaming output lines."""
    line = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def run(self):
        try:
            for ln in self.backend.run_extraction():
                self.line.emit(ln)
        except Exception as e:
            self.line.emit(f"Error: {e}")
        finally:
            self.finished.emit()


class ConfigDialog(QDialog):
    """Interactive editor for the single config.json, split into the
    "chatbot" (query time) and "extraction" (index build time) sections
    raglib itself keeps as two separate config files — see raglib/README.md.
    Populates from the backend's current config, and on Save writes the
    merged JSON back to disk and reloads the backend.
    """

    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.setWindowTitle("Configuration editor")
        self.resize(680, 600)

        cfg = backend.config
        raw = backend._raw_chatbot_config or {}
        raw_extraction = backend._raw_extraction_config or {}
        emb = cfg.embedding
        llm = cfg.llm
        chunking = raw_extraction.get("chunking", {}) if isinstance(raw_extraction.get("chunking"), dict) else {}
        quality = raw_extraction.get("quality", {}) if isinstance(raw_extraction.get("quality"), dict) else {}

        outer = QVBoxLayout(self)
        tabs = QTabWidget()
        outer.addWidget(tabs)

        # ---- General -------------------------------------------------------
        gen = QWidget(); gf = QFormLayout(gen)
        self.project_name = QLineEdit(cfg.project_name)
        self.chromadb_path = QLineEdit(str(raw.get("chromadb_path", "")))
        self.output_dir = QLineEdit(str(raw_extraction.get("output_dir", "")))
        self.use_token_chunking = QCheckBox("Use token-aware chunking (slower, more precise)")
        self.use_token_chunking.setChecked(bool(raw_extraction.get("use_token_chunking", False)))
        gf.addRow("Project name:", self.project_name)
        gf.addRow("ChromaDB path:", self.chromadb_path)
        gf.addRow("Extraction output dir:", self.output_dir)
        gf.addRow("", self.use_token_chunking)
        tabs.addTab(gen, "General")

        # ---- LLM -----------------------------------------------------------
        lw = QWidget(); lf = QFormLayout(lw)
        self.llm_base_url = QLineEdit(llm.base_url)
        self.llm_model = QLineEdit(llm.model)
        self.llm_api_key = QLineEdit(llm.api_key or "")
        self.llm_ssl = QLineEdit(llm.ssl_cert_file or "")
        lf.addRow("Base URL:", self.llm_base_url)
        lf.addRow("Model:", self.llm_model)
        lf.addRow("API key:", self.llm_api_key)
        lf.addRow("SSL cert file:", self.llm_ssl)
        tabs.addTab(lw, "LLM")

        # ---- Embedding -----------------------------------------------------
        ew = QWidget(); ef = QFormLayout(ew)
        self.emb_model = QLineEdit(emb.model)
        self.emb_type = QComboBox(); self.emb_type.addItems(["local", "api"])
        self.emb_type.setCurrentText(emb.type or "local")
        self.emb_base_url = QLineEdit(emb.base_url or "")
        self.emb_api_key = QLineEdit(emb.api_key or "")
        ef.addRow("Model:", self.emb_model)
        ef.addRow("Type:", self.emb_type)
        ef.addRow("Base URL (api):", self.emb_base_url)
        ef.addRow("API key (api):", self.emb_api_key)
        note = QLabel("Must stay identical between extraction and chatbot.")
        note.setStyleSheet("color:#888")
        ef.addRow("", note)
        tabs.addTab(ew, "Embedding")

        # ---- Retrieval -----------------------------------------------------
        rw = QWidget(); rf = QFormLayout(rw)
        self.k_standard = self._spin(1, 200, cfg.k_standard)
        self.k_deep_dive = self._spin(1, 200, cfg.k_deep_dive)
        self.deep_dive_batch_size = self._spin(1, 100, cfg.deep_dive_batch_size)
        self.top_n_after_rerank = self._spin(1, 100, cfg.top_n_after_rerank)
        self.temperature = QDoubleSpinBox(); self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.05); self.temperature.setValue(cfg.temperature)
        self.max_tokens = self._spin(64, 32000, cfg.max_tokens, step=64)
        rf.addRow("k_standard:", self.k_standard)
        rf.addRow("k_deep_dive:", self.k_deep_dive)
        rf.addRow("deep_dive_batch_size:", self.deep_dive_batch_size)
        rf.addRow("top_n_after_rerank:", self.top_n_after_rerank)
        rf.addRow("temperature:", self.temperature)
        rf.addRow("max_tokens (answer):", self.max_tokens)
        tabs.addTab(rw, "Retrieval")

        # ---- Reranker ------------------------------------------------------
        rkw = QWidget(); rkf = QFormLayout(rkw)
        self.reranker_enable = QCheckBox("Enable reranker")
        self.reranker_enable.setChecked(cfg.reranker is not None)
        self.reranker_model = QLineEdit(
            cfg.reranker.model if cfg.reranker else "BAAI/bge-reranker-v2-m3")
        self.reranker_type = QComboBox()
        self.reranker_type.addItems(["cross_encoder", "late_interaction"])
        self.reranker_type.setToolTip(
            "cross_encoder: sentence_transformers.CrossEncoder (default).\n"
            "late_interaction: ColBERT-style MaxSim scoring via "
            "sentence_transformers.MultiVectorEncoder (requires sentence-transformers >= 6.0).")
        if cfg.reranker:
            self.reranker_type.setCurrentText(cfg.reranker.type)
        rkf.addRow("", self.reranker_enable)
        rkf.addRow("Model:", self.reranker_model)
        rkf.addRow("Type:", self.reranker_type)
        self.reranker_enable.toggled.connect(
            lambda on: (self.reranker_model.setEnabled(on), self.reranker_type.setEnabled(on)))
        self.reranker_model.setEnabled(self.reranker_enable.isChecked())
        self.reranker_type.setEnabled(self.reranker_enable.isChecked())
        tabs.addTab(rkw, "Reranker")

        # ---- Agentic -------------------------------------------------------
        ag = cfg.agentic
        agw = QWidget(); agf = QFormLayout(agw)
        self.agentic_enable = QCheckBox("Enable agentic mode")
        self.agentic_enable.setChecked(ag is not None)
        self.agentic_page_index = QLineEdit(
            str(raw.get("agentic", {}).get("page_index_path", "")) if isinstance(raw.get("agentic"), dict)
            else "./salome_docs_extracted/page_index.json")
        self.agentic_max_chars = self._spin(500, 50000, ag.max_chars_per_page if ag else 8000, step=500)
        self.agentic_max_steps = self._spin(1, 100, ag.max_steps if ag else 6)
        self.agentic_section_top_n = self._spin(1, 50, ag.section_top_n if ag else 5)
        self.agentic_section_char_budget = self._spin(
            1000, 100000, ag.section_char_budget if ag else 15000, step=1000)
        self.agentic_debug = QCheckBox("Debug (dump agent traces to debug_traces/)")
        self.agentic_debug.setChecked(bool(ag.debug) if ag else False)
        agf.addRow("", self.agentic_enable)
        agf.addRow("page_index.json:", self.agentic_page_index)
        agf.addRow("max_chars_per_page:", self.agentic_max_chars)
        agf.addRow("max_steps:", self.agentic_max_steps)
        agf.addRow("section_top_n:", self.agentic_section_top_n)
        agf.addRow("section_char_budget:", self.agentic_section_char_budget)
        agf.addRow("", self.agentic_debug)
        self._agentic_fields = [self.agentic_page_index, self.agentic_max_chars,
                                self.agentic_max_steps, self.agentic_section_top_n,
                                self.agentic_section_char_budget, self.agentic_debug]
        self.agentic_enable.toggled.connect(
            lambda on: [w.setEnabled(on) for w in self._agentic_fields])
        for w in self._agentic_fields:
            w.setEnabled(self.agentic_enable.isChecked())
        tabs.addTab(agw, "Agentic")

        # ---- Extraction (chunking + quality) -------------------------------
        xw = QWidget(); xf = QFormLayout(xw)
        self.chunk_max_tokens = self._spin(32, 4096, chunking.get("max_tokens", 384))
        self.chunk_overlap_tokens = self._spin(0, 1024, chunking.get("overlap_tokens", 50))
        self.chunk_char_size = self._spin(128, 20000, chunking.get("char_chunk_size", 1000))
        self.chunk_char_overlap = self._spin(0, 5000, chunking.get("char_overlap", 200))
        self.quality_min_score = QDoubleSpinBox(); self.quality_min_score.setRange(0.0, 5.0)
        self.quality_min_score.setSingleStep(0.1); self.quality_min_score.setValue(quality.get("min_score", 0.3))
        self.quality_min_words = self._spin(0, 10000, quality.get("min_word_count", 50))
        self.quality_substantial = self._spin(0, 10000, quality.get("substantial_word_count", 100))
        xf.addRow("chunking.max_tokens:", self.chunk_max_tokens)
        xf.addRow("chunking.overlap_tokens:", self.chunk_overlap_tokens)
        xf.addRow("chunking.char_chunk_size:", self.chunk_char_size)
        xf.addRow("chunking.char_overlap:", self.chunk_char_overlap)
        xf.addRow("quality.min_score:", self.quality_min_score)
        xf.addRow("quality.min_word_count:", self.quality_min_words)
        xf.addRow("quality.substantial_word_count:", self.quality_substantial)
        tabs.addTab(xw, "Extraction")

        # ---- Modules -------------------------------------------------------
        mw = QWidget(); mv = QVBoxLayout(mw)
        self.modules_table = QTableWidget(0, 6)
        self.modules_table.setHorizontalHeaderLabels(
            ["Name", "Description", "dev_path", "user_path", "methodology_path",
             "url_for_sources_citation"])
        self.modules_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for name, m in (raw_extraction.get("modules", {}) or {}).items():
            if not isinstance(m, dict):
                continue
            self._add_module_row(
                name, m.get("description", ""), m.get("dev_path", ""),
                m.get("user_path", ""), m.get("methodology_path", ""),
                self._citation_to_text(m.get("url_for_sources_citation", "")))
        mv.addWidget(self.modules_table)
        mbtns = QHBoxLayout()
        add_btn = QPushButton("Add module"); add_btn.clicked.connect(lambda: self._add_module_row())
        del_btn = QPushButton("Remove selected"); del_btn.clicked.connect(self._remove_module_row)
        mbtns.addWidget(add_btn); mbtns.addWidget(del_btn); mbtns.addStretch()
        mv.addLayout(mbtns)
        tabs.addTab(mw, "Modules")

        # ---- Buttons -------------------------------------------------------
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        save_btn = bb.button(QDialogButtonBox.Save)
        save_btn.setToolTip("Save to " + backend.default_save_path)
        self.save_as_btn = bb.addButton("Save As...", QDialogButtonBox.ActionRole)
        bb.accepted.connect(self._on_save)
        bb.rejected.connect(self.reject)
        self.save_as_btn.clicked.connect(self._on_save_as)
        outer.addWidget(bb)
        dest = QLabel("Save writes to: " + backend.default_save_path)
        dest.setStyleSheet("color:#888")
        dest.setWordWrap(True)
        outer.addWidget(dest)

    # -- small helpers --
    @staticmethod
    def _spin(lo, hi, val, step=1):
        s = QSpinBox(); s.setRange(lo, hi); s.setSingleStep(step)
        s.setValue(int(val)); return s

    def _add_module_row(self, name="", description="", dev_path="", user_path="",
                        methodology_path="", url_for_sources_citation=""):
        r = self.modules_table.rowCount()
        self.modules_table.insertRow(r)
        for c, val in enumerate((name, description, dev_path, user_path,
                                 methodology_path, url_for_sources_citation)):
            self.modules_table.setItem(r, c, QTableWidgetItem(str(val)))

    def _remove_module_row(self):
        r = self.modules_table.currentRow()
        if r >= 0:
            self.modules_table.removeRow(r)

    def _cell(self, row, col):
        item = self.modules_table.item(row, col)
        return item.text().strip() if item else ""

    @staticmethod
    def _citation_to_text(v):
        """url_for_sources_citation may be a plain string or a {"dev": ..., "user": ...}
        mapping (see raglib/README.md) — render a dict as compact JSON so it
        round-trips through the table cell; a string passes through as-is."""
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
        return str(v) if v else ""

    @staticmethod
    def _citation_from_text(text):
        """Inverse of _citation_to_text: '{...}' parses back to a dict, anything
        else is kept as a plain string; empty text omits the key entirely."""
        text = text.strip()
        if not text:
            return None
        if text.startswith("{"):
            try:
                return json.loads(text)
            except ValueError:
                pass  # not valid JSON -- fall through and keep it as a literal string
        return text

    def build_config(self):
        """Assemble the {"chatbot": ..., "extraction": ...} dict from the
        widgets — each section following raglib's own native config schema
        for that stage — preserving any comment keys present in each
        previously loaded section."""
        chatbot = copy.deepcopy(self.backend._raw_chatbot_config) if self.backend._raw_chatbot_config else {}
        extraction = copy.deepcopy(self.backend._raw_extraction_config) if self.backend._raw_extraction_config else {}

        project_name = self.project_name.text().strip()
        chatbot["project_name"] = project_name
        extraction["project_name"] = project_name
        chatbot["chromadb_path"] = self.chromadb_path.text().strip()
        extraction["output_dir"] = self.output_dir.text().strip()
        extraction["use_token_chunking"] = self.use_token_chunking.isChecked()

        llm = chatbot.get("llm") if isinstance(chatbot.get("llm"), dict) else {}
        llm["base_url"] = self.llm_base_url.text().strip()
        llm["model"] = self.llm_model.text().strip()
        llm["api_key"] = self.llm_api_key.text().strip()
        llm["ssl_cert_file"] = self.llm_ssl.text().strip() or None
        chatbot["llm"] = llm

        emb = chatbot.get("embedding") if isinstance(chatbot.get("embedding"), dict) else {}
        emb["model"] = self.emb_model.text().strip()
        emb["type"] = self.emb_type.currentText()
        emb["base_url"] = self.emb_base_url.text().strip() or None
        emb["api_key"] = self.emb_api_key.text().strip() or None
        chatbot["embedding"] = emb
        # embedding.model must be identical between the two configs (see
        # raglib/README.md) — the dialog only exposes one set of embedding
        # fields, so mirror it into the extraction config too.
        extraction["embedding"] = dict(emb)

        chatbot["k_standard"] = self.k_standard.value()
        chatbot["k_deep_dive"] = self.k_deep_dive.value()
        chatbot["deep_dive_batch_size"] = self.deep_dive_batch_size.value()
        chatbot["top_n_after_rerank"] = self.top_n_after_rerank.value()
        chatbot["temperature"] = round(self.temperature.value(), 4)
        chatbot["max_tokens"] = self.max_tokens.value()

        if self.reranker_enable.isChecked():
            chatbot["reranker"] = {"model": self.reranker_model.text().strip(),
                                   "type": self.reranker_type.currentText()}
        else:
            chatbot["reranker"] = None

        if self.agentic_enable.isChecked():
            chatbot["agentic"] = {
                "page_index_path": self.agentic_page_index.text().strip(),
                "max_chars_per_page": self.agentic_max_chars.value(),
                "max_steps": self.agentic_max_steps.value(),
                "section_top_n": self.agentic_section_top_n.value(),
                "section_char_budget": self.agentic_section_char_budget.value(),
                "debug": self.agentic_debug.isChecked(),
            }
        else:
            chatbot["agentic"] = None

        extraction["chunking"] = {
            "max_tokens": self.chunk_max_tokens.value(),
            "overlap_tokens": self.chunk_overlap_tokens.value(),
            "char_chunk_size": self.chunk_char_size.value(),
            "char_overlap": self.chunk_char_overlap.value(),
        }
        extraction["quality"] = {
            "min_score": round(self.quality_min_score.value(), 4),
            "min_word_count": self.quality_min_words.value(),
            "substantial_word_count": self.quality_substantial.value(),
        }

        modules = {}
        for r in range(self.modules_table.rowCount()):
            name = self._cell(r, 0)
            if not name:
                continue
            module = {
                "description": self._cell(r, 1),
                "dev_path": self._cell(r, 2),
                "user_path": self._cell(r, 3),
            }
            methodology_path = self._cell(r, 4)
            if methodology_path:
                module["methodology_path"] = methodology_path
            citation = self._citation_from_text(self._cell(r, 5))
            if citation:
                module["url_for_sources_citation"] = citation
            modules[name] = module
        extraction["modules"] = modules
        return {"chatbot": chatbot, "extraction": extraction}

    def _save_to(self, path):
        try:
            data = self.build_config()
            self.backend.save_and_reload(path, data)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return False
        return True

    def _on_save(self):
        # Always persist to the standard user location so we never overwrite
        # the bundled config.example.json (which may be the one currently
        # loaded).
        if self._save_to(self.backend.default_save_path):
            self.accept()

    def _on_save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config.json", self.backend.default_save_path,
            "JSON files (*.json);;All files (*)")
        if not path:
            return
        if self._save_to(path):
            self.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SALOME Assistant (RAG / Agentic)")
        self.resize(1000, 720)

        try:
            self.backend = RAGBackend()
        except Exception as e:
            raise EnvironmentError(f"Failed to initialize SALOME Assistant backend: {e}")

        self.robot_icon_path = os.path.join(os.path.dirname(__file__), "salome.jpg")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        main_hsplit = QHBoxLayout()
        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        cfg = self.backend.config

        # --- Connection settings -------------------------------------------
        conn_group = QGroupBox("Connection")
        conn_form = QFormLayout()
        self.config_path_label = QLabel(self.backend.config_path or "(raglib defaults)")
        self.config_path_label.setWordWrap(True)
        self.edit_config_btn = QPushButton("Edit configuration...")
        self.edit_config_btn.clicked.connect(self.handle_edit_config)

        self.base_url_input = QLineEdit(cfg.llm.base_url)
        self.model_input = QLineEdit(cfg.llm.model)
        self.api_key_input = QLineEdit(cfg.llm.api_key or "")
        self.api_key_input.setEchoMode(QLineEdit.PasswordEchoOnEdit)

        conn_form.addRow("Config:", self.config_path_label)
        conn_form.addRow("", self.edit_config_btn)
        conn_form.addRow("LLM base URL:", self.base_url_input)
        conn_form.addRow("Model:", self.model_input)
        conn_form.addRow("API key:", self.api_key_input)
        conn_group.setLayout(conn_form)
        left_panel.addWidget(conn_group)

        # --- Mode -----------------------------------------------------------
        mode_group = QGroupBox("Mode")
        mode_vlayout = QVBoxLayout()
        mode_layout = QHBoxLayout()
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["RAG", "Agentic"])
        self.mode_selector.currentTextChanged.connect(self.update_mode_panels)
        mode_layout.addWidget(self.mode_selector)
        mode_vlayout.addLayout(mode_layout)
        # Explains why "Agentic" is greyed out (no 'agentic' config block,
        # deepagents not installed, or page_index.json missing) — set by
        # _set_agentic_available() after each Connect attempt. Hidden while
        # agentic is available or before the first Connect.
        self.agentic_unavailable_label = QLabel("")
        self.agentic_unavailable_label.setWordWrap(True)
        self.agentic_unavailable_label.setStyleSheet("color: #b35c00; font-size: 11px;")
        self.agentic_unavailable_label.setVisible(False)
        mode_vlayout.addWidget(self.agentic_unavailable_label)
        mode_group.setLayout(mode_vlayout)
        left_panel.addWidget(mode_group)

        # --- RAG parameters -------------------------------------------------
        self.rag_group = QGroupBox("RAG parameters")
        rag_form = QFormLayout()

        self.module_selector = QComboBox()
        self.module_selector.addItem("All")

        self.doc_type_selector = QComboBox()
        self.doc_type_selector.addItems(["All", "Dev", "User"])

        self.k_input = QSpinBox()
        self.k_input.setRange(5, 120)
        self.k_input.setValue(cfg.k_standard)

        self.reranker_checkbox = QCheckBox("Enable reranker")
        self.reranker_checkbox.setEnabled(False)  # enabled after init if available

        self.top_n_input = QSpinBox()
        self.top_n_input.setRange(1, 40)
        self.top_n_input.setValue(cfg.top_n_after_rerank)

        self.style_selector = QComboBox()
        self.style_selector.addItems(list(RESPONSE_STYLES.keys()))

        self.deep_dive_checkbox = QCheckBox("Deep dive mode")

        rag_form.addRow("Module:", self.module_selector)
        rag_form.addRow("Doc type:", self.doc_type_selector)
        rag_form.addRow("Search depth (k):", self.k_input)
        rag_form.addRow("", self.reranker_checkbox)
        rag_form.addRow("Top-N after rerank:", self.top_n_input)
        rag_form.addRow("Response style:", self.style_selector)
        rag_form.addRow("", self.deep_dive_checkbox)
        self.rag_group.setLayout(rag_form)
        left_panel.addWidget(self.rag_group)

        # --- Agentic parameters --------------------------------------------
        # max_steps, temperature and max_chars_per_page are all real
        # per-query overrides in raglib's AgenticChatbot.ask() — each is
        # re-read on every call (tools are rebuilt per-ask with whatever
        # max_chars_per_page was passed, not fixed at Connect time). debug
        # (agent trace dumping) is intentionally left config-only (see
        # ConfigDialog's Agentic tab) as a persistent developer setting
        # rather than a per-question toggle.
        self.agentic_group = QGroupBox("Agentic parameters")
        agentic_form = QFormLayout()
        ag = cfg.agentic

        self.max_steps_input = QSpinBox()
        self.max_steps_input.setRange(1, 99)
        self.max_steps_input.setValue(ag.max_steps if ag else 10)

        self.agentic_style_selector = QComboBox()
        self.agentic_style_selector.addItems(list(RESPONSE_STYLES.keys()))

        self.agentic_max_chars_input = QSpinBox()
        self.agentic_max_chars_input.setRange(500, 50000)
        self.agentic_max_chars_input.setSingleStep(500)
        self.agentic_max_chars_input.setValue(ag.max_chars_per_page if ag else 8000)

        agentic_form.addRow("Max steps:", self.max_steps_input)
        agentic_form.addRow("Response style:", self.agentic_style_selector)
        agentic_form.addRow("max_chars_per_page:", self.agentic_max_chars_input)
        self.agentic_group.setLayout(agentic_form)
        left_panel.addWidget(self.agentic_group)

        # --- Shared generation param ---------------------------------------
        shared_group = QGroupBox("Generation")
        shared_form = QFormLayout()
        self.max_tokens_input = QSpinBox()
        # Matches ConfigDialog's own max_tokens ceiling (see self.max_tokens
        # there) — kept in sync so both accept the same range. A max whose
        # leading digit is low (e.g. the previous 8192) makes QSpinBox reject
        # keystrokes for any typed value starting with a higher digit before
        # the full number is even entered (e.g. typing "9000" gets stuck at
        # "999"), since Qt validates digit-by-digit as you type.
        self.max_tokens_input.setRange(256, 32000)
        self.max_tokens_input.setSingleStep(256)
        self.max_tokens_input.setValue(cfg.max_tokens)
        shared_form.addRow("Max answer tokens:", self.max_tokens_input)
        shared_group.setLayout(shared_form)
        left_panel.addWidget(shared_group)

        # --- Action buttons -------------------------------------------------
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.handle_connect)
        self.build_btn = QPushButton("Build index...")
        self.build_btn.clicked.connect(self.handle_build_index)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.connect_btn)
        btn_row.addWidget(self.build_btn)
        left_panel.addLayout(btn_row)
        left_panel.addStretch()

        # --- Chat area (right) ---------------------------------------------
        self.chat_tabs = QTabWidget()
        self.chat_tabs.setTabsClosable(True)
        self.chat_tabs.tabCloseRequested.connect(self._close_browser_tab)

        self.chat_display = QWebEngineView()
        self.chat_display.setPage(
            ChatWebEnginePage(self._handle_chat_link, self.chat_display))
        self.chat_display.settings().setAttribute(
            QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        self.chat_display.loadFinished.connect(self._scroll_chat_to_bottom)
        self._chat_base_url = QUrl.fromLocalFile(
            os.path.dirname(os.path.abspath(__file__)) + os.sep)
        self._chat_fragments = []
        self._code_blocks = {}
        self._code_block_counter = 0
        self._render_chat()

        self.chat_tabs.addTab(self.chat_display, "Chat")
        # The Chat tab itself is permanent — hide its close button.
        self.chat_tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)
        self.chat_tabs.tabBar().setTabButton(0, QTabBar.LeftSide, None)

        right_panel.addWidget(self.chat_tabs)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        main_hsplit.addWidget(left_widget, 1)
        main_hsplit.addWidget(right_widget, 2)
        self.layout.addLayout(main_hsplit)

        # --- Status ---------------------------------------------------------
        self.status_label = QLabel("Set the LLM endpoint and click Connect.")
        self.layout.addWidget(self.status_label)

        # --- Input area -----------------------------------------------------
        input_layout = QHBoxLayout()
        self.input_field = HistoryLineEdit()
        self.input_field.setPlaceholderText("Connect first, then ask a question...")
        self.input_field.returnPressed.connect(self.handle_ask)
        self.input_field.setEnabled(False)

        self.send_button = QPushButton("Ask")
        self.send_button.clicked.connect(self.handle_ask)
        self.send_button.setEnabled(False)

        self._thinking_timer = QTimer(self)
        self._thinking_timer.setInterval(1000)
        self._thinking_timer.timeout.connect(self._update_thinking_elapsed)
        self._thinking_start = None
        self._thinking_prefix = "Thinking..."

        input_layout.addWidget(self.input_field, 1)
        input_layout.addWidget(self.send_button)
        input_layout.setAlignment(Qt.AlignTop)
        try:
            btn_h = self.input_field.sizeHint().height()
            self.send_button.setFixedHeight(btn_h)
        except Exception:
            pass
        self.layout.addLayout(input_layout)

        # --- Bottom controls ------------------------------------------------
        self.help_btn = QPushButton("Help")
        self.help_btn.clicked.connect(self.show_help)
        self.quit_btn = QPushButton("Quit")
        self.quit_btn.clicked.connect(self.close)
        bottom_controls = QHBoxLayout()
        bottom_controls.addWidget(self.help_btn)
        bottom_controls.addStretch()
        bottom_controls.addWidget(self.quit_btn)
        self.layout.addLayout(bottom_controls)

        self.update_mode_panels()
        # Agentic is selectable right away — setup (deepagents, extraction) is
        # the user's own responsibility, not something the UI should gate on.
        self._set_agentic_available(True)

    # ------------------------------------------------------------------ helpers
    def _set_agentic_available(self, available, reason=None):
        """Keep the Agentic entry in the mode selector always selectable —
        the user manages their own setup (installing deepagents, building
        page_index.json), so the UI shouldn't block picking the mode based on
        auto-detected state. `available`/`reason` only drive a non-blocking
        heads-up (tooltip + label under the selector) when something looks
        off after a Connect attempt; asking a question still surfaces a clear
        error from the backend if agentic truly isn't usable (see
        RAGBackend.ask())."""
        model = self.mode_selector.model()
        item = model.item(1)  # index 1 == "Agentic"
        if item is not None:
            item.setEnabled(True)
            item.setToolTip("" if available else (reason or "Agentic mode may be unavailable."))
        if available or not reason:
            self.agentic_unavailable_label.setVisible(False)
            self.agentic_unavailable_label.setText("")
        else:
            self.agentic_unavailable_label.setText(f"Agentic warning: {reason}")
            self.agentic_unavailable_label.setVisible(True)

    def update_mode_panels(self, *_):
        is_rag = self.mode_selector.currentText() == "RAG"
        self.rag_group.setVisible(is_rag)
        self.agentic_group.setVisible(not is_rag)

    def toggle_inputs(self, enabled):
        self.input_field.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        self.connect_btn.setEnabled(enabled)
        self.build_btn.setEnabled(enabled)
        self.mode_selector.setEnabled(enabled)

    # ------------------------------------------------------------------ config
    def handle_edit_config(self):
        """Open the full config editor; refresh the main panel on save."""
        dlg = ConfigDialog(self.backend, self)
        if dlg.exec_() == QDialog.Accepted:
            self._apply_config_to_widgets("Config saved. Click Connect.")

    def _apply_config_to_widgets(self, status_msg):
        """Refresh the main-panel controls from backend.config after a config
        change, and reset the connection state (the chatbots were dropped)."""
        cfg = self.backend.config
        self.config_path_label.setText(self.backend.config_path or "(raglib defaults)")
        self.base_url_input.setText(cfg.llm.base_url)
        self.model_input.setText(cfg.llm.model)
        self.api_key_input.setText(cfg.llm.api_key or "")
        self.k_input.setValue(cfg.k_standard)
        self.top_n_input.setValue(cfg.top_n_after_rerank)
        self.max_tokens_input.setValue(cfg.max_tokens)
        if cfg.agentic:
            self.max_steps_input.setValue(cfg.agentic.max_steps)
            self.agentic_max_chars_input.setValue(cfg.agentic.max_chars_per_page)
        self.module_selector.clear()
        self.module_selector.addItem("All")
        self.reranker_checkbox.setEnabled(False)
        self.reranker_checkbox.setChecked(False)
        self._set_agentic_available(False)
        self.toggle_inputs(True)
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.status_label.setText(status_msg)

    # ------------------------------------------------------------------ connect
    def handle_connect(self):
        self.toggle_inputs(False)
        self.status_label.setText("Connecting...")
        self.worker = Worker(self.backend, 'init', params={
            'base_url': self.base_url_input.text().strip(),
            'model': self.model_input.text().strip(),
            'api_key': self.api_key_input.text().strip(),
        })
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_connect_finished)
        self.worker.failed.connect(self.on_task_failed)
        self.worker.start()

    def on_connect_finished(self, msg):
        self.connect_btn.setEnabled(True)
        self.build_btn.setEnabled(True)
        self.mode_selector.setEnabled(True)
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.status_label.setText("Ready")
        self.input_field.setFocus()

        # Populate modules detected in the database
        self.module_selector.clear()
        self.module_selector.addItem("All")
        for m in self.backend.available_modules:
            self.module_selector.addItem(m)

        # Reranker availability
        has_reranker = self.backend.has_reranker
        self.reranker_checkbox.setEnabled(has_reranker)
        self.reranker_checkbox.setChecked(has_reranker)

        # Agentic availability
        self._set_agentic_available(self.backend.has_agentic, reason=self.backend.agentic_error)

        self._append_chat(f"<span style='color:green'>System: {self._nl2br(msg)}</span>")

    def on_task_failed(self, err):
        self.toggle_inputs(True)
        self.input_field.setEnabled(self.backend.initialized)
        self.send_button.setEnabled(self.backend.initialized)
        self.status_label.setText("Error")
        try:
            self._stop_thinking_timer()
        except Exception:
            pass
        QMessageBox.critical(self, "Error", err)
        self._append_chat(f"<span style='color:red'>Error: {self._nl2br(err)}</span>")

    # -------------------------------------------------------------------- query
    def handle_ask(self):
        query = self.input_field.text().strip()
        if not query:
            return
        if not self.backend.initialized:
            QMessageBox.information(self, "Not connected", "Click Connect first.")
            return

        self._append_chat(f"<b>You:</b> {query}")
        self.input_field.add_history(query)
        self.input_field.clear()
        self.toggle_inputs(False)
        self._start_thinking_timer("Thinking...")

        mode = self.mode_selector.currentText().lower()
        params = {'query': query, 'mode': mode,
                  'max_tokens': self.max_tokens_input.value()}
        if mode == 'rag':
            style = RESPONSE_STYLES.get(self.style_selector.currentText(), {})
            params.update({
                'module': self.module_selector.currentText(),
                'doc_type': self.doc_type_selector.currentText(),
                'k': self.k_input.value(),
                'temperature': style.get('temperature'),
                'deep_dive': self.deep_dive_checkbox.isChecked(),
                'reranker_enabled': self.reranker_checkbox.isChecked(),
                'top_n': self.top_n_input.value(),
            })
        else:
            agentic_style = RESPONSE_STYLES.get(self.agentic_style_selector.currentText(), {})
            params.update({
                'max_steps': self.max_steps_input.value(),
                'temperature': agentic_style.get('temperature'),
                'max_chars_per_page': self.agentic_max_chars_input.value(),
            })

        self.worker = Worker(self.backend, 'query', params=params)
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_query_finished)
        self.worker.failed.connect(self.on_task_failed)
        self.worker.start()

    def on_query_finished(self, result):
        try:
            self._stop_thinking_timer()
        except Exception:
            pass
        self.toggle_inputs(True)
        self.status_label.setText("Ready")
        self.input_field.setFocus()

        if result.get('error'):
            self._append_chat(
                f"<span style='color:red'>Error: {self._nl2br(result['error'])}</span>")
            # Even on a failed/empty answer, the agent may have retrieved
            # useful pages before giving up — show them so the user isn't
            # left with nothing to go on.
            html_sources = self._render_sources(result.get('sources', []),
                                                result.get('filters', {}))
            if html_sources:
                self._append_chat(html_sources)
            self._append_chat("-" * 30)
            return

        answer = result.get('answer') or "(empty answer)"
        html_answer = self._render_answer(answer)
        html_answer += self._render_sources(result.get('sources', []),
                                            result.get('filters', {}))

        img_html = ""
        try:
            if os.path.exists(self.robot_icon_path):
                img_url = f"file://{self.robot_icon_path}"
                img_html = (f"<img src=\"{img_url}\" width=\"24\" height=\"24\" "
                            f"style=\"vertical-align:middle;margin-right:8px\"/>")
        except Exception:
            img_html = ""

        label = self.mode_selector.currentText()
        self._append_chat(f"<b>Bot ({label}):</b> {img_html}{html_answer}")
        self._append_chat("-" * 30)

    # ------------------------------------------------------------------ extract
    def handle_build_index(self):
        # The index is built from config.json's "extraction" section —
        # raglib's own separate schema for process_docs.py (see raglib/README.md).
        if not self.backend.config_path:
            QMessageBox.information(
                self, "Build index",
                "No config.json is loaded. Use 'Edit configuration...' first.")
            return
        if not self.backend.extraction_modules:
            QMessageBox.warning(
                self, "Build index",
                "The config's 'extraction' section has no 'modules' to extract.\n"
                "Add modules (with dev_path / user_path) to the config, then retry.")
            return
        confirm = QMessageBox.question(
            self, "Build index",
            "Run the extraction pipeline using the loaded config to (re)build "
            "the vector database?\n\n"
            f"Config: {self.backend.config_path}\n\n"
            "This can take a while depending on the documentation size.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        self.toggle_inputs(False)
        self.status_label.setText("Building index...")
        self._append_chat(
            "<span style='color:#888'>--- Building vector database ---</span>")

        self.extract_worker = ExtractionWorker(self.backend)
        self.extract_worker.line.connect(self._on_extract_line)
        self.extract_worker.finished.connect(self._on_extract_finished)
        self.extract_worker.start()

    def _on_extract_line(self, line):
        self._append_chat(f"<span style='color:#888'>{self._escape(line)}</span>")

    def _on_extract_finished(self):
        self.toggle_inputs(True)
        self.status_label.setText("Index build finished. Click Connect to load it.")
        self._append_chat(
            "<span style='color:#888'>--- Done. Click Connect to load the new index. ---</span>")

    # ----------------------------------------------------------------- rendering
    def _append_chat(self, html_fragment):
        """Add one HTML fragment (a "You:"/"Bot:"/status line) to the chat log,
        matching QTextEdit.append()'s one-call-per-block behavior."""
        self._chat_fragments.append(html_fragment)
        self._render_chat()

    def _render_chat(self):
        body = "".join(f'<div class="msg">{frag}</div>' for frag in self._chat_fragments)
        html = (
            "<html><head><meta charset='utf-8'><style>"
            "body{font-family:sans-serif;font-size:14px;margin:8px;}"
            ".msg{margin:4px 0;}"
            "pre{white-space:pre-wrap;background:#f6f8fa;padding:8px;border-radius:4px;}"
            ".code-toolbar{margin:4px 0 0 0;}"
            ".run-btn{display:inline-block;font-size:12px;padding:2px 8px;"
            "border:1px solid #0a7c3b;border-radius:4px;color:#0a7c3b;"
            "text-decoration:none;background:#eafbea;}"
            ".run-btn:hover{background:#d3f5d3;}"
            "</style></head><body>" + body + "</body></html>"
        )
        self.chat_display.setHtml(html, self._chat_base_url)

    def _scroll_chat_to_bottom(self, ok=True):
        self.chat_display.page().runJavaScript(
            "window.scrollTo(0, document.body.scrollHeight);")

    # --------------------------------------------------------- run in SALOME
    def _handle_chat_link(self, url):
        if url.scheme() == 'salome-run':
            try:
                block_id = int(url.path().lstrip('/'))
            except ValueError:
                return
            self._run_code_block(block_id)
        else:
            self._open_link_in_new_tab(url)

    def _run_code_block(self, block_id):
        code = self._code_blocks.get(block_id)
        if code is None:
            return
        self._append_chat(
            "<span style='color:#888'><i>Running code in SALOME...</i></span>")
        self._run_worker = SalomeRunWorker(code)
        self._run_worker.finished.connect(self._on_run_code_finished)
        self._run_worker.failed.connect(self._on_run_code_failed)
        self._run_worker.start()

    def _on_run_code_finished(self, result):
        self._append_chat(self._format_run_result(result))

    def _on_run_code_failed(self, err):
        self._append_chat(
            f"<span style='color:red'><b>SALOME:</b> {self._nl2br(err)}</span>")

    @staticmethod
    def _format_run_result(result):
        out_style = "background:#f6f8fa;padding:8px;border-radius:4px;white-space:pre-wrap;"
        err_style = "color:#b00;background:#fff5f5;padding:8px;border-radius:4px;white-space:pre-wrap;"
        parts = []
        if result.get('stdout'):
            parts.append(f"<pre style='{out_style}'>{MainWindow._escape(result['stdout'])}</pre>")
        if result.get('result') is not None:
            parts.append(f"&rarr; <code>{MainWindow._escape(result['result'])}</code>")
        if result.get('stderr'):
            parts.append(f"<pre style='{err_style}'>{MainWindow._escape(result['stderr'])}</pre>")
        if result.get('error'):
            parts.append(f"<pre style='{err_style}'>{MainWindow._escape(result['error'])}</pre>")
        if not parts:
            parts.append("<i>(no output)</i>")
        return "<b>SALOME:</b> " + "".join(parts)

    # ------------------------------------------------------------- browser tabs
    def _open_link_in_new_tab(self, url):
        """Open a clicked link in a new tab next to the Chat tab."""
        view = QWebEngineView()
        view.setPage(ChatWebEnginePage(self._handle_chat_link, view))
        view.titleChanged.connect(
            lambda title, v=view: self._update_tab_title(v, title))
        view.load(url)
        index = self.chat_tabs.addTab(view, url.toString())
        self.chat_tabs.setCurrentIndex(index)

    def _update_tab_title(self, view, title):
        index = self.chat_tabs.indexOf(view)
        if index > 0:  # never rename the permanent Chat tab
            self.chat_tabs.setTabText(index, title or "Untitled")

    def _close_browser_tab(self, index):
        if index == 0:
            return  # the Chat tab can't be closed
        widget = self.chat_tabs.widget(index)
        self.chat_tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()

    @staticmethod
    def _escape(text):
        return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    @staticmethod
    def _nl2br(text):
        return MainWindow._escape(text).replace("\n", "<br>")

    _FENCE_RE = re.compile(r'```([a-zA-Z0-9_+-]*)\n(.*?)```', re.DOTALL)
    _RUNNABLE_LANGS = {'', 'python', 'python3', 'py'}
    _PRE_OPEN_RE = re.compile(r'(<div class="codehilite">\s*)?<pre>(<code[^>]*>)?')

    def _render_answer(self, answer):
        """Render an LLM answer (Markdown) to styled HTML, adding a
        "Run in SALOME" link before each runnable (Python/untagged) fenced
        code block — clicking it round-trips through salome_mcp_client to
        the SALOME session's PyConsole interpreter (see _handle_chat_link).

        Fenced blocks are extracted from the raw markdown text (in the
        order they appear) and matched by position to the <pre> tags
        markdown produces for them, since that's the only way to recover
        the exact original code from python-markdown's rendered HTML.
        Note this assumes every rendered <pre> that appears before the
        last fenced block came from one — a stray indented (non-fenced)
        code block earlier in the answer would throw off that alignment.
        """
        fences = [(m.group(1).strip().lower(), m.group(2))
                  for m in self._FENCE_RE.finditer(answer)]
        html_answer = markdown.markdown(
            answer, extensions=['fenced_code', 'codehilite', 'nl2br'], output_format='html')
        style = ('background:#f6f8fa;padding:8px;border-radius:4px;white-space:pre-wrap;')

        counter = {'i': 0}

        def _inject(match):
            i = counter['i']
            counter['i'] += 1
            wrapper = match.group(1) or ""
            code_open = match.group(2) or ""
            toolbar = ""
            if i < len(fences) and fences[i][0] in self._RUNNABLE_LANGS:
                block_id = self._code_block_counter
                self._code_block_counter += 1
                self._code_blocks[block_id] = fences[i][1]
                toolbar = (f'<div class="code-toolbar">'
                          f'<a href="salome-run:///{block_id}" class="run-btn">'
                          f'&#9654; Run in SALOME</a></div>')
            return f'{wrapper}{toolbar}<pre style="{style}">{code_open}'

        html_answer = self._PRE_OPEN_RE.sub(_inject, html_answer)
        return html_answer

    def _render_sources(self, sources, filters):
        if not sources:
            return ""
        rows = []
        for i, s in enumerate(sources[:5], 1):
            title = self._escape(str(s.get('title', 'Unknown')))
            mod = self._escape(str(s.get('module', '?')))
            cat = self._escape(str(s.get('doc_category', '?')))
            url = s.get('url') or s.get('filepath') or ''
            if url:
                link = f'<a href="{self._escape(str(url))}">{title}</a>'
            else:
                link = title
            rows.append(f"{i}. [{mod}/{cat}] {link}")
        note = ""
        if filters.get('deep_dive'):
            note = "<br><i>Generated with Deep Dive mode.</i>"
        elif filters.get('mode') == 'agentic':
            note = (f"<br><i>Agentic mode — "
                    f"{filters.get('rounds_used', 1)} round(s).</i>")
        return "<br><br><b>Sources:</b><br>" + "<br>".join(rows) + note

    def show_help(self):
        about_help_text = (
            "<b>SALOME RAG Assistant</b><br><br>"
            "Copyright: CEA 2025.<br>"
            "License: LGPL V2.1<br><br>"
            "1. Set the LLM endpoint (an OpenAI-compatible server, e.g. Ollama) "
            "and click <b>Connect</b>.<br>"
            "2. Choose <b>RAG</b> or <b>Agentic</b> mode and tune the parameters.<br>"
            "3. Type a question and press <b>Ask</b>.<br><br>"
            "Use <b>Build index...</b> to (re)build the vector database from an "
            "extraction config.json."
        )
        QMessageBox.information(self, "Help / About", about_help_text)

    # --- Thinking timer helpers ---
    def _start_thinking_timer(self, prefix="Thinking..."):
        self._thinking_prefix = prefix
        self._thinking_start = time.monotonic()
        self._update_thinking_elapsed()
        self._thinking_timer.start()

    def _stop_thinking_timer(self):
        if self._thinking_timer.isActive():
            self._thinking_timer.stop()
        self._thinking_start = None

    def _update_thinking_elapsed(self):
        if not self._thinking_start:
            return
        elapsed = int(time.monotonic() - self._thinking_start)
        m = elapsed // 60
        s = elapsed % 60
        self.status_label.setText(f"{self._thinking_prefix} {m:02d}:{s:02d}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QGroupBox { font-weight: bold; }
        QLineEdit { padding: 5px; font-size: 14px; }
    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
