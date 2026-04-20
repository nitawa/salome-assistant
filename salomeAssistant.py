#!/bin/env python3
"""
SALOME RAG Assistant — PyQt5 GUI wrapper over the raglib library.

- Uses raglib.chatbot.core.DocumentationChatbot as the RAG engine.
- Extraction and chatbot settings are edited via the Configure menu,
  which round-trips raglib/extraction/config.json and raglib/chatbot/config.json.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)

import markdown

HERE = Path(__file__).resolve().parent

import raglib  # noqa: E402
RAGLIB_DIR = Path(raglib.__file__).resolve().parent
RAGLIB_EXTRACTION_DIR = RAGLIB_DIR / "extraction"
RAGLIB_CHATBOT_DIR = RAGLIB_DIR / "chatbot"
EXTRACTION_CONFIG_PATH = HERE / "salome_assistant_assets" / "salome.documentation.config.json"
CHATBOT_CONFIG_PATH = RAGLIB_CHATBOT_DIR / "config.json"

from raglib.chatbot.core import ChatbotConfig, DocumentationChatbot  # noqa: E402


# --- config I/O ---------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


# --- Workers ------------------------------------------------------------------

class InitWorker(QThread):
    finished_ok = pyqtSignal(object, list, str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, config_path: Path):
        super().__init__()
        self.config_path = config_path

    def _run_and_stream(self, args, **kwargs):
        """Run a command and stream its output to stdout while capturing it."""
        kwargs.setdefault("stdout", subprocess.PIPE)
        kwargs.setdefault("stderr", subprocess.STDOUT)
        kwargs.setdefault("text", True)
        kwargs.setdefault("bufsize", 1)

        full_output = []
        with subprocess.Popen(args, **kwargs) as proc:
            if proc.stdout:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    full_output.append(line)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError("".join(full_output))

    def run(self):
        try:
            # 1. Download models
            self.progress.emit("Downloading/Checking models (sentence-transformers)...")
            download_script = RAGLIB_DIR / "download_models.py"
            # We run it with HF_HUB_OFFLINE=0 to allow downloading if needed
            env = os.environ.copy()
            env["HF_HUB_OFFLINE"] = "0"
            env["TRANSFORMERS_OFFLINE"] = "0"
            try:
                self._run_and_stream([sys.executable, str(download_script)], env=env)
            except RuntimeError as e:
                raise RuntimeError(f"Download models failed:\n{str(e)}")

            # 2. Extract documentation
            self.progress.emit("Extracting documentation (this may take a while)...")
            
            # Expand environment variables in the extraction config
            if not EXTRACTION_CONFIG_PATH.exists():
                raise FileNotFoundError(f"Extraction config not found at {EXTRACTION_CONFIG_PATH}")
            
            with open(EXTRACTION_CONFIG_PATH, "r") as f:
                extract_cfg = json.load(f)
            
            def expand_recursive(d):
                if isinstance(d, dict):
                    return {k: expand_recursive(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [expand_recursive(x) for x in d]
                elif isinstance(d, str):
                    return os.path.expandvars(d)
                return d
            
            expanded_cfg = expand_recursive(extract_cfg)
            temp_config_path = RAGLIB_EXTRACTION_DIR / "temp_salome_config.json"
            with open(temp_config_path, "w") as f:
                json.dump(expanded_cfg, f, indent=2)
            
            extract_script = RAGLIB_EXTRACTION_DIR / "process_docs.py"
            try:
                self._run_and_stream(
                    [sys.executable, str(extract_script), "--config", str(temp_config_path)],
                    cwd=str(RAGLIB_EXTRACTION_DIR)
                )
            except RuntimeError as e:
                raise RuntimeError(f"Documentation extraction failed:\n{str(e)}")

            # 3. Load chatbot configuration and initialize
            self.progress.emit("Loading chatbot configuration...")
            config = ChatbotConfig.load(str(self.config_path))

            # DocumentationChatbot resolves relative chromadb_path against
            # raglib/chatbot/, matching the CLI's working-directory convention.
            cwd_saved = os.getcwd()
            os.chdir(RAGLIB_CHATBOT_DIR)
            try:
                self.progress.emit("Initializing vector DB + LLM...")
                bot = DocumentationChatbot(config)
            finally:
                os.chdir(cwd_saved)

            self.finished_ok.emit(bot, list(bot.available_modules), config.llm.model)
        except Exception as e:
            self.failed.emit(str(e))


class QueryWorker(QThread):
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, bot, question, module, doc_type, deep_dive):
        super().__init__()
        self.bot = bot
        self.question = question
        self.module = module
        self.doc_type = doc_type
        self.deep_dive = deep_dive

    def run(self):
        try:
            result = self.bot.ask(
                self.question,
                module=self.module,
                doc_type=self.doc_type,
                deep_dive=self.deep_dive,
            )
            self.finished_ok.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class TTSWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        try:
            os.system('espeak-ng "{}"'.format(self.text.replace('"', '\\"')))
        except Exception:
            pass
        finally:
            try:
                self.finished.emit()
            except Exception:
                pass


# --- Configure dialog ---------------------------------------------------------

class ModulesTable(QTableWidget):
    """Editable table of extraction modules."""
    COLUMNS = ["Name", "Description", "dev_path", "user_path"]

    def __init__(self, modules: dict, parent=None):
        super().__init__(0, len(self.COLUMNS), parent)
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.load(modules or {})

    def load(self, modules: dict) -> None:
        self.setRowCount(0)
        for name, cfg in modules.items():
            self._add_row(
                name,
                cfg.get("description", ""),
                cfg.get("dev_path", ""),
                cfg.get("user_path", ""),
            )

    def _add_row(self, name="", description="", dev_path="", user_path=""):
        row = self.rowCount()
        self.insertRow(row)
        for col, value in enumerate((name, description, dev_path, user_path)):
            self.setItem(row, col, QTableWidgetItem(value))

    def add_empty_row(self):
        self._add_row()

    def remove_selected(self):
        rows = sorted({i.row() for i in self.selectedIndexes()}, reverse=True)
        for row in rows:
            self.removeRow(row)

    def to_dict(self) -> dict:
        result = {}
        for row in range(self.rowCount()):
            name = (self.item(row, 0).text() if self.item(row, 0) else "").strip()
            if not name:
                continue
            result[name] = {
                "description": (self.item(row, 1).text() if self.item(row, 1) else "").strip(),
                "dev_path": (self.item(row, 2).text() if self.item(row, 2) else "").strip(),
                "user_path": (self.item(row, 3).text() if self.item(row, 3) else "").strip(),
            }
        return result


class ConfigDialog(QDialog):
    """Tabbed editor for extraction/config.json and chatbot/config.json."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure")
        self.resize(820, 680)
        self.extraction_data = load_json(EXTRACTION_CONFIG_PATH) if EXTRACTION_CONFIG_PATH.exists() else {}
        self.chatbot_data = load_json(CHATBOT_CONFIG_PATH) if CHATBOT_CONFIG_PATH.exists() else {}

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_extraction_tab(), "Extraction")
        self.tabs.addTab(self._build_chatbot_tab(), "Chatbot")
        layout.addWidget(self.tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ---- extraction tab ----

    def _build_extraction_tab(self) -> QWidget:
        data = self.extraction_data
        container, form = self._scrollable_form()

        self.ex_project_name = QLineEdit(data.get("project_name", ""))
        form.addRow("project_name", self.ex_project_name)

        self.ex_output_dir = QLineEdit(data.get("output_dir", ""))
        form.addRow("output_dir", self._with_browse(self.ex_output_dir, directory=True))

        self.ex_token_chunking = QCheckBox("use_token_chunking")
        self.ex_token_chunking.setChecked(bool(data.get("use_token_chunking", False)))
        form.addRow(self.ex_token_chunking)

        emb = data.get("embedding") or {}
        self.ex_emb_model = QLineEdit(emb.get("model", "all-MiniLM-L6-v2"))
        self.ex_emb_type = QComboBox()
        self.ex_emb_type.addItems(["local", "api"])
        self.ex_emb_type.setCurrentText(emb.get("type", "local"))
        self.ex_emb_base_url = QLineEdit(emb.get("base_url") or "")
        self.ex_emb_api_key = QLineEdit(emb.get("api_key") or "")
        self.ex_emb_api_key.setEchoMode(QLineEdit.Password)
        form.addRow("embedding.model", self.ex_emb_model)
        form.addRow("embedding.type", self.ex_emb_type)
        form.addRow("embedding.base_url", self.ex_emb_base_url)
        form.addRow("embedding.api_key", self.ex_emb_api_key)

        chunk = data.get("chunking") or {}
        self.ex_max_tokens = self._spin(chunk.get("max_tokens", 384), 1, 8192)
        self.ex_overlap_tokens = self._spin(chunk.get("overlap_tokens", 50), 0, 4096)
        self.ex_char_chunk_size = self._spin(chunk.get("char_chunk_size", 1000), 1, 1_000_000)
        self.ex_char_overlap = self._spin(chunk.get("char_overlap", 200), 0, 1_000_000)
        form.addRow("chunking.max_tokens", self.ex_max_tokens)
        form.addRow("chunking.overlap_tokens", self.ex_overlap_tokens)
        form.addRow("chunking.char_chunk_size", self.ex_char_chunk_size)
        form.addRow("chunking.char_overlap", self.ex_char_overlap)

        qual = data.get("quality") or {}
        self.ex_quality_min_score = QDoubleSpinBox()
        self.ex_quality_min_score.setRange(0.0, 1.0)
        self.ex_quality_min_score.setSingleStep(0.05)
        self.ex_quality_min_score.setValue(float(qual.get("min_score", 0.3)))
        self.ex_quality_min_words = self._spin(qual.get("min_word_count", 50), 0, 1_000_000)
        self.ex_quality_sub_words = self._spin(qual.get("substantial_word_count", 100), 0, 1_000_000)
        form.addRow("quality.min_score", self.ex_quality_min_score)
        form.addRow("quality.min_word_count", self.ex_quality_min_words)
        form.addRow("quality.substantial_word_count", self.ex_quality_sub_words)

        modules_group = QGroupBox("modules")
        modules_layout = QVBoxLayout(modules_group)
        self.ex_modules_table = ModulesTable(data.get("modules") or {})
        modules_layout.addWidget(self.ex_modules_table)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add module")
        add_btn.clicked.connect(self.ex_modules_table.add_empty_row)
        rm_btn = QPushButton("Remove selected")
        rm_btn.clicked.connect(self.ex_modules_table.remove_selected)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rm_btn)
        btn_row.addStretch()
        modules_layout.addLayout(btn_row)
        form.addRow(modules_group)

        return container

    # ---- chatbot tab ----

    def _build_chatbot_tab(self) -> QWidget:
        data = self.chatbot_data
        container, form = self._scrollable_form()

        self.cb_project_name = QLineEdit(data.get("project_name", ""))
        self.cb_chromadb_path = QLineEdit(data.get("chromadb_path", ""))
        form.addRow("project_name", self.cb_project_name)
        form.addRow("chromadb_path", self._with_browse(self.cb_chromadb_path, directory=True))

        llm = data.get("llm") or {}
        self.cb_llm_base_url = QLineEdit(llm.get("base_url", ""))
        self.cb_llm_model = QLineEdit(llm.get("model", ""))
        self.cb_llm_api_key = QLineEdit(llm.get("api_key", ""))
        self.cb_llm_api_key.setEchoMode(QLineEdit.Password)
        self.cb_llm_ssl = QLineEdit(llm.get("ssl_cert_file") or "")
        form.addRow("llm.base_url", self.cb_llm_base_url)
        form.addRow("llm.model", self.cb_llm_model)
        form.addRow("llm.api_key", self.cb_llm_api_key)
        form.addRow("llm.ssl_cert_file", self._with_browse(self.cb_llm_ssl, directory=False))

        emb = data.get("embedding") or {}
        self.cb_emb_model = QLineEdit(emb.get("model", "all-MiniLM-L6-v2"))
        self.cb_emb_type = QComboBox()
        self.cb_emb_type.addItems(["local", "api"])
        self.cb_emb_type.setCurrentText(emb.get("type", "local"))
        self.cb_emb_base_url = QLineEdit(emb.get("base_url") or "")
        self.cb_emb_api_key = QLineEdit(emb.get("api_key") or "")
        self.cb_emb_api_key.setEchoMode(QLineEdit.Password)
        form.addRow("embedding.model", self.cb_emb_model)
        form.addRow("embedding.type", self.cb_emb_type)
        form.addRow("embedding.base_url", self.cb_emb_base_url)
        form.addRow("embedding.api_key", self.cb_emb_api_key)

        rr = data.get("reranker")
        self.cb_rr_enabled = QCheckBox("reranker enabled")
        self.cb_rr_enabled.setChecked(bool(rr))
        self.cb_rr_model = QLineEdit((rr or {}).get("model", "BAAI/bge-reranker-v2-m3"))
        self.cb_rr_type = QComboBox()
        self.cb_rr_type.addItems(["local"])
        self.cb_rr_type.setCurrentText((rr or {}).get("type", "local"))
        form.addRow(self.cb_rr_enabled)
        form.addRow("reranker.model", self.cb_rr_model)
        form.addRow("reranker.type", self.cb_rr_type)

        ag = data.get("agentic")
        self.cb_ag_enabled = QCheckBox("agentic enabled")
        self.cb_ag_enabled.setChecked(bool(ag))
        self.cb_ag_page_index = QLineEdit((ag or {}).get("page_index_path", ""))
        self.cb_ag_max_chars = self._spin((ag or {}).get("max_chars_per_page", 8000), 1, 1_000_000)
        self.cb_ag_max_pages = self._spin((ag or {}).get("max_pages_per_round", 3), 1, 100)
        self.cb_ag_max_pages_r2 = self._spin((ag or {}).get("max_pages_round2", 2), 0, 100)
        form.addRow(self.cb_ag_enabled)
        form.addRow("agentic.page_index_path", self._with_browse(self.cb_ag_page_index, directory=False))
        form.addRow("agentic.max_chars_per_page", self.cb_ag_max_chars)
        form.addRow("agentic.max_pages_per_round", self.cb_ag_max_pages)
        form.addRow("agentic.max_pages_round2", self.cb_ag_max_pages_r2)

        self.cb_k_standard = self._spin(data.get("k_standard", 40), 1, 1000)
        self.cb_k_deep_dive = self._spin(data.get("k_deep_dive", 60), 1, 1000)
        self.cb_deep_batch = self._spin(data.get("deep_dive_batch_size", 10), 1, 1000)
        self.cb_top_n = self._spin(data.get("top_n_after_rerank", 15), 1, 1000)
        self.cb_temp = QDoubleSpinBox()
        self.cb_temp.setRange(0.0, 2.0)
        self.cb_temp.setSingleStep(0.05)
        self.cb_temp.setValue(float(data.get("temperature", 0.0)))
        self.cb_max_tokens = self._spin(data.get("max_tokens", 2000), 1, 1_000_000)
        form.addRow("k_standard", self.cb_k_standard)
        form.addRow("k_deep_dive", self.cb_k_deep_dive)
        form.addRow("deep_dive_batch_size", self.cb_deep_batch)
        form.addRow("top_n_after_rerank", self.cb_top_n)
        form.addRow("temperature", self.cb_temp)
        form.addRow("max_tokens", self.cb_max_tokens)

        return container

    # ---- helpers ----

    @staticmethod
    def _scrollable_form():
        container = QWidget()
        outer = QVBoxLayout(container)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form = QFormLayout(inner)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return container, form

    @staticmethod
    def _spin(value, mn, mx) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(mn, mx)
        sb.setValue(int(value))
        return sb

    @staticmethod
    def _with_browse(line_edit: QLineEdit, directory: bool) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(line_edit)
        btn = QPushButton("Browse...")

        def pick():
            if directory:
                path = QFileDialog.getExistingDirectory(w, "Select directory")
            else:
                path, _ = QFileDialog.getOpenFileName(w, "Select file")
            if path:
                line_edit.setText(path)

        btn.clicked.connect(pick)
        h.addWidget(btn)
        return w

    # ---- save ----

    def _on_save(self):
        try:
            self._apply_extraction()
            self._apply_chatbot()
            save_json(EXTRACTION_CONFIG_PATH, self.extraction_data)
            save_json(CHATBOT_CONFIG_PATH, self.chatbot_data)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        self.accept()

    def _apply_extraction(self):
        d = self.extraction_data
        d["project_name"] = self.ex_project_name.text().strip()
        d["output_dir"] = self.ex_output_dir.text().strip()
        d["use_token_chunking"] = self.ex_token_chunking.isChecked()
        d["modules"] = self.ex_modules_table.to_dict()
        d["embedding"] = {
            **(d.get("embedding") or {}),
            "model": self.ex_emb_model.text().strip(),
            "type": self.ex_emb_type.currentText(),
            "base_url": self.ex_emb_base_url.text().strip() or None,
            "api_key": self.ex_emb_api_key.text().strip() or None,
        }
        d["chunking"] = {
            **(d.get("chunking") or {}),
            "max_tokens": self.ex_max_tokens.value(),
            "overlap_tokens": self.ex_overlap_tokens.value(),
            "char_chunk_size": self.ex_char_chunk_size.value(),
            "char_overlap": self.ex_char_overlap.value(),
        }
        d["quality"] = {
            **(d.get("quality") or {}),
            "min_score": self.ex_quality_min_score.value(),
            "min_word_count": self.ex_quality_min_words.value(),
            "substantial_word_count": self.ex_quality_sub_words.value(),
        }

    def _apply_chatbot(self):
        d = self.chatbot_data
        d["project_name"] = self.cb_project_name.text().strip()
        d["chromadb_path"] = self.cb_chromadb_path.text().strip()
        d["llm"] = {
            **(d.get("llm") or {}),
            "base_url": self.cb_llm_base_url.text().strip(),
            "model": self.cb_llm_model.text().strip(),
            "api_key": self.cb_llm_api_key.text().strip(),
            "ssl_cert_file": self.cb_llm_ssl.text().strip() or None,
        }
        d["embedding"] = {
            **(d.get("embedding") or {}),
            "model": self.cb_emb_model.text().strip(),
            "type": self.cb_emb_type.currentText(),
            "base_url": self.cb_emb_base_url.text().strip() or None,
            "api_key": self.cb_emb_api_key.text().strip() or None,
        }
        d["reranker"] = (
            {"model": self.cb_rr_model.text().strip(), "type": self.cb_rr_type.currentText()}
            if self.cb_rr_enabled.isChecked() else None
        )
        d["agentic"] = (
            {
                "page_index_path": self.cb_ag_page_index.text().strip(),
                "max_chars_per_page": self.cb_ag_max_chars.value(),
                "max_pages_per_round": self.cb_ag_max_pages.value(),
                "max_pages_round2": self.cb_ag_max_pages_r2.value(),
            }
            if self.cb_ag_enabled.isChecked() else None
        )
        d["k_standard"] = self.cb_k_standard.value()
        d["k_deep_dive"] = self.cb_k_deep_dive.value()
        d["deep_dive_batch_size"] = self.cb_deep_batch.value()
        d["top_n_after_rerank"] = self.cb_top_n.value()
        d["temperature"] = self.cb_temp.value()
        d["max_tokens"] = self.cb_max_tokens.value()


# --- Main window --------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SALOME Assistant (RAG based)")
        self.resize(900, 700)
        self.robot_icon_path = str(HERE / "salome_assistant_assets" / "salome.jpg")
        self.last_bot_answer_text = ""
        self.chatbot = None
        self.current_model_name = ""

        # Menu bar
        configure_menu = self.menuBar().addMenu("&Configure")
        configure_action = QAction("Configure...", self)
        configure_action.triggered.connect(self.open_config_dialog)
        configure_menu.addAction(configure_action)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        hsplit = QHBoxLayout()
        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        filters_group = QGroupBox("Query filters")
        filters_layout = QFormLayout(filters_group)
        self.module_selector = QComboBox()
        self.module_selector.addItem("All")
        self.doc_type_selector = QComboBox()
        self.doc_type_selector.addItems(["All", "dev", "user"])
        self.deep_dive_check = QCheckBox("Deep dive")
        filters_layout.addRow("Module:", self.module_selector)
        filters_layout.addRow("Doc type:", self.doc_type_selector)
        filters_layout.addRow(self.deep_dive_check)
        left_panel.addWidget(filters_group)

        self.load_btn = QPushButton("Load / Reload Chatbot")
        self.load_btn.clicked.connect(self.handle_load)
        left_panel.addWidget(self.load_btn)
        left_panel.addStretch()

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        right_panel.addWidget(self.chat_display)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        hsplit.addWidget(left_widget, 1)
        hsplit.addWidget(right_widget, 2)
        root.addLayout(hsplit)

        self.status_label = QLabel("Configure via the Configure menu, then click Load.")
        root.addWidget(self.status_label)

        input_row = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Load the chatbot first, then type here...")
        self.input_field.returnPressed.connect(self.handle_ask)
        self.input_field.setEnabled(False)
        self.send_button = QPushButton("Ask")
        self.send_button.clicked.connect(self.handle_ask)
        self.send_button.setEnabled(False)
        self.audio_button = QPushButton("Audio")
        self.audio_button.setEnabled(False)
        self.audio_button.clicked.connect(self.play_audio)
        input_row.addWidget(self.input_field, 1)
        input_row.addWidget(self.send_button)
        input_row.addWidget(self.audio_button)
        input_row.setAlignment(Qt.AlignTop)
        try:
            btn_h = self.input_field.sizeHint().height()
            self.send_button.setFixedHeight(btn_h)
            self.audio_button.setFixedHeight(btn_h)
        except Exception:
            pass
        root.addLayout(input_row)

        bottom = QHBoxLayout()
        self.help_btn = QPushButton("Help")
        self.help_btn.clicked.connect(self.show_help)
        self.quit_btn = QPushButton("Quit")
        self.quit_btn.clicked.connect(self.close)
        bottom.addWidget(self.help_btn)
        bottom.addStretch()
        bottom.addWidget(self.quit_btn)
        root.addLayout(bottom)

        self._thinking_timer = QTimer(self)
        self._thinking_timer.setInterval(1000)
        self._thinking_timer.timeout.connect(self._update_thinking_elapsed)
        self._thinking_start = None
        self._thinking_prefix = "Thinking..."

    # ---- menu actions ----

    def open_config_dialog(self):
        dlg = ConfigDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self.status_label.setText("Configuration saved. Click Load to apply.")

    # ---- load / ask ----

    def toggle_inputs(self, enabled: bool):
        self.input_field.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        self.load_btn.setEnabled(enabled)

    def handle_load(self):
        if not CHATBOT_CONFIG_PATH.exists():
            QMessageBox.warning(
                self, "Missing configuration",
                f"Chatbot configuration not found at {CHATBOT_CONFIG_PATH}.\n"
                "Open Configure... to create one.",
            )
            return
        self.toggle_inputs(False)
        self._start_thinking_timer("Loading...")
        self.init_worker = InitWorker(CHATBOT_CONFIG_PATH)
        self.init_worker.progress.connect(self.status_label.setText)
        self.init_worker.finished_ok.connect(self._on_init_ok)
        self.init_worker.failed.connect(self._on_init_failed)
        self.init_worker.start()

    def _on_init_ok(self, bot, modules, model_name):
        self._stop_thinking_timer()
        self.chatbot = bot
        self.current_model_name = model_name
        self.module_selector.clear()
        self.module_selector.addItem("All")
        for m in modules:
            self.module_selector.addItem(m)
        self.toggle_inputs(True)
        self.status_label.setText(
            f"Ready — model: {model_name}, modules: {', '.join(modules) or 'none'}"
        )
        self.chat_display.append(
            f"<span style='color:green'>System: chatbot loaded "
            f"({len(modules)} module(s), model {model_name}).</span>"
        )

    def _on_init_failed(self, message):
        self._stop_thinking_timer()
        self.toggle_inputs(True)
        self.status_label.setText("Error loading chatbot")
        QMessageBox.critical(self, "Error", message)

    def handle_ask(self):
        if self.chatbot is None:
            return
        query = self.input_field.text().strip()
        if not query:
            return
        self.chat_display.append(f"<b>You:</b> {query}")
        self.input_field.clear()
        self.toggle_inputs(False)
        self._start_thinking_timer("Thinking...")

        module = self.module_selector.currentText()
        module = None if module == "All" else module
        doc_type = self.doc_type_selector.currentText()
        doc_type = None if doc_type == "All" else doc_type

        self.query_worker = QueryWorker(
            self.chatbot, query, module, doc_type, self.deep_dive_check.isChecked()
        )
        self.query_worker.finished_ok.connect(self._on_query_ok)
        self.query_worker.failed.connect(self._on_query_failed)
        self.query_worker.start()

    def _on_query_ok(self, result: dict):
        self._stop_thinking_timer()
        self.toggle_inputs(True)
        self.input_field.setFocus()

        error = result.get("error")
        if error:
            self.chat_display.append(f"<span style='color:red'>Error: {error}</span>")
            self.status_label.setText("Error")
            return

        answer = result.get("answer") or ""

        # Try RST → HTML first (the LLM sometimes emits RST), fall back to Markdown.
        try:
            from docutils.core import publish_parts
            html_answer = publish_parts(source=answer, writer_name="html5").get("html_body", "")
        except Exception:
            html_answer = markdown.markdown(
                answer, extensions=["fenced_code", "codehilite"], output_format="html"
            )
        html_answer = html_answer.replace(
            "<pre><code",
            '<pre style="background:#f6f8fa;padding:8px;border-radius:4px;white-space:pre-wrap;"><code',
        )
        html_answer = html_answer.replace(
            "<pre>",
            '<pre style="background:#f6f8fa;padding:8px;border-radius:4px;white-space:pre-wrap;">',
        )

        img_html = ""
        if os.path.exists(self.robot_icon_path):
            img_html = (
                f'<img src="file://{self.robot_icon_path}" width="24" height="24" '
                'style="vertical-align:middle;margin-right:8px"/>'
            )

        self.chat_display.append(
            f"<b>Bot ({self.current_model_name}):</b> {img_html}{html_answer}"
        )
        sources = result.get("sources") or []
        if sources:
            src_lines = "<br>".join(
                f"• {s.get('title', '?')} "
                f"<i>({s.get('module', '?')}/{s.get('doc_category', '?')})</i>"
                for s in sources[:5]
            )
            self.chat_display.append(
                f"<div style='color:#555;font-size:12px'>Sources:<br>{src_lines}</div>"
            )
        self.chat_display.append("-" * 30)
        self.last_bot_answer_text = answer
        self.audio_button.setEnabled(True)
        self.status_label.setText("Ready")

    def _on_query_failed(self, message: str):
        self._stop_thinking_timer()
        self.toggle_inputs(True)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Query error", message)

    # ---- misc ----

    def show_help(self):
        text = (
            "<b>SALOME RAG Assistant</b><br><br>"
            "Copyright: CEA 2025.<br>"
            "License: LGPL V2.1<br><br>"
            "1. Open <b>Configure</b> → Extraction / Chatbot to edit the JSON configs.<br>"
            "2. Click <b>Load / Reload Chatbot</b> to initialize.<br>"
            "3. Type a question and press <b>Ask</b>."
        )
        QMessageBox.information(self, "Help / About", text)

    def play_audio(self):
        text = getattr(self, "last_bot_answer_text", "")
        if not text:
            return
        self.audio_button.setEnabled(False)
        self.tts_worker = TTSWorker(text)
        self.tts_worker.finished.connect(lambda: self.audio_button.setEnabled(True))
        self.tts_worker.start()

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
        m, s = elapsed // 60, elapsed % 60
        self.status_label.setText(f"{self._thinking_prefix} {m:02d}:{s:02d}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(
        """
        QGroupBox { font-weight: bold; }
        QTextEdit { font-size: 14px; }
        QLineEdit { padding: 5px; font-size: 14px; }
        """
    )
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
