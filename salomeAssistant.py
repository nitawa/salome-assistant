#!/bin/env python3
import sys
import os
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                             QLabel, QProgressBar, QMessageBox, QComboBox, QGroupBox, QCheckBox,
                             QDoubleSpinBox, QSpinBox)

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from rag_engine import RAGBackend
import markdown

class Worker(QThread):
    """Background thread to handle heavy RAG operations without freezing GUI."""
    finished = pyqtSignal(str)
    status_update = pyqtSignal(str)
    
    def __init__(self, backend, task_type, query=None, model_name=None, reload_vectors=False,
                 temperature=0.1, max_new_tokens=512, chunk_size=700, chunk_overlap=100,
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.backend = backend
        self.task_type = task_type
        self.query = query
        self.model_name = model_name
        self.reload_vectors = reload_vectors
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

    def run(self):
        try:
            if self.task_type in ('init_all', 'switch_model'):
                self.status_update.emit(f"Loading Documents & {self.model_name}...")

                # Ensure vector DB exists or is reloaded from HTML based on user choice
                doc_res = self.backend.ensure_vector_db(reload=self.reload_vectors,
                                                       chunk_size=self.chunk_size,
                                                       chunk_overlap=self.chunk_overlap,
                                                       embedding_model=self.embedding_model)

                # Load Model
                model_res = self.backend.initialize_llm(self.model_name,
                                                       temperature=self.temperature,
                                                       max_new_tokens=self.max_new_tokens)

                self.finished.emit(f"{doc_res}\n{model_res}")

            elif self.task_type == 'switch_model':
                # fallback (shouldn't normally be reached now)
                self.status_update.emit(f"Switching to {self.model_name}...")
                res = self.backend.initialize_llm(self.model_name)
                self.finished.emit(res)

            elif self.task_type == 'query':
                self.status_update.emit("Thinking...")
                answer = self.backend.ask(self.query)
                self.finished.emit(answer)

        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SALOME Assistant (RAG based)")
        self.resize(900, 700)
        try:
            doc_dir = os.environ["DOCUMENTATION_ROOT_DIR"]
        except:
            raise EnvironmentError("Critical Error: 'DOCUMENTATION_ROOT_DIR' environment variable is not set.")
        self.backend = RAGBackend(doc_dir=doc_dir)
        self.robot_icon_path = os.path.join(os.path.dirname(__file__), "salome.jpg")

        # --- UI Setup ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Main horizontal split: left for settings, right for chat
        main_hsplit = QHBoxLayout()
        left_panel  = QVBoxLayout()
        right_panel = QVBoxLayout()

        # 1. Settings Area (left panel)
        settings_group  = QGroupBox("Settings")
        settings_layout = QVBoxLayout()

        self.model_selector = QComboBox()
        self.model_selector.addItems(list(self.backend.models_config.keys()))
        self.model_label = QLabel("Select Model:")

        # Embedding model selector
        self.embedding_label = QLabel("Embeddings:")
        self.embedding_selector = QComboBox()
        self.embedding_selector.addItems([
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/all-MiniLM-L6-v2"
        ])

        # Make labels and combo boxes consistent width and alignment
        label_width   = 100
        control_width = 280
        self.model_label.setFixedWidth(label_width)
        self.embedding_label.setFixedWidth(label_width)
        self.model_selector.setFixedWidth(control_width)
        self.embedding_selector.setFixedWidth(control_width)

        self.reload_label = QLabel("Process RAG input documentation:")
        self.reload_checkbox = QCheckBox("")
        self.reload_checkbox.setChecked(True)

        # Temperature and max tokens
        self.temp_label = QLabel("Temperature:")
        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(0.0, 2.0)
        self.temp_input.setSingleStep(0.01)
        self.temp_input.setValue(0.05)

        self.max_tokens_label = QLabel("Max Tokens:")
        self.max_tokens_input = QSpinBox()
        self.max_tokens_input.setRange(1, 4098)
        self.max_tokens_input.setValue(1024)

        # Chunk size and overlap
        self.chunk_size_label = QLabel("Chunk Size:")
        self.chunk_size_input = QSpinBox()
        self.chunk_size_input.setRange(128, 10000)
        self.chunk_size_input.setValue(512)

        self.chunk_overlap_label = QLabel("Chunk Overlap:")
        self.chunk_overlap_input = QSpinBox()
        self.chunk_overlap_input.setRange(0, 2000)
        self.chunk_overlap_input.setValue(128)

        self.load_btn = QPushButton("Load / Switch Model")
        self.load_btn.clicked.connect(self.handle_model_switch)
        self.quit_btn = QPushButton("Quit")
        self.quit_btn.clicked.connect(self.close)

        # Create rows (horizontal) for label+control pairs, then stack vertically
        row_model = QHBoxLayout()
        row_model.addWidget(self.model_label)
        row_model.addWidget(self.model_selector)
        row_model.setAlignment(Qt.AlignLeft)

        row_embedding = QHBoxLayout()
        row_embedding.addWidget(self.embedding_label)
        row_embedding.addWidget(self.embedding_selector)
        row_embedding.setAlignment(Qt.AlignLeft)

        row_reload = QHBoxLayout()
        row_reload.addWidget(self.reload_label)
        row_reload.addWidget(self.reload_checkbox)

        row_temp = QHBoxLayout()
        row_temp.addWidget(self.temp_label)
        row_temp.addWidget(self.temp_input)

        row_max = QHBoxLayout()
        row_max.addWidget(self.max_tokens_label)
        row_max.addWidget(self.max_tokens_input)

        row_chunk = QHBoxLayout()
        row_chunk.addWidget(self.chunk_size_label)
        row_chunk.addWidget(self.chunk_size_input)

        row_overlap = QHBoxLayout()
        row_overlap.addWidget(self.chunk_overlap_label)
        row_overlap.addWidget(self.chunk_overlap_input)

        row_buttons = QHBoxLayout()
        row_buttons.addWidget(self.load_btn)

        settings_layout.addLayout(row_model)
        settings_layout.addLayout(row_embedding)
        settings_layout.addLayout(row_reload)
        settings_layout.addLayout(row_temp)
        settings_layout.addLayout(row_max)
        settings_layout.addLayout(row_chunk)
        settings_layout.addLayout(row_overlap)
        settings_layout.addLayout(row_buttons)
        settings_group.setLayout(settings_layout)
        left_panel.addWidget(settings_group)

        # 2. Chat Area (right panel)
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        right_panel.addWidget(self.chat_display)

        # assemble split: wrap layouts in widgets and add with stretch so
        # the right panel occupies two-thirds and the left one-third
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)

        main_hsplit.addWidget(left_widget, 1)
        main_hsplit.addWidget(right_widget, 2)
        self.layout.addLayout(main_hsplit)

        # 3. Status
        self.status_label = QLabel("Please select a model and click Load.")
        self.layout.addWidget(self.status_label)
        # status_label is used to show state (e.g. "Thinking...")

        # 4. Input Area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("First load a model, then type here...")
        self.input_field.returnPressed.connect(self.handle_ask)
        self.input_field.setEnabled(False)
        
        self.send_button = QPushButton("Ask")
        self.send_button.clicked.connect(self.handle_ask)
        self.send_button.setEnabled(False)

        # Timer for elapsed 'Thinking...' shown when asking
        self._thinking_timer = QTimer(self)
        self._thinking_timer.setInterval(1000)
        self._thinking_timer.timeout.connect(self._update_thinking_elapsed)
        self._thinking_start = None
        self._thinking_prefix = "Thinking..."

        # Make the input field expand to fill space up to the Ask button
        input_layout.addWidget(self.input_field, 1)
        input_layout.addWidget(self.send_button)
        # Align controls to the top of the row and make Ask match input height
        input_layout.setAlignment(Qt.AlignTop)
        try:
            btn_h = self.input_field.sizeHint().height()
            self.send_button.setFixedHeight(btn_h)
        except Exception:
            pass
        self.layout.addLayout(input_layout)

        # Bottom control row: Help (left) --- Quit (right)
        self.help_btn = QPushButton("Help")
        self.help_btn.clicked.connect(self.show_help)

        bottom_controls = QHBoxLayout()
        bottom_controls.addWidget(self.help_btn)
        bottom_controls.addStretch()
        bottom_controls.addWidget(self.quit_btn)
        self.layout.addLayout(bottom_controls)

    def toggle_inputs(self, enabled):
        """Helper to enable/disable UI elements during processing."""
        self.input_field.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        self.load_btn.setEnabled(enabled)
        self.model_selector.setEnabled(enabled)
        self.embedding_selector.setEnabled(enabled)
        # keep quit/help available at all times; if desired, enable/disable here

    def handle_model_switch(self):
        selected_model = self.model_selector.currentText()
        selected_embedding = self.embedding_selector.currentText()
        
        self.toggle_inputs(False)
        self.status_label.setText("Thinking...")
        
        # Decide if we need to load docs too (first run) or just switch model
        task = 'init_all' if self.backend.vector_db is None else 'switch_model'

        self.worker = Worker(
            self.backend,
            task_type=task,
            model_name=selected_model,
            reload_vectors=self.reload_checkbox.isChecked(),
            temperature=self.temp_input.value(),
            max_new_tokens=self.max_tokens_input.value(),
            chunk_size=self.chunk_size_input.value(),
            chunk_overlap=self.chunk_overlap_input.value()
            , embedding_model=selected_embedding
        )
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_init_finished)
        self.worker.start()

    def handle_ask(self):
        query = self.input_field.text().strip()
        if not query:
            return

        self.chat_display.append(f"<b>You:</b> {query}")
        self.input_field.clear()
        self.toggle_inputs(False)
        self._start_thinking_timer("Thinking...")

        self.worker = Worker(self.backend, task_type='query', query=query)
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_query_finished)
        self.worker.start()

    def on_init_finished(self, result):
        self.toggle_inputs(True)
        self.status_label.setText("Ready")
        self.chat_display.append(f"<span style='color:green'>System: {result}</span>")
        
        if "Error" in result:
            QMessageBox.critical(self, "Error", result)
        
        # TODO SAVE the embeddings

    def on_query_finished(self, answer):
        # Stop elapsed timer and restore UI state
        try:
            self._stop_thinking_timer()
        except Exception:
            pass
        self.toggle_inputs(True)
        self.status_label.setText("Ready")
        self.input_field.setFocus()
        # Try to detect/handle reStructuredText (RST) responses first using docutils,
        # otherwise fall back to Markdown conversion. This prevents raw RST markup
        # from showing in the chat when the LLM emits RST.
        html_answer = ""
        try:
            from docutils.core import publish_parts
            parts = publish_parts(source=answer, writer_name='html5')
            html_answer = parts.get('html_body', '')
        except Exception:
            # fallback to Markdown
            html_answer = markdown.markdown(answer, extensions=['fenced_code', 'codehilite'], output_format='html5')

        # Add minimal inline styling for code blocks so they render nicely in QTextEdit
        html_answer = html_answer.replace('<pre><code', '<pre style="background:#f6f8fa;padding:8px;border-radius:4px;white-space:pre-wrap;"><code')
        html_answer = html_answer.replace('<pre>', '<pre style="background:#f6f8fa;padding:8px;border-radius:4px;white-space:pre-wrap;">')

        # Prepend robot icon if available
        img_html = ""
        try:
            if os.path.exists(self.robot_icon_path):
                img_url = f"file://{self.robot_icon_path}"
                img_html = f"<img src=\"{img_url}\" width=\"24\" height=\"24\" style=\"vertical-align:middle;margin-right:8px\"/>"
        except Exception:
            img_html = ""

        self.chat_display.append(f"<b>Bot ({self.backend.current_model_name}):</b> {img_html}{html_answer}")
        self.chat_display.append("-" * 30)

    def show_help(self):
        about_help_text = (
            "<b>SALOME RAG Assistant</b><br><br>"
            "Copyright: CEA 2025.<br>"
            "License: LGPL V2.1<br><br>"
            "Use 'Load / Switch Model' to initialize models and vectors.<br>"
            "Then type a question and press 'Ask'."
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
    
    # Simple styling
    app.setStyleSheet("""
        QGroupBox { font-weight: bold; }
        QTextEdit { font-size: 14px; }
        QLineEdit { padding: 5px; font-size: 14px; }
    """)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
