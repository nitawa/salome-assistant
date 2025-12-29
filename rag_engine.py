import os
import glob
from langchain_community.document_loaders import BSHTMLLoader
# Corrected Import:
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import torch

from bs4 import BeautifulSoup # Import BeautifulSoup
import urllib.parse # For resolving URLs

class RAGBackend:
    def __init__(self, doc_dir):
        self.doc_dir = doc_dir
        self.vector_db = None
        self.llm_pipeline = None
        self.current_model_name = None
        # Directory where the FAISS vector DB will be persisted/loaded
        self.vector_db_path = os.path.join(os.path.dirname(__file__), "vector_db")
        
        self.models_config = {
            "Mistral-7B (High Quality, Requires GPU/High RAM)": {
                "id": "mistralai/Mistral-7B-Instruct-v0.3",
                "type": "causal",
                "task": "text-generation"
            },
            "Flan-T5-Base (Fast, CPU Friendly)": {
                "id": "google/flan-t5-base",
                "type": "seq2seq",
                "task": "text2text-generation"
            }
        }

# ... inside your RAGBackend class ...
    def load_and_process_documents(self, chunk_size=700, chunk_overlap=100,
                                   embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """Recursively finds all HTML files using os.walk and discovers linked HTML files.

        chunk_size and chunk_overlap control text splitting.
        """
        if self.vector_db:
            return "Documents already loaded."

        # Initialize sets and lists for file discovery
        all_html_files = set()  # Stores all unique HTML files to process
        files_to_explore = []   # Queue of files whose links need to be explored

        # 1. Initial discovery: Find all HTML files directly in self.doc_dir
        # root: current directory path
        # dirs: list of subdirectories in current root
        # files: list of files in current root
        for root, dirs, files in os.walk(self.doc_dir):
            for filename in files:
                if filename.lower().endswith(".html"):
                    full_path = os.path.abspath(os.path.join(root, filename))
                    if full_path not in all_html_files:
                        all_html_files.add(full_path)
                        files_to_explore.append(full_path) # Add to exploration queue

        if not all_html_files:
            return f"No HTML files found in '{self.doc_dir}' (checked all subfolders)."

        # 2. Link discovery loop: Explore links within found files
        doc_root_abs = os.path.abspath(self.doc_dir) # Absolute path to doc directory for comparison

        while files_to_explore:
            current_file_path = files_to_explore.pop(0) # Get next file to explore
            print(f"Exploring links in {current_file_path}")

            try:
                with open(current_file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f, 'html.parser')

                # Find all <a> tags with href attributes
                for a_tag in soup.find_all('a', href=True):
                    link = a_tag['href']
                    self._process_discovered_link(current_file_path, link, doc_root_abs, all_html_files, files_to_explore)

                # Find all <iframe> tags with src attributes
                for iframe_tag in soup.find_all('iframe', src=True):
                    link = iframe_tag['src']
                    self._process_discovered_link(current_file_path, link, doc_root_abs, all_html_files, files_to_explore)

            except Exception as e:
                print(f"Error reading or parsing {current_file_path} for links: {e}")

        if not all_html_files:
            return f"No HTML files found or discoverable in '{self.doc_dir}'."

        # 3. Load and process all discovered documents
        documents = []
        for file_path in all_html_files:
            print(f"Loading {file_path}")
            try:
                loader = BSHTMLLoader(file_path, open_encoding='utf-8')
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if not documents:
            return "Found files, but failed to extract content."

        # Text splitting using the new package
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_db = FAISS.from_documents(texts, embeddings)
        
        return f"Successfully indexed {len(texts)} chunks from {len(all_html_files)} files."

    def load_vector_db_from_disk(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """Attempt to load a persisted FAISS vector DB from disk. Returns message or None if missing."""
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        if os.path.exists(self.vector_db_path):
            try:
                self.vector_db = FAISS.load_local(self.vector_db_path, embeddings)
                return f"Loaded vector db from {self.vector_db_path}"
            except Exception as e:
                return f"Failed to load vector db from disk: {e}"
        return None

    def save_vector_db(self):
        """Persist the current in-memory FAISS vector DB to disk."""
        if not self.vector_db:
            return "No vector DB in memory to save."
        try:
            os.makedirs(self.vector_db_path, exist_ok=True)
            self.vector_db.save_local(self.vector_db_path)
            return f"Saved vector db to {self.vector_db_path}"
        except Exception as e:
            return f"Failed to save vector db: {e}"

    def ensure_vector_db(self, reload=False, chunk_size=700, chunk_overlap=100,
                         embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """Ensure self.vector_db is available.

        If reload=True: always reprocess HTML and save vector DB to disk.
        If reload=False: attempt to load from disk, and if missing, build from HTML and save.
        Returns a short status message.
        """
        if reload:
            # Force rebuild from HTML and persist
            res = self.load_and_process_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                   embedding_model=embedding_model)
            save_res = self.save_vector_db()
            return f"Rebuilt from HTML. {res}\n{save_res}"

        # Try loading from disk first
        load_res = self.load_vector_db_from_disk(embedding_model=embedding_model)
        if load_res:
            return load_res

        # Not found on disk -> build from HTML and persist
        res = self.load_and_process_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              embedding_model=embedding_model)
        save_res = self.save_vector_db()
        return f"Vector DB not found on disk; built from HTML. {res}\n{save_res}"

    def _process_discovered_link(self, base_file_path, link, doc_root_abs, all_html_files, files_to_explore):
        """Helper to process a discovered link, resolve its path, and add to discovery queue if valid."""
        parsed_link = urllib.parse.urlparse(link)
        
        # Ignore external URLs (those with a scheme like http/https or a network location)
        if parsed_link.scheme or parsed_link.netloc:
            return

        resolved_path = None
        if os.path.isabs(link): # If link starts with a / (absolute path from filesystem root)
            resolved_path = os.path.normpath(link)
        else: # Relative path from the base_file_path's directory
            resolved_path = os.path.normpath(os.path.join(os.path.dirname(base_file_path), link))

        resolved_path_abs = os.path.abspath(resolved_path)

        # Ensure it's an HTML file and within the specified document directory
        if resolved_path_abs.lower().endswith(".html") and resolved_path_abs.startswith(doc_root_abs):
            if resolved_path_abs not in all_html_files:
                all_html_files.add(resolved_path_abs)
                files_to_explore.append(resolved_path_abs)
                print(f"  Discovered new HTML file: {resolved_path_abs}")

    def initialize_llm(self, model_display_name, temperature=0.1, max_new_tokens=512):
        config = self.models_config.get(model_display_name)
        if not config:
            raise ValueError("Invalid model selection")

        if self.current_model_name == model_display_name and self.llm_pipeline:
            return f"Model {model_display_name} is already active."

        model_id = config["id"]
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if config["type"] == "causal":
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="auto" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        pipe = pipeline(
            config["task"],
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            device=device if config["type"] != "causal" else None
        )

        self.llm_pipeline = HuggingFacePipeline(pipeline=pipe)
        self.current_model_name = model_display_name
        return f"Loaded {model_display_name}"

    def ask(self, query):
        if not self.vector_db or not self.llm_pipeline:
            errorMsg= "Error: System not fully initialized."
            if not self.vector_db:
                errorMsg+="Vector_db is not initialized."
            if not self.llm_pipeline:
                errorMsg+="LLM pipeline not initialized."
            return errorMsg

        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        if "Mistral" in self.current_model_name:
            prompt = f"<s>[INST] Answer the question based on the context below.\n\nContext:\n{context_text}\n\nQuestion:\n{query} [/INST]"
        else:
            prompt = f"Context:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer:"

        response = self.llm_pipeline.invoke(prompt)
        
        if "Mistral" in self.current_model_name and "[/INST]" in response:
             response = response.split("[/INST]")[-1].strip()
        
        return response
