# salome-assistant

Tested on personal laptop: RAM 32 GB + RTX 5070 8 GB.

---

## Installation via `requirements.txt` (manual method)

```bash
apt install python3-venv espeak-ng
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt

export DOCUMENTATION_ROOT_DIR=<path to the SALOME documentation directory>

chmod u+x salomeAssistant.py
./salomeAssistant.py
```

---

## Installation via `pyproject.toml` (recommended method)

This method installs the assistant into an **isolated** Python environment,
completely separate from SALOME's embedded Python, preventing any dependency
conflicts (PyQt5, transformers, etc.).

### System prerequisites

```bash
sudo apt install python3-venv espeak-ng
```

### 1. Create a dedicated virtual environment

```bash
python3 -m venv SALOMEAssistant
```

### 2. Install the package

```bash
SALOMEAssistant/bin/pip install .
```

The `salome-assistant` launcher script is then available inside the venv:

```
SALOMEAssistant/bin/salome-assistant
```

### 3. Set the environment variables

```bash
# Path to the SALOME documentation directory (required at runtime)
export DOCUMENTATION_ROOT_DIR=<path to the SALOME documentation directory>

# Path to the launcher installed in the venv
# (used by the SALOME GUI to display the menu entry)
export SALOME_ASSISTANT=SALOMEAssistant/bin/salome-assistant
```

### 4. Run the assistant in standalone mode

```bash
export DOCUMENTATION_ROOT_DIR=<path to the SALOME documentation directory>
SALOMEAssistant/bin/salome-assistant
```

---

## SALOME GUI integration

When the `SALOME_ASSISTANT` environment variable is set before starting SALOME,
a **"SALOME Assistant"** entry is automatically added to the *Plugins* menu.
Clicking it launches the assistant inside its isolated environment, with no
interference with SALOME's own Python runtime.

```
Plugins menu
└── SALOME Assistant    ← visible only when SALOME_ASSISTANT is set
```
