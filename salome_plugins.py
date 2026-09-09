
import salome_pluginsmanager

DEMO_IS_ACTIVATED = False

def runSalomeAssistant(context):
  # All names are local or imported here to be self-contained.
  # salome_plugins.py is executed via exec(code, globals(), {}) so any
  # module-level name defined in this file lands in the discarded locals
  # dict and is invisible to functions at call time.
  import os
  import subprocess
  from PyQt5.Qt import QMessageBox

  import getpass
  import tempfile
  pid_file = os.path.join(
    tempfile.gettempdir(),
    'salome-assistant-{}.pid'.format(getpass.getuser())
  )

  def _is_process_running(pid):
    """Return True if the process with the given PID is still alive.
    Uses os.kill(pid, 0) which works on both Linux and Windows:
      - raises PermissionError  -> process exists (we just cannot signal it)
      - raises OSError          -> process does not exist
    """
    try:
      import psutil
      return psutil.pid_exists(pid)
    except PermissionError:
      print("ERROR: permission issues")
      return True   # process alive, insufficient privilege to signal it
    except OSError:
      print("ERROR: OS errors")
      return False  # process no longer exists

  # Singleton guard via PID file (survives plugin reloads on both platforms).
  try:
    with open(pid_file) as f:
      pid = int(f.read().strip())
    if _is_process_running(pid):
      QMessageBox.information(
        None,
        "SALOME Assistant",
        "SALOME Assistant is already running."
      )
      return
  except (OSError, ValueError):
    pass  # no PID file or invalid content -> proceed to launch

  try:
    salome_assistant = os.environ.get('SALOME_ASSISTANT')
  # Start the in-process code-execution server the assistant's "Run in
  # SALOME" button talks to. Best-effort: the assistant window is still
  # useful for chat if this fails, it just loses the Run button.
  
    # minimal environment to be copied -
    env = os.environ.copy()
    for var in (
        'PYTHONPATH',
        'QT_PLUGIN_PATH',
        'QT_QPA_PLATFORM_PLUGIN_PATH',
        'QT_QPA_FONTDIR',
        'LD_LIBRARY_PATH',
    ):
      env.pop(var, None)
    proc = subprocess.Popen([salome_assistant], start_new_session=True, env=env)
    with open(pid_file, 'w') as f:
      f.write(str(proc.pid))
  except Exception as e:
    QMessageBox.warning(
      None,
      "SALOME Assistant",
      "Failed to launch SALOME Assistant:\n" + str(e)
    )

salome_pluginsmanager.AddFunction('SALOME Assistant (experimental)',
                                  'Launch the SALOME Assistant (RAG)',
                                  runSalomeAssistant)

# -------------------------------------------------------------------------
# Example 6: MCP bridge - let an external AI agent (e.g. via an MCP server)
# execute python commands in this session's python console. Auto-started
# so it is ready as soon as SALOME starts; set SALOME_MCP_BRIDGE_DISABLE=1
# to opt out. See salome_mcp_bridge.py and mcp/salome_mcp_server.py.
import os
if os.environ.get('SALOME_MCP_BRIDGE_DISABLE') != '1':
  try:
    import salome_mcp_bridge
    salome_mcp_bridge.start()
  except Exception as e:
    print("[MCP bridge] failed to start:", e)
