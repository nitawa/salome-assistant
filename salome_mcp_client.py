"""Client for the code-execution bridge started inside a running SALOME
session by ``salome_mcp_bridge.py`` (see that module's docstring for the
exact wire format: a single JSON object per line over a loopback TCP
socket, not HTTP/JSON-RPC).

Runs in the assistant's own isolated process -- always a separate OS
process from SALOME (see ``salome_plugins.py: runSalomeAssistant``, which
launches it with a stripped env via subprocess.Popen) -- so this talks to
SALOME purely over loopback TCP, discovering the endpoint via a per-user
temp file the bridge writes on startup.

Stdlib-only by design, matching this repo's other GUI-side modules.
"""
import os
import json
import socket
import getpass
import tempfile


class SalomeNotAvailable(RuntimeError):
    """No running SALOME session with the assistant's MCP bridge was found."""


def _discovery_path():
    return os.path.join(
        tempfile.gettempdir(),
        "salome-mcp-bridge-{}.json".format(getpass.getuser()),
    )


def discover():
    """Return (host, port, token), or (None, None, None) if no bridge is discoverable."""
    try:
        with open(_discovery_path()) as f:
            data = json.load(f)
        return data["host"], data["port"], data["token"]
    except (OSError, ValueError, KeyError):
        return None, None, None


def run_python(code, timeout=60):
    """Execute code in the SALOME session's PyConsole interpreter (its
    shared __main__ namespace), via salome_mcp_bridge's TCP/JSON-lines
    protocol.

    Returns {"stdout": str, "stderr": str, "result": str|None, "error": None}.
    Raises SalomeNotAvailable if no bridge is reachable.
    """
    host, port, token = discover()
    if host is None:
        raise SalomeNotAvailable(
            "No running SALOME session found. Start SALOME with the "
            "assistant plugin loaded to enable Run (the MCP bridge starts "
            "automatically with the plugin unless SALOME_MCP_BRIDGE_DISABLE=1).")

    request = (json.dumps({"token": token, "code": code}) + "\n").encode("utf-8")
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.sendall(request)
            with sock.makefile("rb") as rf:
                line = rf.readline()
    except OSError as e:
        raise SalomeNotAvailable(
            "Could not reach the SALOME session ({}). It may have been "
            "closed or restarted.".format(e)) from e

    if not line:
        raise SalomeNotAvailable(
            "SALOME session closed the connection with no response.")

    try:
        response = json.loads(line.decode("utf-8"))
    except ValueError as e:
        raise SalomeNotAvailable(
            "Bad response from SALOME session: {}".format(e)) from e

    return {
        "stdout": response.get("stdout", ""),
        "stderr": response.get("stderr", ""),
        "result": response.get("result"),
        "error": None,
    }
