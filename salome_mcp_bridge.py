# -*- coding: utf-8 -*-
# Copyright (C) 2010-2026  CEA, EDF, OPEN CASCADE
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
#
# See http://www.salome-platform.org/ or email : webmaster.salome@opencascade.com

"""
In-process TCP bridge letting an external MCP server run Python commands
inside the *running* SALOME session, in the same namespace as the
interactive PythonConsole (PyInterp).

This module is started from salome_plugins.py, i.e. it runs inside the
embedded CPython interpreter that also backs PyConsole/PyInterp. There is
a single interpreter/GIL for the whole process, so code executed here via
exec()/eval() against sys.modules['__main__'].__dict__ sees and mutates
the exact same globals the user sees when typing in the Python console
(variables, "salome"/"study" objects, etc. are shared both ways).

Companion piece: mcp/salome_mcp_server.py, a standalone MCP server process
(started by the MCP client, e.g. Claude Code) that talks to this bridge
over the loopback socket described by the token file.
"""

import sys
import os
import io
import json
import socket
import socketserver
import threading
import contextlib
import traceback
import getpass
import secrets
import tempfile

DEFAULT_PORT = 8765

# Serializes MCP-originated requests against each other only. Commands
# typed by hand in the interactive console run on PyInterp_Dispatcher's own
# worker thread and are NOT serialized against this lock -- a command
# arriving from the agent at the exact same instant as one typed in the
# console can race on the shared __main__ namespace. Acceptable for the
# intended single-operator use case; not a guarantee under heavy concurrent use.
_exec_lock = threading.RLock()

_server = None
_server_thread = None


def _token_file():
    return os.path.join(
        tempfile.gettempdir(),
        "salome-mcp-bridge-%s.json" % getpass.getuser(),
    )


def _console_namespace():
    """Same dict PyInterp_Interp uses as global/local context for the console."""
    return sys.modules["__main__"].__dict__


def _run_one(code):
    ns = _console_namespace()
    out, err = io.StringIO(), io.StringIO()
    result = None
    ok = True
    with _exec_lock:
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    compiled = compile(code, "<mcp>", "eval")
                except SyntaxError:
                    compiled = None
                if compiled is not None:
                    result = eval(compiled, ns, ns)
                else:
                    exec(compile(code, "<mcp>", "exec"), ns, ns)
        except Exception:
            ok = False
            err.write(traceback.format_exc())
    return {
        "ok": ok,
        "stdout": out.getvalue(),
        "stderr": err.getvalue(),
        "result": None if result is None else repr(result),
    }


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline()
        if not line:
            return
        try:
            req = json.loads(line.decode("utf-8"))
        except Exception as exc:
            self._reply({"ok": False, "stderr": "bad request: %s" % exc})
            return
        if req.get("token") != self.server.token:
            self._reply({"ok": False, "stderr": "bad token"})
            return
        self._reply(_run_one(req.get("code", "")))

    def _reply(self, payload):
        self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start(port=None):
    """Idempotent: safe to call again (e.g. plugin file reload)."""
    global _server, _server_thread
    if _server is not None:
        return _server

    if port is None:
        port = int(os.environ.get("SALOME_MCP_BRIDGE_PORT", DEFAULT_PORT))
    srv = _Server(("127.0.0.1", port), _Handler)
    srv.token = secrets.token_hex(16)

    th = threading.Thread(target=srv.serve_forever, name="salome-mcp-bridge", daemon=True)
    th.start()

    token_path = _token_file()
    with open(token_path, "w") as f:
        json.dump(
            {"host": "127.0.0.1", "port": srv.server_address[1], "token": srv.token},
            f,
        )
    try:
        os.chmod(token_path, 0o600)
    except OSError:
        pass

    _server, _server_thread = srv, th
    print(
        "[MCP bridge] listening on 127.0.0.1:%d (token file: %s)"
        % (srv.server_address[1], token_path)
    )
    return srv


def stop():
    global _server, _server_thread
    if _server is not None:
        _server.shutdown()
        _server.server_close()
        try:
            os.remove(_token_file())
        except OSError:
            pass
        _server, _server_thread = None, None
