#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2025  CEA
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
Entry point for the salome-assistant console script.

This module is the target of the [project.scripts] entry in pyproject.toml:
    salome-assistant = "salome_assistant_run:main"

It launches the SALOME Assistant Qt application in an environment isolated
from SALOME's own embedded Python (the venv in which this package is installed
provides that isolation at the OS level).
"""

import sys
import os


def main():
    # Ensure the directory containing this file (and salomeAssistant.py /
    # rag_engine.py) is on sys.path so that local imports resolve correctly
    # regardless of the current working directory.
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    from PyQt5.QtWidgets import QApplication
    from salomeAssistant import MainWindow

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QGroupBox { font-weight: bold; }
        QTextEdit  { font-size: 14px; }
        QLineEdit  { padding: 5px; font-size: 14px; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
