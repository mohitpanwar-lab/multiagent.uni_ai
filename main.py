#!/usr/bin/env python
# coding: utf-8

"""
main.py — CLI launcher (terminal chat, no UI)
All the actual logic lives in engine.py so it can be shared with the
Streamlit and Flask UIs. Run this if you just want to chat in the terminal.
"""

from engine import main

if __name__ == "__main__":
    main()
