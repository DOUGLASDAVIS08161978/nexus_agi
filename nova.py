#!/usr/bin/env python3
"""
nova.py — Always launches the latest Nova version.
Douglas: just run  python nova.py
"""
import runpy, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
runpy.run_module("nova_asi_v29", run_name="__main__", alter_sys=True)
