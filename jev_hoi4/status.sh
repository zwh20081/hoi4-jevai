#!/bin/bash
# Status of the parallel collection (see jevai/status.py).
cd "$(dirname "$0")/.." && PYTHONPATH=jev_hoi4 .venv/Scripts/python.exe -m jevai.status
