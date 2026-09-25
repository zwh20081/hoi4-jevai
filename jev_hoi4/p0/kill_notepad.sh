#!/bin/bash
# -debug opens error.log in notepad whenever an error is logged; that steals focus from hoi4. Keep closing it.
while true; do taskkill //IM notepad.exe //F >/dev/null 2>&1; sleep 2; done
