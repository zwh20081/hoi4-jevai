"""Build the JevAI release: the mod folder with the runner frozen into jevai.exe (OpenVINO runtime bundled, no Python
needed) and the exported model next to it.

    python -m mods.jevai.package --model temp/ov/hoi4-v1 --out temp/release/jevai

Layout of the release (copy the folder into Documents/Paradox Interactive/Hearts of Iron IV/mod/ and add jevai.mod):
    jevai/descriptor.mod, common/, events/, localisation/, history/  the HOI4 mod
    jevai/runner/jevai.exe (+ _internal/), *.cmd                    the runner and its two launch shortcuts
    jevai/runner/model/                                             jev_npu.xml/.bin, jev_cpu.xml/.bin, tokenizer.json, jev.json
    jevai.mod                                                       descriptor for the launcher (path = mod/jevai)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the JevAI release folder")
    ap.add_argument("--model", required=True, help="exported model folder (trainer.export --out)")
    ap.add_argument("--out", default=os.path.join(ROOT, "temp", "release", "jevai"))
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for part in ("descriptor.mod", "common", "events", "history", "localisation"):
        src = os.path.join(HERE, part)
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(out, part))
    work = os.path.join(ROOT, "temp", "release", "build")
    entry = os.path.join(work, "jevai_main.py")
    os.makedirs(work, exist_ok=True)
    with open(entry, "w", encoding="utf-8") as f:
        f.write("from mods.jevai.runtime.runner import main\nmain()\n")
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--console", "--name", "jevai",
                    "--distpath", os.path.join(out, "runner_dist"), "--workpath", os.path.join(work, "pyi"),
                    "--specpath", work, "--paths", ROOT, "--collect-all", "openvino", "--collect-binaries", "tokenizers",
                    entry], check=True)
    shutil.move(os.path.join(out, "runner_dist", "jevai"), os.path.join(out, "runner"))
    os.rmdir(os.path.join(out, "runner_dist"))
    shutil.copytree(a.model, os.path.join(out, "runner", "model"))
    for name, args, note in (("start_hoi4_with_jevai.cmd", "", "Starts HOI4 and the JevAI model together."),
                             ("attach_to_running_game.cmd", " --no-launch", "Use when HOI4 was started from the Paradox launcher.")):
        with open(os.path.join(out, "runner", name), "w", encoding="utf-8", newline="\r\n") as f:
            f.write(f'@echo off\nrem {note}\n"%~dp0jevai.exe"{args}\npause\n')
    with open(os.path.join(HERE, "descriptor.mod"), encoding="utf-8-sig") as f:
        desc = f.read().rstrip()
    with open(os.path.join(os.path.dirname(out), "jevai.mod"), "w", encoding="utf-8") as f:
        f.write(desc + '\npath="mod/jevai"\n')
    size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(out) for fn in fns)
    print(f"release in {out} ({size / 2**30:.2f} GB) + {os.path.join(os.path.dirname(out), 'jevai.mod')}")


if __name__ == "__main__":
    main()
