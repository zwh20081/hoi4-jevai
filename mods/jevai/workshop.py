"""Publish the JevAI release folder (mods.jevai.package --out) as a Hearts of Iron IV Steam Workshop item, through the
running Steam client and the game's own steam_api64.dll, as the Paradox launcher uploads mods. The first run creates
the item and writes its id into descriptor.mod as remote_file_id (in this folder and in the release); later runs
update that item. The page text is workshop.txt (Steam BBCode; {id} becomes the item id).

    python -m mods.jevai.workshop temp/release/jevai --note "what changed" [--visibility private]

Needs Steam running, logged in with an account that owns HOI4. A new item starts private.
"""
from __future__ import annotations

import argparse
import ctypes as C
import os
import re
import sys
import time

from .runtime.runner import _read, find_game

HERE = os.path.dirname(os.path.abspath(__file__))
APP = 394360  # Hearts of Iron IV
TITLE = "JevAI - neural network AI strategy"
VISIBILITY = {"public": 0, "friends": 1, "private": 2, "unlisted": 3}
STATUS = ["", "preparing", "preparing content", "uploading content", "uploading preview", "committing"]
u64, P = C.c_uint64, C.c_void_p


class CreateItemResult(C.Structure):  # CreateItemResult_t, callback 3403
    _fields_ = [("result", C.c_int), ("id", u64), ("legal", C.c_bool)]


class SubmitResult(C.Structure):  # SubmitItemUpdateResult_t, callback 3404
    _fields_ = [("result", C.c_int), ("legal", C.c_bool), ("id", u64)]


class Strings(C.Structure):  # SteamParamStringArray_t
    _fields_ = [("strings", C.POINTER(C.c_char_p)), ("n", C.c_int)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Upload the JevAI release folder to the Steam Workshop")
    ap.add_argument("content", help="the release folder (mods.jevai.package --out)")
    ap.add_argument("--note", default="", help="change note on the item's page")
    ap.add_argument("--visibility", choices=VISIBILITY, help="default: private for a new item, unchanged for an update")
    a = ap.parse_args(argv)
    content = os.path.abspath(a.content)
    preview = os.path.join(content, "thumbnail.png")
    descriptors = [os.path.join(HERE, "descriptor.mod"), os.path.join(content, "descriptor.mod")]
    if not all(map(os.path.isfile, descriptors + [preview])):
        sys.exit(f"{content} is not a release folder (descriptor.mod, thumbnail.png)")
    page = _read(os.path.join(HERE, "workshop.txt"))
    if len(page.replace("{id}", "0" * 12).encode()) > 8000:
        sys.exit("workshop.txt is over Steam's 8000-byte description limit")
    game = find_game()
    if not game:
        sys.exit("Hearts of Iron IV not found in the Steam libraries")
    os.environ["SteamAppId"] = os.environ["SteamGameId"] = str(APP)
    api = C.CDLL(os.path.join(os.path.dirname(game), "steam_api64.dll"))

    def fn(name, restype, *argtypes):
        f = getattr(api, name)
        f.restype, f.argtypes = restype, list(argtypes)
        return f

    def newest(prefix):  # the flat methods call through the DLL's own interface version: the newest one it exports
        return fn(next(n for n in (f"{prefix}{v:03d}" for v in range(40, 0, -1)) if hasattr(api, n)), P)()

    err = C.create_string_buffer(1024)
    if fn("SteamAPI_InitFlat", C.c_int, C.c_char_p)(err) != 0:
        sys.exit(f"Steam: {err.value.decode(errors='replace')} (running, and logged in with an account that owns HOI4?)")
    ugc, utils = newest("SteamAPI_SteamUGC_v"), newest("SteamAPI_SteamUtils_v")
    run_callbacks = fn("SteamAPI_RunCallbacks", None)
    completed = fn("SteamAPI_ISteamUtils_IsAPICallCompleted", C.c_bool, P, u64, C.POINTER(C.c_bool))
    call_result = fn("SteamAPI_ISteamUtils_GetAPICallResult", C.c_bool, P, u64, P, C.c_int, C.c_int, C.POINTER(C.c_bool))

    def wait(call, out, callback, tick=lambda: None):
        failed = C.c_bool()
        while not completed(utils, call, C.byref(failed)):
            run_callbacks()
            tick()
            time.sleep(0.2)
        if not call_result(utils, call, C.byref(out), C.sizeof(out), callback, C.byref(failed)) or out.result != 1:
            sys.exit(f"Steam call {callback} failed: EResult {out.result} (partner.steamgames.com/doc/api/steam_api#EResult)")
        if out.legal:
            print("Accept the Workshop legal agreement, or the item stays hidden: "
                  "https://steamcommunity.com/sharedfiles/workshoplegalagreement")
        return out

    m = re.search(r'^remote_file_id="(\d+)"', _read(descriptors[0]), re.M)
    new = m is None
    if new:
        fid = wait(fn("SteamAPI_ISteamUGC_CreateItem", u64, P, C.c_uint32, C.c_int)(ugc, APP, 0),  # 0: community item
                   CreateItemResult(), 3403).id
        print(f"created Workshop item {fid}", flush=True)
    else:
        fid = int(m.group(1))
    for d in descriptors:  # the launcher links an installed mod to its Workshop item through remote_file_id
        text = _read(d)
        if f'remote_file_id="{fid}"' not in text:
            with open(d, "w", encoding="utf-8", newline="\n") as f:
                f.write(re.sub(r"(?m)^(name\s*=.*\n)", rf'\g<1>remote_file_id="{fid}"\n', text, count=1))

    h = fn("SteamAPI_ISteamUGC_StartItemUpdate", u64, P, C.c_uint32, u64)(ugc, APP, fid)
    tags = re.findall(r'"([^"]+)"', re.search(r"tags\s*=\s*\{([^}]*)\}", _read(descriptors[1])).group(1))
    arr = (C.c_char_p * len(tags))(*(t.encode() for t in tags))
    def text_field(k, v):
        return fn(f"SteamAPI_ISteamUGC_SetItem{k}", C.c_bool, P, u64, C.c_char_p)(ugc, h, v.encode())

    ok = {"title": text_field("Title", TITLE),
          "description": text_field("Description", page.replace("{id}", str(fid))),
          "content": text_field("Content", content),
          "preview": text_field("Preview", preview),
          "tags": fn("SteamAPI_ISteamUGC_SetItemTags", C.c_bool, P, u64, C.POINTER(Strings), C.c_bool)(
              ugc, h, C.byref(Strings(C.cast(arr, C.POINTER(C.c_char_p)), len(tags))), False)}
    if new or a.visibility:
        ok["visibility"] = fn("SteamAPI_ISteamUGC_SetItemVisibility", C.c_bool, P, u64, C.c_int)(
            ugc, h, VISIBILITY[a.visibility or "private"])
    if not all(ok.values()):
        sys.exit(f"Steam refused: {', '.join(k for k, v in ok.items() if not v)}")

    progress = fn("SteamAPI_ISteamUGC_GetItemUpdateProgress", C.c_int, P, u64, C.POINTER(u64), C.POINTER(u64))
    shown = [0.0]

    def tick():
        if time.time() - shown[0] >= 15:
            shown[0] = time.time()
            done, total = u64(), u64()
            s = progress(ugc, h, C.byref(done), C.byref(total))
            print(f"  {STATUS[s] if 0 <= s < len(STATUS) else s}: {done.value / 2**20:,.0f} of {total.value / 2**20:,.0f} MB",
                  flush=True)

    wait(fn("SteamAPI_ISteamUGC_SubmitItemUpdate", u64, P, u64, C.c_char_p)(ugc, h, a.note.encode()), SubmitResult(),
         3404, tick)
    fn("SteamAPI_Shutdown", None)()
    print(f"uploaded: https://steamcommunity.com/sharedfiles/filedetails/?id={fid}")


if __name__ == "__main__":
    main()
