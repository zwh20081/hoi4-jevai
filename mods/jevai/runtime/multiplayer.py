"""Multiplayer glue without Steam: the pairing code, an orders file on the wire, this PC's role and the margin before a
set applies (docs/superpowers/specs/2026-09-28-multiplayer-design.md; the relay is runner.relay, Steam is
steamlobby.py)."""
from __future__ import annotations

import base64
import json
import os
import zlib

ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"  # 32 symbols without 0/O and 1/I
CODE_LEN = 7  # 32**7 > 2**32: one code per Steam account id
MARGIN_DAYS = 3  # a set applies this many game days after the host's newest game date (about 2 s at speed 5)


def code_for(account_id: int) -> str:
    """The pairing code of a Steam account: its account id (the Steam ID's low 32 bits) in ALPHABET."""
    n, out = account_id & 0xFFFFFFFF, []
    for _ in range(CODE_LEN):
        n, r = divmod(n, len(ALPHABET))
        out.append(ALPHABET[r])
    return "".join(reversed(out))


def valid_code(code: str) -> bool:
    return len(code) == CODE_LEN and all(c in ALPHABET for c in code.upper())


def pack(text: str) -> str:
    """An orders file as lobby data: zlib + base64, about 1 KB for a hundred countries (a value holds 8 KB)."""
    return base64.b64encode(zlib.compress(text.encode("utf-8"), 9)).decode("ascii")


def unpack(wire: str) -> str:
    return zlib.decompress(base64.b64decode(wire)).decode("utf-8")


def load_role(home: str) -> dict | None:
    """This PC's part in multiplayer games (home/multiplayer.json): {"role": "host"} or {"role": "client", "code": ...},
    maybe with "steam_api"; None without a valid one."""
    try:
        with open(os.path.join(home, "multiplayer.json"), encoding="utf-8") as f:
            r = json.load(f)
    except (OSError, ValueError):
        return None
    ok = isinstance(r, dict) and (r.get("role") == "host" or (r.get("role") == "client" and valid_code(str(r.get("code", "")))))
    return r if ok else None


def save_role(home: str, role: str, code: str | None = None, steam_api: str | None = None):
    os.makedirs(home, exist_ok=True)
    r = {"role": role} | ({"code": code.upper()} if code else {}) | ({"steam_api": steam_api} if steam_api else {})
    with open(os.path.join(home, "multiplayer.json"), "w", encoding="utf-8") as f:
        json.dump(r, f)
