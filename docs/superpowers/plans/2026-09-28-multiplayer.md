# JevAI Multiplayer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** JevAI steers the AI countries of a HOI4 multiplayer game without desyncs: the host's runner decides, and every player's game applies the same orders on the same game day.

**Architecture:** The host's runner writes its orders file with a target date in a `date >` guard; a relay child process (`jevai.exe --relay`) is the only process that loads `steam_api64.dll` and carries the file, compressed, through an invisible Steam lobby tagged with the host's pairing code; each client's relay writes it byte for byte to its own orders file. Roles (`--host`, `--join CODE`) live in `%LOCALAPPDATA%\jevai\multiplayer.json`; a multiplayer game without a role keeps JevAI off.

**Tech Stack:** Python 3.13 (stdlib only: ctypes, zlib, base64, json), Steamworks flat C API through `ctypes`, HOI4 script, PyInstaller release (`mods.jevai.package`).

**Spec:** `docs/superpowers/specs/2026-09-28-multiplayer-design.md`

## Global Constraints

- Windows 10/11; run checks and packaging with `C:/Users/33489/.conda/envs/ov/python.exe` (called `PY` below) from the repo root.
- `mods/jevai/runtime/` uses relative imports only (it ships inside the mod); no new third-party packages.
- HOI4 Steam app id `394360`; a lobby data value holds at most 8 KB; a set applies `MARGIN_DAYS = 3` game days after the host's newest game date.
- Only the relay process (and `jevai.exe --host` for a moment) loads `steam_api64.dll`.
- Checks live in `temp/checks/check_runner.py` (gitignored, no framework); `PY -m temp.checks.check_runner` must end with `all passed`. Its helpers `check`, `fresh`, `write`, `captured`, `stamp`, `jev`, `burst`, `header`, `run_session`, `MONTHLY`, `MAJORS` already exist.
- Checks never stop the real runner, touch the real Startup shortcut or the real `%LOCALAPPDATA%\jevai`: patch `runner.HOME`, `runner.STARTUP_LINK`, `runner.stop_others`.
- No HOI4 window during Tasks 1-7 without the maintainer's OK (it takes the screen).
- Committed files must not describe the maintainer's own Steam DLL setup.
- Code style: match `runner.py` (docstrings that say why, lines up to ~120 columns, no new abstractions).

## File Structure

- Create `mods/jevai/runtime/steamlobby.py`: ctypes wrapper for the Steam lobby calls (init as HOI4, create/find/join, lobby data). Nothing loads until `Lobby.open()`.
- Create `mods/jevai/runtime/multiplayer.py`: Steam-free glue: pairing code, wire format, role file, margin.
- Modify `mods/jevai/runtime/runner.py`: `day_date`, dated `orders_file`, `GameLog.now/.multiplayer`, `me`, `default_steam_api`, `start_relay`, `publish`, `fetch`, `relay`, `set_role`, `session`, `main`.
- Modify `mods/jevai/package.py`: `host.cmd`, `join.cmd` next to `install.cmd`.
- Modify `README.md`, `mods/jevai/workshop.txt`, `CLAUDE.md`: multiplayer instructions.
- Test-only (gitignored): `temp/checks/lobby_spike.py`, additions to `temp/checks/check_runner.py`.

---

### Task 1: Steam lobby wrapper and spike

**Files:**
- Create: `mods/jevai/runtime/steamlobby.py`
- Create (throwaway): `temp/checks/lobby_spike.py`

**Interfaces:**
- Produces: `steamlobby.SteamError(RuntimeError)`; `steamlobby.Lobby(dll: str)` with `.open()`, `.account_id() -> int`, `.create(code: str) -> int`, `.find(code: str) -> int` (0 if none), `.join(lobby: int)`, `.set(key: str, value: str) -> bool`, `.get(key: str) -> str`, `.tick()`, attribute `.lobby: int`.

- [ ] **Step 1: Write `mods/jevai/runtime/steamlobby.py`**

```python
"""Steam lobbies through a steam_api64.dll's flat C API: the channel between JevAI runners in a multiplayer game
(docs/superpowers/specs/2026-09-28-multiplayer-design.md). Only call results, polled; no callbacks, which older DLLs
cannot hand to Python. Nothing loads until Lobby.open(), and only the relay process calls it: a replaced or old DLL
can crash the process that loads it."""
from __future__ import annotations

import ctypes as C
import os
import re
import tempfile
import time

APP = 394360  # Hearts of Iron IV
KEY = "jevai"  # the lobby data key holding the host's pairing code
INVISIBLE, EQUAL, WORLDWIDE = 3, 0, 3  # ELobbyType, ELobbyComparison, ELobbyDistanceFilter
u64, P = C.c_uint64, C.c_void_p


class LobbyCreated(C.Structure):  # LobbyCreated_t, callback 513
    _fields_ = [("result", C.c_int), ("lobby", u64)]


class LobbyEnter(C.Structure):  # LobbyEnter_t, callback 504
    _fields_ = [("lobby", u64), ("permissions", C.c_uint32), ("locked", C.c_bool), ("response", C.c_uint32)]


class LobbyMatchList(C.Structure):  # LobbyMatchList_t, callback 510
    _fields_ = [("count", C.c_uint32)]


class SteamError(RuntimeError):
    pass


class Lobby:
    def __init__(self, dll: str):
        self.dll, self.lobby = dll, 0

    def _fn(self, name, restype, *argtypes):
        f = getattr(self.api, name)
        f.restype, f.argtypes = restype, list(argtypes)
        return f

    def _iface(self, accessor: str, getter: str, version: str):
        """An interface whose layout matches this DLL's flat methods: its own newest accessor (SDK 1.48+), else through
        ISteamClient with the version the DLL was built for (named inside it, else `version`)."""
        found = [v for v in range(1, 40) if hasattr(self.api, f"{accessor}{v:03d}")]
        if found:
            return self._fn(f"{accessor}{found[-1]:03d}", P)()
        m = re.search(re.escape(version[:-3].encode()) + rb"\d{3}", self.raw)
        client = self._fn("SteamClient", P)()
        user, pipe = self._fn("SteamAPI_GetHSteamUser", C.c_int)(), self._fn("SteamAPI_GetHSteamPipe", C.c_int)()
        return self._fn(getter, P, P, C.c_int, C.c_int, C.c_char_p)(client, user, pipe, m.group() if m else version.encode())

    def open(self):
        """Start the Steam API as HOI4 (steam_appid.txt in a temporary working folder) and get the interfaces."""
        if not os.path.isfile(self.dll):
            raise SteamError(f"no steam_api64.dll at {self.dll}")
        with open(self.dll, "rb") as f:
            self.raw = f.read()
        self.api = C.CDLL(os.path.abspath(self.dll))
        cwd, err, ok = os.getcwd(), C.create_string_buffer(1024), False
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with open(os.path.join(tmp, "steam_appid.txt"), "w") as f:
                f.write(str(APP))
            os.chdir(tmp)
            try:
                if hasattr(self.api, "SteamAPI_InitFlat"):  # SDK 1.58+
                    ok = self._fn("SteamAPI_InitFlat", C.c_int, P)(C.addressof(err)) == 0
                else:
                    ok = self._fn("SteamAPI_Init", C.c_bool)()
            except OSError as e:  # an access violation inside the DLL
                raise SteamError(f"{self.dll} crashed ({e})") from e
            finally:
                os.chdir(cwd)
        if not ok:
            raise SteamError(f"Steam API init failed {err.value.decode(errors='replace')}: is Steam running and logged in?")
        self.mm = self._iface("SteamAPI_SteamMatchmaking_v", "SteamAPI_ISteamClient_GetISteamMatchmaking", "SteamMatchMaking009")
        self.utils = self._iface("SteamAPI_SteamUtils_v", "SteamAPI_ISteamClient_GetISteamUtils", "SteamUtils009")
        self.user = self._iface("SteamAPI_SteamUser_v", "SteamAPI_ISteamClient_GetISteamUser", "SteamUser020")
        self.tick = self._fn("SteamAPI_RunCallbacks", None)

    def _wait(self, call: int, out: C.Structure, callback: int, timeout: float = 30.0):
        if not call:
            raise SteamError(f"Steam refused call {callback}")
        completed = self._fn("SteamAPI_ISteamUtils_IsAPICallCompleted", C.c_bool, P, u64, C.POINTER(C.c_bool))
        result = self._fn("SteamAPI_ISteamUtils_GetAPICallResult", C.c_bool, P, u64, P, C.c_int, C.c_int, C.POINTER(C.c_bool))
        failed, t = C.c_bool(), time.time()
        while not completed(self.utils, call, C.byref(failed)):
            if time.time() - t > timeout:
                raise SteamError(f"Steam call {callback} timed out")
            self.tick()
            time.sleep(0.1)
        if not result(self.utils, call, C.byref(out), C.sizeof(out), callback, C.byref(failed)) or failed.value:
            raise SteamError(f"Steam call {callback} failed")
        return out

    def account_id(self) -> int:
        return self._fn("SteamAPI_ISteamUser_GetSteamID", u64, P)(self.user) & 0xFFFFFFFF

    def create(self, code: str) -> int:
        """An invisible lobby (found by search, not shown to friends) tagged with the pairing code."""
        call = self._fn("SteamAPI_ISteamMatchmaking_CreateLobby", u64, P, C.c_int, C.c_int)(self.mm, INVISIBLE, 16)
        r = self._wait(call, LobbyCreated(), 513)
        if r.result != 1:
            raise SteamError(f"CreateLobby: EResult {r.result}")
        self.lobby = r.lobby
        if not self.set(KEY, code):
            raise SteamError("could not tag the lobby with its code")
        return self.lobby

    def find(self, code: str) -> int:
        """The lobby tagged with this code, or 0."""
        self._fn("SteamAPI_ISteamMatchmaking_AddRequestLobbyListStringFilter", None, P, C.c_char_p, C.c_char_p, C.c_int)(
            self.mm, KEY.encode(), code.encode(), EQUAL)
        self._fn("SteamAPI_ISteamMatchmaking_AddRequestLobbyListDistanceFilter", None, P, C.c_int)(self.mm, WORLDWIDE)
        r = self._wait(self._fn("SteamAPI_ISteamMatchmaking_RequestLobbyList", u64, P)(self.mm), LobbyMatchList(), 510)
        return self._fn("SteamAPI_ISteamMatchmaking_GetLobbyByIndex", u64, P, C.c_int)(self.mm, 0) if r.count else 0

    def join(self, lobby: int):
        r = self._wait(self._fn("SteamAPI_ISteamMatchmaking_JoinLobby", u64, P, u64)(self.mm, lobby), LobbyEnter(), 504)
        if r.response != 1:  # k_EChatRoomEnterResponseSuccess
            raise SteamError(f"JoinLobby: response {r.response}")
        self.lobby = lobby

    def set(self, key: str, value: str) -> bool:
        return self._fn("SteamAPI_ISteamMatchmaking_SetLobbyData", C.c_bool, P, u64, C.c_char_p, C.c_char_p)(
            self.mm, self.lobby, key.encode(), value.encode())

    def get(self, key: str) -> str:
        v = self._fn("SteamAPI_ISteamMatchmaking_GetLobbyData", C.c_char_p, P, u64, C.c_char_p)(self.mm, self.lobby, key.encode())
        return (v or b"").decode()
```

- [ ] **Step 2: Write the spike `temp/checks/lobby_spike.py`**

```python
"""Spike (Task 1 of docs/superpowers/plans/2026-09-28-multiplayer.md): the Steam lobby calls from a second process.
Terminal 1: host; terminal 2: client with the printed code.

    PY -m temp.checks.lobby_spike host --steam-api DLL [--seconds 60]
    PY -m temp.checks.lobby_spike client CODE --steam-api DLL [--seconds 60]
"""
import argparse
import sys
import time

from mods.jevai.runtime import steamlobby


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["host", "client"])
    ap.add_argument("code", nargs="?")
    ap.add_argument("--steam-api", required=True)
    ap.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()
    lobby = steamlobby.Lobby(a.steam_api)
    t = time.time()
    lobby.open()
    print(f"open: {time.time() - t:.1f}s, account id {lobby.account_id()}", flush=True)
    end = time.time() + a.seconds
    if a.role == "host":
        code = f"SPIKE{lobby.account_id() % 100:02d}"
        t = time.time()
        lobby.create(code)
        print(f"lobby {lobby.lobby} created in {time.time() - t:.1f}s; code {code}", flush=True)
        while time.time() < end:
            print(f"set: {lobby.set('orders', f'{time.time():.3f}')}", flush=True)
            lobby.tick()
            time.sleep(2)
        return 0
    t = time.time()
    found = lobby.find(a.code)
    print(f"find: lobby {found} in {time.time() - t:.1f}s", flush=True)
    if not found:
        return 1
    try:
        t = time.time()
        lobby.join(found)
        print(f"joined in {time.time() - t:.1f}s", flush=True)
    except steamlobby.SteamError as e:
        print(f"join: {e} (expected with the host's own account); reading as a non-member", flush=True)
        lobby.lobby = found
    last = None
    while time.time() < end:
        lobby.tick()
        v = lobby.get("orders")
        if v and v != last:
            print(f"update seen {time.time() - float(v):.2f}s after it was set", flush=True)
            last = v
        time.sleep(0.2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run the spike with an SDK 1.48+ `steam_api64.dll`** (Steam running and logged in; any stock DLL from another Steam app works)

Run in two terminals: `PY -m temp.checks.lobby_spike host --steam-api "<SDK 1.48+ DLL>" --seconds 40`, then `PY -m temp.checks.lobby_spike client <printed code> --steam-api "<same DLL>" --seconds 30`.
Expected: host prints `open: ...s, account id N`, `lobby L created`, `set: True` lines; client prints `find: lobby L` (not 0). With one Steam account the join may fail and updates may not show; note it, it is covered in Task 8.

- [ ] **Step 4: Run the spike with a pre-1.48 stock DLL** (HOI4's own `steam_api64.dll` if it is Valve's; else a stock copy of the same version)

Same commands with that DLL. Expected: the same lines (this exercises the `SteamClient()` path of `_iface`).

- [ ] **Step 5: Run the spike once more with HOI4 running** (ask the maintainer to start HOI4; the main menu is enough; never start it yourself)

Expected: the same lines while `hoi4.exe` runs. **If any of Steps 3-5 fails (no lobby, `set: False`, a crash), stop and report to the maintainer; the relay design depends on it.**

- [ ] **Step 6: Commit**

```bash
git add mods/jevai/runtime/steamlobby.py
git commit -m "runtime.steamlobby: Steam lobby calls for the multiplayer relay (spiked with both DLL kinds)"
```

---

### Task 2: Pairing code, wire format and role file

**Files:**
- Create: `mods/jevai/runtime/multiplayer.py`
- Test: `temp/checks/check_runner.py`

**Interfaces:**
- Produces: `multiplayer.ALPHABET: str`, `CODE_LEN = 7`, `MARGIN_DAYS = 3`, `code_for(account_id: int) -> str`, `valid_code(code: str) -> bool`, `pack(text: str) -> str`, `unpack(wire: str) -> str`, `load_role(home: str) -> dict | None`, `save_role(home: str, role: str, code: str | None = None, steam_api: str | None = None)`.

- [ ] **Step 1: Write the failing check** (add to `temp/checks/check_runner.py`; add `from mods.jevai.runtime import multiplayer` to its imports and `check_multiplayer` to the list in `main()`)

```python
def check_multiplayer():
    codes = [multiplayer.code_for(i) for i in (0, 1, 31, 32, 2**31, 2**32 - 1)]
    check("code_for: 7 symbols of the alphabet, one per account", len(set(codes)) == len(codes)
          and all(len(c) == 7 and set(c) <= set(multiplayer.ALPHABET) for c in codes), codes)
    check("code_for: only the account id (the Steam ID's low 32 bits) counts",
          multiplayer.code_for(76561198000000123) == multiplayer.code_for(76561198000000123 & 0xFFFFFFFF))
    check("valid_code: codes in any case, nothing else",
          all(multiplayer.valid_code(c) and multiplayer.valid_code(c.lower()) for c in codes)
          and not any(multiplayer.valid_code(s) for s in ("", "ABC", "ABCDEFGH", "0000000", "IIIIIII")))
    text = "# JevAI orders for 1936.7.1 (NPU), from 1936.7.4\ninstant_effect = {\n" + "".join(
        f"\t\tif = {{ limit = {{ country_exists = T{i:02d} }} T{i:02d} = {{ set_variable = {{ jev_want = {i % 7} }} }} }}\n"
        for i in range(100)) + "}\n"
    wire = multiplayer.pack(text)
    check("pack/unpack: the exact orders file, far under a lobby value's 8 KB",
          multiplayer.unpack(wire) == text and wire.isascii() and len(wire) < 8000, len(wire))
    h = os.path.join(fresh("mp_roles"), "home")
    check("load_role: none before any", multiplayer.load_role(h) is None)
    multiplayer.save_role(h, "host")
    check("save_role/load_role: host", multiplayer.load_role(h) == {"role": "host"}, multiplayer.load_role(h))
    multiplayer.save_role(h, "client", "abcdefg", "D:/x/steam_api64.dll")
    check("save_role/load_role: client, the code upper-cased, the DLL kept",
          multiplayer.load_role(h) == {"role": "client", "code": "ABCDEFG", "steam_api": "D:/x/steam_api64.dll"},
          multiplayer.load_role(h))
    write(os.path.join(h, "multiplayer.json"), '{"role": "client", "code": "nope"}')
    check("load_role: a client without a valid code is no role", multiplayer.load_role(h) is None)
    write(os.path.join(h, "multiplayer.json"), "{garbage")
    check("load_role: an unreadable file is no role", multiplayer.load_role(h) is None)
```

- [ ] **Step 2: Run it and see it fail**

Run: `PY -m temp.checks.check_runner`
Expected: `FAIL check_multiplayer  -> ...` (ImportError / no module `multiplayer`).

- [ ] **Step 3: Write `mods/jevai/runtime/multiplayer.py`**

```python
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
```

- [ ] **Step 4: Run and see it pass**

Run: `PY -m temp.checks.check_runner`
Expected: every `check_multiplayer` line `PASS`, last line `all passed`.

- [ ] **Step 5: Commit**

```bash
git add mods/jevai/runtime/multiplayer.py
git commit -m "runtime.multiplayer: pairing code, wire format and role file"
```

---

### Task 3: Target dates in the orders file; GameLog knows the session kind and the game date

**Files:**
- Modify: `mods/jevai/runtime/runner.py` (`game_day` neighbourhood, `GameLog`, `orders_file`)
- Test: `temp/checks/check_runner.py`

**Interfaces:**
- Produces: `runner.day_date(day: int) -> str`; `runner.orders_file(postures: dict[str, int], date: str, device: str, target: str | None = None) -> str`; `GameLog.now: int` (game day of the newest dated line, 0 before any) and `GameLog.multiplayer: bool | None` (None before a session starts).

- [ ] **Step 1: Write the failing checks** (add, and list both functions in `main()`)

```python
def check_orders_dates():
    days = range(runner.game_day("1936.1.1"), runner.game_day("1938.1.1"))
    bad = [d for d in days if runner.game_day(runner.day_date(d)) != d]
    check("day_date: game_day's inverse over two years", not bad
          and runner.day_date(runner.game_day("1936.3.1") - 1) == "1936.2.28", bad[:3])
    text = runner.orders_file({"GER": 2, "ITA": 0}, "1936.7.1", "NPU", "1936.7.4")
    check("orders_file: the set applies from its target day", "date > 1936.7.3" in text
          and "GER = { set_variable = { jev_want = 2 } }" in text and text.count("{") == text.count("}"), text)
    check("orders_file: without a target, from the update's own day (single-player: at once)",
          "date > 1936.2.28" in runner.orders_file({"GER": 1}, "1936.3.1", "CPU"))
    check("orders_file: an empty set has no date guard", "date >" not in runner.orders_file({}, "no update yet", "-"))


def check_gamelog_session():
    u = fresh("gamelog_session")
    path = os.path.join(u, "logs", "game.log")
    write(path, stamp("1936.1.1") + "Executing History\n")
    log = runner.GameLog(u)
    log.poll()
    check("GameLog: no game session yet", log.multiplayer is None)
    write(path, stamp("1936.6.1") + "\n\n\t\t[[ Launching MULTIPLAYER-game ]]\n\tStart-date: 1936.6.1.2\n"
          + stamp("1936.6.5") + "tick\n", "a")
    log.poll()
    check("GameLog: a multiplayer session and the newest game date",
          log.multiplayer is True and log.now == runner.game_day("1936.6.5"), (log.multiplayer, log.now))
    write(path, "\t\t[[ Launching SINGLEPLAYER-game ]]\n", "a")
    log.poll()
    check("GameLog: then a single-player session", log.multiplayer is False)
    write(path, "x\n")
    log.poll()
    check("GameLog: a new game.log forgets both", log.multiplayer is None and log.now == 0, (log.multiplayer, log.now))
```

- [ ] **Step 2: Run and see them fail**

Run: `PY -m temp.checks.check_runner`
Expected: `FAIL check_orders_dates -> AttributeError: ... 'day_date'` and `FAIL check_gamelog_session -> AttributeError: ... 'multiplayer'`.

- [ ] **Step 3: Add `day_date` right after `game_day` in `runner.py`**

```python
def day_date(day: int) -> str:
    """game_day's inverse: the game date "y.m.d" of a day count."""
    y, r = divmod(day - 1, 365)
    m = max(i for i, s in enumerate(MONTH_START) if s <= r)
    return f"{y}.{m + 1}.{r - MONTH_START[m] + 1}"
```

- [ ] **Step 4: Teach `GameLog` the session kind and the newest date**

In `GameLog.reset`, replace the line with:

```python
        self.pos, self.mode, self.period, self.bursts, self.latest, self.days, self.jev = 0, None, None, {}, {}, set(), False
        self.now, self.multiplayer = 0, None  # newest game day; None until the game logs "[[ Launching ...-game ]]"
```

In `GameLog.poll`, replace the start of the line loop:

```python
        for line in chunk[:end].decode("utf-8", "replace").splitlines():
            m = LINE.match(line)
            if not m:
                continue
            date = tuple(map(int, m.groups()[:3]))
```

with:

```python
        for line in chunk[:end].decode("utf-8", "replace").splitlines():
            if "[[ Launching " in line:  # a game session starts: single- or multiplayer
                self.multiplayer = "MULTIPLAYER" in line
                continue
            m = LINE.match(line)
            if not m:
                continue
            date = tuple(map(int, m.groups()[:3]))
            self.now = 365 * date[0] + MONTH_START[date[1] - 1] + date[2]
```

- [ ] **Step 5: Replace `orders_file` in `runner.py`**

```python
def orders_file(postures: dict[str, int], date: str, device: str, target: str | None = None) -> str:
    """An OOB file whose instant_effect sets each country's wanted posture; the mod applies it to AI countries.
    It holds no units, so loading it changes nothing but those variables. The set takes effect from `target` (default:
    the update's own day, at once): every game of a multiplayer session reloads the file daily and applies the set
    on the same game day, since the postures are synchronized state."""
    lines = [f"# JevAI orders for {date} ({device}), from {target or date}; rewritten by the runner, reloaded daily by "
             "the mod\n", "instant_effect = {\n"]
    if postures:
        lines.append(f"\tif = {{ limit = {{ date > {day_date(game_day(target or date) - 1)} }}\n")
        for tag, k in sorted(postures.items()):
            lines.append(f"\t\tif = {{ limit = {{ country_exists = {tag} }} {tag} = {{ set_variable = {{ jev_want = {k} }} }} }}\n")
        lines.append("\t}\n")
    lines.append("}\n")
    return "".join(lines)
```

- [ ] **Step 6: Run and see everything pass**

Run: `PY -m temp.checks.check_runner`
Expected: `all passed` (the session checks still pass: their orders now sit inside the date guard).

- [ ] **Step 7: Commit**

```bash
git add mods/jevai/runtime/runner.py
git commit -m "runner: orders apply from a target date; GameLog knows the session kind and the game date"
```

---

### Task 4: The relay

**Files:**
- Modify: `mods/jevai/runtime/runner.py` (imports, `child_command`, new functions after `write_atomic`)
- Test: `temp/checks/check_runner.py`

**Interfaces:**
- Consumes: `steamlobby.Lobby`, `steamlobby.SteamError` (Task 1); `multiplayer.code_for`, `pack`, `unpack` (Task 2).
- Produces: `runner.me() -> list[str]`; `runner.default_steam_api() -> str`; `runner.start_relay(a, role: dict, orders: str) -> subprocess.Popen`; `runner.publish(lobby, orders: str, last: str | None) -> str | None`; `runner.fetch(lobby, orders: str, last: str | None) -> str | None`; `runner.relay(role: str, orders: str, code: str | None, dll: str) -> int` (returns 1 on a Steam failure, else runs until killed).

- [ ] **Step 1: Write the failing check** (add; list `check_relay` in `main()`)

```python
class StopRelay(Exception):
    pass


class FakeLobby:
    """Steam's lobby service in memory: every instance shares `store` (lobby id -> data)."""
    store, fail, max_ticks = {}, None, 2

    def __init__(self, dll):
        self.dll, self.lobby, self.ticks, self.sets = dll, 0, 0, 0

    def open(self):
        if FakeLobby.fail:
            raise runner.steamlobby.SteamError(FakeLobby.fail)

    def account_id(self):
        return 12345

    def create(self, code):
        self.lobby = 7
        FakeLobby.store[7] = {"jevai": code}
        return 7

    def find(self, code):
        return next((i for i, d in FakeLobby.store.items() if d.get("jevai") == code), 0)

    def join(self, lobby):
        self.lobby = lobby

    def set(self, key, value):
        self.sets += 1
        FakeLobby.store[self.lobby][key] = value
        return True

    def get(self, key):
        return FakeLobby.store.get(self.lobby, {}).get(key, "")

    def tick(self):
        self.ticks += 1
        if self.ticks > FakeLobby.max_ticks:
            raise StopRelay


def check_relay():
    u = fresh("relay")
    host_orders, client_orders = os.path.join(u, "host", "JEVAI_orders.txt"), os.path.join(u, "client", "JEVAI_orders.txt")
    text = runner.orders_file({"GER": 2}, "1936.7.1", "NPU", "1936.7.4")
    write(host_orders, text)
    os.makedirs(os.path.dirname(client_orders), exist_ok=True)
    FakeLobby.store, FakeLobby.fail = {7: {"jevai": "X"}}, None
    host = FakeLobby("dll")
    host.lobby = 7
    last = runner.publish(host, host_orders, None)
    again = runner.publish(host, host_orders, last)
    check("publish: sends a changed orders file once", last == text and again == last and host.sets == 1
          and runner.multiplayer.unpack(FakeLobby.store[7]["orders"]) == text, host.sets)
    client = FakeLobby("dll")
    client.lobby = 7
    seen = runner.fetch(client, client_orders, None)
    check("fetch: writes the host's file byte for byte",
          open(client_orders, encoding="utf-8").read() == text and seen == FakeLobby.store[7]["orders"])
    os.utime(client_orders, (1, 1))
    runner.fetch(client, client_orders, seen)
    check("fetch: leaves an unchanged set alone", os.path.getmtime(client_orders) == 1)
    saved_home, saved_lobby = runner.HOME, runner.steamlobby.Lobby
    runner.HOME, runner.steamlobby.Lobby = os.path.join(u, "home"), FakeLobby
    os.makedirs(runner.HOME, exist_ok=True)
    logfile = os.path.join(runner.HOME, "jevai.log")
    try:
        FakeLobby.store = {}
        try:
            runner.relay("host", host_orders, None, "dll")
        except StopRelay:
            pass
        code = runner.multiplayer.code_for(12345)
        logtext = open(logfile, encoding="utf-8").read()
        check("relay host: opens the lobby with its code and sends the orders", f"--join {code}" in logtext
              and "relay: sent JevAI orders for 1936.7.1" in logtext
              and runner.multiplayer.unpack(FakeLobby.store[7]["orders"]) == text, logtext)
        try:
            runner.relay("client", client_orders + ".2", code, "dll")
        except StopRelay:
            pass
        check("relay client: finds the host by its code and writes its orders",
              open(client_orders + ".2", encoding="utf-8").read() == text)
        FakeLobby.fail = "Steam is not running"
        check("relay: a Steam failure stops only the relay, with a log line",
              runner.relay("host", host_orders, None, "dll") == 1
              and "relay: stopped: Steam is not running" in open(logfile, encoding="utf-8").read())
    finally:
        runner.HOME, runner.steamlobby.Lobby, FakeLobby.fail = saved_home, saved_lobby, None
```

- [ ] **Step 2: Run and see it fail**

Run: `PY -m temp.checks.check_runner`
Expected: `FAIL check_relay -> AttributeError: ... 'publish'`.

- [ ] **Step 3: Import the new modules in `runner.py`** (next to `from .pack import Packer`)

```python
from . import multiplayer, steamlobby
```

- [ ] **Step 4: Replace `child_command` in `runner.py` with `me` + `child_command`**

```python
def me() -> list[str]:
    """This runner again, as a child process: jevai.exe itself when frozen, else this module."""
    return [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", __spec__.name]


def child_command(model_dir: str, device: str) -> list[str]:
    return me() + ["--compile", device, "--model", model_dir]
```

- [ ] **Step 5: Add the relay after `write_atomic` in `runner.py`**

```python
def default_steam_api() -> str:
    """The game's own steam_api64.dll, or "" when HOI4 is not in a Steam library."""
    exe = find_game()
    return os.path.join(os.path.dirname(exe), "steam_api64.dll") if exe else ""


def start_relay(a, role: dict, orders: str) -> subprocess.Popen:
    """The Steam relay of this multiplayer game (jevai.exe --relay), stopped when the game closes."""
    dll = role.get("steam_api") or a.steam_api or default_steam_api()
    cmd = me() + ["--relay", role["role"], "--orders", orders, "--steam-api", dll]
    return subprocess.Popen(cmd + (["--code", role["code"]] if role.get("code") else []),
                            creationflags=NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)


def publish(lobby, orders: str, last: str | None) -> str | None:
    """Host relay, one poll: put the orders file in the lobby when it changed; returns what the lobby holds."""
    text = _read(orders)
    return text if text and text != last and lobby.set("orders", multiplayer.pack(text)) else last


def fetch(lobby, orders: str, last: str | None) -> str | None:
    """Client relay, one poll: write the host's newest orders file when it changed; returns the last set seen."""
    wire = lobby.get("orders")
    if not wire or wire == last:
        return last
    write_atomic(orders, multiplayer.unpack(wire))
    return wire


def relay(role: str, orders: str, code: str | None, dll: str) -> int:
    """Child mode (jevai.exe --relay host|client): the only process that loads steam_api64.dll, so a replaced or old DLL
    can only crash this one. The host's relay puts each new orders file in its lobby; a client's relay finds the host's
    lobby by code and writes each set to its own orders file, byte for byte. Runs until the runner stops it; logs to
    jevai.log."""
    out = open(os.path.join(HOME, "jevai.log"), "a", encoding="utf-8")

    def say(msg: str):
        out.write(f"[{time.strftime('%H:%M:%S')}] relay: {msg}\n")
        out.flush()

    lobby = steamlobby.Lobby(dll)
    try:
        lobby.open()
        if role == "host":
            code = multiplayer.code_for(lobby.account_id())
            lobby.create(code)
            say(f"hosting; the other players run jevai.exe --join {code}")
        else:
            waited = False
            while not (found := lobby.find(code)):
                if not waited:
                    say(f"waiting for the host {code} (its JevAI opens the lobby when its game starts)")
                    waited = True
                time.sleep(10)
            lobby.join(found)
            say(f"following the host {code}")
    except steamlobby.SteamError as e:
        say(f"stopped: {e}")
        return 1
    last = None
    while True:
        lobby.tick()
        new = publish(lobby, orders, last) if role == "host" else fetch(lobby, orders, last)
        if new != last:
            say(("sent " if role == "host" else "received ") + _read(orders).split("\n", 1)[0].lstrip("# "))
            last = new
        time.sleep(1)
```

- [ ] **Step 6: Run and see everything pass**

Run: `PY -m temp.checks.check_runner`
Expected: `all passed` (`check_relay` takes about 5 s: the relay loop sleeps 1 s per round).

- [ ] **Step 7: Commit**

```bash
git add mods/jevai/runtime/runner.py
git commit -m "runner: the multiplayer relay (jevai.exe --relay) over a Steam lobby"
```

---

### Task 5: Session roles

**Files:**
- Modify: `mods/jevai/runtime/runner.py` (`session`)
- Test: `temp/checks/check_runner.py` (`run_session`, new `check_sessions_mp`)

**Interfaces:**
- Consumes: `multiplayer.load_role`, `multiplayer.MARGIN_DAYS` (Task 2); `day_date`, `GameLog.now`, `GameLog.multiplayer`, `orders_file(..., target)` (Task 3); `start_relay` (Task 4).
- Produces: `session(a, mod, pid)` behaviour: loads the model only once the game logs `[[ Launching ...-game ]]`; in multiplayer, decides only as host, stamps `target = day_date(log.now + MARGIN_DAYS)`, runs the relay for the session's length; without a role keeps JevAI off and says why once.

- [ ] **Step 1: Extend the test harness** in `temp/checks/check_runner.py`

Add module-level lists next to `run_session`:

```python
RELAYS, MODELS = [], []  # relays started/killed and models built by the last run_session
```

Change the `run_session` signature and body: signature

```python
def run_session(name, events, end, device="CPU", per_call=0.0, fail=False, mp=False, role=None, launched=True):
```

replace its first `write(... "game.log" ...)` line with

```python
    write(os.path.join(u, "logs", "game.log"), stamp("1936.1.1") + "Executing History\n"
          + (f"\t\t[[ Launching {'MULTIPLAYER' if mp else 'SINGLEPLAYER'}-game ]]\n" if launched else ""))
    RELAYS.clear()
    home = os.path.join(u, "home")
    if role:
        multiplayer.save_role(home, role["role"], role.get("code"))
```

in `StubModel.__init__` add as first line `MODELS.append(dev)`; extend the `saved` names with `"HOME", "start_relay"` and set, next to the other patches:

```python
    class Proc:
        def kill(self):
            RELAYS.append("killed")

    runner.HOME = home
    runner.start_relay = lambda a, r, orders: RELAYS.append(r["role"]) or Proc()
```

and add `steam_api=None` to the `Namespace(...)` passed to `session`.

- [ ] **Step 2: Write the failing checks** (add; list `check_sessions_mp` in `main()`)

```python
def check_sessions_mp():
    ev = [(1005, header("1936.6.1", "majors", "month") + burst("1936.6.1", MONTHLY))]
    MODELS.clear()
    dates, out, orders = run_session("s_nolaunch", ev, 1100, launched=False)
    check("session: no model before the game session starts", not dates and not MODELS, (dates, MODELS))
    dates, out, orders = run_session("s_mp_norole", ev, 1100, mp=True)
    check("session multiplayer without a role: JevAI stays off and says why once", not dates
          and out.count("without a JevAI role") == 1 and "jev_want" not in orders and not RELAYS, (dates, out, RELAYS))
    dates, out, orders = run_session("s_mp_host", ev, 1100, mp=True, role={"role": "host"})
    check("session multiplayer host: decides; the relay runs for the session", dates == ["1936.6.1"]
          and RELAYS == ["host", "killed"], (dates, RELAYS))
    check("session multiplayer host: the set applies 3 game days after the newest game date", "date > 1936.6.3" in orders, orders)
    MODELS.clear()
    dates, out, orders = run_session("s_mp_client", ev, 1100, mp=True, role={"role": "client", "code": "ABCDEFG"})
    check("session multiplayer client: no model, no decisions; the relay follows the host", not dates and not MODELS
          and RELAYS == ["client", "killed"], (dates, MODELS, RELAYS))
```

- [ ] **Step 3: Run and see them fail**

Run: `PY -m temp.checks.check_runner`
Expected: `FAIL session: no model before the game session starts` and the three multiplayer checks fail; the older session checks still pass.

- [ ] **Step 4: Replace `session` in `runner.py`**

```python
def session(a, mod: str, pid: int):
    """Steer one running game until its process `pid` exits: once the game session starts, load the English names and
    the model (prepared into the cache already, normally: seconds), then decide on the updates the mod logs, once per
    period the player chose. In a multiplayer game only the host (jevai.exe --host) decides: every order set applies
    multiplayer.MARGIN_DAYS after the host's newest game date, and the relay carries it to the other players' games.
    Says once why, if the game logs nothing for JevAI."""
    orders = os.path.join(mod, "history", "units", ORDERS + ".txt")
    write_atomic(orders, orders_file({}, "no update yet", "-"))  # not the last game's
    log = GameLog(a.userdir)
    role = multiplayer.load_role(HOME)
    print(f"reading {log.path}; orders go to {orders}", flush=True)
    ready: dict = {}

    def load():
        try:
            exe = find_game()
            ready["names"] = english_names(([os.path.dirname(exe)] if exe else []) + active_mods(a.userdir))
            md = model_dir(a, mod)
            device = prepare(md, a.device)  # a warm cache: seconds; else compiles within the time limits
            if device is None:
                raise RuntimeError("no device could run the model (see above)")
            ready["model"] = Model(md, device)
        except Exception as e:  # noqa: BLE001 - reported below; the game keeps running without the model
            ready["error"] = e

    loader = proc = None
    model = names_en = last = None  # last: game day of the last decision
    per_country = 0.0  # seconds, from the last batch
    started = checked = time.time()
    explained = told = False
    try:
        while True:
            if time.time() - checked > 5:
                checked = time.time()
                if pid not in pids("hoi4.exe"):
                    return
                if not explained and (why := silence(a.userdir, log, checked - started)):
                    print(why, flush=True)
                    explained = True
            log.poll()
            if log.multiplayer is None:  # the menus: no game session yet
                time.sleep(1)
                continue
            if log.multiplayer and role and proc is None:
                proc = start_relay(a, role, orders)
                print("multiplayer game: JevAI " + ("decides for everyone" if role["role"] == "host"
                                                    else f"applies the orders of the host {role['code']}"), flush=True)
            if log.multiplayer and (role is None or role["role"] == "client"):
                if role is None and not told:
                    print("multiplayer game without a JevAI role, so JevAI stays off: the host runs jevai.exe --host, "
                          "the other players jevai.exe --join CODE", flush=True)
                    told = True
                time.sleep(1)
                continue
            if loader is None:
                loader = threading.Thread(target=load, daemon=True)
                loader.start()
            if model is None:
                if "error" in ready:
                    print(f"model failed to load: {ready['error']}; JevAI sits out this game", flush=True)
                    while pid in pids("hoi4.exe"):
                        time.sleep(5)
                    return
                if loader.is_alive():
                    time.sleep(1)
                    continue
                model, names_en = ready["model"], ready["names"]
                per_country = 1.5 if model.device == "NPU" else 10.0
                log.complete(quiet=0)  # what was logged before the model was ready is history; decide from the next update
                print("waiting for the game's next update (the 1st of each month, or weekly with the weekly period)",
                      flush=True)
            done = log.complete()
            if not done:
                time.sleep(1)
                continue
            date, reported = done[-1]  # if the game ran ahead of the model, the older bursts are stale: decide the newest
            now = f"[{time.strftime('%H:%M:%S')}] {date}"
            if log.mode in (None, "off"):
                print(f"{now}: " + ("waiting for the JevAI choice in the start-of-game event" if log.mode is None
                                    else "JevAI is off for this game (vanilla AI)"), flush=True)
                continue
            period = log.period or ("week" if log.mode == "majors" else "month")  # saves from JevAI 0.2 log no period
            if not due(period, last, game_day(date)):
                continue
            recs, names, humans, majors = log_records([p for ls in log.latest.values() for p in ls], names_en, date)
            humans |= set(filter(None, a.player.split(",")))
            elig = [t for t in reported if t in recs and t not in humans and recs[t]["mil"] + recs[t]["civ"] >= a.min_factories
                    and (log.mode == "all" or t in majors)]
            if not elig:
                continue
            head = f"{now} ({log.mode}, {period})"
            if (est := len(elig) * per_country) > 20:  # a CPU takes minutes for a big batch: say so before it starts
                print(f"{head}: deciding {len(elig)} countries on {model.device}, about "
                      + (f"{est:.0f}s" if est < 120 else f"{est / 60:.0f} min"), flush=True)
            t0 = time.time()
            chosen = {}
            for t in elig:
                u = model.scores(state_text(recs[t], names, recs), a.horizon)
                best, cur = int(np.argmax(u)), int(recs[t]["jev"].get("pos") or 0) - 1  # cur: posture in force, -1 none
                # near-ties flip with tiny changes (even the date): switch only for a clear gain
                chosen[t] = (cur if 0 <= cur < len(u) and u[best] - u[cur] < a.stick else best) + 1
            target = day_date(log.now + multiplayer.MARGIN_DAYS) if log.multiplayer else date
            write_atomic(orders, orders_file(chosen, date, model.device, target))
            last, per_country = game_day(date), (time.time() - t0) / len(elig)
            top = sorted(chosen.items(), key=lambda kv: -(recs[kv[0]]["mil"] + recs[kv[0]]["civ"]))[:6]
            skipped = f"; {len(done) - 1} older updates skipped" if len(done) > 1 else ""
            print(f"{head}: {len(chosen)} countries in {time.time() - t0:.0f}s on {model.device}{skipped}; "
                  + ", ".join(f"{t} {POSTURES[k - 1]}" for t, k in top), flush=True)
    finally:
        if proc:
            proc.kill()
```

- [ ] **Step 5: Run and see everything pass**

Run: `PY -m temp.checks.check_runner`
Expected: `all passed`.

- [ ] **Step 6: Commit**

```bash
git add mods/jevai/runtime/runner.py
git commit -m "runner: multiplayer roles in the session (host decides with a target date, clients follow, no role: off)"
```

---

### Task 6: `--host`, `--join`, `--steam-api` and the relay child on the command line

**Files:**
- Modify: `mods/jevai/runtime/runner.py` (new `set_role`, `main`)
- Test: `temp/checks/check_runner.py`

**Interfaces:**
- Consumes: `multiplayer.valid_code`, `save_role`, `code_for` (Task 2); `steamlobby.Lobby` (Task 1); `relay`, `default_steam_api` (Task 4).
- Produces: `runner.set_role(a) -> int` (`a` has `join`, `host`, `steam_api`); `main` routes `--relay`/`--orders`/`--code` to `relay`, `--host`/`--join` to `set_role` (both before the single-runner mutex).

- [ ] **Step 1: Write the failing check** (add; list `check_cli_roles` in `main()`)

```python
def check_cli_roles():
    u = fresh("cli")
    saved_home, saved_lobby = runner.HOME, runner.steamlobby.Lobby
    runner.HOME, runner.steamlobby.Lobby, FakeLobby.fail = os.path.join(u, "home"), FakeLobby, None
    try:
        r, out = captured(runner.set_role, Namespace(join="abcdefg", host=False, steam_api=None))
        check("--join: saves the client role", r == 0
              and multiplayer.load_role(runner.HOME) == {"role": "client", "code": "ABCDEFG"}, out)
        r, out = captured(runner.set_role, Namespace(join="nope", host=False, steam_api=None))
        check("--join: refuses a wrong code and keeps the role", r == 1
              and multiplayer.load_role(runner.HOME)["code"] == "ABCDEFG", out)
        r, out = captured(runner.set_role, Namespace(join=None, host=True, steam_api="D:/x/steam_api64.dll"))
        check("--host: saves the host role with its DLL and prints the code to share", r == 0
              and f"--join {multiplayer.code_for(12345)}" in out
              and multiplayer.load_role(runner.HOME) == {"role": "host", "steam_api": "D:/x/steam_api64.dll"}, out)
        FakeLobby.fail = "Steam is not running"
        r, out = captured(runner.set_role, Namespace(join=None, host=True, steam_api=None))
        check("--host without Steam: still the host; the code comes later in jevai.log", r == 0 and "jevai.log" in out, out)
        FakeLobby.fail = None
        r, out = captured(runner.main, ["--join", "BCDEFGH"])
        check("main --join: routed to set_role (while another runner may run)", r == 0
              and multiplayer.load_role(runner.HOME)["code"] == "BCDEFGH", out)
    finally:
        runner.HOME, runner.steamlobby.Lobby, FakeLobby.fail = saved_home, saved_lobby, None
```

- [ ] **Step 2: Run and see it fail**

Run: `PY -m temp.checks.check_runner`
Expected: `FAIL check_cli_roles -> AttributeError: ... 'set_role'`.

- [ ] **Step 3: Add `set_role` before `main` in `runner.py`**

```python
def set_role(a) -> int:
    """jevai.exe --host / --join CODE: this PC's part in multiplayer games, kept in HOME/multiplayer.json for the running
    JevAI, which reads it when a game starts; single-player games ignore it."""
    if a.join:
        if not multiplayer.valid_code(a.join):
            print(f"{a.join} is not a JevAI code (7 letters and digits, printed by the host's jevai.exe --host)")
            return 1
        multiplayer.save_role(HOME, "client", a.join, a.steam_api)
        print(f"multiplayer games: this PC applies the orders of the host {a.join.upper()} (from the next game)")
        return 0
    multiplayer.save_role(HOME, "host", steam_api=a.steam_api)
    lobby = steamlobby.Lobby(a.steam_api or default_steam_api())
    try:
        lobby.open()
        code = multiplayer.code_for(lobby.account_id())
    except steamlobby.SteamError as e:
        print(f"multiplayer games: this PC hosts. Steam did not answer ({e}); the code to share appears in jevai.log "
              "when a multiplayer game starts")
        return 0
    print(f"multiplayer games: this PC hosts. The other players run: jevai.exe --join {code}")
    return 0
```

- [ ] **Step 4: Wire it into `main`**

After `ap.add_argument("--compile", ...)` add:

```python
    ap.add_argument("--host", action="store_true", help="multiplayer: this PC hosts; prints the code the others join with")
    ap.add_argument("--join", metavar="CODE", help="multiplayer: this PC applies the orders of the host with this code")
    ap.add_argument("--steam-api", help="multiplayer: the steam_api64.dll to use (default: the game's)")
    ap.add_argument("--relay", choices=["host", "client"], help=argparse.SUPPRESS)  # child mode, see relay
    ap.add_argument("--orders", help=argparse.SUPPRESS)
    ap.add_argument("--code", help=argparse.SUPPRESS)
```

After `a.mod = os.path.abspath(a.mod)` add:

```python
    if a.relay:
        return relay(a.relay, a.orders, a.code, a.steam_api or default_steam_api())
```

Replace

```python
    if sys.stdout:  # a process started without a console has none
```

with

```python
    if hasattr(sys.stdout, "reconfigure"):  # a process started without a console has none
```

and after the `if a.install or a.uninstall:` block (before `k32 = ...`) add:

```python
    if a.host or a.join:
        return set_role(a)
```

- [ ] **Step 5: Run and see everything pass**

Run: `PY -m temp.checks.check_runner`
Expected: `all passed`.

- [ ] **Step 6: Commit**

```bash
git add mods/jevai/runtime/runner.py
git commit -m "runner: jevai.exe --host / --join CODE / --steam-api; the relay child mode"
```

---

### Task 7: Player instructions, helper scripts, packaged smoke test

**Files:**
- Modify: `mods/jevai/package.py` (the `install.cmd`/`uninstall.cmd` loop), `README.md`, `mods/jevai/workshop.txt`, `CLAUDE.md`

**Interfaces:**
- Consumes: the CLI of Task 6.
- Produces: `runner\host.cmd` and `runner\join.cmd` in the release.

- [ ] **Step 1: Add the helper scripts in `mods/jevai/package.py`**

Replace the `for name, args, note in (...)` loop and its body with:

```python
    scripts = (("install.cmd", " --install", "Run once: prepares the model (about 2 minutes on an Intel NPU, seconds on a CPU); "
                                           "JevAI then starts with Windows and steers every game you start from the launcher."),
               ("uninstall.cmd", " --uninstall", "Stops JevAI starting with Windows and stops the background runner."),
               ("host.cmd", " --host", "Multiplayer host: run once; share the code it prints with the other players."),
               ("join.cmd", " --join %CODE%", "Multiplayer: run once with the host's code."))
    for name, args, note in scripts:
        with open(os.path.join(out, "runner", name), "w", encoding="utf-8", newline="\r\n") as f:
            ask = 'set /p CODE=JevAI code from the host: \n' if name == "join.cmd" else ""
            f.write(f'@echo off\nrem {note}\n{ask}"%~dp0jevai.exe"{args}\npause\n')
```

- [ ] **Step 2: Build and smoke-test the packaged exe** (no Steam, no game window)

Run:
```bash
PY -m mods.jevai.package --model temp/ov/hoi4-v1
T=$(mktemp -d); W=$(cygpath -w "$T")
LOCALAPPDATA="$W" temp/release/jevai/runner/jevai.exe --join bcdefgh; cat "$T/jevai/multiplayer.json"; echo
LOCALAPPDATA="$W" temp/release/jevai/runner/jevai.exe --relay client --orders "$W\\o.txt" --code BCDEFGH --steam-api "$W\\none.dll"; echo "exit $?"
cat "$T/jevai/jevai.log"; cat temp/release/jevai/runner/join.cmd
```
Expected: `{"role": "client", "code": "BCDEFGH"}`; `exit 1`; the log ends with `relay: stopped: no steam_api64.dll at ...none.dll`; `join.cmd` asks for the code and runs `jevai.exe --join %CODE%`.

- [ ] **Step 3: Update the docs**

In `README.md` (lines 41-42), replace the sentence `Multiplayer is not` / `supported: each player's game would read its own orders.` (it spans the line break) with the text below, then re-wrap that paragraph at about 118 columns:

```
Multiplayer: every player needs JevAI in the playset and the runner installed. The host runs `runner\host.cmd` (or
`jevai.exe --host`) once and shares the code it prints; the others run `runner\join.cmd` (or `jevai.exe --join CODE`)
once. The host's model decides for everyone: each order set travels through a Steam lobby and every game applies it on
the same game day, 3 days after the host decided (a player whose runner misses a set goes out of sync until HOI4's
resync).
```

In `mods/jevai/workshop.txt`, replace `[*]Multiplayer is not supported: each player's game would read its own orders.` with:

```
[*]Multiplayer: every player subscribes and runs install.cmd. The host runs [b]host.cmd[/b] in the runner folder once and shares the code it prints; the others run [b]join.cmd[/b] once and enter that code. The host's model decides for everyone.
```

and replace `不支持多人游戏。` with:

```
多人游戏：所有玩家都要订阅并运行 install.cmd。主机在 runner 文件夹运行一次 host.cmd 并分享它显示的代码，其他玩家各运行一次 join.cmd 并输入该代码。由主机的模型为所有人决策。
```

In `CLAUDE.md`, after the paragraph that ends with `lines in game.log show postures being applied).`, add a blank line and this paragraph:

```
Multiplayer (`docs/superpowers/specs/2026-09-28-multiplayer-design.md`): the postures are synchronized state, so every
game must apply identical orders on the same day. `jevai.exe --host` / `--join CODE` (`host.cmd` / `join.cmd`) store
the role in `%LOCALAPPDATA%\jevai\multiplayer.json`; in a game that logs `[[ Launching MULTIPLAYER-game ]]` only the
host decides, the orders file guards its set with `date >` (the host's newest game date + `multiplayer.MARGIN_DAYS`),
and a relay child (`jevai.exe --relay`, the only process that loads `steam_api64.dll`, `runtime/steamlobby.py`)
carries the file through an invisible Steam lobby tagged with the host's code; without a role JevAI stays off.
```

- [ ] **Step 4: Commit**

```bash
git add mods/jevai/package.py README.md mods/jevai/workshop.txt CLAUDE.md
git commit -m "Multiplayer instructions; host.cmd and join.cmd in the release"
```

---

### Task 8: Two-machine acceptance test, then release

Needs the maintainer: two PCs (or a PC and a VM) with **legitimate** HOI4 copies on two Steam accounts, the same game version, JevAI from the Workshop and the same playset. Do not start games or take screens without the maintainer.

- [ ] **Step 1: Install the new build on both machines** (copy `temp/release/jevai` over the local test copy, or publish to the Workshop as private first), run `install.cmd` on both, `host.cmd` on the host, `join.cmd` with the printed code on the client.
- [ ] **Step 2: Play** a multiplayer game (host starts it, client joins), answer the JevAI questions (major powers, every month), speed 5, three game months after the host's first decision line in `jevai.log`.
- [ ] **Step 3: Check the logs.** On both machines: `jevai.log` has `relay: sent ...` (host) / `relay: received ...` (client) for every set; `JEV|ACT` lines in both `game.log` files are identical (`grep "JEV|ACT" game.log | cut -d] -f2-` on both, then `diff`), and a `JEV|ACT` follows each set on its target day, which also settles CLAUDE.md's open question whether `load_oob` re-reads the file every day; no "not in sync" message in either game.
- [ ] **Step 4: Failure path.** Kill the client's relay (`taskkill /IM jevai.exe` for the `--relay` process only) before the next set: the client must go out of sync at that set's target date, and HOI4's resync must recover it.
- [ ] **Step 5: Release** (only after Steps 2-4 pass): bump `mods/jevai/descriptor.mod` to `version="0.4.0"`, `PY -m mods.jevai.package --model temp/ov/hoi4-v1`, upload with `PY -m mods.jevai.workshop temp/release/jevai --note "0.4.0: multiplayer: the host's model decides for everyone (host.cmd / join.cmd)"` (plus `--steam-api` where the game's DLL is not Valve's), confirm the new `time_updated` through `ISteamRemoteStorage/GetPublishedFileDetails`, then:

```bash
git add mods/jevai/descriptor.mod
git commit -m "JevAI 0.4.0: multiplayer"
git push origin main
```
