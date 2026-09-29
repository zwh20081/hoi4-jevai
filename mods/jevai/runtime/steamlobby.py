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

    def _iface(self, accessor: str, getter: str, version: str, per_user: bool = True):
        """An interface whose layout matches this DLL's flat methods: its own newest accessor (SDK 1.48+), else through
        ISteamClient with the version the DLL was built for (named inside it, else `version`). ISteamClient's getters
        take the user and the pipe, except per-pipe interfaces like utils, which take only the pipe."""
        found = [v for v in range(1, 40) if hasattr(self.api, f"{accessor}{v:03d}")]
        if found:
            return self._fn(f"{accessor}{found[-1]:03d}", P)()
        m = re.search(re.escape(version[:-3].encode()) + rb"\d{3}", self.raw)
        client = self._fn("SteamClient", P)()
        user, pipe = self._fn("SteamAPI_GetHSteamUser", C.c_int)(), self._fn("SteamAPI_GetHSteamPipe", C.c_int)()
        handles = (user, pipe) if per_user else (pipe,)
        return self._fn(getter, P, P, *[C.c_int] * len(handles), C.c_char_p)(client, *handles, m.group() if m else version.encode())

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
        self.utils = self._iface("SteamAPI_SteamUtils_v", "SteamAPI_ISteamClient_GetISteamUtils", "SteamUtils009", per_user=False)
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
