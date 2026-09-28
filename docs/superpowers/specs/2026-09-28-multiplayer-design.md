# JevAI in multiplayer: design

Status: proposed, 2026-09-28. Goal: JevAI steers the AI countries of a multiplayer game without desyncs.

## What the game does (research, 2026-09-28)

- **Lockstep.** Every player's game runs the whole simulation; only commands (player clicks, AI actions) travel
  over the network, and the host compares state checksums with each client. Any difference is a desync
  (`GameSynchronizationHelper::CalcChecksums`, "Out of Synch" in `hoi4.exe`; Paradox's desync-reporting post).
- **Scripts run on every machine.** `on_actions`, events and scripted effects are part of the simulation, so the
  mod's `JEV|` lines are identical in every player's `game.log`.
- **No console** in multiplayer ("Console not available in multiplayer or ironman mode.").
- **The AI acts through commands** ("AI tried to post an invalid command", gated by `GetAI()->IsCommandsAllowed()`),
  computed in parallel threads and barred from the synchronized random numbers. Most likely one machine computes it
  and sends its choices; not confirmed by Paradox, and it does not change this design.
- **JevAI's postures are synchronized state.** Measured in single-player with the developer console: after
  `e SOV jev_set_posture_2`, `dump_synchronized_members` gained the dynamic modifier `jev_posture_2`, the variable
  `jev_posture=2` and four `persistent_strategy` entries, and `compare_to_last_checksum` reported
  "Gamestate checksum and Country Modifiers". A game that does not apply the host's orders desyncs at once, so a
  host-only runner (clients without orders) cannot work: **every game must apply the same orders on the same day.**
- **Orders enter the game from a local file:** the mod's `on_daily` reads `history/units/JEVAI_orders.txt` with
  `load_oob`, on every machine, from that machine's disk.
- **Steam from a second process:** HOI4's stock `steam_api64.dll` has the matchmaking lobby functions
  (`CreateLobby`, `JoinLobby`, `SetLobbyData`, `GetLobbyData`, `RequestLobbyList`) but not
  `ISteamNetworkingMessages` or manual callback dispatch. `mods.jevai.workshop` already runs the Steam API this way
  next to a running game.
- **Detection:** `game.log` starts a multiplayer session with `[[ Launching MULTIPLAYER-game ]]`.
- `on_startup` never runs for a loaded save; since 0.3.2 `jev_announce` logs the range and period at every weekly and
  monthly update, which multiplayer (usually loaded saves) relies on.

## Design

### Roles

- The host runs `jevai.exe --host` once. It prints a pairing code that stays the same for the host's Steam account
  (derived from the account id).
- Every other player runs `jevai.exe --join CODE` once per host.
- The role goes to `%LOCALAPPDATA%\jevai\multiplayer.json`, which the installed background runner reads; a manual
  `jevai.exe --host/--join` while the runner is running only writes the file. Single-player games ignore the role.
- In a multiplayer game (the `game.log` marker) without a role, the runner decides nothing, keeps the orders file
  empty and logs: "multiplayer game: the host runs jevai.exe --host, the other players jevai.exe --join CODE".
- Only the host loads the model. Clients never decide.

### Channel: a Steam lobby through a relay process

- `jevai.exe --relay` is started by the runner in multiplayer games and is the only process that loads
  `steam_api64.dll` (the game's, or `--steam-api`), so a broken or old DLL can only crash the relay. It starts the
  Steam API under HOI4's app id (`steam_appid.txt` in its own working folder, as the uploader does) and uses only
  call results polled through `ISteamUtils`, no callbacks.
- Host relay: creates an invisible lobby with the pairing code as lobby data, then publishes each new order set
  with `SetLobbyData("orders", ...)`.
- Client relay: finds the lobby with `RequestLobbyList` filtered on the code, joins it, polls `GetLobbyData` every
  second and writes each new set to its orders file.
- Wire format: `TARGET|TAG:k,TAG:k,...` (a few hundred bytes; lobby data values hold up to 8 KB).
- The relay logs every set it publishes or receives, with its target date and arrival time.

### Orders with a target date

- The host's runner decides as it does today, then stamps the set with a target date: the newest game date in its
  `game.log` plus 3 days. Single-player uses the same format with today's date, so nothing changes there.
- The orders file wraps all order lines in `if = { limit = { date > TARGET - 1 day } ... }`. Every game reloads the
  file daily, so every game applies the set on the same game day. Later reloads of the same set change nothing
  (`jev_follow_orders` only acts when the wanted posture changed).
- All runners empty the orders file when HOI4 starts, as today, so every game starts from the same empty file.
- A client that misses a set (relay down, network) desyncs at the target date; HOI4's resync (the host re-sends its
  save) repairs it. The relay log says which set was missed.

### Everything else stays

- The start-of-game questions go to one player (the event fires once); the answer is a synchronized event option,
  and `jev_announce` logs it on every machine.
- Every player needs the same load order: clients run the installed runner too, which patches it as today.

## Testing

- Offline checks (`temp/checks`): stamping and the file format, the wire format both ways, roles and multiplayer
  detection, a client with a fake lobby writing the same file as the host.
- Spike before building the relay: the lobby calls from a second process while HOI4 runs, and how long lobby data
  takes to reach another account.
- In game, two machines with legitimate copies (two Steam accounts that own HOI4, the same game version, JevAI from
  the Workshop, the same playset): the host runs `--host`, the client `--join`. Play three game months at speed 5
  after the first decision: both `game.log` files must show identical `JEV|ACT` lines and no desync. Stopping the
  client's relay mid-game must desync, and resync must recover.

## Out of scope

- Pairing without a code (for example through Steam friends in the same session).
- Clients deciding on their own: NPU and CPU rounding can flip near-tie choices, and a slow CPU can miss the target
  date; either is a desync.
- Host migration, and anything for players who join mid-game beyond joining the lobby before the next target date.

## Risks

- Lobby data propagation delay under load; 3 days at speed 5 is about 2 to 4 seconds of real time.
- A second Steam API process under HOI4's app id next to the game (works for the uploader; unproven for hours).
- Replaced or old `steam_api64.dll` in players' game folders: the relay crashes, the runner logs it, and
  `--steam-api` points it at another app's stock DLL.
