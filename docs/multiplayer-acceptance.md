# Experimental multiplayer acceptance

The implementation and offline checks do not establish that two HOI4 simulations remain synchronized. The maintainer currently has no second test environment and requested an experimental build with the v2 OpenVINO model. Do not turn the pending checks below into a claim of validated multiplayer support.

Use two machines and two Steam accounts that own HOI4, matching game versions, matching playsets and the exact same JevAI release. Keep the original logs separately before beginning. Do not change the host/client role during a running session.

1. Run `host.cmd` once on the host and share the printed code. Run `join.cmd` on the client and enter that code. Start both runners before entering the multiplayer game.
2. Confirm only the host loads a model. The client should report following the host; a multiplayer session without a role must remain off.
3. After the host's first decision, play at least three game months. Retain both game logs and runner logs. Check every sent/received set and its target date, including successive changed files, to verify the daily `load_oob` reread behavior.
4. Check both game windows for desyncs, then compare logs offline:

   ```powershell
   python -m dev_tools.hoi4.check_multiplayer_logs --host-game host-game.log --client-game client-game.log --out multiplayer-audit.json
   ```

   The tool checks the latest multiplayer session only. Empty logs never pass. It compares the game date/hour, country and posture, ignores wall-clock time, and reports one-sided actions. Passing these checks is evidence about those logs, not proof of account identity, timely relay delivery, or the absence of every in-game synchronization problem.

5. In a separate failure-path test, stop only the client's relay before the next set. Record what HOI4 does and whether resynchronization restores matching orders. Do not claim recovery is verified until this has actually been observed.
6. Repeat with the intended Steam DLL setup and with a real game running while the relay connects. Earlier lobby spikes passed with two DLL versions but did not overlap a running HOI4 instance.

For any failure, preserve both logs, the release/model hashes, game version, playset, target date and both local orders files. Do not overwrite an earlier report; use a new output filename for each attempt. No private credentials or Steam tokens are needed in the report.

The v2 model uses a three-month prediction horizon. The packaged model config, not an old six-month command-line default, should select that horizon unless the user explicitly overrides it. CPU/NPU export equivalence does not prove improved gameplay; the model card contains the measured offline evaluation and its limitations.
