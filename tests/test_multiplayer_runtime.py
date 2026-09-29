"""Offline multiplayer contracts; no Steam, model inference, or real runner processes."""
import base64
import contextlib
import io
import json
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import Mock, patch

from mods.jevai.runtime import multiplayer, runner, steamlobby


class MultiplayerTests(unittest.TestCase):
    def test_wire_roundtrip_and_limits(self):
        text = runner.orders_file({"GER": 2}, "1936.7.1", "CPU", "1936.7.4")
        self.assertEqual(multiplayer.unpack(multiplayer.pack(text)), text)
        bad = ["?", "A" * 8001, multiplayer.pack("ok") + "AAAA",
               base64.b64encode(zlib.compress(b"x" * (multiplayer.MAX_TEXT + 1))).decode()]
        for wire in bad:
            with self.subTest(wire=wire[:20]), self.assertRaises((ValueError, zlib.error)):
                multiplayer.unpack(wire)
        with self.assertRaises(ValueError):
            multiplayer.pack("x" * (multiplayer.MAX_TEXT + 1))

    def test_roles_are_atomic_and_normalized(self):
        with tempfile.TemporaryDirectory() as home:
            multiplayer.save_role(home, "client", "abcdefg")
            self.assertEqual(multiplayer.load_role(home), {"role": "client", "code": "ABCDEFG"})
            path = Path(home, "multiplayer.json")
            path.write_text('{"role":"client","code":"bcdefgh"}')
            self.assertEqual(multiplayer.load_role(home)["code"], "BCDEFGH")
            with patch.object(multiplayer.os, "replace", side_effect=OSError("locked")):
                with self.assertRaises(OSError):
                    multiplayer.save_role(home, "host")
            self.assertEqual(json.loads(path.read_text())["role"], "client")
            self.assertEqual([p.name for p in Path(home).iterdir()], ["multiplayer.json"])

    def test_date_guard_and_calendar(self):
        for day in range(runner.game_day("1936.1.1"), runner.game_day("1938.1.1")):
            self.assertEqual(runner.game_day(runner.day_date(day)), day)
        text = runner.orders_file({"GER": 2}, "1936.7.1", "CPU", "1936.7.4")
        self.assertIn("date > 1936.7.3", text)
        self.assertEqual(text.count("{"), text.count("}"))
        self.assertNotIn("date >", runner.orders_file({}, "not ready", "-"))

    def test_log_generation_changes_on_rollback_and_new_game(self):
        with tempfile.TemporaryDirectory() as home:
            path = Path(home, "logs", "game.log")
            path.parent.mkdir()
            path.write_text('[[ Launching MULTIPLAYER-game ]]\n[1][1936.6.9.1][game]: tick\n')
            log = runner.GameLog(home)
            log.poll()
            generation = log.generation
            with path.open("a") as out:
                out.write('[1][1936.1.1.1][game]: loaded save\n')
            log.poll()
            self.assertGreater(log.generation, generation)
            self.assertTrue(log.multiplayer)
            generation = log.generation
            with path.open("a") as out:
                out.write('[[ Launching SINGLEPLAYER-game ]]\n')
            log.poll()
            self.assertGreater(log.generation, generation)
            self.assertFalse(log.multiplayer)

    def test_relay_roundtrip_and_failed_publish(self):
        with tempfile.TemporaryDirectory() as home:
            host, client = str(Path(home, "host.txt")), str(Path(home, "client.txt"))
            text = runner.orders_file({"GER": 2}, "1936.7.1", "CPU", "1936.7.4")
            Path(host).write_text(text)
            lobby = Mock()
            lobby.set.return_value = True
            self.assertEqual(runner.publish(lobby, host, None), text)
            runner.publish(lobby, host, text)
            lobby.set.assert_called_once()
            wire = lobby.set.call_args.args[1]
            lobby.get.return_value = wire
            self.assertEqual(runner.fetch(lobby, client, None), wire)
            self.assertEqual(Path(client).read_text(), text)
            lobby.set.return_value = False
            with self.assertRaises(steamlobby.SteamError):
                runner.publish(lobby, host, None)

    def test_relay_rejects_owner_change(self):
        with tempfile.TemporaryDirectory() as home:
            lobby = Mock()
            lobby.account_id.return_value = 12345
            lobby.owner.return_value = 54321
            with patch.object(runner, "HOME", home), patch.object(steamlobby, "Lobby", return_value=lobby):
                self.assertEqual(runner.relay("host", str(Path(home, "orders")), None, "fake.dll"), 1)
            self.assertIn("owner no longer matches", Path(home, "jevai.log").read_text())
            lobby.set.assert_not_called()

    def test_interactive_join_and_eof(self):
        with tempfile.TemporaryDirectory() as home, patch.object(runner, "HOME", home), \
                contextlib.redirect_stdout(io.StringIO()):
            with patch("builtins.input", return_value="abcdefg"):
                self.assertEqual(runner.main(["--join"]), 0)
            with patch("builtins.input", side_effect=EOFError):
                self.assertEqual(runner.main(["--join"]), 1)
            self.assertEqual(multiplayer.load_role(home)["code"], "ABCDEFG")


if __name__ == "__main__":
    unittest.main()
