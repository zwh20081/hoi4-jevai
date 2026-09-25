"""JevAI runtime: the Python side of the mod. It reads what the game writes (history_dump months, game.log),
turns each country into the text the model reads, packs model inputs and types console commands into the game.
Relative imports only, so it runs from the repo (mods.jevai.runtime) and from an installed mod (runtime)."""
