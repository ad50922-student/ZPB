import numpy as np
from c4 import C4
from game_runner import GameRunner
from ultimate_ttt import UTTT

if __name__ == "__main__":
    game_runner = GameRunner(UTTT, None, None, 0, 1, None)
    outcome, game_info = game_runner.run()