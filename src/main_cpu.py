import numpy as np
from c4 import C4
from game_runner import GameRunner
from src.mcts import MCTS
from ultimate_ttt import UTTT

if __name__ == "__main__":
    game_runner = GameRunner(UTTT, None,
                             MCTS(search_time_limit=5.0, search_steps_limit=np.inf, vanilla=False),
                             0, 1, None)
    outcome, game_info = game_runner.run()