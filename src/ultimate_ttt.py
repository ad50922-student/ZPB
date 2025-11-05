import numpy as np
from mcts import State
from numba import jit
from numba import int8

__version__ = "1.0.1"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"


class UTTT(State):
    """
    Class for states of Connect 4 game.

    Attributes:
        M (int):
            number of rows in the board, defaults to ``6``.
        N (int):
            number of columns in the board, defaults to ``7``.
        SYMBOLS (List):
            list of strings representing disc symbols (black, white) or ``"."`` for empty cell.
    """
    M = 9
    N = 9
    E = 11
    SYMBOLS = ["\u25CB", ".", "\u25CF"]  # or: ["O", ".", "X"]

    def __init__(self, parent=None):
        """
        Constructor (ordinary or copying) of ``C4`` instances - states of Connect 4 game.

        Args:
            parent (State):
                reference to parent state object.
        """
        super().__init__(parent)
        if self.parent:
            self.board = np.copy(self.parent.board)
            self.extra_info = np.copy(self.parent.extra_info)
        else:
            self.board = np.zeros((UTTT.M, UTTT.N), dtype=np.int8)
            self.extra_info = np.zeros(UTTT.E, dtype=np.int8)

    @staticmethod
    def class_repr():
        """
        Returns a string representation of class ``C4`` (meant to instantiate states of Connect 4 game), informing about the size of board.

        Returns:
            str: string representation of class ``C4`` (meant to instantiate states of Connect 4 game), informing about the size of board.
        """
        return f"{UTTT.__name__}_{UTTT.M}x{UTTT.N}"

    def __str__(self):
        """
        Returns a string representation of this Ultimate Tic Tac Toe board (9x9 grid).

        Each cell shows 'X', 'O', or blank. Dividers separate the 3x3 small boards.
        """
        s = ""
        for i in range(UTTT.M):
            s += "|"
            for j in range(UTTT.N):
                val = self.board[i, j]
                s += self.SYMBOLS[val + 1] + "|"

                if (j + 1) % 3 == 0 and j < 8:
                    s += "  |"
            s += "\n"

            if (i + 1) % 3 == 0 and i < 8:
                s += "=" * 29 + "\n"
        return s

    def take_action_job(self, action_index):
        """
        Drops a disc into column indicated by the action_index and returns ``True`` if the action is legal (column not full yet).
        Otherwise, does no changes and returns ``False``.

        Args:
            action_index (int):
                index of column where to drop a disc.

        Returns:
            action_legal (bool):
                boolean flag indicating if the specified action was legal and performed.
        """

        """
        Struktura danych board [9x9]:
        
        Tablica dwuwymiarowa 9x9 przechowująca 81 pozycji, indeksy i od 0 do 8 oraz j od 0 do 8.
        Każdy element tablicy może zawierać tylko następujące wartości:
            0, jeżeli na pozycji nie zostało postawione ani krzyżyk ani kółko;
            1, jeżeli na pozycji został postawiony krzyżyk;
            -1, jeżeli na pozycji zostało postawione kołko
        
        Index, czyli numer ruchu to wartość zależna od pozycji którą wybrał obecnie rozgrywający gracz, od 0 do 80.
        Index = i * 9 + j (np. i = 2, j = 4, to 2 * 9 + 4 = 22).
        
        Natomiast konwersja odwrotna - z numeru ruchu (index) do indeksów tablicy:
            i = index // 9 
            j = index % 9
        
        Wektor jednowymiarowy extra_info (typ danych = byte) składa się z jedenastu elementów.
        Pierwsze 9 elementów (indeksy od 0 do 8) określają stan każdej z podtablic 3x3.
        Stany mogą być nastepujące:
            0, jeżeli gra na tej podtablicy nie została jeszcze rozpoczęta (wszystkie elementy podtablicy = 0);
            1, jeżeli tą podtablicę wygrał krzyżyk (wtedy ta podtablica jest już wyłączona z dalszej gry);
            -1, jeżeli tą podtablicę wygrało kółko (wtedy ta podtablica jest już wyłączona z dalszej gry);
            2, jeżeli trwa gra na tej podtablicy (nikt nie wygrał, ale nie ma sytuacji patowej)
            -2, jeżeli doszło do sytuacji patowej (wszystkie pozycje podtablicy są zajęte, ale nikt nie wygrał)
        
        Ostatnie dwa elementy (indeksy 9 i 10) wskazują na indeksy podtablicy (I od 0 do 2, J od 0 do 2), na której
        należy wykonać następny ruch. Po rozpoczęciu gry, te wartości automatycznie muszą przybrać wartości 
        I = 1 oraz J = 1, aby wskazywać na środkową podtablicę 3x3 (zawsze na niej należy wykonać pierwszy ruch).
        W kolejnych turach, indeksy są  już obliczane na podstawie indeksów i, j tablicy 9x9 wskazujących na pozycję,
        na której został zagrany poprzedni ruch:
            I = i % 3;
            J = j % 3
        """

        j = action_index
        next_board = [self.extra_info[9], self.extra_info[10]]
        # if self.extra_info[j] == UTTT.E:
        #     return False [2, 3, 3, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0,2, 0, 0, 2, 0, 0, 2, 0, 0]
        i = UTTT.M - 1 - self.extra_info[j]
        self.board[i, j] = self.turn
        self.extra_info[j] += 1
        self.turn *= -1
        return True

    def compute_outcome_job(self):
        """
        Computes and returns the game outcome for this state in compliance with rules of Connect 4 game:
        {-1, 1} denoting a win for the minimizing or maximizing player, respectively, if he connected at least 4 his discs;
        0 denoting a tie, when the board is filled and no line of 4 exists;
        ``None`` when the game is ongoing.

        Returns:
            outcome ({-1, 0, 1} or ``None``)
                game outcome for this state.
        """
        j = self.last_action_index
        i = UTTT.M - self.extra_info[j]
        if True:  # a bit faster outcome via numba
            numba_outcome = UTTT.compute_outcome_job_numba_jit(UTTT.M, UTTT.N, self.turn, i, j, self.board)
            if numba_outcome != 0:
                return numba_outcome
        else:  # a bit slower outcome via pure Python (inactive now)
            last_token = -self.turn
            # N-S
            total = 0
            for k in range(1, 4):
                if i - k < 0 or self.board[i - k, j] != last_token:
                    break
                total += 1
            for k in range(1, 4):
                if i + k >= UTTT.M or self.board[i + k, j] != last_token:
                    break
                total += 1
            if total >= 3:
                return last_token
                # E-W
            total = 0
            for k in range(1, 4):
                if j + k >= UTTT.N or self.board[i, j + k] != last_token:
                    break
                total += 1
            for k in range(1, 4):
                if j - k < 0 or self.board[i, j - k] != last_token:
                    break
                total += 1
            if total >= 3:
                return last_token
                # NE-SW
            total = 0
            for k in range(1, 4):
                if i - k < 0 or j + k >= UTTT.N or self.board[i - k, j + k] != last_token:
                    break
                total += 1
            for k in range(1, 4):
                if i + k >= UTTT.M or j - k < 0 or self.board[i + k, j - k] != last_token:
                    break
                total += 1
            if total >= 3:
                return last_token
                # NW-SE
            total = 0
            for k in range(1, 4):
                if i - k < 0 or j - k < 0 or self.board[i - k, j - k] != last_token:
                    break
                total += 1
            for k in range(1, 4):
                if i + k >= C4.M or j + k >= C4.N or self.board[i + k, j + k] != last_token:
                    break
                total += 1
            if total >= 3:
                return last_token
        if np.sum(self.board == 0) == 0:  # draw
            return 0
        return None

    @staticmethod
    @jit(int8(int8, int8, int8, int8, int8, int8[:, :]), nopython=True, cache=True)
    def compute_outcome_job_numba_jit(M, N, turn, last_i, last_j, board):
        """Called by ``compute_outcome_job`` for faster outcomes."""
        last_token = -turn
        i, j = last_i, last_j
        # N-S
        total = 0
        for k in range(1, 4):
            if i - k < 0 or board[i - k, j] != last_token:
                break
            total += 1
        for k in range(1, 4):
            if i + k >= M or board[i + k, j] != last_token:
                break
            total += 1
        if total >= 3:
            return last_token
            # E-W
        total = 0
        for k in range(1, 4):
            if j + k >= N or board[i, j + k] != last_token:
                break
            total += 1
        for k in range(1, 4):
            if j - k < 0 or board[i, j - k] != last_token:
                break
            total += 1
        if total >= 3:
            return last_token
        # NE-SW
        total = 0
        for k in range(1, 4):
            if i - k < 0 or j + k >= N or board[i - k, j + k] != last_token:
                break
            total += 1
        for k in range(1, 4):
            if i + k >= M or j - k < 0 or board[i + k, j - k] != last_token:
                break
            total += 1
        if total >= 3:
            return last_token
        # NW-SE
        total = 0
        for k in range(1, 4):
            if i - k < 0 or j - k < 0 or board[i - k, j - k] != last_token:
                break
            total += 1
        for k in range(1, 4):
            if i + k >= M or j + k >= N or board[i + k, j + k] != last_token:
                break
            total += 1
        if total >= 3:
            return last_token
        return 0

    def take_random_action_playout(self):
        """
        Picks a uniformly random action from actions available in this state and returns the result of calling ``take_action`` with the action index as argument.

        Returns:
            child (State):
                result of ``take_action`` call for the random action.
        """
        j_indexes = np.where(self.column_fills < UTTT.M)[0]
        j = np.random.choice(j_indexes)
        child = self.take_action(j)
        return child

    def get_board(self):
        """
        Returns the board of this state (a two-dimensional array of bytes).

        Returns:
            board (ndarray[np.int8, ndim=2]):
                board of this state (a two-dimensional array of bytes).
        """
        return self.board

    def get_extra_info(self):
        """
        Returns additional information associated with this state, as one-dimensional array of bytes,
        informing about fills of columns (how many discs have been dropped in each column).

        Returns:
            extra_info (ndarray[np.int8, ndim=1] or ``None``):
                one-dimensional array with additional information associated with this state - fills of columns.
        """
        return self.column_fills

    @staticmethod
    def action_name_to_index(action_name):
        """
        Returns an action's index (numbering from 0) based on its name. E.g., name ``"0"``, denoting a drop into the leftmost column, maps to index ``0``.

        Args:
            action_name (str):
                name of an action.
        Returns:
            action_index (int):
                index corresponding to the given name.
        """
        return int(action_name)

    @staticmethod
    def action_index_to_name(action_index):
        """
        Returns an action's name based on its index (numbering from 0). E.g., index ``0`` maps to name ``"0"``, denoting a drop into the leftmost column.

        Args:
            action_index (int):
                index of an action.
        Returns:
            action_name (str):
                name corresponding to the given index.
        """
        return str(action_index)

    @staticmethod
    def get_board_shape():
        """
        Returns a tuple with shape of boards for Connect 4 game.

        Returns:
            shape (tuple(int, int)):
                shape of boards related to states of this class.
        """
        return (UTTT.M, UTTT.N)

    @staticmethod
    def get_extra_info_memory():
        """
        Returns amount of memory (in bytes) needed to memorize additional information associated with Connect 4 states, i.e., the memory for fills of columns.
        That number is equal to the number of columns.

        Returns:
            extra_info_memory (int):
                number of bytes required to memorize fills of columns.
        """
        return UTTT.N

    @staticmethod
    def get_max_actions():
        """
        Returns the maximum number of actions (the largest branching factor) equal to the number of columns.

        Returns:
            max_actions (int):
                maximum number of actions (the largest branching factor) equal to the number of columns.
        """
        return UTTT.N