import numpy as np
from mcts import State
from numba import jit, njit
from numba import int8

__version__ = "1.0.1"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"


class UTTT(State):
    """
    Class for states of Ultimate Tic Tac Toe.
    """
    M = 9
    N = 9
    E = 11
    SYMBOLS = ["O", ".", "X"]  # or: ["O", ".", "X"]

    def __init__(self, parent=None):
        """
        Constructor (ordinary or copying) of ``UTTT`` instances - states of UTTT game.

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
            self.extra_info[9] = 1  # startowy ruch na środkowej podtablicy
            self.extra_info[10] = 1  # startowy ruch na środkowej podtablicy

    @staticmethod
    def class_repr():
        """
        Returns a string representation of class ``UTTT`` (meant to instantiate states of UTTT game), informing about the size of board.

        Returns:
            str: string representation of class ``UTTT`` (meant to instantiate states of UTTT game), informing about the size of board.
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

    @staticmethod
    def _check_subboard_winner(sb):
        # przekazujemy do funkcji podtablicę 3x3
        # sprawdzamy wiersze, kolumny i przekątne
        # Funkcja może zwracać trzy wartości: 1, jeżeli podtablicę wygrał krzyżyk, -1 jeżeli podtablicę wygrało kółko
        # Albo 0 jeżeli nie ma zwycięzcy

        # Sprawdzenie, czy wygrał ktoś w wierszu
        for r in range(3):
            if sb[r, 0] != 0 and sb[r, 0] == sb[r, 1] and sb[r, 1] == sb[r, 2]:
                return sb[r, 0]

        # Sprawdzenie, czy wygrał ktoś w jakiejś kolumnie
        for c in range(3):
            if sb[0, c] != 0 and sb[0, c] == sb[1, c] and sb[1, c] == sb[2, c]:
                return sb[0, c]

        # Sprawdzenie, czy wygrał ktoś na głównej przekątnej albo na przeciwprzekątnej
        if sb[0, 0] != 0 and sb[0, 0] == sb[1, 1] and sb[1, 1] == sb[2, 2]:
            return sb[0, 0]

        if sb[0, 2] != 0 and sb[0, 2] == sb[1, 1] and sb[1, 1] == sb[2, 0]:
            return sb[0, 2]

        # Jeżeli nikt nie wygrał, zwracane jest 0
        return 0

    def take_action(self, action_index):
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

        Uwaga - jeżeli kolejna podtablica jest niemożliwa do zagrania (jest wygrana albo wszystkie pola są zajęte),
        to w indeksie 9 i 10 extra_info będą wartości -1 i -1. To oznacza, że kolejny ruch może zostać wykonany
        w dowolnej podtablicy.
        """

        # Konwersja indexu ruchu do i, j
        i = action_index // 9
        j = action_index % 9

        # Obliczamy indeks podtablicy 3x3
        I = i // 3
        J = j // 3

        # Sprawdzenie, czy pole jest wolne
        if self.board[i, j] != 0:
            print(f"Niepoprawny ruch! Indeks ({i},{j}) jest już zajęty.")
            return None

        # Sprawdzenie, czy ruch jest zgodny z wyznaczoną podtablicą
        I_now = self.extra_info[9]
        J_now = self.extra_info[10]

        if I_now != -1 and J_now != -1:
            table_now_state = self.extra_info[I_now * 3 + J_now]
            if table_now_state in (0, 2):  # podtablica aktywna
                if I != I_now or J != J_now:
                    print(f"Nieprawidłowy ruch! Ruch musi być wykonany w podtablicy ({I_now},{J_now}), a nie ({I},{J})")
                    return None

        # Tworzymy nowy obiekt stanu gry
        child = UTTT(parent=self)

        # Wykonujemy ruch
        child.board[i, j] = self.turn

        # Sprawdzamy zwycięzcę podtablicy
        sub_board = child.board[I * 3:(I + 1) * 3, J * 3:(J + 1) * 3]
        won = self._check_subboard_winner(sub_board)

        if won != 0:
            child.extra_info[I * 3 + J] = won
        elif np.all(sub_board != 0):
            child.extra_info[I * 3 + J] = -2  # pat
        else:
            child.extra_info[I * 3 + J] = 2  # gra trwa

        # Wyznaczamy kolejną podtablicę
        I_next = i % 3
        J_next = j % 3
        next_state = child.extra_info[I_next * 3 + J_next]

        if next_state in (0, 2):
            child.extra_info[9] = I_next
            child.extra_info[10] = J_next
        else:
            child.extra_info[9] = -1
            child.extra_info[10] = -1

        # Zmieniamy gracza
        child.turn = -self.turn

        return child

    def compute_outcome(self):

        # M przechowuje stany wszystkich podtablic.
        M = self.extra_info[:9]

        # Wszystkie możliwe scenariusze zwycięstw głównej tablicy 9x9
        win_scenarios = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]

        for scenario in win_scenarios:
            a, b, c = scenario
            if M[a] == M[b] == M[c] and M[a] in (1, -1): # Sprawdzamy czy któryś z graczy wygrał
                return M[a]  # 1 jeżeli wygrał krzyżyk, -1 jeżeli wygrało kółko

        # Jeżeli nikt nie wygrał
        finished = True
        for s in M:
            if s in (0, 2):  # Jeżeli jeszcze któraś z podtablic jest aktywna
                finished = False # Gra trwa dalej
                break

        if finished: # Jeżeli wszystkie podtablice są zajęte
            return 0  # remis na głównej tablicy 9x9

        # W przeciwnym przypadku gra trwa
        return None

    @njit
    def compute_outcome_job_numba_jit(extra_info):
        M = extra_info[:9]

        win_scenarios = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]

        for scenario in win_scenarios:
            a, b, c = scenario
            if M[a] == M[b] == M[c] and M[a] in (1, -1):
                return M[a]

        finished = True
        for s in M:
            if s in (0, 2):
                finished = False
                break

        if finished:
            return 0

        return 2

    def take_random_action_playout(self):
        child = UTTT(parent=self)  # kopiujemy obecny stan

        I_now = child.extra_info[9]
        J_now = child.extra_info[10]

        # Lista wszystkich możliwych ruchów
        possible_moves = []

        if I_now == -1 and J_now == -1:
            # Dowolna podtablica aktywna
            for I in range(3):
                for J in range(3):
                    sub_index = I * 3 + J
                    if child.extra_info[sub_index] in (0, 2):  # podtablica aktywna
                        sub_board = child.board[I * 3:(I + 1) * 3, J * 3:(J + 1) * 3]
                        free_positions = np.argwhere(sub_board == 0)
                        for pos in free_positions:
                            i, j = pos
                            possible_moves.append((i + I * 3, j + J * 3))
        else:
            # Tylko wskazana podtablica
            sub_board = child.board[I_now * 3:(I_now + 1) * 3, J_now * 3:(J_now + 1) * 3]
            free_positions = np.argwhere(sub_board == 0)
            for pos in free_positions:
                i, j = pos
                possible_moves.append((i + I_now * 3, j + J_now * 3))

        if not possible_moves:
            return None  # brak dostępnych ruchów

        # losowy wybór ruchu
        i, j = possible_moves[np.random.randint(len(possible_moves))]
        action_index = i * 9 + j
        child.take_action(action_index)
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
        return self.extra_info

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