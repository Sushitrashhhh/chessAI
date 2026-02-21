"""
alpha_beta.py
=============
Classical Alpha-Beta Minimax search engine for the Chess AI.

Features
--------
* Iterative Deepening Alpha-Beta (IDAB)       — time-bounded search
* Quiescence Search (QS)                      — avoids horizon effect on captures
* Transposition Table (Zobrist-keyed dict)    — avoid re-evaluating positions
* Move Ordering                               — MVV-LVA captures first, then killers
* Two evaluation modes
    1. Heuristic  — piece values + piece-square tables (no model needed)
    2. Neural Net — uses ChessModel.predict() value head as leaf evaluator
                    (plugged in via constructor; much stronger when trained)

Usage
-----
    # Heuristic only (no model needed)
    player = AlphaBetaPlayer(max_depth=4)
    move   = player.select_move(board)

    # Neural-net evaluator
    from neural_network import ChessModel
    model  = ChessModel()
    player = AlphaBetaPlayer(max_depth=4, model=model)
    move   = player.select_move(board)

    # Time-limited (recommended for play)
    player = AlphaBetaPlayer(max_depth=6, time_limit=2.0)
    move   = player.select_move(board)
"""

import chess
import time
import math
from typing import Optional, Dict, Tuple

from config import NUM_SIMULATIONS   # kept for parity; not used directly here

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INF = float('inf')

# Piece material values (centipawns)
PIECE_VALUES: Dict[int, int] = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

# ---------------------------------------------------------------------------
# Piece-Square Tables (PST)
# White's perspective, a1=index 0, h8=index 63.
# Source: adapted from CPW (Chess Programming Wiki) tables.
# ---------------------------------------------------------------------------

# fmt: off
_PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

_KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

_BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

_ROOK_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

_QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

_KING_MID_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

_KING_END_PST = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]
# fmt: on

PST: Dict[int, list] = {
    chess.PAWN:   _PAWN_PST,
    chess.KNIGHT: _KNIGHT_PST,
    chess.BISHOP: _BISHOP_PST,
    chess.ROOK:   _ROOK_PST,
    chess.QUEEN:  _QUEEN_PST,
    chess.KING:   _KING_MID_PST,   # swapped to endgame table when queens off
}


# ---------------------------------------------------------------------------
# Heuristic Evaluation
# ---------------------------------------------------------------------------

def _is_endgame(board: chess.Board) -> bool:
    """Simple endgame detection: no queens, or both sides have <= 1 minor piece."""
    queens = board.pieces(chess.QUEEN, chess.WHITE) | board.pieces(chess.QUEEN, chess.BLACK)
    if not queens:
        return True
    # Both sides have queen but very little material
    white_mat = sum(
        PIECE_VALUES[pt] * len(board.pieces(pt, chess.WHITE))
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK)
    )
    black_mat = sum(
        PIECE_VALUES[pt] * len(board.pieces(pt, chess.BLACK))
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK)
    )
    return white_mat + black_mat <= 1300   # roughly rook + minor piece each


def _pst_score(square: int, piece_type: int, color: bool, endgame: bool) -> int:
    """Return PST bonus for a piece on a given square (White's perspective)."""
    if piece_type == chess.KING and endgame:
        table = _KING_END_PST
    else:
        table = PST[piece_type]

    if color == chess.WHITE:
        # PST is indexed from a1 (row 0 = rank 1) — flip rank for white
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        idx  = (7 - rank) * 8 + file
    else:
        # Black uses the table as-is (mirrored)
        idx = square  # a8=56 in PST order for black

    return table[idx] if 0 <= idx < 64 else 0


def heuristic_eval(board: chess.Board) -> float:
    """
    Static evaluation of *board* in centipawns from White's perspective.

    Returns a value in roughly [-1, 1] (divided by 10000 for NN compatibility).
    Positive → White is better, Negative → Black is better.
    """
    if board.is_checkmate():
        # The side to move is in checkmate → they lose
        return -1.0 if board.turn == chess.WHITE else 1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
        return 0.0

    eg = _is_endgame(board)
    score = 0

    for square, piece in board.piece_map().items():
        val  = PIECE_VALUES[piece.piece_type]
        pst  = _pst_score(square, piece.piece_type, piece.color, eg)
        sign = 1 if piece.color == chess.WHITE else -1
        score += sign * (val + pst)

    # Normalise to [-1, 1] range (max material ~4000cp → king excluded)
    return max(-1.0, min(1.0, score / 4000.0))


# ---------------------------------------------------------------------------
# Move Ordering
# ---------------------------------------------------------------------------

# MVV-LVA table: attacker (rows) × victim (cols) — higher = better capture
_MVV_LVA = {
    (a, v): PIECE_VALUES[v] - PIECE_VALUES[a] // 10
    for a in PIECE_VALUES
    for v in PIECE_VALUES
}


def _move_score(board: chess.Board, move: chess.Move, killer1=None, killer2=None) -> int:
    """
    Assign an ordering score to *move* (higher = try first).

    Priority:
        1. Captures (MVV-LVA)
        2. Promotions
        3. Killer moves (non-capture that caused beta-cutoff before)
        4. Everything else
    """
    if board.is_capture(move):
        attacker = board.piece_type_at(move.from_square)
        victim   = board.piece_type_at(move.to_square)
        if victim is None:
            # En-passant
            victim = chess.PAWN
        return 10000 + _MVV_LVA.get((attacker, victim), 0)

    if move.promotion:
        return 9000 + PIECE_VALUES.get(move.promotion, 0)

    if move == killer1:
        return 8000
    if move == killer2:
        return 7000

    return 0


def _ordered_moves(board: chess.Board, killer1=None, killer2=None):
    """Return legal moves sorted by _move_score descending."""
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: _move_score(board, m, killer1, killer2), reverse=True)
    return moves


# ---------------------------------------------------------------------------
# Transposition Table entry
# ---------------------------------------------------------------------------

EXACT  = 0
LBOUND = 1   # alpha (lower bound)
UBOUND = 2   # beta  (upper bound)


class TTEntry:
    __slots__ = ('depth', 'score', 'flag', 'best_move')

    def __init__(self, depth, score, flag, best_move=None):
        self.depth     = depth
        self.score     = score
        self.flag      = flag
        self.best_move = best_move


# ---------------------------------------------------------------------------
# Alpha-Beta Engine
# ---------------------------------------------------------------------------

class AlphaBetaEngine:
    """
    Iterative-deepening Alpha-Beta search engine.

    Parameters
    ----------
    model       : ChessModel or None
                  If provided, model.predict(board) value is used at leaf nodes
                  instead of the heuristic evaluator. Requires a trained model
                  for meaningful results.
    max_depth   : int  — hard depth limit (used when time_limit is None)
    time_limit  : float or None — seconds per move (preferred over max_depth)
    use_nn_eval : bool — whether to use the neural-net value at leaf nodes.
                  Automatically True when model is provided.
    """

    def __init__(
        self,
        model=None,
        max_depth: int = 4,
        time_limit: Optional[float] = None,
        use_nn_eval: bool = True,
    ):
        self.model       = model
        self.max_depth   = max_depth
        self.time_limit  = time_limit
        self.use_nn_eval = use_nn_eval and model is not None

        # State reset per search
        self._tt:      Dict[int, TTEntry] = {}
        self._killers: Dict[int, list]    = {}   # ply → [killer1, killer2]
        self._nodes    = 0
        self._start    = 0.0
        self._abort    = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, board: chess.Board) -> Tuple[Optional[chess.Move], float]:
        """
        Find the best move for the side to move using iterative deepening.

        Returns
        -------
        (best_move, score)
            score is from the perspective of the side to move,
            normalised to [-1, 1].
        """
        self._tt      = {}
        self._killers = {}
        self._nodes   = 0
        self._abort   = False
        self._start   = time.time()

        legal = list(board.legal_moves)
        if not legal:
            return None, 0.0
        if len(legal) == 1:
            return legal[0], 0.0

        best_move  = legal[0]
        best_score = -INF

        depth_limit = self.max_depth if self.time_limit is None else 99

        for depth in range(1, depth_limit + 1):
            move, score = self._root_search(board, depth)

            if self._abort:
                break   # Time ran out — keep result from previous depth

            best_move  = move
            best_score = score

            if self.time_limit is None:
                continue  # Fixed depth — always run to max_depth

            elapsed = time.time() - self._start
            if elapsed >= self.time_limit * 0.5:
                # Heuristic: if we've used half our time, don't start deeper
                break

        return best_move, best_score

    def get_stats(self) -> dict:
        """Return search statistics from the last search call."""
        return {
            'nodes_searched': self._nodes,
            'tt_size':        len(self._tt),
            'elapsed':        round(time.time() - self._start, 3),
        }

    # ------------------------------------------------------------------
    # Root search
    # ------------------------------------------------------------------

    def _root_search(
        self, board: chess.Board, depth: int
    ) -> Tuple[Optional[chess.Move], float]:
        """Alpha-beta at the root — returns (best_move, score)."""
        alpha = -INF
        beta  =  INF
        best_move  = None
        best_score = -INF

        for move in _ordered_moves(board):
            if self._timed_out():
                self._abort = True
                break

            board.push(move)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, ply=1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move  = move
            if score > alpha:
                alpha = score

        return best_move, best_score

    # ------------------------------------------------------------------
    # Alpha-Beta (negamax form)
    # ------------------------------------------------------------------

    def _alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
    ) -> float:
        """
        Negamax alpha-beta.
        Returns score from the perspective of the side to move at *board*.
        """
        self._nodes += 1

        if self._timed_out():
            self._abort = True
            return 0.0

        # --- Terminal / draw ---
        if board.is_game_over():
            return self._terminal_score(board)

        # --- Transposition table lookup ---
        key = board._transposition_key()
        tt_entry = self._tt.get(key)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == EXACT:
                return tt_entry.score
            if tt_entry.flag == LBOUND:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == UBOUND:
                beta  = min(beta,  tt_entry.score)
            if alpha >= beta:
                return tt_entry.score

        # --- Quiescence search at leaves ---
        if depth <= 0:
            return self._quiescence(board, alpha, beta)

        # --- Move loop ---
        killers   = self._killers.get(ply, [None, None])
        k1, k2    = killers[0] if len(killers) > 0 else None, \
                    killers[1] if len(killers) > 1 else None

        best_score = -INF
        best_move  = None
        orig_alpha = alpha
        found_move = False

        for move in _ordered_moves(board, k1, k2):
            board.push(move)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if self._abort:
                return 0.0

            found_move = True

            if score > best_score:
                best_score = score
                best_move  = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                # Beta cut-off — store killer if not a capture
                if not board.is_capture(move):
                    self._store_killer(ply, move)
                break

        if not found_move:
            return self._terminal_score(board)

        # --- Store in transposition table ---
        if best_score <= orig_alpha:
            flag = UBOUND
        elif best_score >= beta:
            flag = LBOUND
        else:
            flag = EXACT

        self._tt[key] = TTEntry(depth, best_score, flag, best_move)

        return best_score

    # ------------------------------------------------------------------
    # Quiescence Search
    # ------------------------------------------------------------------

    def _quiescence(
        self, board: chess.Board, alpha: float, beta: float, depth: int = 0
    ) -> float:
        """
        Search captures (and promotions) until a quiet position is reached.
        Prevents the horizon effect where a piece hangs just beyond the horizon.
        """
        self._nodes += 1

        if board.is_game_over():
            return self._terminal_score(board)

        # Stand-pat: the side to move can choose not to capture
        stand_pat = self._evaluate(board)

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Limit quiescence depth to avoid blowup
        if depth >= 6:
            return alpha

        # Only look at captures and promotions
        for move in board.generate_pseudo_legal_captures():
            if not board.is_legal(move):
                continue

            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, depth + 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, board: chess.Board) -> float:
        """
        Evaluate from the perspective of the side to move.
        Delegates to the neural network if available, otherwise heuristic.
        """
        if self.use_nn_eval and self.model is not None:
            try:
                _, value = self.model.predict(board)
                # model returns value from White's perspective; flip for Black
                return value if board.turn == chess.WHITE else -value
            except Exception:
                pass  # Fall through to heuristic on error

        raw = heuristic_eval(board)
        return raw if board.turn == chess.WHITE else -raw

    def _terminal_score(self, board: chess.Board) -> float:
        """Score a terminal (game-over) position."""
        if board.is_checkmate():
            return -1.0          # Side to move is mated
        return 0.0               # Draw

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _timed_out(self) -> bool:
        if self.time_limit is None:
            return False
        return (time.time() - self._start) >= self.time_limit

    def _store_killer(self, ply: int, move: chess.Move):
        """Store a killer move for the given ply."""
        killers = self._killers.get(ply, [])
        if move not in killers:
            killers.insert(0, move)
        self._killers[ply] = killers[:2]   # Keep only 2 killers per ply


# ---------------------------------------------------------------------------
# Player Wrapper (matches AlphaZeroPlayer interface in mcts.py)
# ---------------------------------------------------------------------------

class AlphaBetaPlayer:
    """
    Wrapper around AlphaBetaEngine with the same interface as AlphaZeroPlayer,
    making it a drop-in replacement for head-to-head benchmarking.

    Parameters
    ----------
    model       : ChessModel or None
    max_depth   : int  — max search depth (used if time_limit is None)
    time_limit  : float or None — seconds per move (recommended for play)
    use_nn_eval : bool — use neural network value head at leaf nodes
    verbose     : bool — print search info after each move
    """

    def __init__(
        self,
        model=None,
        max_depth: int = 4,
        time_limit: Optional[float] = 2.0,
        use_nn_eval: bool = True,
        verbose: bool = False,
    ):
        self.engine  = AlphaBetaEngine(
            model=model,
            max_depth=max_depth,
            time_limit=time_limit,
            use_nn_eval=use_nn_eval,
        )
        self.verbose = verbose

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Select the best move for the current position."""
        move, score = self.engine.search(board)

        if self.verbose:
            stats = self.engine.get_stats()
            print(
                f"[AlphaBeta] move={move} score={score:+.3f} "
                f"nodes={stats['nodes_searched']:,} "
                f"time={stats['elapsed']}s"
            )

        return move

    def get_move_with_policy(self, board: chess.Board):
        """
        Mimic AlphaZeroPlayer.get_move_with_policy() for compatibility.
        Policy is a uniform distribution over legal moves (Alpha-Beta
        doesn't produce a policy distribution).
        """
        move = self.select_move(board)
        legal = list(board.legal_moves)
        uniform_prob = 1.0 / len(legal) if legal else 0.0
        policy = {m: uniform_prob for m in legal}
        return move, policy


# ---------------------------------------------------------------------------
# Benchmarking utility
# ---------------------------------------------------------------------------

def benchmark(
    model=None,
    depth: int = 4,
    time_limit: Optional[float] = None,
    fen: str = chess.STARTING_FEN,
):
    """
    Quick benchmark: run one search from *fen* and print statistics.

    Example
    -------
        from alpha_beta import benchmark
        benchmark(depth=5)
    """
    board  = chess.Board(fen)
    player = AlphaBetaPlayer(
        model=model,
        max_depth=depth,
        time_limit=time_limit,
        use_nn_eval=(model is not None),
        verbose=True,
    )

    print(f"\n[Benchmark] FEN: {fen}")
    print(f"[Benchmark] Depth: {depth}  TimeLimit: {time_limit}s")
    t0   = time.time()
    move = player.select_move(board)
    t1   = time.time()

    stats = player.engine.get_stats()
    print(f"[Benchmark] Best move : {move}")
    print(f"[Benchmark] Nodes     : {stats['nodes_searched']:,}")
    print(f"[Benchmark] TT size   : {stats['tt_size']:,}")
    print(f"[Benchmark] Elapsed   : {t1-t0:.3f}s")
    nps = stats['nodes_searched'] / max(t1 - t0, 1e-9)
    print(f"[Benchmark] NPS       : {nps:,.0f}")
    return move


# ---------------------------------------------------------------------------
# Head-to-head: AlphaBeta vs MCTS
# ---------------------------------------------------------------------------

def play_ab_vs_mcts(
    ab_model=None,
    mcts_model=None,
    ab_depth: int = 4,
    ab_time: Optional[float] = 2.0,
    mcts_sims: int = 100,
    verbose: bool = True,
) -> float:
    """
    Play one game: Alpha-Beta (White) vs MCTS (Black).

    Returns
    -------
    float : 1.0 = Alpha-Beta wins, -1.0 = MCTS wins, 0.0 = draw
    """
    from mcts import AlphaZeroPlayer
    from neural_network import ChessModel

    if mcts_model is None:
        mcts_model = ChessModel()

    ab_player   = AlphaBetaPlayer(
        model=ab_model,
        max_depth=ab_depth,
        time_limit=ab_time,
        verbose=verbose,
    )
    mcts_player = AlphaZeroPlayer(mcts_model, num_simulations=mcts_sims)

    board     = chess.Board()
    move_num  = 0
    max_moves = 200

    while not board.is_game_over() and move_num < max_moves:
        if board.turn == chess.WHITE:
            move = ab_player.select_move(board)
            if verbose:
                print(f"  [{move_num+1}] AB (White): {move}")
        else:
            move = mcts_player.select_move(board)
            if verbose:
                print(f"  [{move_num+1}] MCTS (Black): {move}")

        if move is None:
            break

        board.push(move)
        move_num += 1

    result_str = board.result()
    if result_str == '1-0':
        result = 1.0
    elif result_str == '0-1':
        result = -1.0
    else:
        result = 0.0

    if verbose:
        print(f"\n[Game Over] Result: {result_str}  "
              f"({'AB wins' if result>0 else 'MCTS wins' if result<0 else 'Draw'})")

    return result


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("  Alpha-Beta Engine — standalone test")
    print("=" * 60)

    arg = sys.argv[1] if len(sys.argv) > 1 else 'bench'

    if arg == 'bench':
        benchmark(depth=5)

    elif arg == 'vs_mcts':
        print("\nPlaying Alpha-Beta (White) vs MCTS (Black) ...")
        result = play_ab_vs_mcts(ab_depth=4, ab_time=2.0, mcts_sims=50, verbose=True)
        print(f"\nFinal result: {result}")

    else:
        print(f"Unknown argument '{arg}'. Use: 'bench' or 'vs_mcts'")
