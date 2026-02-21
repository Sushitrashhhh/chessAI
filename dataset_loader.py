"""
dataset_loader.py
=================
Loads real chess games from games.csv into the ReplayBuffer for
supervised pre-training before self-play begins.

CSV columns used:
    - moves          : SAN move string, space-separated (e.g. "e4 e5 Nf3 ...")
    - winner         : "white" | "black" | "draw"
    - white_rating   : integer ELO of the white player
    - black_rating   : integer ELO of the black player
    - opening_eco    : ECO code (logged but not used for training)
    - opening_name   : Opening name (logged but not used for training)

How positions are labelled
--------------------------
Policy  : Uniform distribution over legal moves at each ply.
          (We don't know the search tree, so we give equal credit
           to all legal moves and let the value signal drive learning.)
Value   : Game outcome from the perspective of the side to move.
          +1.0  → side to move eventually won
          -1.0  → side to move eventually lost
           0.0  → draw
"""

import chess
import pandas as pd
from tqdm import tqdm

from config import (
    GAMES_CSV_PATH,
    MIN_ELO_FILTER,
    MAX_GAMES_TO_LOAD,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _outcome_to_value(winner: str, side_to_move: bool) -> float:
    """
    Map the game result to a value from the perspective of the side to move.

    Args:
        winner      : 'white', 'black', or anything else (draw)
        side_to_move: chess.WHITE (True) or chess.BLACK (False)

    Returns:
        float in {-1.0, 0.0, 1.0}
    """
    if winner == 'white':
        result = 1.0
    elif winner == 'black':
        result = -1.0
    else:
        result = 0.0

    # Flip sign for Black's perspective
    return result if side_to_move == chess.WHITE else -result


def _parse_game(moves_str: str, winner: str):
    """
    Replay a single game from its SAN move string.

    Policy is ONE-HOT: the move actually played gets probability 1.0, all
    others get 0.0.  This is behaviour cloning — far stronger than uniform
    because the network learns which moves humans chose, not just outcomes.

    Args:
        moves_str : space-separated SAN moves (e.g. "e4 e5 Nf3 Nc6 ...")
        winner    : 'white' | 'black' | anything else

    Returns:
        list of (chess.Board, policy_dict, value) tuples, one per ply.
        Returns an empty list if the move string cannot be fully parsed.
    """
    board = chess.Board()
    game_data = []

    for san in moves_str.split():
        if board.is_game_over():
            break

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        # --- Parse the actual move that was played ---
        try:
            move_played = board.parse_san(san)
        except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            break  # Malformed SAN — discard rest of game

        if not board.is_legal(move_played):
            break

        # --- One-hot policy: only the played move gets probability 1.0 ---
        policy = {m: 0.0 for m in legal_moves}
        policy[move_played] = 1.0

        # --- Value from the perspective of the side to move ---
        value = _outcome_to_value(winner, board.turn)

        game_data.append((board.copy(), policy, value))

        board.push(move_played)

    return game_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_games_into_buffer(
    replay_buffer,
    csv_path: str = GAMES_CSV_PATH,
    min_elo: int = MIN_ELO_FILTER,
    max_games: int = MAX_GAMES_TO_LOAD,
    verbose: bool = True,
) -> dict:
    """
    Read games.csv, filter by ELO, replay every game move-by-move, and
    populate *replay_buffer* with labelled (board, policy, value) tuples.

    Args:
        replay_buffer : trainer.ReplayBuffer instance
        csv_path      : path to the CSV file
        min_elo       : minimum rating for both players (quality filter)
        max_games     : maximum number of games to load
        verbose       : print progress information

    Returns:
        dict with loading statistics:
            'games_loaded'     : number of games successfully parsed
            'games_skipped'    : number of games dropped (bad ELO / parse error)
            'positions_added'  : total board positions added to buffer
    """
    if verbose:
        print(f"\n[DatasetLoader] Reading {csv_path} ...")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[DatasetLoader] ERROR: '{csv_path}' not found.")
        return {'games_loaded': 0, 'games_skipped': 0, 'positions_added': 0}

    # --- Validate required columns ---
    required_cols = {'moves', 'winner', 'white_rating', 'black_rating'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[DatasetLoader] ERROR: CSV is missing columns: {missing}")
        return {'games_loaded': 0, 'games_skipped': 0, 'positions_added': 0}

    # --- ELO filter ---
    original_len = len(df)
    df = df[
        (df['white_rating'].apply(pd.to_numeric, errors='coerce') >= min_elo) &
        (df['black_rating'].apply(pd.to_numeric, errors='coerce') >= min_elo)
    ].dropna(subset=['moves', 'winner'])

    if verbose:
        print(f"[DatasetLoader] {original_len:,} total games → "
              f"{len(df):,} after ELO ≥ {min_elo} filter")

    # Cap to max_games
    df = df.head(max_games).reset_index(drop=True)

    if verbose:
        print(f"[DatasetLoader] Processing up to {len(df):,} games ...")

    # --- Process games ---
    games_loaded = 0
    games_skipped = 0
    positions_before = len(replay_buffer)

    iterator = tqdm(df.iterrows(), total=len(df), desc="Loading CSV games") \
        if verbose else df.iterrows()

    for _, row in iterator:
        game_data = _parse_game(str(row['moves']), str(row['winner']).lower())

        if game_data:
            replay_buffer.add_game(game_data)
            games_loaded += 1
        else:
            games_skipped += 1

    positions_added = len(replay_buffer) - positions_before

    if verbose:
        print(f"\n[DatasetLoader] Done!")
        print(f"  Games loaded   : {games_loaded:,}")
        print(f"  Games skipped  : {games_skipped:,}")
        print(f"  Positions added: {positions_added:,}")
        print(f"  Buffer size    : {len(replay_buffer):,}\n")

    return {
        'games_loaded': games_loaded,
        'games_skipped': games_skipped,
        'positions_added': positions_added,
    }


def get_dataset_stats(csv_path: str = GAMES_CSV_PATH) -> dict:
    """
    Print a quick summary of the dataset without loading it into memory fully.

    Returns:
        dict with basic stats, or an empty dict on error.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[DatasetLoader] '{csv_path}' not found.")
        return {}

    stats = {
        'total_games'   : len(df),
        'columns'       : list(df.columns),
        'winners'       : df['winner'].value_counts().to_dict() if 'winner' in df else {},
        'avg_white_elo' : df['white_rating'].mean() if 'white_rating' in df else None,
        'avg_black_elo' : df['black_rating'].mean() if 'black_rating' in df else None,
        'avg_turns'     : df['turns'].mean() if 'turns' in df else None,
    }

    print("\n[DatasetLoader] Dataset Statistics")
    print(f"  Total games   : {stats['total_games']:,}")
    print(f"  Columns       : {', '.join(stats['columns'])}")
    if stats['winners']:
        for k, v in stats['winners'].items():
            print(f"  Winner={k:<6} : {v:,} ({100*v/stats['total_games']:.1f}%)")
    if stats['avg_white_elo']:
        print(f"  Avg White ELO : {stats['avg_white_elo']:.0f}")
    if stats['avg_black_elo']:
        print(f"  Avg Black ELO : {stats['avg_black_elo']:.0f}")
    if stats['avg_turns']:
        print(f"  Avg turns/game: {stats['avg_turns']:.1f}")
    print()

    return stats
