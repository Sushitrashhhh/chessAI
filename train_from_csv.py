"""
train_from_csv.py
=================
Standalone supervised pre-training script from games.csv.

What this does
--------------
1. Reads games.csv and filters games by ELO rating
2. Replays every game move-by-move, storing:
      board state  →  as a 14-plane tensor
      policy       →  ONE-HOT (move actually played = 1.0)
      value        →  +1.0 / -1.0 / 0.0 from side-to-move's perspective
3. Shuffles all positions and trains the neural network using:
      Cross-entropy loss  on the policy head  (behaviour cloning)
      MSE loss            on the value head   (outcome prediction)
4. Saves a checkpoint after every epoch AND a final 'latest.pt' model

Why this matters
----------------
* Without pre-training: the AI plays random-looking moves initially
* After  pre-training: the AI immediately plays recognisable chess
  (develops pieces, controls centre, avoids blunders)
* Self-play afterwards then refines it beyond human-imitation level

Usage
-----
    python train_from_csv.py                        # defaults from config.py
    python train_from_csv.py --min_elo 1800         # higher quality games only
    python train_from_csv.py --epochs 10            # more epochs
    python train_from_csv.py --max_games 5000       # quick test run
"""

import os
import sys
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from neural_network import ChessModel, board_to_tensor, move_to_index
from dataset_loader import load_games_into_buffer, get_dataset_stats
from config import (
    GAMES_CSV_PATH, MIN_ELO_FILTER, MAX_GAMES_TO_LOAD,
    CSV_TRAIN_EPOCHS, CSV_BATCH_SIZE, MODEL_PATH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PositionStore:
    """
    Simple list-backed store for (board, policy, value) triples.
    Unlike ReplayBuffer (deque with maxlen), this has NO size cap so all
    positions from the CSV are kept in memory.
    """

    def __init__(self):
        self._data = []

    def add_game(self, game_data):
        self._data.extend(game_data)

    def shuffle(self):
        random.shuffle(self._data)

    def __len__(self):
        return len(self._data)

    def get_batch(self, start: int, size: int):
        chunk = self._data[start: start + size]
        boards   = [item[0] for item in chunk]
        policies = [item[1] for item in chunk]
        values   = [item[2] for item in chunk]
        return boards, policies, values


def _format_time(seconds: float) -> str:
    """Human-readable elapsed time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _estimate_remaining(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "?"
    rate = done / elapsed
    remaining = (total - done) / rate
    return _format_time(remaining)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_from_csv(
    csv_path:  str   = GAMES_CSV_PATH,
    min_elo:   int   = MIN_ELO_FILTER,
    max_games: int   = MAX_GAMES_TO_LOAD,
    epochs:    int   = CSV_TRAIN_EPOCHS,
    batch_size: int  = CSV_BATCH_SIZE,
    save_every: int  = 1,
    resume:    bool  = False,
):
    """
    Full supervised training pipeline from games.csv.

    Args:
        csv_path   : path to the CSV file
        min_elo    : minimum ELO rating for both players (quality filter)
        max_games  : maximum number of games to load
        epochs     : number of training epochs over the full dataset
        batch_size : mini-batch size
        save_every : save checkpoint every N epochs
        resume     : if True, load 'latest.pt' before training
    """
    print("\n" + "=" * 62)
    print("  Chess AI — Supervised Training from games.csv")
    print("=" * 62)
    print(f"  CSV        : {csv_path}")
    print(f"  Min ELO    : {min_elo}")
    print(f"  Max games  : {max_games:,}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch_size}")
    print("=" * 62)

    # -----------------------------------------------------------------------
    # 1. Dataset statistics
    # -----------------------------------------------------------------------
    get_dataset_stats(csv_path)

    # -----------------------------------------------------------------------
    # 2. Load ALL positions into an uncapped store
    # -----------------------------------------------------------------------
    store = PositionStore()
    load_games_into_buffer(store, csv_path, min_elo, max_games)

    if len(store) == 0:
        print("\n[ERROR] No positions loaded. Check that games.csv exists "
              "and contains games above the ELO threshold.")
        return None

    print(f"\nTotal positions in memory : {len(store):,}")
    batches_per_epoch = len(store) // batch_size
    print(f"Batches per epoch         : {batches_per_epoch:,}")

    # -----------------------------------------------------------------------
    # 3. Build / load model
    # -----------------------------------------------------------------------
    os.makedirs(MODEL_PATH, exist_ok=True)
    model = ChessModel()

    if resume:
        latest = os.path.join(MODEL_PATH, 'latest.pt')
        if os.path.exists(latest):
            model.load(latest)
            print(f"\n[Resume] Loaded weights from {latest}")
        else:
            print(f"\n[Resume] No checkpoint found at {latest}, starting fresh.")

    print(f"\nDevice  : {model.device}")
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"Params  : {total_params:,}")

    # -----------------------------------------------------------------------
    # 4. Training loop
    # -----------------------------------------------------------------------
    history = {'epoch': [], 'total_loss': [], 'policy_loss': [], 'value_loss': []}
    global_start = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\n{'─'*62}")
        print(f"  Epoch {epoch}/{epochs}")
        print(f"{'─'*62}")

        # Shuffle positions at the start of every epoch
        store.shuffle()

        epoch_losses = {'total': [], 'policy': [], 'value': []}
        epoch_start  = time.time()

        pbar = tqdm(
            range(batches_per_epoch),
            desc=f"  Epoch {epoch}",
            unit="batch",
            ncols=80,
            dynamic_ncols=True,
        )

        for batch_idx in pbar:
            start = batch_idx * batch_size
            boards, policies, values = store.get_batch(start, batch_size)

            losses = model.train_step(boards, policies, values)

            epoch_losses['total'].append(losses['total_loss'])
            epoch_losses['policy'].append(losses['policy_loss'])
            epoch_losses['value'].append(losses['value_loss'])

            # Live loss in progress bar every 50 batches
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'loss': f"{losses['total_loss']:.4f}",
                    'pol':  f"{losses['policy_loss']:.4f}",
                    'val':  f"{losses['value_loss']:.4f}",
                })

        # ── Epoch summary ──────────────────────────────────────────────────
        avg_total  = float(np.mean(epoch_losses['total']))
        avg_policy = float(np.mean(epoch_losses['policy']))
        avg_value  = float(np.mean(epoch_losses['value']))
        epoch_time = time.time() - epoch_start
        elapsed    = time.time() - global_start
        eta        = _estimate_remaining(elapsed, epoch, epochs)

        print(f"\n  ✔ Epoch {epoch} done in {_format_time(epoch_time)}   "
              f"ETA: {eta}")
        print(f"     Total Loss  : {avg_total:.4f}")
        print(f"     Policy Loss : {avg_policy:.4f}  "
              f"(↓ = learning which moves to play)")
        print(f"     Value  Loss : {avg_value:.4f}  "
              f"(↓ = learning who is winning)")

        history['epoch'].append(epoch)
        history['total_loss'].append(avg_total)
        history['policy_loss'].append(avg_policy)
        history['value_loss'].append(avg_value)

        # ── Save checkpoint ────────────────────────────────────────────────
        if epoch % save_every == 0:
            ckpt_path = os.path.join(MODEL_PATH, f'csv_epoch_{epoch:03d}.pt')
            model.save(ckpt_path)
            print(f"     💾 Checkpoint → {ckpt_path}")

    # -----------------------------------------------------------------------
    # 5. Save final model
    # -----------------------------------------------------------------------
    latest_path    = os.path.join(MODEL_PATH, 'latest.pt')
    pretrain_path  = os.path.join(MODEL_PATH, 'pretrained_supervised.pt')
    model.save(latest_path)
    model.save(pretrain_path)

    total_time = time.time() - global_start
    print(f"\n{'=' * 62}")
    print("  Training Complete!")
    print(f"{'=' * 62}")
    print(f"  Total time        : {_format_time(total_time)}")
    print(f"  Final total loss  : {history['total_loss'][-1]:.4f}")
    print(f"  Final policy loss : {history['policy_loss'][-1]:.4f}")
    print(f"  Final value loss  : {history['value_loss'][-1]:.4f}")
    print(f"  Model saved       → {latest_path}")
    print(f"  Model saved       → {pretrain_path}")
    print(f"{'=' * 62}")
    print("\n  Next step: Run self-play to refine beyond imitation level.")
    print("    python start.py  →  Option 3  (Full Training)\n")

    # Print loss curve summary
    print("  Loss curve:")
    for e, tl, pl, vl in zip(
        history['epoch'],
        history['total_loss'],
        history['policy_loss'],
        history['value_loss'],
    ):
        bar = '█' * int(tl * 20)
        print(f"    Epoch {e:>2}: total={tl:.4f} pol={pl:.4f} val={vl:.4f}  {bar}")

    return model, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised pre-training from games.csv"
    )
    parser.add_argument('--csv',        default=GAMES_CSV_PATH, help='Path to CSV')
    parser.add_argument('--min_elo',    type=int, default=MIN_ELO_FILTER,
                        help='Minimum ELO for both players')
    parser.add_argument('--max_games',  type=int, default=MAX_GAMES_TO_LOAD,
                        help='Max games to load')
    parser.add_argument('--epochs',     type=int, default=CSV_TRAIN_EPOCHS,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=CSV_BATCH_SIZE,
                        help='Mini-batch size')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest.pt if it exists')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    train_from_csv(
        csv_path   = args.csv,
        min_elo    = args.min_elo,
        max_games  = args.max_games,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        save_every = args.save_every,
        resume     = args.resume,
    )
