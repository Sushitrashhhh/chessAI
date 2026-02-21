"""
Test script to verify Chess AI installation and components.
Run this after installation to ensure everything works.
"""

def test_imports():
    """Test if all required packages are installed."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ❌ PyTorch not found: {e}")
        return False
    
    try:
        import chess
        print(f"  ✅ python-chess")
    except ImportError as e:
        print(f"  ❌ python-chess not found: {e}")
        return False
    
    try:
        import pygame
        print(f"  ✅ pygame {pygame.version.ver}")
    except ImportError as e:
        print(f"  ❌ pygame not found: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ❌ NumPy not found: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print(f"  ✅ tqdm")
    except ImportError as e:
        print(f"  ❌ tqdm not found: {e}")
        return False

    try:
        import pandas as pd
        print(f"  ✅ pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ❌ pandas not found: {e}")
        return False

    return True


def test_modules():
    """Test if project modules can be imported."""
    print("\nTesting project modules...")
    
    try:
        import config
        print(f"  ✅ config.py")
    except Exception as e:
        print(f"  ❌ config.py: {e}")
        return False
    
    try:
        import neural_network
        print(f"  ✅ neural_network.py")
    except Exception as e:
        print(f"  ❌ neural_network.py: {e}")
        return False
    
    try:
        import mcts
        print(f"  ✅ mcts.py")
    except Exception as e:
        print(f"  ❌ mcts.py: {e}")
        return False
    
    try:
        import trainer
        print(f"  ✅ trainer.py")
    except Exception as e:
        print(f"  ❌ trainer.py: {e}")
        return False
    
    try:
        import engine_new
        print(f"  ✅ engine_new.py")
    except Exception as e:
        print(f"  ❌ engine_new.py: {e}")
        return False

    try:
        import dataset_loader
        print(f"  ✅ dataset_loader.py")
    except Exception as e:
        print(f"  ❌ dataset_loader.py: {e}")
        return False

    try:
        import alpha_beta
        print(f"  ✅ alpha_beta.py")
    except Exception as e:
        print(f"  ❌ alpha_beta.py: {e}")
        return False

    return True


def test_neural_network():
    """Test neural network creation and forward pass."""
    print("\nTesting neural network...")
    
    try:
        from neural_network import ChessModel, board_to_tensor
        import chess
        import torch
        
        # Create model
        model = ChessModel()
        print("  ✅ Model created")
        
        # Test board encoding
        board = chess.Board()
        tensor = board_to_tensor(board)
        
        expected_shape = (14, 8, 8)
        if tensor.shape == expected_shape:
            print(f"  ✅ Board encoding shape: {tensor.shape}")
        else:
            print(f"  ❌ Board encoding shape wrong: {tensor.shape} (expected {expected_shape})")
            return False
        
        # Test prediction
        move_probs, value = model.predict(board)
        
        if isinstance(move_probs, dict) and len(move_probs) > 0:
            print(f"  ✅ Prediction works: {len(move_probs)} legal moves")
        else:
            print(f"  ❌ Prediction failed")
            return False
        
        if isinstance(value, float):
            print(f"  ✅ Value prediction: {value:.3f}")
        else:
            print(f"  ❌ Value prediction failed")
            return False
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠️  CUDA not available (will use CPU - slower)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Neural network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcts():
    """Test MCTS implementation."""
    print("\nTesting MCTS...")
    
    try:
        from mcts import MCTS, MCTSNode
        from neural_network import ChessModel
        import chess
        
        model = ChessModel()
        mcts = MCTS(model, num_simulations=10)  # Just 10 for testing
        
        board = chess.Board()
        
        # Test MCTS search
        move, policy = mcts.search(board, temperature=1.0)
        
        if move is not None:
            print(f"  ✅ MCTS selected move: {move.uci()}")
        else:
            print(f"  ❌ MCTS returned None")
            return False
        
        if isinstance(policy, dict) and len(policy) > 0:
            print(f"  ✅ MCTS policy: {len(policy)} moves")
        else:
            print(f"  ❌ MCTS policy failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ MCTS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_engine():
    """Test chess engine."""
    print("\nTesting chess engine...")
    
    try:
        from engine_new import ChessEngine
        import chess
        
        engine = ChessEngine(num_simulations=10)  # Fast for testing
        print("  ✅ Engine created")
        
        board = chess.Board()
        move = engine.get_best_move(board)
        
        if move is not None:
            print(f"  ✅ Engine selected move: {move.uci()}")
        else:
            print(f"  ❌ Engine returned None")
            return False
        
        value = engine.evaluate_position(board)
        print(f"  ✅ Position evaluation: {value:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer():
    """Test trainer (without actually training)."""
    print("\nTesting trainer...")
    
    try:
        from trainer import Trainer, ReplayBuffer
        
        trainer = Trainer()
        print("  ✅ Trainer created")
        
        buffer = ReplayBuffer(max_size=100)
        print("  ✅ Replay buffer created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alpha_beta():
    """Test the Alpha-Beta engine."""
    print("\nTesting Alpha-Beta engine...")

    try:
        from alpha_beta import AlphaBetaPlayer, heuristic_eval
        import chess

        board = chess.Board()

        # Heuristic eval on starting position should be near 0
        score = heuristic_eval(board)
        if abs(score) < 0.1:
            print(f"  ✅ Heuristic eval starting pos: {score:.4f} (near 0 ✓)")
        else:
            print(f"  ⚠️  Heuristic eval starting pos: {score:.4f} (expected ~0)")

        # Depth-2 search is fast enough for a test
        player = AlphaBetaPlayer(max_depth=2, time_limit=None)
        move = player.select_move(board)

        if move is not None and board.is_legal(move):
            print(f"  ✅ Alpha-Beta selected legal move: {move.uci()}")
        else:
            print(f"  ❌ Alpha-Beta returned illegal/None move")
            return False

        return True

    except Exception as e:
        print(f"  ❌ Alpha-Beta test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loader():
    """Test dataset_loader can import and report stats."""
    print("\nTesting dataset loader...")

    try:
        from dataset_loader import get_dataset_stats
        import os

        if os.path.exists('games.csv'):
            stats = get_dataset_stats('games.csv')
            if stats.get('total_games', 0) > 0:
                print(f"  ✅ games.csv found: {stats['total_games']:,} games")
            else:
                print("  ⚠️  games.csv found but appears empty")
        else:
            print("  ⚠️  games.csv not found — skipping (not required to run)")

        return True

    except Exception as e:
        print(f"  ❌ Dataset loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Chess AI - System Test")
    print("="*60)
    
    tests = [
        ("Dependencies",    test_imports),
        ("Project Modules", test_modules),
        ("Neural Network",  test_neural_network),
        ("MCTS",            test_mcts),
        ("Engine",          test_engine),
        ("Trainer",         test_trainer),
        ("Alpha-Beta",      test_alpha_beta),
        ("Dataset Loader",  test_dataset_loader),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run 'python start.py' for interactive menu")
        print("  2. Run 'python ChessGame.py' to play")
        print("  3. Run 'python examples.py' to see examples")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")
        print("\nCommon fixes:")
        print("  - Run: pip install -r requirements.txt")
        print("  - Check Python version (need 3.8+)")
        print("  - Check file permissions")
    
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
