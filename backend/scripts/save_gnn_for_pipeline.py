"""
Save the current GNN model from the notebook in the format expected by the Railtracks pipeline.
Run this (or the equivalent) after training in the GNN notebook, with model and config in scope.

Usage from Python (e.g. after running the GNN notebook):
  from src.model.gnn_inference import GraphSAGE_AML
  from pathlib import Path
  import torch
  model = ...  # your trained GraphSAGE_AML
  path = Path("model/run_1_GraphSAGE_A+B_(Synergy).pkl")
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save({
      "state_dict": model.state_dict(),
      "config": {"input_dim": x_train.shape[1], "hidden_dim": 64, "num_layers": 3, "dropout": 0.3},
  }, path)
"""
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    print("Save from the GNN notebook with:")
    print("  torch.save({'state_dict': model.state_dict(), 'config': {...}}, 'model/run_1_GraphSAGE_A+B_(Synergy).pkl')")
