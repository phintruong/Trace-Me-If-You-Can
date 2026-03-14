"""
Save a trained GNN model in the format expected by the backend pipeline.

Run after training in a notebook with model and config in scope:

    import torch
    from pathlib import Path

    path = Path("backend/model/run_1_GraphSAGE_A+B_(Synergy).pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": "GraphSAGE",
        "input_dim": x_train.shape[1],
        "feature_set": "A+B",
        "config": {
            "input_dim": x_train.shape[1],
            "hidden_dim": 64,
            "num_layers": 3,
            "dropout": 0.3,
            "aggr": "mean",
        },
    }, path)

The backend checkpoint loader (backend/app/models/gnn_models.py:load_gnn_model)
also accepts the simpler state_dict+config format for backwards compatibility.
"""

if __name__ == "__main__":
    print("This is a reference script. Run the code above from your training notebook.")
    print("See backend/app/models/gnn_models.py for supported checkpoint formats.")
