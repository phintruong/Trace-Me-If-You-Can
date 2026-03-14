"""Unified GNN model zoo for AML. Loads both state_dict and model_state_dict checkpoint formats."""

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCN_AML(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.3, num_classes=2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.cls(x)
        return F.log_softmax(x, dim=1)


class GraphSAGE_AML(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.3, aggr="mean", num_classes=2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x_prev = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if x.shape == x_prev.shape:
                x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.cls(x)
        return F.log_softmax(x, dim=1)


class GAT_AML(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, heads=2, dropout=0.3, num_classes=2, gat_concat=False):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=gat_concat, dropout=dropout))
        in_dim = (hidden_dim * heads) if gat_concat else hidden_dim
        for _ in range(max(num_layers - 2, 0)):
            self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=gat_concat, dropout=dropout))
            in_dim = (hidden_dim * heads) if gat_concat else hidden_dim
        self.convs.append(GATConv(in_dim, hidden_dim, heads=1, concat=False, dropout=dropout))
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = F.elu(x)
        x = self.cls(x)
        return F.log_softmax(x, dim=1)


def build_model(model_name: str, input_dim: int, cfg: Dict[str, Any]) -> nn.Module:
    """Factory: instantiate GNN from checkpoint config."""
    if model_name == "GCN":
        return GCN_AML(
            input_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    if model_name == "GraphSAGE":
        return GraphSAGE_AML(
            input_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            aggr=cfg.get("aggr", "mean"),
        )
    if model_name == "GAT":
        return GAT_AML(
            input_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            heads=cfg["heads"],
            dropout=cfg["dropout"],
            gat_concat=cfg.get("gat_concat", False),
        )
    raise ValueError(f"Unknown model architecture: {model_name}")


def _is_backend_checkpoint(obj: Any) -> bool:
    """Checkpoint format: model_state_dict, config, input_dim, model_name, feature_set."""
    return isinstance(obj, dict) and {"model_state_dict", "config", "input_dim", "model_name", "feature_set"}.issubset(obj.keys())


def load_gnn_model(
    model_path: str | Path,
    input_dim: int | None = None,
    device: torch.device | None = None,
) -> tuple[nn.Module, int]:
    """
    Load GNN from .pkl. Accepts:
    - Backend format: model_state_dict, config, input_dim, model_name, feature_set
    - Src/notebook format: state_dict, config (with optional input_dim in config)
    Returns (model, input_dim).
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"GNN model not found: {path}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(path, map_location=device, weights_only=False)

    if isinstance(data, dict):
        if _is_backend_checkpoint(data):
            cfg = data["config"]
            inp_dim = int(data["input_dim"])
            model_name = data["model_name"]
            model = build_model(model_name, inp_dim, cfg)
            model.load_state_dict(data["model_state_dict"], strict=True)
        elif "state_dict" in data:
            cfg = data.get("config", {})
            inp_dim = int(input_dim or cfg.get("input_dim", 0))
            if inp_dim <= 0:
                raise ValueError("input_dim required when checkpoint has state_dict only")
            model = GraphSAGE_AML(
                input_dim=inp_dim,
                hidden_dim=cfg.get("hidden_dim", 64),
                num_layers=cfg.get("num_layers", 3),
                dropout=cfg.get("dropout", 0.3),
                aggr=cfg.get("aggr", "mean"),
            )
            model.load_state_dict(data["state_dict"], strict=True)
        elif "config" in data and "input_dim" in data["config"]:
            cfg = data["config"]
            inp_dim = int(cfg["input_dim"])
            model = GraphSAGE_AML(
                input_dim=inp_dim,
                hidden_dim=cfg.get("hidden_dim", 64),
                num_layers=cfg.get("num_layers", 3),
                dropout=cfg.get("dropout", 0.3),
            )
            if "state_dict" in data:
                model.load_state_dict(data["state_dict"], strict=True)
            else:
                raise ValueError("Checkpoint has config but no state_dict")
        else:
            raise ValueError("Unrecognized checkpoint format: need state_dict or model_state_dict + config + input_dim")
    elif isinstance(data, nn.Module):
        model = data
        inp_dim = input_dim or getattr(model, "input_dim", None)
        if inp_dim is None:
            raise ValueError("input_dim required when checkpoint is raw nn.Module")
    else:
        raise TypeError("Expected dict or nn.Module in .pkl")

    model.to(device)
    model.eval()
    return model, inp_dim
