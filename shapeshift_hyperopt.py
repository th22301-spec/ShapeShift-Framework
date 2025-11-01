import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv
from sklearn.model_selection import train_test_split
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import copy
from torch.utils.data import TensorDataset, DataLoader  # NEW

# -----------------------------
# Load data
# -----------------------------
solid_df = pd.read_excel("SOLID-FEA-dataset.xlsx")
mex_df = pd.read_excel("MEX-FEA-dataset.xlsx")

mex_df.columns = [col.strip().lower() for col in mex_df.columns]
solid_df.columns = [col.strip().lower() for col in solid_df.columns]

# -----------------------------
# Best practice preprocessing:
#   - One-hot stays 0/1 (no scaling)
#   - Scale ONLY continuous numerics
#   - Use same scaler for both datasets' targets
# -----------------------------
ohe_pattern = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe_part    = OneHotEncoder(sparse=False, handle_unknown='ignore')

pattern_ohe = ohe_pattern.fit_transform(mex_df[['infill pattern']])
part_ohe    = ohe_part.fit_transform(mex_df[['part number']])

# Scale only numeric inputs for MEX
num_cols_mex = ['infill density (%)', 'layer thickness (mm)']
scaler_input = StandardScaler()
num_scaled_mex = scaler_input.fit_transform(mex_df[num_cols_mex].values)

# Concatenate: [scaled numerics | one-hot pattern | one-hot part]
mex_inputs_np = np.concatenate([num_scaled_mex, pattern_ohe, part_ohe], axis=1)
mex_targets_np = mex_df[['torsional stiffness (n.m/deg)']].values  # keep 2D

# Geometry inputs (continuous) scale; stiffness targets share same scaler as MEX targets
geom_cols = ['diameter(mm)', 'key width(mm)', 'key depth(mm)']
scaler_geom = StandardScaler()
geometry_inputs_np = scaler_geom.fit_transform(solid_df[geom_cols].values)

scaler_stiff = StandardScaler()
mex_targets_np = scaler_stiff.fit_transform(mex_targets_np)
solid_targets_np = scaler_stiff.transform(solid_df[['torsional stiffness (n.m/deg)']].values)

# Tensors
mex_inputs = torch.tensor(mex_inputs_np, dtype=torch.float32)
mex_targets = torch.tensor(mex_targets_np, dtype=torch.float32)
geometry_inputs = torch.tensor(geometry_inputs_np, dtype=torch.float32)
solid_targets = torch.tensor(solid_targets_np, dtype=torch.float32)

# -----------------------------
# Train/val split
# -----------------------------
train_geom_x, val_geom_x, train_geom_y, val_geom_y = train_test_split(
    geometry_inputs, solid_targets, test_size=0.2, random_state=42
)
train_x, val_x, train_y, val_y = train_test_split(
    mex_inputs, mex_targets, test_size=0.2, random_state=42
)

# -----------------------------
# Models
# -----------------------------
class ForwardModel(nn.Module):
    def __init__(self, layers=2, hidden_dim=64, activation=nn.ReLU):
        super().__init__()
        act = activation
        net = [nn.Linear(3, hidden_dim), act()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden_dim, hidden_dim), act()]
        net.append(nn.Linear(hidden_dim, 1))
        self.fc = nn.Sequential(*net)

    def forward(self, x):
        return self.fc(x)

class GeometryGNN(nn.Module):
    def __init__(self, input_dim, gnn_type='GIN', hidden_dim=64, activation=nn.ReLU, gnn_layers=1):
        super().__init__()
        self.act_cls = activation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation()
        )
        self.init_nodes = nn.Parameter(torch.randn(3, hidden_dim))
        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            if gnn_type == 'GIN':
                conv = GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        activation(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                )
            elif gnn_type == 'GCN':
                conv = GCNConv(hidden_dim, hidden_dim)
            else:
                raise ValueError("gnn_type must be 'GIN' or 'GCN'")
            self.gnn_layers.append(conv)
        self.decoder = nn.Linear(hidden_dim, 1)

        # Fixed edge_index for triangle graph of 3 nodes
        self.register_buffer(
            "edge_index",
            torch.tensor([[0,1,2,1,2,0],[1,2,0,0,1,2]], dtype=torch.long)
        )

    def forward(self, x):
        encoded = self.encoder(x)  # [B, H]
        out_geos = []
        for i in range(x.size(0)):
            node_feats = self.init_nodes + encoded[i]  # [3, H]
            for conv in self.gnn_layers:
                node_feats = conv(node_feats, self.edge_index)
                node_feats = self.act_cls()(node_feats)
            pred_geo = self.decoder(node_feats).squeeze()  # [3]
            out_geos.append(pred_geo)
        return torch.stack(out_geos)  # [B, 3]

# -----------------------------
# Training helpers (mini-batch)
# -----------------------------
def train_with_early_stop(
    model,
    optimizer,
    train_x,
    train_y,
    val_x,
    val_y,
    loss_fn,
    max_epochs=10000,
    patience=100,
    batch_size=None,
    shuffle=True,
):
    """Generic trainer with mini-batching (if batch_size provided)."""
    if batch_size is None:
        batch_size = len(train_x)

    train_ds = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    best_model = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = loss_fn(val_pred, val_y)

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopped at epoch {epoch}, best val loss: {best_loss:.4f}")
            break

    model.load_state_dict(best_model)
    return model

# -----------------------------
# Independent grids
# -----------------------------
forward_grid = {
    "hidden_dim": [16, 32],
    "mlp_layers": [2,3],
    "activation": [nn.Tanh, nn.ReLU],
    "lr": [1e-2, 1e-1],
    "batch_size": [16, 32],  # forward now mini-batch
}

inverse_grid = {
    "hidden_dim": [16, 32],
    "gnn_layers": [2,3],
    "batch_size": [16, 32],
    "activation": [nn.Tanh, nn.ReLU],
    "lr": [ 1e-2, 1e-1],
    "gnn_type": ["GIN", "GCN"],
}

# Build Cartesian product of independent grids
f_keys, f_vals = zip(*forward_grid.items())
i_keys, i_vals = zip(*inverse_grid.items())

experiments = []
for f_tuple in itertools.product(*f_vals):
    fcfg = dict(zip(f_keys, f_tuple))
    for i_tuple in itertools.product(*i_vals):
        icfg = dict(zip(i_keys, i_tuple))
        experiments.append({"fwd": fcfg, "inv": icfg})

loss_fn = nn.MSELoss()
results = []

# -----------------------------
# Grid search loop
# -----------------------------
for i, cfg in enumerate(experiments):
    fcfg = cfg["fwd"]
    icfg = cfg["inv"]
    print(f"\nExperiment {i+1}/{len(experiments)}")
    print("  Forward:", {k: (v.__name__ if hasattr(v, "__name__") else v) for k, v in fcfg.items()})
    print("  Inverse:", {k: (v.__name__ if hasattr(v, "__name__") else v) for k, v in icfg.items()})

    # Models
    forward_model = ForwardModel(
        layers=fcfg["mlp_layers"],
        hidden_dim=fcfg["hidden_dim"],
        activation=fcfg["activation"],
    )

    inverse_model = GeometryGNN(
        input_dim=mex_inputs.shape[1],
        gnn_type=icfg["gnn_type"],
        hidden_dim=icfg["hidden_dim"],
        activation=icfg["activation"],
        gnn_layers=icfg["gnn_layers"]
    )

    # Optimizers (independent LRs)
    optimizer_fwd = torch.optim.Adam(forward_model.parameters(), lr=fcfg["lr"])
    optimizer_inv = torch.optim.Adam(inverse_model.parameters(), lr=icfg["lr"])

    # ---- Train forward model (geometry -> stiffness) with its own batch size
    forward_model = train_with_early_stop(
        model=forward_model,
        optimizer=optimizer_fwd,
        train_x=train_geom_x,
        train_y=train_geom_y,
        val_x=val_geom_x,
        val_y=val_geom_y,
        loss_fn=loss_fn,
        batch_size=fcfg["batch_size"],
    )

    # ---- Train inverse model via forward model supervision (mini-batch on MEX)
    best_model = copy.deepcopy(inverse_model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 100

    inv_train_ds = TensorDataset(train_x, train_y)
    inv_loader = DataLoader(inv_train_ds, batch_size=icfg["batch_size"], shuffle=True, drop_last=False)

    for epoch in range(100000):
        inverse_model.train()
        for bx, by in inv_loader:
            optimizer_inv.zero_grad()
            pred_geom = inverse_model(bx)        # [B, 3]
            pred_stiff = forward_model(pred_geom)  # [B, 1]
            loss = loss_fn(pred_stiff, by)
            loss.backward()
            optimizer_inv.step()

        # Validation
        inverse_model.eval()
        with torch.no_grad():
            pred_geom_val = inverse_model(val_x)
            pred_stiff_val = forward_model(pred_geom_val)
            inverse_val_loss = loss_fn(pred_stiff_val, val_y).item()
            forward_val_loss = loss_fn(forward_model(val_geom_x), val_geom_y).item()

        avg_val_loss = (inverse_val_loss + forward_val_loss) / 2.0

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(inverse_model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Inverse early stopped at epoch {epoch}, best avg val loss: {best_val_loss:.4f}")
            break

    inverse_model.load_state_dict(best_model)

    # ---- Final metrics snapshot
    inverse_model.eval()
    with torch.no_grad():
        pred_geom_train = inverse_model(train_x)
        pred_stiff_train = forward_model(pred_geom_train)
        train_loss = loss_fn(pred_stiff_train, train_y).item()

        inverse_val_loss = loss_fn(forward_model(inverse_model(val_x)), val_y).item()
        forward_val_loss = loss_fn(forward_model(val_geom_x), val_geom_y).item()
        avg_val_loss = (inverse_val_loss + forward_val_loss) / 2.0

    results.append((fcfg, icfg, train_loss, inverse_val_loss, forward_val_loss, avg_val_loss))

# -----------------------------
# Report top configs
# -----------------------------
results.sort(key=lambda x: x[5])

print("\nTop 10 Configurations (by avg validation loss):")
print("-" * 80)
for rank, (fcfg, icfg, train_loss, inv_val, fwd_val, avg) in enumerate(results[:10], start=1):
    print(f"Rank {rank}")
    print("  Forward MLP:")
    print(f"    Hidden Dim   : {fcfg['hidden_dim']}")
    print(f"    Layers       : {fcfg['mlp_layers']}")
    print(f"    Activation   : {fcfg['activation'].__name__}")
    print(f"    Batch Size   : {fcfg['batch_size']}")
    print(f"    Learning Rate: {fcfg['lr']}")
    print("  Inverse GNN:")
    print(f"    GNN Type     : {icfg['gnn_type']}")
    print(f"    Hidden Dim   : {icfg['hidden_dim']}")
    print(f"    Layers       : {icfg['gnn_layers']}")
    print(f"    Batch Size   : {icfg['batch_size']}")
    print(f"    Activation   : {icfg['activation'].__name__}")
    print(f"    Learning Rate: {icfg['lr']}")
    print(f"  Inv Val Loss   : {inv_val:.4f}")
    print(f"  Fwd Val Loss   : {fwd_val:.4f}")
    print(f"  Avg Val Loss   : {avg:.4f}")
    print("-" * 80)
