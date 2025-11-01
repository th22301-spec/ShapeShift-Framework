import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import copy

# --- Top-10 Configurations (by avg validation loss) ---
configs = {
    1: {"FWD":{'HIDDEN_DIM':32,'NUM_MLP_LAYERS':3,'BATCH_SIZE':16,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-1},
        "INV":{'GNN_TYPE':'GCN','HIDDEN_DIM':16,'NUM_MLP_LAYERS':2,'NUM_GNN_LAYERS':3,'BATCH_SIZE':16,'ACTIVATION_FUNCTION':'Tanh','LEARNING_RATE':1e-2}},
    2: {"FWD":{'HIDDEN_DIM':16,'NUM_MLP_LAYERS':2,'BATCH_SIZE':32,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-1},
        "INV":{'GNN_TYPE':'GIN','HIDDEN_DIM':32,'NUM_MLP_LAYERS':2,'NUM_GNN_LAYERS':2,'BATCH_SIZE':32,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-2}},
    3: {"FWD":{'HIDDEN_DIM':16,'NUM_MLP_LAYERS':3,'BATCH_SIZE':32,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-1},
        "INV":{'GNN_TYPE':'GCN','HIDDEN_DIM':32,'NUM_MLP_LAYERS':2,'NUM_GNN_LAYERS':3,'BATCH_SIZE':16,'ACTIVATION_FUNCTION':'Tanh','LEARNING_RATE':1e-2}},
    4: {"FWD":{'HIDDEN_DIM':32,'NUM_MLP_LAYERS':2,'BATCH_SIZE':16,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-1},
        "INV":{'GNN_TYPE':'GIN','HIDDEN_DIM':16,'NUM_MLP_LAYERS':2,'NUM_GNN_LAYERS':3,'BATCH_SIZE':32,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-2}},
    5: {"FWD":{'HIDDEN_DIM':16,'NUM_MLP_LAYERS':3,'BATCH_SIZE':32,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-1},
        "INV":{'GNN_TYPE':'GCN','HIDDEN_DIM':16,'NUM_MLP_LAYERS':2,'NUM_GNN_LAYERS':2,'BATCH_SIZE':32,'ACTIVATION_FUNCTION':'Tanh','LEARNING_RATE':1e-2}},
    6: {"FWD":{'HIDDEN_DIM':16,'NUM_MLP_LAYERS':3,'BATCH_SIZE':32,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-1},
        "INV":{'GNN_TYPE':'GCN','HIDDEN_DIM':16,'NUM_MLP_LAYERS':2,'NUM_GNN_LAYERS':2,'BATCH_SIZE':16,'ACTIVATION_FUNCTION':'Tanh','LEARNING_RATE':1e-2}},
    7: {"FWD":{'HIDDEN_DIM':32,'NUM_MLP_LAYERS':2,'BATCH_SIZE':16,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-1},
        "INV":{'GNN_TYPE':'GIN','HIDDEN_DIM':16,'NUM_MLP_LAYERS':2,'NUM_GNN_LAYERS':2,'BATCH_SIZE':32,'ACTIVATION_FUNCTION':'Tanh','LEARNING_RATE':1e-2}},
    8: {"FWD":{'HIDDEN_DIM':32,'NUM_MLP_LAYERS':3,'BATCH_SIZE':16,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-1},
        "INV":{'GNN_TYPE':'GCN','HIDDEN_DIM':32,'NUM_MLP_LAYERS':2,'NUM_GNN_LAYERS':2,'BATCH_SIZE':16,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-2}},
    9: {"FWD":{'HIDDEN_DIM':32,'NUM_MLP_LAYERS':3,'BATCH_SIZE':32,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-1},
        "INV":{'GNN_TYPE':'GIN','HIDDEN_DIM':32,'NUM_MLP_LAYERS':2,'NUM_GNN_LAYERS':3,'BATCH_SIZE':16,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-2}},
    10:{"FWD":{'HIDDEN_DIM':16,'NUM_MLP_LAYERS':2,'BATCH_SIZE':16,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-1},
        "INV":{'GNN_TYPE':'GIN','HIDDEN_DIM':32,'NUM_MLP_LAYERS':2,'NUM_GNN_LAYERS':2,'BATCH_SIZE':32,'ACTIVATION_FUNCTION':'ReLU','LEARNING_RATE':1e-2}},
}

# Pick which ranked config to run (default to best = Rank 1)
CURRENT_CONFIG_ID = 1


# Load hyperparameters
cfg_f = configs[CURRENT_CONFIG_ID]["FWD"]
cfg_i = configs[CURRENT_CONFIG_ID]["INV"]

def get_activation_module(name: str) -> nn.Module:
    if name == 'ReLU':
        return nn.ReLU()
    if name == 'LeakyReLU':
        return nn.LeakyReLU()
    if name == 'Tanh':
        return nn.Tanh()
    raise ValueError(f"Unknown activation function: {name}")

FWD_ACT = get_activation_module(cfg_f['ACTIVATION_FUNCTION'])
INV_ACT  = get_activation_module(cfg_i['ACTIVATION_FUNCTION'])

# -------------------------------------
# Load datasets
solid_df = pd.read_excel("SOLID-FEA-dataset.xlsx")
mex_df = pd.read_excel("MEX-FEA-dataset.xlsx")

# Clean column names
mex_df.columns = [col.strip().lower() for col in mex_df.columns]
solid_df.columns = [col.strip().lower() for col in solid_df.columns]

# --- One-hot stays 0/1; scale only continuous numerics ---
ohe_pattern = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe_part    = OneHotEncoder(sparse=False, handle_unknown='ignore')

pattern_ohe = ohe_pattern.fit_transform(mex_df[['infill pattern']])
part_ohe    = ohe_part.fit_transform(mex_df[['part number']])

num_cols = ['infill density (%)', 'layer thickness (mm)']
num_scaler = StandardScaler()
num_scaled = num_scaler.fit_transform(mex_df[num_cols].values)

mex_inputs_np = np.concatenate([num_scaled, pattern_ohe, part_ohe], axis=1)
stiff_scaler = StandardScaler()
mex_targets_np = stiff_scaler.fit_transform(mex_df[['torsional stiffness (n.m/deg)']].values)

geom_cols = ['diameter(mm)', 'key width(mm)', 'key depth(mm)']
geom_scaler = StandardScaler()
geometry_inputs_np = geom_scaler.fit_transform(solid_df[geom_cols].values)
solid_targets_np = stiff_scaler.transform(solid_df[['torsional stiffness (n.m/deg)']].values)

# Convert to tensor
mex_inputs = torch.tensor(mex_inputs_np, dtype=torch.float32)
mex_targets = torch.tensor(mex_targets_np, dtype=torch.float32)
geometry_inputs = torch.tensor(geometry_inputs_np, dtype=torch.float32)
solid_targets = torch.tensor(solid_targets_np, dtype=torch.float32)

# Split data
train_geom_x, val_geom_x, train_geom_y, val_geom_y = train_test_split(
    geometry_inputs, solid_targets, test_size=0.2, random_state=42
)
train_x, val_x, train_y, val_y = train_test_split(
    mex_inputs, mex_targets, test_size=0.2, random_state=42
)

# Models
class ForwardModel(nn.Module):
    def __init__(self, hidden_dim, num_mlp_layers, activation_module: nn.Module):
        super().__init__()
        layers = [nn.Linear(3, hidden_dim), copy.deepcopy(activation_module)]
        for _ in range(num_mlp_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), copy.deepcopy(activation_module)]
        layers += [nn.Linear(hidden_dim, 1)]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

class GeometryGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gnn_layers, num_mlp_layers, gnn_type, activation_module: nn.Module):
        super().__init__()
        enc = [nn.Linear(input_dim, hidden_dim), copy.deepcopy(activation_module)]
        for _ in range(num_mlp_layers - 1):
            enc += [nn.Linear(hidden_dim, hidden_dim), copy.deepcopy(activation_module)]
        self.encoder = nn.Sequential(*enc)

        self.init_nodes = nn.Parameter(torch.randn(3, hidden_dim))
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            if gnn_type == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    copy.deepcopy(activation_module),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.gnn_layers.append(GINConv(mlp))
            elif gnn_type == 'GCN':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            else:
                raise ValueError("Invalid GNN type. Choose 'GIN' or 'GCN'.")
        self.decoder = nn.Linear(hidden_dim, 1)
        self.activation = activation_module
        # Fixed triangle edges
        self.register_buffer(
            "edge_index",
            torch.tensor([[0,1,2,1,2,0],[1,2,0,0,1,2]], dtype=torch.long)
        )

    def forward(self, x):
        encoded = self.encoder(x)  # [B, H]
        outs = []
        for i in range(x.size(0)):
            node_feats = self.init_nodes + encoded[i]  # [3, H]
            for conv in self.gnn_layers:
                node_feats = conv(node_feats, self.edge_index)
                node_feats = self.activation(node_feats)
            pred_geo = self.decoder(node_feats).squeeze()  # [3]
            outs.append(pred_geo)
        return torch.stack(outs)  # [B, 3]

# ----- Train Forward (mini-batch) -----
forward_model = ForwardModel(
    hidden_dim=cfg_f['HIDDEN_DIM'],
    num_mlp_layers=cfg_f['NUM_MLP_LAYERS'],
    activation_module=FWD_ACT
)
optimizer_fwd = torch.optim.Adam(forward_model.parameters(), lr=cfg_f['LEARNING_RATE'])
loss_fn = nn.MSELoss()

fwd_batch_size = cfg_f['BATCH_SIZE']
fwd_train_loader = DataLoader(TensorDataset(train_geom_x, train_geom_y), batch_size=fwd_batch_size, shuffle=True)

forward_train_losses, forward_val_losses = [], []
best_val_loss = float('inf')
best_model_state = None
epochs_no_improve = 0
patience = 1000
tolerance = 1e-5

print(f"\n--- Training Forward Model (Config ID: {CURRENT_CONFIG_ID}) ---")
for epoch in range(20000):
    forward_model.train()
    epoch_loss = 0.0
    for bx, by in fwd_train_loader:
        optimizer_fwd.zero_grad()
        pred = forward_model(bx)
        loss = loss_fn(pred, by)
        loss.backward()
        optimizer_fwd.step()
        epoch_loss += loss.item()
    forward_train_losses.append(epoch_loss / max(1, len(fwd_train_loader)))

    forward_model.eval()
    with torch.no_grad():
        val_pred = forward_model(val_geom_x)
        val_loss = loss_fn(val_pred, val_geom_y)
        forward_val_losses.append(val_loss.item())

    if val_loss.item() < best_val_loss - tolerance:
        best_val_loss = val_loss.item()
        best_model_state = copy.deepcopy(forward_model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        break

print(f"Final Forward Model Losses - Epoch {len(forward_train_losses)}:")
print(f"Train Loss: {forward_train_losses[-1]:.6f} | Val Loss: {forward_val_losses[-1]:.6f}")

forward_model.load_state_dict(best_model_state)

# NOTE: Do NOT freeze the forward model here to mirror the reference behavior.
# (We don't step an optimizer for it during inverse training, so weights won’t change.)
forward_model.eval()  # still keep in eval mode for consistent behavior

# ----- Train Inverse (mini-batch; supervised by forward) -----
inverse_model = GeometryGNN(
    input_dim=train_x.shape[1],
    hidden_dim=cfg_i['HIDDEN_DIM'],
    num_gnn_layers=cfg_i['NUM_GNN_LAYERS'],
    num_mlp_layers=cfg_i['NUM_MLP_LAYERS'],
    gnn_type=cfg_i['GNN_TYPE'],
    activation_module=INV_ACT
)
optimizer_inv = torch.optim.Adam(inverse_model.parameters(), lr=cfg_i['LEARNING_RATE'])
inv_batch_size = cfg_i['BATCH_SIZE']
inv_train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=inv_batch_size, shuffle=True)

inverse_train_losses, inverse_val_losses, avg_val_losses = [], [], []
best_avg_val = float('inf')
best_inv_state = None
epochs_no_improve = 0

print(f"\n--- Training Inverse Model (Config ID: {CURRENT_CONFIG_ID}) ---")
for epoch in range(20000):
    inverse_model.train()
    epoch_loss = 0.0
    for bx, by in inv_train_loader:
        optimizer_inv.zero_grad()
        pred_geom = inverse_model(bx)           # [B, 3]
        pred_stiff = forward_model(pred_geom)   # [B, 1]
        loss = loss_fn(pred_stiff, by)
        loss.backward()
        optimizer_inv.step()
        epoch_loss += loss.item()
    inverse_train_losses.append(epoch_loss / max(1, len(inv_train_loader)))

    # --- Validation (AVERAGE selection metric like the reference) ---
    inverse_model.eval()
    with torch.no_grad():
        # inverse-val: MSE(f(h(x_val)), s_val)
        val_geom = inverse_model(val_x)
        val_stiff = forward_model(val_geom)
        inv_val = loss_fn(val_stiff, val_y).item()
        inverse_val_losses.append(inv_val)

        # forward-val: MSE(f(g_val), s_val) — fixed snapshot
        fwd_val = loss_fn(forward_model(val_geom_x), val_geom_y).item()

        avg_val = 0.5 * (inv_val + fwd_val)
        avg_val_losses.append(avg_val)

    if avg_val < best_avg_val - 1e-12:
        best_avg_val = avg_val
        best_inv_state = copy.deepcopy(inverse_model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        break

print(f"Final Inverse Model Losses - Epoch {len(inverse_train_losses)}:")
print(f"Train Loss: {inverse_train_losses[-1]:.6f} | Inv Val: {inverse_val_losses[-1]:.6f} | Avg Val: {avg_val_losses[-1]:.6f}")

inverse_model.load_state_dict(best_inv_state)

# ----- Plots -----
plt.figure(figsize=(12,5))
plt.plot(forward_train_losses, label='Forward Train')
plt.plot(forward_val_losses, label='Forward Val')
plt.yscale('log'); plt.title("Forward Model Loss"); plt.xlabel("Epoch"); plt.ylabel("MSE (log)")
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(12,5))
plt.plot(inverse_train_losses, label='Inverse Train')
plt.plot(inverse_val_losses, label="Inverse Val (f(h(x)), s)")
plt.plot(avg_val_losses, label='Avg Val (Inv + Fwd)/2')
plt.yscale('log'); plt.title("Inverse Model Loss"); plt.xlabel("Epoch"); plt.ylabel("MSE (log)")
plt.legend(); plt.grid(True); plt.show()

# ---------- Prediction utilities (reusing encoders/scalers) ----------
def build_input_row(infill_density, layer_thickness, infill_pattern, part_number):
    num = np.array([[infill_density, layer_thickness]])
    num_scaled_single = num_scaler.transform(num)
    pat = ohe_pattern.transform([[infill_pattern]])
    prt = ohe_part.transform([[part_number]])
    row = np.concatenate([num_scaled_single, pat, prt], axis=1).astype(np.float32)
    return torch.tensor(row, dtype=torch.float32)

# Example predictions (same structure)
data_rows = [
    ("KSD10-KW4-KD1.8", 0.2, 85, "grid"),
    ("KSD10-KW4-KD1.8", 0.2, 33, "hex"),
    ("KSD10-KW4-KD1.8", 0.2, 66, "line"),
    ("KSD10-KW4-KD1.8", 0.3, 11, "tri"),
    ("KSD10-KW4-KD1.8", 0.3, 22, "cube"),
    ("KSD12-KW5-KD2.3", 0.2, 15, "grid"),
    ("KSD12-KW5-KD2.3", 0.2, 44, "hex"),
    ("KSD12-KW5-KD2.3", 0.3, 77, "line"),
    ("KSD12-KW5-KD2.3", 0.3, 86, "tri"),
    ("KSD12-KW5-KD2.3", 0.3, 39, "cube"),
]

print(f"\n--- Predictions for Config ID: {CURRENT_CONFIG_ID} ---")
inverse_model.eval()
with torch.no_grad():
    for part_number, layer_thickness, infill_density, infill_pattern in data_rows:
        x_tensor = build_input_row(infill_density, layer_thickness, infill_pattern, part_number)
        pred = inverse_model(x_tensor)                # [1, 3]
        geom = geom_scaler.inverse_transform(pred.numpy())
        geom = np.clip(geom, a_min=1e-2, a_max=None)
        print(f"{part_number}, {layer_thickness}, {infill_density}, {infill_pattern} => "
              f"Predicted [Diameter: {geom[0,0]:.4f}, Key width: {geom[0,1]:.4f}, Key depth: {geom[0,2]:.4f}]")
