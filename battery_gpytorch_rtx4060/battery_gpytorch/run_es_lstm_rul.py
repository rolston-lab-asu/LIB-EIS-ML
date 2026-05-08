"""
ES-LSTM Remaining Useful Life Prediction
Hithesh Rai Purushothama | CAS 523 | Arizona State University

Capacity-only (no EIS) temporal RUL prediction using exponential smoothing
and a 2-layer LSTM with two input features:
  1. Smoothed SOH window (30 steps)
  2. Cycle-index: actual elapsed cycles / global_max_cycles

Cycle-index uses a 2× scale factor for new_dataset (A cells), which are
measured every 2 cycles vs every 1 cycle for ca_dataset (CA cells).

Best result: mean RMSE 38.6 cycles across 7 test cells (CA train → A test).

Usage:
    python run_es_lstm_rul.py
    python run_es_lstm_rul.py --epochs 400 --out output/es_lstm
    python run_es_lstm_rul.py --data_root /path/to/data --out output/es_lstm
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str,
                    default=str(Path(__file__).parent / 'data'),
                    help='Path to the data/ directory (default: ./data)')
parser.add_argument('--epochs',     type=int,   default=400)
parser.add_argument('--lr',         type=float, default=1e-3)
parser.add_argument('--batch',      type=int,   default=64)
parser.add_argument('--hidden',     type=int,   default=64)
parser.add_argument('--layers',     type=int,   default=2)
parser.add_argument('--window',     type=int,   default=30)
parser.add_argument('--alpha',      type=float, default=0.3)
parser.add_argument('--out',        type=str,   default='output/es_lstm',
                    help='Directory for plots, metrics, and model checkpoint')
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
print(f'PyTorch: {torch.__version__}')
if device.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')

# ── Paths & cell lists ────────────────────────────────────────────────────────
DATA_ROOT = Path(args.data_root)
CA_DIR    = DATA_ROOT / 'ca_dataset'
NEW_DIR   = DATA_ROOT / 'new_dataset'
MULTI_DIR = DATA_ROOT / 'multitemp_dataset'

TRAIN_CELLS = ['CA1', 'CA2', 'CA3', 'CA4', 'CA5', 'CA7', 'CA8']
MULTI_CELLS = ['N10_CB1', 'N10_CB2', 'N10_CB3', 'N10_CB4']  # N20 excluded (<30 RUL pts)
TEST_CELLS  = ['A1', 'A2', 'A3', 'A4', 'A5', 'A7', 'A8']   # A6 excluded (no RUL file)
# Note: A3 is a DNF cell (never reached 80% capacity threshold); its RUL labels
# count down to end-of-experiment rather than physical EoL.

# Global max actual cycles across all CA training cells (used to normalise
# cycle-index for A test cells, which are measured every 2 actual cycles).
GLOBAL_MAX_CYCLES = 600   # CA7 runs 529 measurements × 1 cyc/meas = 529; 600 is a safe ceiling

# ── Data loading ──────────────────────────────────────────────────────────────
def load_cell(directory, cell):
    cap = np.loadtxt(directory / f'cap_{cell}.txt')
    rul = np.loadtxt(directory / f'rul_{cell}.txt')
    return cap, rul

train_raw = {c: load_cell(CA_DIR,    c) for c in TRAIN_CELLS}
multi_raw = {c: load_cell(MULTI_DIR, c) for c in MULTI_CELLS}
test_raw  = {c: load_cell(NEW_DIR,   c) for c in TEST_CELLS}

print('\nTraining cells (ca_dataset):')
for c in TRAIN_CELLS:
    cap, rul = train_raw[c]
    print(f'  {c}: cap={len(cap)}  rul={len(rul)}  rul[0]={rul[0]:.0f}  '
          f'cap[0]={cap[0]:.0f}  cap[-1]={cap[-1]:.1f}')
print('\nMulti-temp cells (N10, added to training):')
for c in MULTI_CELLS:
    cap, rul = multi_raw[c]
    print(f'  {c}: cap={len(cap)}  rul={len(rul)}  rul[0]={rul[0]:.0f}  '
          f'cap[0]={cap[0]:.0f}  cap[-1]={cap[-1]:.1f}')
print('\nTest cells (new_dataset):')
for c in TEST_CELLS:
    cap, rul = test_raw[c]
    print(f'  {c}: cap={len(cap)}  rul={len(rul)}  rul[0]={rul[0]:.0f}  '
          f'cap[0]={cap[0]:.0f}  cap[-1]={cap[-1]:.1f}')

# ── Degradation curves ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
cmap = plt.cm.tab10
for i, c in enumerate(TRAIN_CELLS):
    cap, _ = train_raw[c]
    axes[0].plot(cap / cap[0], color=cmap(i / 7), label=c, lw=1.5, alpha=0.85)
axes[0].axhline(0.8, color='k', ls='--', lw=0.8, alpha=0.4, label='80% threshold')
axes[0].set_xlabel('Measurement point'); axes[0].set_ylabel('SOH')
axes[0].set_title('Training — ca_dataset (full degradation)')
axes[0].legend(fontsize=8, frameon=False)
for i, c in enumerate(TEST_CELLS):
    cap, _ = test_raw[c]
    axes[1].plot(cap / cap[0], color=cmap(i / 7), label=c, lw=1.5, alpha=0.85)
axes[1].axhline(0.8, color='k', ls='--', lw=0.8, alpha=0.4)
axes[1].set_xlabel('Measurement point'); axes[1].set_ylabel('SOH')
axes[1].set_title('Test — new_dataset (partial degradation)')
axes[1].legend(fontsize=8, frameon=False)
plt.suptitle('Capacity Degradation Curves', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'degradation_curves.png', dpi=150); plt.close()
print('\nSaved: degradation_curves.png')

# ── Exponential smoothing ─────────────────────────────────────────────────────
def exponential_smooth(x, alpha=args.alpha):
    s = np.empty_like(x, dtype=float)
    s[0] = x[0]
    for t in range(1, len(x)):
        s[t] = alpha * x[t] + (1.0 - alpha) * s[t - 1]
    return s

# ── Sequence construction ─────────────────────────────────────────────────────
def make_sequences_full(cap, window=args.window):
    """
    Full-history sequences for CA training cells (1 cycle per measurement).
    RUL normalised to [1→0] over the cell's full observed life.
    Cycle-index: t / (len(cap)-1), consistent with GLOBAL_MAX_CYCLES denominator
    when the training cell's eol ~ GLOBAL_MAX_CYCLES.
    """
    eol      = len(cap) - 1
    soh_full = exponential_smooth(cap / cap[0])
    rul_norm = np.arange(eol, -1, -1, dtype=np.float32) / eol
    X, y = [], []
    for t in range(window, len(soh_full)):
        cycle_norm = float(t / eol)
        soh_win    = soh_full[t - window:t].astype(np.float32)
        X.append(np.stack([soh_win,
                           np.full(window, cycle_norm, dtype=np.float32)], axis=1))
        y.append(float(rul_norm[t]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def make_sequences(cap, rul, cycle_scale=1, window=args.window):
    """
    RUL-window sequences for N10 multitemp (cycle_scale=1) and
    new_dataset test cells (cycle_scale=2, measured every 2 actual cycles).
    Cycle-index uses absolute position in actual cycles / GLOBAL_MAX_CYCLES.
    """
    soh_full    = exponential_smooth(cap / cap[0])
    soh_aligned = soh_full[-len(rul):]
    rul_norm    = rul / rul[0]
    offset      = len(cap) - len(rul)     # start index of RUL window in cap array
    X, y = [], []
    for t in range(window, len(soh_aligned)):
        abs_t      = cycle_scale * (offset + t)
        cycle_norm = float(abs_t / GLOBAL_MAX_CYCLES)
        soh_win    = soh_aligned[t - window:t].astype(np.float32)
        X.append(np.stack([soh_win,
                           np.full(window, cycle_norm, dtype=np.float32)], axis=1))
        y.append(float(rul_norm[t]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ── Build training set ────────────────────────────────────────────────────────
X_parts, y_parts = [], []
print('\nBuilding training sequences:')
for c in TRAIN_CELLS:
    cap, _ = train_raw[c]
    Xc, yc = make_sequences_full(cap)
    X_parts.append(Xc); y_parts.append(yc)
    print(f'  {c}: {len(Xc)} sequences (full history, cycle_scale=1)')
for c in MULTI_CELLS:
    cap, rul = multi_raw[c]
    Xc, yc   = make_sequences(cap, rul, cycle_scale=1)
    X_parts.append(Xc); y_parts.append(yc)
    print(f'  {c}: {len(Xc)} sequences (RUL window, cycle_scale=1)')

X_train = np.concatenate(X_parts)   # (N, W, 2)
y_train = np.concatenate(y_parts)
print(f'\nTotal training sequences: {len(X_train)}  shape: {X_train.shape}')

# Test sequences — cycle_scale=2 (new_dataset measured every 2 actual cycles)
test_seqs = {c: make_sequences(*test_raw[c], cycle_scale=2) for c in TEST_CELLS}
print('Test sequences:')
for c in TEST_CELLS:
    print(f'  {c}: {len(test_seqs[c][0])} sequences (cycle_scale=2)')

# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, hidden=args.hidden, layers=args.layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):          # x: (B, W, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)

model = LSTMForecaster().to(device)
print(f'\nModel: {sum(p.numel() for p in model.parameters()):,} parameters')

# ── Training ──────────────────────────────────────────────────────────────────
X_t = torch.tensor(X_train)        # (N, W, 2) — no extra dim needed
y_t = torch.tensor(y_train)
loader  = DataLoader(TensorDataset(X_t, y_t), batch_size=args.batch, shuffle=True)
opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
sched   = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
loss_fn = nn.MSELoss()

print(f'\nTraining for {args.epochs} epochs...')
train_losses = []
model.train()
for ep in range(1, args.epochs + 1):
    ep_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss += loss.item() * len(xb)
    ep_loss /= len(X_train)
    train_losses.append(ep_loss)
    sched.step()
    if ep % 50 == 0:
        print(f'  Epoch {ep:4d}/{args.epochs}  MSE={ep_loss:.6f}  RMSE={ep_loss**0.5:.4f}')

torch.save(model.state_dict(), OUT / 'model.pt')
print('Saved: model.pt')

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(train_losses, lw=1.5, color='steelblue')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (normalised RUL)')
ax.set_title('Training loss')
plt.tight_layout()
plt.savefig(OUT / 'training_loss.png', dpi=150); plt.close()
print('Saved: training_loss.png')

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
results = {}
print(f'\n{"Cell":<8} {"N":>5} {"RMSE_norm":>10} {"RMSE_cyc":>10} {"MAE_cyc":>9}')
print('-' * 48)
for c in TEST_CELLS:
    cap, rul  = test_raw[c]
    X_c, y_c  = test_seqs[c]
    with torch.no_grad():
        pred_norm = model(torch.tensor(X_c).to(device)).cpu().numpy()
    pred_norm = np.clip(pred_norm, 0.0, 1.0)
    pred_rul  = pred_norm * rul[0]
    true_rul  = y_c * rul[0]
    rmse_n = float(np.sqrt(np.mean((pred_norm - y_c) ** 2)))
    rmse_c = float(np.sqrt(np.mean((pred_rul - true_rul) ** 2)))
    mae_c  = float(np.mean(np.abs(pred_rul - true_rul)))
    results[c] = dict(true=true_rul.tolist(), pred=pred_rul.tolist(),
                      rmse_n=rmse_n, rmse_c=rmse_c, mae_c=mae_c)
    dnf = ' [DNF]' if c == 'A3' else ''
    print(f'{c:<8} {len(y_c):>5} {rmse_n:>10.4f} {rmse_c:>10.1f} {mae_c:>9.1f}{dnf}')

mean_rmse = np.mean([v['rmse_c'] for v in results.values()])
mean_mae  = np.mean([v['mae_c']  for v in results.values()])
print(f'{"Mean":<8} {"":>5} {"":>10} {mean_rmse:>10.1f} {mean_mae:>9.1f}')

with open(OUT / 'metrics.json', 'w') as f:
    json.dump({c: {k: v for k, v in r.items() if k not in ('true', 'pred')}
               for c, r in results.items()}, f, indent=2)
print('Saved: metrics.json')

# ── RUL prediction plots ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, c in zip(axes.flatten(), TEST_CELLS):
    r = results[c]
    ax.plot(r['true'], 'b-',  lw=1.5, label='Actual')
    ax.plot(r['pred'], 'r--', lw=1.5, label='Predicted')
    title = f'{c}  RMSE={r["rmse_c"]:.0f}  MAE={r["mae_c"]:.0f} cyc'
    if c == 'A3':
        title += ' [DNF]'
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Step'); ax.set_ylabel('RUL')
    ax.legend(fontsize=7, frameon=False)
axes.flatten()[-1].set_visible(False)
plt.suptitle('ES-LSTM RUL Prediction — Test Cells (new_dataset)', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'rul_predictions.png', dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(5, 5))
lo = 0
hi = max(max(results[c]['true']) for c in TEST_CELLS) * 1.05
ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5)
for i, c in enumerate(TEST_CELLS):
    ax.scatter(results[c]['true'], results[c]['pred'],
               s=8, alpha=0.6, color=plt.cm.tab10(i / 7), label=c)
ax.set_xlabel('Actual RUL'); ax.set_ylabel('Predicted RUL')
ax.set_title(f'Predicted vs Actual  (mean RMSE={mean_rmse:.0f} cyc)')
ax.legend(fontsize=8, frameon=False, markerscale=2)
plt.tight_layout()
plt.savefig(OUT / 'scatter.png', dpi=150); plt.close()

print('Saved: rul_predictions.png  scatter.png')
print(f'\nDone. All outputs in: {OUT.resolve()}')
