# app.py
import streamlit as st
import numpy as np
import pickle, os, torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="FedSim — Shubh", layout="wide")

# ---------- Config ----------
GLOBAL_CHECKPOINT = "global_checkpoint.pth"
GLOBAL_META = "global_meta.pkl"
DATA_FILE = "data_for_app.npz"
CLIENTS_FILE = "clients.pkl"
RNG = np.random.RandomState(42)
DEVICE = torch.device("cpu")  # keep CPU for Streamlit / portability

# ---------- Utilities & Model ----------
class SmallMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        x = self.fc1(x); x = self.relu(x); x = self.fc2(x); return x

def state_cpu(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def safe_load_state_into(model, state_dict, report_fn=print):
    """
    Load only matching keys/shape from state_dict into model.
    Returns dict with lists: loaded_keys, skipped_keys.
    """
    model_sd = model.state_dict()
    loaded = []
    skipped = []
    for k, v in state_dict.items():
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                model_sd[k] = v.clone().to(model_sd[k].device)
                loaded.append(k)
            else:
                skipped.append((k, v.shape, model_sd[k].shape))
        else:
            skipped.append((k, "missing_in_model", None))
    model.load_state_dict(model_sd)
    if loaded:
        report_fn(f"[safe_load] loaded keys: {loaded}")
    if skipped:
        report_fn(f"[safe_load] skipped keys (shape mismatch or missing): {skipped}")
    return {"loaded": loaded, "skipped": skipped}

def train_local(model, X, y, epochs=3, batch_size=32, lr=1e-3, verbose=False):
    """
    Train the provided PyTorch model on numpy X,y arrays and return state dict on CPU.
    """
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.long).to(DEVICE)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for e in range(epochs):
        losses = []
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward(); opt.step()
            losses.append(loss.item())
        if verbose:
            st.write(f"  epoch {e+1}/{epochs} loss={np.mean(losses):.4f}")
    return state_cpu(model)

def eval_state(state, X, y):
    """
    Evaluate a state dict (cpu tensors) on numpy X,y. Returns (acc, preds).
    """
    out_dim = state["fc2.bias"].shape[0]
    m = SmallMLP(in_dim=X.shape[1], hidden=64, out_dim=out_dim)
    safe_load_state_into(m, state, report_fn=lambda s: st.write(s))
    m.eval()
    with torch.no_grad():
        out = m(torch.tensor(X, dtype=torch.float32))
        preds = torch.argmax(out, dim=1).numpy()
    return accuracy_score(y, preds), preds

def fed_avg(state_dicts, weights=None):
    """
    Weighted average of a list of state dicts. Assumes identical keys and shapes.
    """
    import torch
    n = len(state_dicts)
    if weights is None:
        weights = [1.0/n]*n
    avg = {}
    for k in state_dicts[0].keys():
        # stack and multiply by weights (convert to float for safe sum)
        stacked = torch.stack([sd[k].float()*w for sd,w in zip(state_dicts, weights)], dim=0)
        avg[k] = torch.sum(stacked, dim=0)
    return avg

def l2_norms(state):
    w1 = state['fc1.weight'].float()
    w2 = state['fc2.weight'].float()
    return float(torch.norm(w1).item()), float(torch.norm(w2).item())

# ---------- Persistence helpers ----------
def save_global_checkpoint(state_dict, meta):
    """
    meta: dict e.g. {'in_dim': 19, 'feature_cols': [...]} - small jsonable dict
    """
    torch.save(state_dict, GLOBAL_CHECKPOINT)
    with open(GLOBAL_META, "wb") as f:
        pickle.dump(meta, f)

def load_global_checkpoint():
    if not os.path.exists(GLOBAL_CHECKPOINT) or not os.path.exists(GLOBAL_META):
        return None, None
    try:
        sd = torch.load(GLOBAL_CHECKPOINT, map_location="cpu")
    except Exception as e:
        st.warning(f"Failed to load global checkpoint: {e}")
        return None, None
    with open(GLOBAL_META, "rb") as f:
        meta = pickle.load(f)
    return sd, meta

# ---------- Load data ----------
if not os.path.exists(DATA_FILE) or not os.path.exists(CLIENTS_FILE):
    st.error("Missing data files. Place 'data_for_app.npz' and 'clients.pkl' next to app.py.")
    st.stop()

data = np.load(DATA_FILE)
X_all, y_all = data['X'], data['y']
with open(CLIENTS_FILE,'rb') as f:
    clients = pickle.load(f)  # dict client_id -> (X,y)

# ---------- Session state for models/params ----------
if 'clients' not in st.session_state:
    st.session_state.clients = clients  # dict client_id -> (X,y)
if 'local_params' not in st.session_state:
    st.session_state.local_params = {}   # client_id -> state_dict

# Attempt to load persisted global checkpoint (and validate shapes)
loaded_sd, loaded_meta = load_global_checkpoint()
if 'global_params' not in st.session_state:
    if loaded_sd is not None and loaded_meta is not None:
        # check feature dim agreement
        expected_in = loaded_meta.get('in_dim', None)
        if expected_in is None or expected_in == X_all.shape[1]:
            st.session_state.global_params = loaded_sd
            st.session_state.global_meta = loaded_meta
            st.success("Loaded persisted global checkpoint.")
        else:
            st.warning(f"Persisted global checkpoint expects in_dim={expected_in},"
                       f" current dataset in_dim={X_all.shape[1]}. Ignoring checkpoint.")
            # reinit new global model and do not persist
            model = SmallMLP(in_dim=X_all.shape[1], hidden=64, out_dim=len(np.unique(y_all)))
            st.session_state.global_params = state_cpu(model)
            st.session_state.global_meta = {'in_dim': X_all.shape[1]}
    else:
        # no checkpoint — initialize fresh
        model = SmallMLP(in_dim=X_all.shape[1], hidden=64, out_dim=len(np.unique(y_all)))
        st.session_state.global_params = state_cpu(model)
        st.session_state.global_meta = {'in_dim': X_all.shape[1]}

# ---------- Layout ----------
st.title("Federated Learning Simulator — Shubh")
left, right = st.columns([1,2])

with left:
    st.header("Clients")
    sid = st.selectbox("Select client", list(st.session_state.clients.keys()))
    st.write("Samples:", st.session_state.clients[sid][0].shape[0],
             "Label counts:", np.bincount(st.session_state.clients[sid][1]))
    if st.button("Train selected client locally (1 epoch)"):
        Xc, yc = st.session_state.clients[sid]
        # build model using client's feature size
        m = SmallMLP(in_dim=Xc.shape[1], hidden=64, out_dim=len(np.unique(y_all)))
        # start from global params if compatible; else use partial load
        report_lines = []
        def rep(s): report_lines.append(s)
        try:
            safe_load_state_into(m, st.session_state.global_params, report_fn=rep)
        except Exception as e:
            rep(f"safe_load error: {e}")
        for line in report_lines: st.write(line)
        st.write("Training ...")
        new_state = train_local(m, Xc, yc, epochs=1, batch_size=32, lr=1e-3, verbose=True)
        st.session_state.local_params[sid] = new_state
        st.success(f"Saved local params for {sid}")

    if st.button("Show selected client params summary"):
        if sid in st.session_state.local_params:
            s = st.session_state.local_params[sid]
            n1,n2 = l2_norms(s)
            st.write(f"L2 norms — fc1: {n1:.4f}, fc2: {n2:.4f}")
            st.write("first 6 fc1.weight values:", s['fc1.weight'].flatten()[:6].numpy())
        else:
            st.warning("No local params for this client yet.")

    if st.button("Add new client (duplicate with noise)"):
        cid = f"client_{len(st.session_state.clients)}"
        ref = list(st.session_state.clients.values())[0]
        Xref, yref = ref
        Xnew = Xref + RNG.normal(0, 0.01, Xref.shape)
        st.session_state.clients[cid] = (Xnew, yref.copy())
        st.success(f"Added {cid}")

with right:
    st.header("Federated Controls")
    n_local = st.number_input("Local epochs per client during sim (each selected)", min_value=1, max_value=10, value=1)
    fraction = st.slider("Client fraction per round", 0.1, 1.0, 1.0, step=0.1)
    if st.button("Simulate Federated Round"):
        client_ids = list(st.session_state.clients.keys())
        k = max(1, int(len(client_ids)*fraction))
        chosen = RNG.choice(client_ids, k, replace=False).tolist()
        st.write("Selected clients:", chosen)
        state_list = []
        weights = []
        for cid in chosen:
            Xc, yc = st.session_state.clients[cid]
            m = SmallMLP(in_dim=Xc.shape[1], hidden=64, out_dim=len(np.unique(y_all)))
            # choose starting state: prefer client's local params, else try global
            start_state = st.session_state.local_params.get(cid, st.session_state.global_params)
            # safe load (in case dims differ)
            rep_lines = []
            safe_load_state_into(m, start_state, report_fn=lambda s: rep_lines.append(s))
            for r in rep_lines: st.write(r)
            st.write(f" Training {cid} locally ({n_local} epochs) ...")
            s = train_local(m, Xc, yc, epochs=n_local, batch_size=32, lr=1e-3, verbose=False)
            st.session_state.local_params[cid] = s
            state_list.append(s)
            weights.append(len(Xc))
        # normalize weights and aggregate
        weights = [w/sum(weights) for w in weights]
        agg = fed_avg(state_list, weights=weights)
        st.session_state.global_params = agg
        # persist global checkpoint + meta
        meta = {'in_dim': X_all.shape[1], 'n_clients': len(st.session_state.clients)}
        save_global_checkpoint(agg, meta)
        st.success("Federated aggregation complete — global model updated and saved.")
        n1,n2 = l2_norms(agg)
        st.write("Global param norms (fc1, fc2):", (round(n1,4), round(n2,4)))
        acc, _ = eval_state(agg, X_all, y_all)
        st.write("Global model accuracy on full dataset:", round(acc,4))

    st.markdown("---")
    st.header("Global model")
    if st.button("Show global params summary"):
        g = st.session_state.global_params
        n1,n2 = l2_norms(g)
        st.write(f"Global L2 norms — fc1: {n1:.4f}, fc2: {n2:.4f}")
        st.write("First 6 global fc1.weight values:", g['fc1.weight'].flatten()[:6].numpy())

    st.markdown("---")
    st.header("Predict")
    mode = st.radio("Prediction input", ["Random test sample", "Pick client sample"])
    if mode == "Random test sample":
        idx = RNG.randint(0, X_all.shape[0])
        x_in = X_all[idx:idx+1]
        y_true = int(y_all[idx])
        st.write("True label:", y_true)
    else:
        cid = st.selectbox("Client for sample", list(st.session_state.clients.keys()))
        Xc, yc = st.session_state.clients[cid]
        i = RNG.randint(0, len(Xc))
        x_in = Xc[i:i+1]; y_true = int(yc[i])
        st.write("True label:", y_true)

    if st.button("Predict with global model"):
        try:
            acc_local, preds = eval_state(st.session_state.global_params, x_in, np.array([y_true]))
            st.write("Global model prediction:", int(preds[0]), " — correct?" , bool(preds[0]==y_true))
        except Exception as e:
            st.error(f"Prediction failed: {e}. Likely feature-dimension mismatch. Check global in_dim vs sample in_dim.")

st.sidebar.header("Quick controls")
if st.sidebar.button("Reset local params"):
    st.session_state.local_params = {}; st.sidebar.success("Cleared local params")
if st.sidebar.button("Reset global model"):
    model = SmallMLP(in_dim=X_all.shape[1], hidden=64, out_dim=len(np.unique(y_all)))
    st.session_state.global_params = state_cpu(model)
    # overwrite persisted checkpoint too
    save_global_checkpoint(st.session_state.global_params, {'in_dim': X_all.shape[1], 'n_clients': len(st.session_state.clients)})
    st.sidebar.success("Reset global model and saved checkpoint")
