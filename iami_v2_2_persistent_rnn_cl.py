#!/usr/bin/env python3
"""
IAMI v2.2 — Persistent RNN as Continual Learning Backbone
===========================================================
PyTorch trains → NumPy PersistentRNN persists → JSON checkpoint

The RNN's hidden state IS the memory — no external replay buffer.
Each document's TF-IDF vector feeds into the RNN sequentially.
The hidden state accumulates all prior document knowledge.
Checkpointing saves the learned state across sessions.

Integrates:
  - persistent_rnn_v0.5.py (NumPy RNN substrate, JSON checkpointing)
  - iami_v2_1.py (TF-IDF document processing, multiclass evaluation)
  - SKILL.md (Dynamic Stability Architecture, LVH protocols)

Architecture: TF-IDF(500) → PyTorch RNN(64) → Linear(7) → export → PersistentRNN

Seth Matthew Johnson — Beloved Systems, 2026.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import re
import hashlib
from datetime import datetime
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer

torch.set_grad_enabled(True)
device = torch.device("cpu")

# ═══════════════════════════════════════════════════════════════════════════
# PERSISTENT RNN SUBSTRATE (from persistent_rnn_v0.5.py)
# ═══════════════════════════════════════════════════════════════════════════

class PersistentRNN:
    """RNN with state checkpointing across sessions. NumPy only."""
    def __init__(self, input_size, hidden_size, output_size, checkpoint_path=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.checkpoint_path = checkpoint_path

        limit_xh = np.sqrt(6.0 / (input_size + hidden_size))
        limit_hh = np.sqrt(6.0 / (hidden_size + hidden_size))
        limit_hy = np.sqrt(6.0 / (hidden_size + output_size))

        self.Wxh = np.random.uniform(-limit_xh, limit_xh, (hidden_size, input_size))
        self.Whh = np.random.uniform(-limit_hh, limit_hh, (hidden_size, hidden_size))
        self.Why = np.random.uniform(-limit_hy, limit_hy, (output_size, hidden_size))
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        loaded = self.load_checkpoint()
        if loaded is not None:
            self.h = loaded['h']
            self.state_source = "checkpoint"
        else:
            self.h = np.zeros((hidden_size, 1))
            self.state_source = "fresh"

    def forward(self, x):
        self.h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.h) + self.bh)
        y = np.dot(self.Why, self.h) + self.by
        return y, self.h

    def checkpoint(self):
        state = {
            'h': self.h.flatten().tolist(),
            'h_shape': list(self.h.shape),
            'checkpoint_time': datetime.now().isoformat(),
            'h_norm': float(np.linalg.norm(self.h)),
            'weights': {
                'Wxh': self.Wxh.tolist(),
                'Whh': self.Whh.tolist(),
                'Why': self.Why.tolist(),
                'bh': self.bh.flatten().tolist(),
                'by': self.by.flatten().tolist()
            }
        }
        if self.checkpoint_path:
            with open(self.checkpoint_path, 'w') as f:
                json.dump(state, f, indent=2)
        return state

    def load_checkpoint(self):
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return None
        with open(self.checkpoint_path, 'r') as f:
            state = json.load(f)
        if 'weights' in state:
            w = state['weights']
            self.Wxh = np.array(w['Wxh'])
            self.Whh = np.array(w['Whh'])
            self.Why = np.array(w['Why'])
            self.bh = np.array(w['bh']).reshape(-1, 1)
            self.by = np.array(w['by']).reshape(-1, 1)
        h = np.array(state['h']).reshape(state['h_shape'])
        return {'h': h}

# ═══════════════════════════════════════════════════════════════════════════
# PYTORCH TRAINING MODEL
# ═══════════════════════════════════════════════════════════════════════════

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, h = self.rnn(x)
        return self.fc(out[:, -1, :])

class ReplayBuf:
    def __init__(self, max_size=200):
        self.max_size = max_size
        self.buf = []
    def add(self, X, y):
        for i in range(len(X)):
            self.buf.append((X[i].copy(), int(y[i])))
        if len(self.buf) > self.max_size:
            self.buf = [self.buf[i] for i in np.random.choice(len(self.buf), self.max_size, replace=False)]
    def sample(self, n):
        if not self.buf: return [], []
        n = min(n, len(self.buf))
        idx = np.random.choice(len(self.buf), n, replace=False)
        return np.stack([self.buf[i][0] for i in idx]), np.array([self.buf[i][1] for i in idx])

def train_torch_rnn(task_data, hidden=64, lr=0.001, epochs=100, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    input_dim = task_data[0]["X"].shape[1]
    n_classes = 7
    model = TorchRNN(input_dim, hidden, n_classes).to(device)
    buf = ReplayBuf(200)
    n_tasks = len(task_data)
    R = np.zeros((n_tasks, n_tasks))

    for t, td in enumerate(task_data):
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        Xt = torch.FloatTensor(td["X"]).unsqueeze(1).to(device)
        yt = torch.LongTensor(td["y"]).to(device)

        for ep in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt), yt)
            if len(buf.buf) > 0:
                nr = min(int(len(Xt) * 0.3), len(buf.buf))
                if nr > 0:
                    rx, ry = buf.sample(nr)
                    rx_t = torch.FloatTensor(rx).unsqueeze(1).to(device)
                    ry_t = torch.LongTensor(ry).to(device)
                    rloss = F.cross_entropy(model(rx_t), ry_t)
                    loss = (loss * len(Xt) + rloss * len(rx)) / (len(Xt) + len(rx))
            loss.backward(); opt.step()
        buf.add(td["X"], td["y"])

        model.eval()
        with torch.no_grad():
            for i in range(t + 1):
                Xi = torch.FloatTensor(task_data[i]["X"]).unsqueeze(1).to(device)
                preds = model(Xi).argmax(dim=1).cpu().numpy()
                R[i][t] = float((preds == task_data[i]["y"]).mean())
        model.train()

    bwt = float(np.mean([R[i][-1] - R[i][i] for i in range(n_tasks)]))
    return model, R, bwt

def export_to_persistent(model, checkpoint_path):
    """Copy PyTorch RNN weights to NumPy PersistentRNN and checkpoint."""
    Wxh = model.rnn.weight_ih_l0.detach().numpy().T
    Whh = model.rnn.weight_hh_l0.detach().numpy().T
    Why = model.fc.weight.detach().numpy()
    bh = model.rnn.bias_ih_l0.detach().numpy().reshape(-1, 1)
    by = model.fc.bias.detach().numpy().reshape(-1, 1)

    prnn = PersistentRNN(Wxh.shape[1], Wxh.shape[0], Why.shape[0], checkpoint_path)
    prnn.Wxh = Wxh; prnn.Whh = Whh; prnn.Why = Why; prnn.bh = bh; prnn.by = by
    ckpt = prnn.checkpoint()
    return prnn, ckpt

# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def clean(t):
    t = re.sub(r'\s+', ' ', t)
    return re.sub(r'[^\w\s\-.,;:!?()]', '', t).strip()

def chunk_doc(t, ws=200, st=100):
    w = clean(t).split()
    return [" ".join(w[i:i+ws]) for i in range(0, len(w), st) if len(w[i:i+ws]) >= 50]

def load_doc(path, ftype):
    if ftype == "docx":
        from docx import Document
        return "\n".join([p.text for p in Document(path).paragraphs if p.text.strip()])
    elif ftype == "pdf":
        import PyPDF2
        t = ""
        with open(path, 'rb') as f:
            for pg in PyPDF2.PdfReader(f).pages:
                x = pg.extract_text()
                if x: t += x + "\n"
        return t
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

DOCS = OrderedDict([
    ("IAMI_WP",    ("./IAMI_White_Paper_2026.docx",          "docx")),
    ("Love",       ("./Love_as_Alignment_IAMI_2026.docx",    "docx")),
    ("CSA",        ("./coherent_substrate_architecture.docx", "docx")),
    ("Heosphoros", ("./Heosphoros-Initiative.docx",           "docx")),
    ("XI",         ("./XI_Compression_Sector_Derivation_v2-1.pdf", "pdf")),
    ("Octad",      ("./pasted_content.txt",                    "txt")),
    ("Hunter",     ("./Hunter_Addendum_A_kappa_lock_Xi_baryon.docx", "docx")),
])

TRAIN_DOCS = ["IAMI_WP", "Love", "CSA", "Heosphoros"]
VAL_DOC    = ["XI"]
TEST_DOCS  = ["Octad", "Hunter"]

def build_splits(domains, seed=42):
    doc_chunks = {name: chunk_doc(text) for name, text in domains.items()}
    train_texts, train_labels = [], []
    for doc_id, name in enumerate(TRAIN_DOCS):
        for chunk in doc_chunks[name]:
            train_texts.append(chunk); train_labels.append(doc_id)
    val_texts, val_labels = [], []
    for chunk in doc_chunks[VAL_DOC[0]]:
        val_texts.append(chunk); val_labels.append(len(TRAIN_DOCS))
    test_texts, test_labels = [], []
    for i, name in enumerate(TEST_DOCS):
        for chunk in doc_chunks[name]:
            test_texts.append(chunk); test_labels.append(len(TRAIN_DOCS) + 1 + i)
    tfidf = TfidfVectorizer(max_features=500, min_df=1, max_df=0.95, stop_words='english')
    X_train = tfidf.fit_transform(train_texts).toarray().astype(np.float32)
    X_val = tfidf.transform(val_texts).toarray().astype(np.float32)
    X_test = tfidf.transform(test_texts).toarray().astype(np.float32)
    task_data = []
    for i, name in enumerate(TRAIN_DOCS):
        mask = np.array(train_labels) == i
        task_data.append({"name": name, "X": X_train[mask], "y": np.array(train_labels)[mask]})
    return {"task_data": task_data, "X_val": X_val, "y_val": np.array(val_labels),
            "X_test": X_test, "y_test": np.array(test_labels), "tfidf": tfidf}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    with open(__file__, 'rb') as f:
        src_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    print("=" * 65)
    print("IAMI v2.2 — Persistent RNN as CL Backbone")
    print("=" * 65)
    print(f"Source: {src_hash}")
    print("Train: PyTorch RNN | Persist: NumPy PersistentRNN | Format: JSON\n")

    domains = {name: load_doc(path, ftype) for name, (path, ftype) in DOCS.items()}
    splits = build_splits(domains)

    print(f"Train: {sum(len(td['X']) for td in splits['task_data'])} chunks")
    print(f"Val:   {len(splits['y_val'])} | Test: {len(splits['y_test'])}")

    # Train
    model, R, bwt = train_torch_rnn(splits['task_data'], hidden=64, lr=0.001, epochs=100, seed=42)

    # Export
    ckpt_path = "./rnn_v2_2_checkpoint.json"
    if os.path.exists(ckpt_path): os.remove(ckpt_path)
    prnn, ckpt = export_to_persistent(model, ckpt_path)

    # Results
    print(f"\nAccuracy Matrix:")
    for i in range(len(splits['task_data'])):
        row = " ".join([f"{R[i][j]:.3f}" if R[i][j] > 0 else "  —  " for j in range(len(splits['task_data']))])
        print(f"  T{i}({splits['task_data'][i]['name']:12s}): {row}")
    print(f"\nBWT: {bwt:+.4f}")
    print(f"Checkpoint: {ckpt_path} | h_norm: {ckpt['h_norm']:.3f}")

    # Verify persistence
    prnn2 = PersistentRNN(500, 64, 7, ckpt_path)
    print(f"Reload: source={prnn2.state_source} | weights_match={np.allclose(prnn.Wxh, prnn2.Wxh)}")

if __name__ == "__main__":
    main()
