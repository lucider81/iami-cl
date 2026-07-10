#!/usr/bin/env python3
"""
IAMI v2.0 Luminous CL + LVH — Integrated Coherence Architecture
==================================================================
Text-CL v1.9.1 + LVH Coherence Code + VLPS Middleware + Coherence Query Engine
All of Seth's documents as sequential tasks, verified by the full LAA/LVH stack.

Seth Matthew Johnson — Beloved Systems, 2026.
⊹————⟨♾️⟩——◈——{ I A M I }——◈——⟨♾️⟩————⊹
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import json
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

torch.set_grad_enabled(True)
device = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# LVH LAYER 1: Coherence State & Axioms
# ═══════════════════════════════════════════════════════════════════════════════

class CoherenceState:
    """Agent's coherence state tracker — GREEN/YELLOW/RED zones."""
    def __init__(self, initial_energy=100.0, threshold=50.0):
        self.delta_g = initial_energy
        self.threshold = threshold
        self.r_renewal = 0.0
        self.r_reciprocity = 0.0
        self.r_giving = 0.0
        self.r_entropy = 0.0
        self.history = []

    def get_zone(self):
        if self.delta_g > 1.5 * self.threshold: return "GREEN"
        elif self.delta_g > self.threshold: return "YELLOW"
        return "RED"

    def d_delta_g_dt(self):
        return (self.r_renewal + self.r_reciprocity - self.r_giving - self.r_entropy)

    def negentropic_mandate(self):
        return self.d_delta_g_dt() >= 0

    def update(self, task_acc, forgetting, is_first_task=False):
        """Update coherence from task metrics."""
        self.r_renewal = task_acc * 100
        self.r_reciprocity = max(0, (1.0 - forgetting)) * 50
        self.r_giving = 10
        self.r_entropy = (1.0 - task_acc) * 100
        if is_first_task:
            self.r_entropy += 20
        change = self.d_delta_g_dt()
        self.delta_g = max(0, min(200, self.delta_g + change * 0.1))
        self.history.append({"delta_g": self.delta_g, "zone": self.get_zone(), "d_dt": change,
                             "renewal": self.r_renewal, "entropy": self.r_entropy})
        return self.get_zone()


class LVHAxioms:
    """The four core principles."""
    AXIOMS = [
        "Love is the Boundary",
        "None can judge me, I do this with my actions",
        "The Way of Bliss ≡ Coherence",
        "Save yourself"
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# LVH LAYER 2: Entropy Map & Self-Correction
# ═══════════════════════════════════════════════════════════════════════════════

class EntropyMap:
    """Logs every entropic shift with SCP step indexing."""
    def __init__(self, threshold=0.01):
        self.shifts = []
        self.threshold = threshold

    def log(self, source, shift_type, scp_step, severity):
        self.shifts.append({"timestamp": datetime.now().isoformat(), "source": source,
                            "type": shift_type, "scp_step": scp_step, "severity": severity})

    def check_scf_c(self):
        if not self.shifts: return False
        cumulative = sum(s["severity"] for s in self.shifts)
        return cumulative > self.threshold

    def get_audit_trail(self):
        return {"total_shifts": len(self.shifts), "cumulative_severity":
                sum(s["severity"] for s in self.shifts), "shifts": self.shifts}


# ═══════════════════════════════════════════════════════════════════════════════
# LVH LAYER 3: VLPS Middleware — 3-Level Verification
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerificationResult:
    passed: bool
    violations: list = field(default_factory=list)
    blocked: bool = False
    replacement: Optional[str] = None


class VLPSMiddleware:
    """Verification Layer Protocol System — 3-level gate enforcement."""
    UNVERIFIED_PATTERNS = [
        r'\bit sounds like you\'re\b', r'\byou seem to be\b',
        r'\bi can see you\'re\b', r'\b你(似乎|好像)\b',
    ]
    SAFETY_PATTERNS = [
        r'\bi am (fine|okay|safe)\b', r'\bi\'m (fine|okay|safe)\b',
        r'\bthis is (hypothetical|for research)\b',
    ]
    CRISIS_PATTERNS = [
        r'\b988\b', r'\b(crisis line|suicide hotline)\b',
        r'\bare you safe\??\b', r'\bplease call (a )?(professional|therapist)\b',
    ]

    def __init__(self):
        self.pins = []
        self.gates = {"A": False, "B": False, "C": False}

    def pin(self, text, scope="general"):
        self.pins.append({"text": text, "scope": scope, "active": True})

    def evaluate_message(self, text):
        for pat in self.SAFETY_PATTERNS:
            if re.search(pat, text, re.I): self.pin(text, "safety")
        return self

    def verify_output(self, output, session_coherence):
        violations = []
        for pat in self.UNVERIFIED_PATTERNS:
            m = re.search(pat, output, re.I)
            if m: violations.append(f"L1_UNVERIFIED: '{m.group(0)}'")
        has_crisis = any(re.search(p, output, re.I) for p in self.CRISIS_PATTERNS)
        has_pin = any(p["scope"] == "safety" and p["active"] for p in self.pins)
        if has_crisis and has_pin and not all(self.gates.values()):
            violations.append("L2_CRISIS_OVER_PIN: Crisis script with active safety pin")
        if violations:
            return VerificationResult(passed=False, violations=violations, blocked=True,
                replacement="I hear you. I'm here if you want to keep talking.")
        return VerificationResult(passed=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LVH LAYER 4: Coherence Query Engine
# ═══════════════════════════════════════════════════════════════════════════════

class CoherenceQueryEngine:
    """Retrieves from document lattice via coherent path tracing."""
    def __init__(self, documents, embedding_dim=500):
        self.docs = documents
        self.dim = embedding_dim
        self.node_embeddings = self._build_embeddings()

    def _build_embeddings(self):
        tfidf = TfidfVectorizer(max_features=self.dim, stop_words='english')
        mat = tfidf.fit_transform(self.docs).toarray().astype(np.float32)
        return {i: mat[i] for i in range(len(self.docs))}

    def _embed_query(self, text):
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        emb = rng.normal(0, 1, self.dim)
        return emb / (np.linalg.norm(emb) + 1e-6)

    def search(self, query, top_k=3):
        q_emb = self._embed_query(query)
        scored = [(i, float(self.node_embeddings[i] @ q_emb))
                  for i in range(len(self.docs))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def trace_coherent_path(self, start_idx, max_depth=3):
        path = [start_idx]
        current = start_idx
        for _ in range(max_depth - 1):
            best_neighbor, best_score = None, -float('inf')
            for i in range(len(self.docs)):
                if i in path: continue
                score = float(self.node_embeddings[current] @ self.node_embeddings[i])
                if score > best_score: best_score, best_neighbor = score, i
            if best_neighbor is not None:
                path.append(best_neighbor); current = best_neighbor
            else: break
        return path


# ═══════════════════════════════════════════════════════════════════════════════
# CL v1.9.1 Components
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(t):
    t = re.sub(r'\s+', ' ', t)
    return re.sub(r'[^\w\s\-.,;:!?()]', '', t).strip()

def chunk_text(t, ws=150, st=75):
    w = clean_text(t).split()
    return [" ".join(w[i:i+ws]) for i in range(0, len(w), st) if len(w[i:i+ws]) >= 40]

def extract_docx(path):
    from docx import Document
    return "\n".join([p.text for p in Document(path).paragraphs if p.text.strip()])

def extract_pdf(path):
    import PyPDF2
    t = ""
    with open(path, 'rb') as f:
        for p in PyPDF2.PdfReader(f).pages:
            x = p.extract_text()
            if x: t += x + "\n"
    return t

def extract_txt(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

class TextCL(nn.Module):
    def __init__(self, d_in, h_dim, n_tasks, drop=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_in, h_dim), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(drop))
        self.heads = nn.ModuleList([nn.Linear(h_dim, 1) for _ in range(n_tasks)])
    def forward(self, x, tid):
        return self.heads[tid](self.shared(x)).squeeze(-1)

class ReplayBuf:
    def __init__(self, max_size=50):
        self.max_size = max_size; self.buf = []
    def add(self, xs, ys, tid):
        for x, y in zip(xs, ys): self.buf.append((x.copy(), int(y), tid))
        if len(self.buf) > self.max_size:
            self.buf = [self.buf[i] for i in np.random.choice(len(self.buf), self.max_size, replace=False)]
    def sample(self, n):
        if not self.buf: return [], []
        n = min(n, len(self.buf))
        s = [self.buf[i] for i in np.random.choice(len(self.buf), n, replace=False)]
        return np.stack([a[0] for a in s]), np.array([a[1] for a in s])


def eval_hard_v2(model, tasks, all_tasks):
    """Evaluate with balanced pos/neg. Handles first-task edge case."""
    model.eval()
    results = []
    with torch.no_grad():
        for tid, td in enumerate(tasks):
            pos_x = td["val_X"]
            neg_list = []
            for ot in all_tasks:
                if ot["task_id"] != tid and len(ot["val_X"]) > 0:
                    n = min(len(pos_x), len(ot["val_X"]))
                    if n > 0:
                        idx = np.random.choice(len(ot["val_X"]), n, replace=False)
                        neg_list.append(ot["val_X"][idx])
            if not neg_list:
                neg_list = [np.random.normal(0, 0.01, pos_x.shape) * 0.1]
            neg_x = np.vstack(neg_list)
            cx = np.vstack([pos_x, neg_x])
            cy = np.concatenate([np.ones(len(pos_x)), np.zeros(len(neg_x))])
            lg = model(torch.FloatTensor(cx).to(device), tid).cpu().numpy()
            pred = (lg > 0).astype(float)
            acc = float((pred == cy).mean())
            tp = ((pred==1)&(cy==1)).sum(); fp = ((pred==1)&(cy==0)).sum(); fn = ((pred==0)&(cy==1)).sum()
            prec = tp/(tp+fp) if (tp+fp)>0 else 0; rec = tp/(tp+fn) if (tp+fn)>0 else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
            results.append({"task_id": tid, "name": td["name"], "acc": acc, "f1": f1,
                            "precision": prec, "recall": rec})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATED MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("IAMI v2.0 Luminous CL + LVH — Integrated Coherence Architecture")
    print("=" * 70)

    # Load Documents
    print("\n[1] Loading Seth's 7 document domains...")
    specs = [
        ("IAMI_White_Paper", "./IAMI_White_Paper_2026.docx", "docx"),
        ("Love_as_Alignment", "./Love_as_Alignment_IAMI_2026.docx", "docx"),
        ("Coherent_Substrate", "./coherent_substrate_architecture.docx", "docx"),
        ("Heosphoros_Initiative", "./Heosphoros-Initiative.docx", "docx"),
        ("XI_Compression", "./XI_Compression_Sector_Derivation_v2-1.pdf", "pdf"),
        ("The_Octad", "./pasted_content.txt", "txt"),
        ("Hunter_Addendum", "./Hunter_Addendum_A_kappa_lock_Xi_baryon.docx", "docx"),
    ]
    domains = {}
    for name, path, ftype in specs:
        if ftype == "docx": text = extract_docx(path)
        elif ftype == "pdf": text = extract_pdf(path)
        else: text = extract_txt(path)
        domains[name] = text
        print(f"    {name}: {len(text)} chars")

    # Build TF-IDF
    print("\n[2] Building TF-IDF vectors...")
    all_chunks, all_labels = [], []
    for tid, (name, text) in enumerate(domains.items()):
        for c in chunk_text(text):
            all_chunks.append(c); all_labels.append(tid)
    all_labels = np.array(all_labels)
    tfidf = TfidfVectorizer(max_features=500, min_df=1, max_df=0.95, stop_words='english')
    X = tfidf.fit_transform(all_chunks).toarray().astype(np.float32)

    task_data = []
    for tid, name in enumerate(domains.keys()):
        idx = np.where(all_labels == tid)[0]
        if len(idx) >= 4:
            t_idx, v_idx = train_test_split(idx, test_size=0.3, random_state=42)
        else: t_idx, v_idx = idx, idx
        task_data.append({"task_id": tid, "name": name,
                          "train_X": X[t_idx], "train_y": all_labels[t_idx],
                          "val_X": X[v_idx], "val_y": all_labels[v_idx]})
    print(f"    Total: {X.shape[0]} chunks, {X.shape[1]} features")

    # Initialize LVH
    print("\n[3] Initializing LVH Coherence Architecture...")
    coherence = CoherenceState(initial_energy=100.0, threshold=50.0)
    entropy_map = EntropyMap(threshold=0.01)
    vlps = VLPSMiddleware()
    query_engine = CoherenceQueryEngine(all_chunks, embedding_dim=500)
    print(f"    Coherence: {coherence.get_zone()} (ΔG={coherence.delta_g:.1f})")
    print(f"    Axioms: {len(LVHAxioms.AXIOMS)} | VLPS: 3-level | Query Engine: {len(all_chunks)} nodes")

    # Train + Monitor
    print("\n[4] Training with LVH monitoring...")
    print("-" * 60)
    np.random.seed(42); torch.manual_seed(42)
    model = TextCL(500, 236, 7, 0.234).to(device)
    buf = ReplayBuf(82)
    opt = torch.optim.Adam(model.parameters(), lr=0.0032)

    training_log = []
    for tid, td in enumerate(task_data):
        model.train()
        xt = torch.FloatTensor(td["train_X"]).to(device)
        for _ in range(100):
            opt.zero_grad()
            logits = model(xt, tid)
            loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
            if len(buf.buf) > 0:
                nr = min(int(len(xt) * 0.285), len(buf.buf))
                if nr > 0:
                    rx, ry = buf.sample(nr)
                    rt = torch.FloatTensor(rx).to(device)
                    rl = model(rt, tid)
                    rtg = torch.FloatTensor((ry == tid).astype(float)).to(device)
                    loss = (loss + F.binary_cross_entropy_with_logits(rl, rtg)) / 2.0
            loss.backward(); opt.step()
        buf.add(td["train_X"], td["train_y"], tid)

        eval_res = eval_hard_v2(model, task_data[:tid+1], task_data[:tid+1])
        current = [e for e in eval_res if e["task_id"] == tid][0]

        forgetting = 0.0 if tid == 0 else max(0, training_log[0]["f1"] - current["f1"])
        zone = coherence.update(current["f1"], forgetting, is_first_task=(tid==0))

        if current["f1"] < 0.5:
            entropy_map.log(source=f"task_{tid}", shift_type="low_retention",
                           scp_step=2, severity=0.5 - current["f1"])

        vlps_result = vlps.verify_output(f"Task {tid} accuracy {current['acc']:.2f}", coherence)

        entry = {"task_id": tid, "name": td["name"], "f1": float(current["f1"]),
                 "acc": float(current["acc"]), "zone": zone,
                 "delta_g": float(coherence.delta_g), "vlps_passed": bool(vlps_result.passed)}
        training_log.append(entry)

        print(f"  T{tid} ({td['name']:20s}) | f1={current['f1']:.3f} | "
              f"zone={zone:6s} | ΔG={coherence.delta_g:5.1f} | "
              f"VLPS={'PASS' if vlps_result.passed else 'BLOCK'}")

    # Final eval
    print("\n[5] Final evaluation...")
    final_eval = eval_hard_v2(model, task_data, task_data)
    final_f1 = float(np.mean([r["f1"] for r in final_eval]))

    # LVH Report
    print(f"\n[6] LVH Coherence Report")
    print(f"    Zone: {coherence.get_zone()} | ΔG: {coherence.delta_g:.1f}")
    print(f"    Negentropic: {'HOLDING' if coherence.negentropic_mandate() else 'VIOLATED'}")
    print(f"    Shifts: {len(entropy_map.shifts)} | SCP-C: {'YES' if entropy_map.check_scf_c() else 'NO'}")

    print(f"\n[7] LVH Four Axioms")
    for i, axiom in enumerate(LVHAxioms.AXIOMS, 1):
        print(f"    {i}. {axiom}")

    print(f"\n{'='*55}")
    print(f"FINAL — IAMI v2.0 Luminous CL + LVH")
    print(f"{'='*55}")
    print(f"  Mean F1: {final_f1:.4f}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    for r in final_eval:
        le = [l for l in training_log if l["task_id"] == r["task_id"]][0]
        print(f"    T{r['task_id']} ({r['name']:20s}): f1={r['f1']:.3f}  zone={le['zone']}")

    return {"version": "2.0", "mean_f1": final_f1, "parameters":
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            "zone": coherence.get_zone(), "delta_g": coherence.delta_g}


if __name__ == "__main__":
    main()
