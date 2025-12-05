# quantum_llm_probe.py
import pennylane as qml
import torch
from torch import nn
from math import log2, ceil

# Tiny vocab: noun number (sing/plur) + verbs + eos
vocab = ["cat", "cats", "dog", "dogs", "is", "are", "<eos>"]
token_to_id = {t: i for i, t in enumerate(vocab)}
n_vocab = len(vocab)
n_qubits = ceil(log2(n_vocab))  # 3 qubits → 8 basis states

# Training data: context → target verb with agreement
train_samples = [
    (["cat"], "is"),
    (["cats"], "are"),
    (["dog"], "is"),
    (["dogs"], "are"),
    (["cat", "<eos>"], "is"),
    (["cats", "<eos>"], "are"),
]

class QuantumHead(nn.Module):
    def __init__(self, n_qubits, d_embed):
        super().__init__()
        self.d_embed = d_embed
        self.embed = nn.Embedding(n_vocab, d_embed)
        self.ctx_proj = nn.Linear(d_embed, n_qubits)  # project to rotation angles
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(angles, weights):
            # Angle embedding
            for i in range(n_qubits):
                qml.RX(angles[i], wires=i)
                qml.RZ(angles[i], wires=i)
            # Simple entangler + trainable layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Data re-uploading with trainable rotations
            for i in range(n_qubits):
                qml.RY(weights[i, 0] * angles[i] + weights[i, 1], wires=i)
            return qml.probs(wires=range(n_qubits))

        self.circuit = circuit
        self.weights = nn.Parameter(torch.randn(n_qubits, 2) * 0.1)

    def forward(self, ctx_token_ids):
        # Average embeddings over context (length 1–2 here)
        emb = self.embed(ctx_token_ids).mean(dim=0)
        angles = self.ctx_proj(emb)
        probs_full = self.circuit(angles, self.weights)  # length 2^n_qubits
        probs = probs_full[:n_vocab]  # truncate to vocab size
        return torch.log(probs + 1e-9)  # log-probs for stability

def encode_ctx(tokens):
    return torch.tensor([token_to_id[t] for t in tokens], dtype=torch.long)

def train():
    model = QuantumHead(n_qubits=n_qubits, d_embed=8)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.NLLLoss()

    for step in range(400):
        total = 0.0
        for ctx, tgt in train_samples:
            ctx_ids = encode_ctx(ctx)
            logp = model(ctx_ids)
            target_id = torch.tensor([token_to_id[tgt]])
            loss = loss_fn(logp.unsqueeze(0), target_id)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if step % 50 == 0:
            print(f"step {step}, loss {total/len(train_samples):.4f}")
    return model

def inspect(model):
    print("\nProbabilities:")
    for ctx, tgt in train_samples:
        ctx_ids = encode_ctx(ctx)
        with torch.no_grad():
            logp = model(ctx_ids)
            p = logp.exp()
        top = torch.topk(p, k=3)
        shown = [(vocab[i], float(p[i])) for i in top.indices]
        print(f"context={ctx} target={tgt} -> top3={shown}")

if __name__ == "__main__":
    model = train()
    inspect(model)
