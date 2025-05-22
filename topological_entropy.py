import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigs
from tqdm import tqdm


# -------------------- helpers --------------------
def set_seed(seed=42):
    torch.manual_seed(seed);
    np.random.seed(seed)


def nucleus_indices(probs: np.ndarray, top_p: float) -> np.ndarray:
    order = np.argsort(-probs)
    cumsum = np.cumsum(probs[order])
    return order[: np.searchsorted(cumsum, top_p) + 1]


def trim_ctx(inp: torch.Tensor, max_len: int) -> torch.Tensor:
    """Keep the last `max_len` tokens of a 2‑D [1, L] tensor."""
    return inp[:, -max_len:]


# -------------------- config --------------------
set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()

MAX_CTX = 512  # hard cap
TOP_P = 0.90
STEPS = 10000
BATCH_SIZE = 64

enc = tokenizer("This", return_tensors='pt').to(device)
enc['input_ids'] = trim_ctx(enc['input_ids'], MAX_CTX)
enc['attention_mask'] = torch.ones_like(enc['input_ids'])

V = model.config.vocab_size
eigenvalues, generated_tokens = [], []

# ==================== main loop ====================
for t in tqdm(range(STEPS), desc="Generating"):
    # -------- 1) logits for current context --------
    with torch.no_grad():
        logits = model(**enc).logits[0, -1]
    probs_t = torch.softmax(logits, dim=-1).cpu().numpy()
    S_t = nucleus_indices(probs_t, TOP_P)

    # -------- 2) build adjacency by batching over α --------
    A = lil_matrix((V, V), dtype=float)
    base_ids_full = enc['input_ids'][0]  # [L]
    base_ids = base_ids_full[-(MAX_CTX - 1):]  # leave room for α
    for i in range(0, len(S_t), BATCH_SIZE):
        batch_α = S_t[i:i + BATCH_SIZE]
        prefix = base_ids.repeat(len(batch_α), 1)  # (B, ≤511)
        suffix = torch.tensor(batch_α, device=device).unsqueeze(1)
        ids_b = torch.cat([prefix, suffix], dim=1)  # (B, ≤512)
        mask_b = torch.ones_like(ids_b, device=device)

        with torch.no_grad():
            logits_b = model(input_ids=ids_b,
                             attention_mask=mask_b).logits[:, -1, :]
        probs_b = torch.softmax(logits_b, dim=-1).cpu().numpy()
        for row, α in enumerate(batch_α):
            betas = nucleus_indices(probs_b[row], TOP_P)
            A.rows[α], A.data[α] = list(betas), [1.0] * len(betas)

    # -------- 3) dominant eigenvalue --------
    λ_max, _ = eigs(A.tocsr(), k=1, which='LM')
    eigenvalues.append(float(np.real(λ_max[0])))

    # -------- 4) sample next token --------
    with torch.no_grad():
        generated = model.generate(
            input_ids=trim_ctx(enc['input_ids'], MAX_CTX),
            attention_mask=trim_ctx(enc['attention_mask'], MAX_CTX),
            do_sample=True,
            max_new_tokens=1,
            # top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id
        )
    last_id = generated[0, -1].item()
    generated_tokens.append(tokenizer.decode([last_id], skip_special_tokens=True))

    # -------- 5) update context (trim immediately) --------
    enc = {
        'input_ids': trim_ctx(generated, MAX_CTX),
        'attention_mask': torch.ones_like(trim_ctx(generated, MAX_CTX))
    }

    if t % 100 == 0: torch.cuda.empty_cache()
    print(f"[{t + 1}] λ_max={eigenvalues[-1]:.4f} generated='{generated_tokens[-1]}'")

# -------------------- save & plot --------------------
pd.DataFrame({'token': generated_tokens,
              'eigenvalue': eigenvalues}).to_csv('eigenvalues_tokens.csv', index=False)
print("Saved to eigenvalues_tokens.csv")

plt.figure(figsize=(12, 4))
plt.plot(range(1, STEPS + 1), eigenvalues, marker='o', markersize=2)
plt.xlabel('Step');
plt.ylabel('Top eigenvalue λ_max')
plt.title('Eigenvalue trajectory');
plt.grid(True);
plt.tight_layout()
plt.show()
