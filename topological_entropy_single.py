import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- nucleus sampling ----------------
def nucleus_indices(probs: np.ndarray, top_p: float) -> np.ndarray:
    """
    Return indices of the smallest prefix whose cumulative prob ≥ top_p.
    """
    order = np.argsort(-probs)
    cumsum = np.cumsum(probs[order])
    return order[: np.searchsorted(cumsum, top_p) + 1]

# ---------------- initialization ----------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model.eval()
V = model.config.vocab_size

text = "This "
enc = tokenizer(text, return_tensors='pt').to(device)

top_p = 0.9
steps = 5
eigenvalues = []

# ---------------- main loop ----------------
for t in tqdm(range(steps), desc="Generating"):
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits[0, -1]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # 2. Use top-p nucleus sampling for S_t
    S_t = nucleus_indices(probs, top_p)

    # 3. Compute S_next[α] for each α in S_t using top-p nucleus
    S_next = {}
    for α in S_t:
        new_ids = torch.cat([enc['input_ids'][0], torch.tensor([α], device=device)], dim=0).unsqueeze(0)
        with torch.no_grad():
            outα = model(new_ids)
        probs_α = torch.softmax(outα.logits[0, -1], dim=-1).cpu().numpy()
        S_next[α] = nucleus_indices(probs_α, top_p)

    # 4. Build sparse adjacency matrix A
    A_lil = lil_matrix((V, V), dtype=int)
    for α, betas in S_next.items():
        for β in betas:
            A_lil[α, β] = 1
    A = A_lil.tocsr()

    # 5. Compute top eigenvalue
    vals, _ = eigs(A, k=1, which='LM')
    eigenvalues.append(np.real(vals[0]))

    # 6. Sample one token using top-p sampling
    enc = model.generate(
        input_ids=enc['input_ids'],
        attention_mask=enc['attention_mask'],
        do_sample=True,
        # top_p=top_p,
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id
    )
    enc = {'input_ids': enc, 'attention_mask': torch.ones_like(enc)}

    # 7. Decode and print
    text_so_far = tokenizer.decode(enc['input_ids'][0], skip_special_tokens=True)
    print(f"\nStep {t+1} generated text:\n{text_so_far}\n{'-'*50}")

# 8. Plot eigenvalues
plt.figure(figsize=(8,4))
plt.plot(range(1, steps+1), eigenvalues, marker='o')
plt.xlabel('Generation Step t')
plt.ylabel('Top Eigenvalue λ_max')
plt.title('Eigenvalue vs. Generation Step (50 steps)')
plt.grid(True)
plt.tight_layout()
plt.show()
