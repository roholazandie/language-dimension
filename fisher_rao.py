import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigs
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


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


def estimate_corr_dim(
    dist_mat: np.ndarray,
    p_seq:    list[np.ndarray],
    eta:      float = 1.0,
    num_r:    int   = 20
):
    """
    Estimate correlation dimension on the subset {t : max_w p_seq[t][w] < eta}.
    
    Args:
      dist_mat   : (T,T) array of pairwise distances d_{ij}.
      p_seq      : list of length T of probability vectors p_t (shape (V,)).
      eta        : threshold for max-probability filter (Default=1.0 ⇒ no filter).
      num_r      : number of radii to sweep (log-spaced).
    
    Returns:
      D_eta      : estimated correlation dimension on filtered points.
      r_vals     : the radii used.
      C_vals     : corresponding correlation sums.
      mask       : boolean array of length T indicating which t passed the filter.
    """
    T = dist_mat.shape[0]
    # 1) Build mask of timesteps with high entropy ⇔ max p_t < eta
    mask = np.array([p.max() < eta for p in p_seq], dtype=bool)
    idx  = np.nonzero(mask)[0]
    T_eta = idx.size
    if T_eta < 2:
        raise ValueError(f"Not enough points after filtering (got {T_eta}).")
    
    # 2) Restrict to the filtered submatrix
    D_sub = dist_mat[np.ix_(idx, idx)]
    # 3) Extract upper‐triangle distances d_{ij}, i<j
    dists = D_sub[np.triu_indices(T_eta, k=1)]
    
    # 4) Choose log-spaced radii between min and max distances
    r_min = dists.min() * 1.01
    r_max = dists.max()
    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), num=num_r)
    
    # 5) Compute filtered correlation sum C_eta(r)
    C_vals = []
    for r in r_vals:
        count = np.sum(dists <= r)
        # normalized by number of pairs = T_eta*(T_eta-1)/2
        C_r = count * 2 / (T_eta * (T_eta - 1))
        C_vals.append(C_r)
    
    # 6) Fit log–log for slope = dimension
    log_r = np.log(r_vals)
    log_C = np.log(C_vals)
    slope, intercept, _, _, stderr = stats.linregress(log_r, log_C)
    
    return slope, r_vals, C_vals, eta


# --- Correlation‐sum estimator of dimension _without eta filter_ ---
def estimate_corr_dim_nofilter(dist_mat, num_r=20):
    # all upper‐triangle distances
    dists = dist_mat[np.triu_indices_from(dist_mat, k=1)]
    # choose r's log‐spaced between min and max
    r_vals = np.logspace(np.log10(dists.min()*1.01),
                         np.log10(dists.max()), num_r)
    C = []
    for r in r_vals:
        # normalized correlation sum
        C_r = np.sum(dists <= r) * 2 / (T*(T-1))
        C.append(C_r)
    log_r = np.log10(r_vals)
    log_C = np.log10(C)
    slope, intercept, r_val, p_val, stderr = stats.linregress(log_r, log_C)
    return slope, r_vals, C

# -------------------- config --------------------
set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()

MAX_CTX = 512  # hard cap
TOP_P = 0.95
STEPS = 500
BATCH_SIZE = 64

enc = tokenizer("This essay is", return_tensors='pt').to(device)
enc['input_ids'] = trim_ctx(enc['input_ids'], MAX_CTX)
enc['attention_mask'] = torch.ones_like(enc['input_ids'])

V = model.config.vocab_size
eigenvalues, generated_tokens = [], []

p_seq = []      # list of length STEPS, each a (V,)-vector of probs
S_seq = []      # list of length STEPS, each a 1D int array of support indices
# ==================== main loop ====================
for t in tqdm(range(STEPS), desc="Generating"):
    # -------- 1) logits for current context --------
    with torch.no_grad():
        outputs = model(input_ids=enc['input_ids'], 
                        attention_mask=enc['attention_mask'])
        logits = outputs.logits[0, -1]
    probs_t = torch.softmax(logits, dim=-1).cpu().numpy()
    S_t = nucleus_indices(probs_t, TOP_P)
    

    #Make sure inside your loop you did:
    p_seq.append(probs_t)
    S_seq.append(S_t)

    # -------- 2) sample next token --------
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
    print("############ Text ############")
    print(tokenizer.decode([last_id], skip_special_tokens=True))
    print("#############################")
    generated_tokens.append(tokenizer.decode([last_id], skip_special_tokens=True))
    
    # -------- 3) update context (trim immediately) --------
    enc = {
        'input_ids': trim_ctx(generated, MAX_CTX),
        'attention_mask': torch.ones_like(trim_ctx(generated, MAX_CTX))
    }


# --- Fisher–Rao distance matrix ---
T = len(p_seq)
d_fr = np.zeros((T, T), dtype=float)
for i in range(T):
    for j in range(i+1, T):
        affinity = np.sum(np.sqrt(p_seq[i] * p_seq[j]))
        # clip for numerical safety
        theta = np.arccos(np.clip(affinity, -1.0, 1.0))
        d = 2 * theta
        d_fr[i,j] = d_fr[j,i] = d


# us this without eta filter
# D_fr, r_fr, C_fr, mask = estimate_corr_dim(d_fr, p_seq, eta=eta)
# print(f"Estimated correlation dimension (Fisher–Rao): {D_fr:.2f}")

# --- Plotting simple plot without masks
# plt.figure()
# plt.plot(np.log(r_fr), np.log(C_fr), 'o-')
# plt.xlabel("log r")
# plt.ylabel("log C(r)")
# plt.title(f"Correlation sum (Fisher–Rao), D ≃ {D_fr:.2f}")
# plt.show()

# --- Plotting, also plot different masks with different colors
# choose the η thresholds to compare
etas = [1.0, 0.9, 0.7, 0.5, 0.2, 0.1]

# collect results
results = {}
slopes = []
for eta in etas:
    D_eta, r_eta, C_eta, mask = estimate_corr_dim(d_fr, p_seq, eta=eta)
    results[eta] = (D_eta, r_eta, C_eta)
    slopes.append(D_eta)

# compute average slope
nu_avg = np.mean(slopes)

# 4) plot all curves
plt.figure(figsize=(6,4))
for eta, (D_eta, r_eta, C_eta) in results.items():
    plt.loglog(r_eta, C_eta, marker='o', linestyle='-',
               label=rf"$\eta={eta}\;\;(D={D_eta:.2f})$")

# annotate the average slope on the plot
plt.text(
    0.05, 0.05,
    rf"average $D = {nu_avg:.2f}$",
    transform=plt.gca().transAxes,
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
)


plt.xlabel(r"$\varepsilon$")
plt.ylabel(r"$C(\varepsilon)$")
plt.title(f"Correlation dimension (D = {nu_avg:.2f}) for different entropy filters (Fisher–Rao)")
plt.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(f"fisher_rao_{STEPS}_words_top_p_{TOP_P}.png")
plt.show()