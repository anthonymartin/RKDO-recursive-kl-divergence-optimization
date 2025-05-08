import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# ----- Mean-Teacher with jitter & high LR -----
n = 8
d = 3
steps = 35
temp = 0.5
alpha = 0.3          # EMA coefficient
lr = 2.5             # large learning rate for visible motion
jitter = 0.12        # per-step augment jitter
rng = np.random.default_rng(7)

def normalize(v): 
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def compute_q(emb):
    sims = emb @ emb.T / temp
    np.fill_diagonal(sims, -np.inf)
    exp = np.exp(sims - sims.max(axis=1, keepdims=True))
    q = exp / exp.sum(axis=1, keepdims=True)
    np.fill_diagonal(q, 0.0)
    return q

# init embeddings
emb = normalize(rng.standard_normal((n, d)))
logits_prev = emb @ emb.T / temp
np.fill_diagonal(logits_prev, -np.inf)

teacher_frames = []
student_frames = []

for t in range(steps):
    # ----- forward with jitter for student -----
    aug = normalize(emb + jitter * rng.standard_normal(emb.shape))
    q = compute_q(aug)
    student_frames.append(q)
    
    # ----- update teacher (EMA of logits) -----
    logits_curr = emb @ emb.T / temp
    np.fill_diagonal(logits_curr, -np.inf)
    teacher_logits = (1 - alpha) * logits_prev + alpha * logits_curr
    targets = np.argmax(teacher_logits, axis=1)
    teacher_dist = np.zeros_like(q)
    teacher_dist[np.arange(n), targets] = 1.0
    teacher_frames.append(teacher_dist)
    
    # ----- gradient: pull each node toward its teacher target -----
    grad = np.zeros_like(emb)
    for i, j in enumerate(targets):
        if i == j:
            continue
        # move i and j closer; simple symmetric pull
        grad[i] -= emb[j]
        grad[j] -= emb[i]
    emb = normalize(emb - lr * grad)
    logits_prev = logits_curr.copy()

# ----- Animation -----
pos = nx.circular_layout(range(n))
fig, axes = plt.subplots(2, 1, figsize=(4, 6))
plt.tight_layout()

def draw(ax, W, title):
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if i != j and W[i, j] > 1e-3:
                G.add_edge(i, j, weight=W[i, j])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, node_color="#ffffff", edgecolors="#000000")
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)],
            width=0.5 + 6 * w, edge_color=[plt.cm.turbo(w)],
            arrowstyle="->", arrowsize=8, alpha=0.9
        )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

def update(f):
    draw(axes[0], teacher_frames[f], f"Mean-Teacher: EMA pseudo-labels (t={f})")
    draw(axes[1], student_frames[f], "Student logits → soft q")
    
ani = FuncAnimation(fig, update, frames=steps, interval=250, blit=False)
gif_path = "./data/mean_teacher_lively.gif"
ani.save(gif_path, writer=PillowWriter(fps=6))
gif_path
