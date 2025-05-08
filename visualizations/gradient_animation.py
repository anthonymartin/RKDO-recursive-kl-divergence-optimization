import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
n_nodes = 8
d = 3
total_steps = 40
alpha = 0.2
temperature = 0.5
lr = 3.0
jitter = 0.15  # random perturbation to mimic data augmentations
seed = 5
rng = np.random.default_rng(seed)

emb = rng.standard_normal((n_nodes, d))
emb /= np.linalg.norm(emb, axis=1, keepdims=True)

def compute_q(embeddings):
    sims = embeddings @ embeddings.T / temperature
    np.fill_diagonal(sims, -np.inf)
    exp = np.exp(sims - np.max(sims, axis=1, keepdims=True))
    q = exp / exp.sum(axis=1, keepdims=True)
    np.fill_diagonal(q, 0.0)
    return q

q_prev = compute_q(emb)
p_prev = q_prev.copy()

qs = [q_prev]
ps = [p_prev]

for step in range(1, total_steps):
    # Simulate fresh augmentations: jitter embeddings slightly before forward pass
    aug_emb = emb + jitter * rng.standard_normal(emb.shape)
    aug_emb /= np.linalg.norm(aug_emb, axis=1, keepdims=True)

    q_curr = compute_q(aug_emb)

    # Supervisor based on previous q
    p_t = (1 - alpha) * p_prev + alpha * q_prev

    # gradient direction: (q_curr - p_t)
    diff = q_curr - p_t

    # Compute a simple cosine-sim gradient wrt emb
    grad = np.zeros_like(emb)
    for i in range(n_nodes):
        grad_i = ((diff[i][:, None] + diff[:, i][:, None]) * emb).sum(axis=0)
        grad[i] = grad_i / temperature
    # Apply gradient
    emb -= lr * grad
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    q_new = compute_q(emb)

    qs.append(q_new)
    ps.append(p_t)

    q_prev = q_new
    p_prev = p_t

# Visualization
pos = nx.circular_layout(range(n_nodes))
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
plt.tight_layout()

def draw_graph(ax, mat, title):
    ax.clear()
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                G.add_edge(i, j, weight=mat[i, j])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600, node_color="#ffffff", edgecolors="#000000", linewidths=1.0)
    for (u, v, d) in G.edges(data=True):
        w = d["weight"]
        if w < 1e-3:
            continue
        color = plt.cm.plasma(w)
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)],
                               width=0.5 + 6.0 * w, alpha=0.85,
                               edge_color=[color], arrowstyle="->", arrowsize=9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    ax.set_title(title)
    ax.set_axis_off()

def update(frame):
    draw_graph(axes[0], qs[frame], f"q (response) – step {frame}")
    draw_graph(axes[1], ps[frame], f"p (EMA) – step {frame}")

anim = FuncAnimation(fig, update, frames=total_steps, interval=200, blit=False)
gif_path = "./data/rkdo_grad_lively.gif"
anim.save(gif_path, writer=PillowWriter(fps=6))
gif_path
