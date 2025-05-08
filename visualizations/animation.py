import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
n_nodes = 8
total_steps = 120
alpha = 0.5
temp = 0.5
seed = 42
rng = np.random.default_rng(seed)

# Fixed layout
pos = nx.circular_layout(range(n_nodes))

def compute_q(embeddings):
    sims = embeddings @ embeddings.T / temp
    np.fill_diagonal(sims, -np.inf)
    exp = np.exp(sims - np.max(sims, axis=1, keepdims=True))
    q = exp / exp.sum(axis=1, keepdims=True)
    np.fill_diagonal(q, 0.0)
    return q

# Initial embeddings
emb = rng.standard_normal((n_nodes, 3))
emb /= np.linalg.norm(emb, axis=1, keepdims=True)

qs, ps = [], []

# Step 0
q0 = compute_q(emb)
qs.append(q0)
ps.append(q0.copy())

# Simulate RKDO updates
for t in range(1, total_steps):
    emb += 0.03 * rng.standard_normal(emb.shape)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q_t = compute_q(emb)
    p_t = (1 - alpha) * ps[-1] + alpha * qs[-1]
    qs.append(q_t)
    ps.append(p_t)

# Build animation
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
plt.tight_layout()
node_opts = dict(node_size=600, node_color="#ffffff", edgecolors="#000000", linewidths=1.0)

def draw_graph(ax, mat, title):
    ax.clear()
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                G.add_edge(i, j, weight=mat[i, j])
    nx.draw_networkx_nodes(G, pos, ax=ax, **node_opts)
    for (u, v, d) in G.edges(data=True):
        w = d["weight"]
        if w < 1e-3:
            continue
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)],
            width=0.5 + 4.0 * w, alpha=0.8, arrowstyle="->", arrowsize=9
        )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    ax.set_title(title)
    ax.set_axis_off()

def update(frame):
    draw_graph(axes[0], qs[frame], f"q (response) – step {frame}")
    draw_graph(axes[1], ps[frame], f"p (EMA) – step {frame}")

anim = FuncAnimation(fig, update, frames=total_steps, interval=300, blit=False)

# Save GIF
gif_path = "./data/rkdo_evolution.gif"
anim.save(gif_path, writer=PillowWriter(fps=4))
gif_path
