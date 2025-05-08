import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# -------------------------
# Utility: compute soft neighborhood q
def compute_q(embeddings, temp):
    sims = embeddings @ embeddings.T / temp
    np.fill_diagonal(sims, -np.inf)
    exp = np.exp(sims - sims.max(axis=1, keepdims=True))
    q = exp / exp.sum(axis=1, keepdims=True)
    np.fill_diagonal(q, 0.0)
    return q

# Gradient for RKDO
def rkdo_step(emb, q_prev, p_prev, alpha, lr, temp):
    p_t = (1 - alpha) * p_prev + alpha * q_prev
    q_curr = compute_q(emb, temp)
    diff = q_curr - p_t
    grad = ((diff[:,:,None] + diff.T[:,:,None]) * emb[None,:,:]).sum(axis=1) / temp
    emb -= lr * grad
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q_next = compute_q(emb, temp)
    return emb, q_next, p_t

# MeanTeacher step
def mean_teacher_step(emb, logits_prev, alpha, lr, temp):
    logits_curr = emb @ emb.T / temp
    np.fill_diagonal(logits_curr, -np.inf)
    teacher = (1 - alpha) * logits_prev + alpha * logits_curr
    targets = np.argmax(teacher, axis=1)
    grad = np.zeros_like(emb)
    for i, j in enumerate(targets):
        if i == j: continue
        grad[i] -= emb[j]
        grad[j] -= emb[i]
    emb -= lr * grad
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q_next = compute_q(emb, temp)
    # Teacher distribution: one‐hot on argmax (pseudo-label)
    teacher_dist = np.zeros_like(q_next)
    teacher_dist[np.arange(len(targets)), targets] = 1.0
    return emb, q_next, teacher_dist, logits_curr

# BYOL step (fixed positive, clockwise neighbor)
def byol_step(emb, lr):
    n = emb.shape[0]
    grad = np.zeros_like(emb)
    for i in range(n):
        j = (i + 1) % n
        grad[i] -= emb[j]
        grad[j] -= emb[i]
    emb -= lr * grad
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb

# -------------------------
# Simulation parameters
n = 8
d = 3
steps = 35
temp = 0.5
alpha = 0.25
rng = np.random.default_rng(3)

emb_rkdo = rng.standard_normal((n,d)); emb_rkdo /= np.linalg.norm(emb_rkdo, axis=1, keepdims=True)
emb_mt   = rng.standard_normal((n,d)); emb_mt   /= np.linalg.norm(emb_mt, axis=1, keepdims=True)
emb_byol = rng.standard_normal((n,d)); emb_byol /= np.linalg.norm(emb_byol, axis=1, keepdims=True)

q_r_prev = compute_q(emb_rkdo, temp)
p_r_prev = q_r_prev.copy()

logits_prev = emb_mt @ emb_mt.T / temp
np.fill_diagonal(logits_prev, -np.inf)

frames_r_q, frames_r_p = [], []
frames_mt_q, frames_mt_teacher = [], []
frames_byol_q, frames_byol_teacher = [], []

# Precompute BYOL teacher (static)
byol_teacher = np.zeros((n, n))
for i in range(n):
    byol_teacher[i, (i+1)%n] = 1.0

for step in range(steps):
    # store
    frames_r_q.append(q_r_prev)
    frames_r_p.append(p_r_prev)
    frames_mt_q.append(compute_q(emb_mt, temp))
    frames_mt_teacher.append(byol_teacher*0)  # placeholder, will replace later
    frames_byol_q.append(compute_q(emb_byol, temp))
    frames_byol_teacher.append(byol_teacher)

    # RKDO update
    emb_rkdo, q_r_next, p_r_t = rkdo_step(emb_rkdo, q_r_prev, p_r_prev, alpha, lr=2.0, temp=temp)
    q_r_prev, p_r_prev = q_r_next, p_r_t

    # MeanTeacher update
    emb_mt, q_mt_next, teacher_dist, logits_prev = mean_teacher_step(emb_mt, logits_prev, alpha, lr=1.5, temp=temp)
    frames_mt_teacher[-1] = teacher_dist
    # update for next
    logits_prev = emb_mt @ emb_mt.T / temp
    np.fill_diagonal(logits_prev, -np.inf)

    # BYOL update
    emb_byol = byol_step(emb_byol, lr=1.5)

# -------------------------
# Visualization
pos = nx.circular_layout(range(n))
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
plt.tight_layout()

def draw_graph(ax, weights, title):
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if i!=j and weights[i,j] > 1e-3:
                G.add_edge(i,j,weight=weights[i,j])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=350, node_color="#ffffff", edgecolors="#000000")
    edges = G.edges(data=True)
    for u,v,d in edges:
        w = d["weight"]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u,v)],
                               width=0.5+5*w, edge_color=[plt.cm.plasma(w)],
                               arrowstyle="->", arrowsize=7, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)

def update(frame):
    # RKDO
    draw_graph(axes[0,0], frames_r_p[frame], f"RKDO Teacher pᵗ (step {frame})")
    draw_graph(axes[1,0], frames_r_q[frame], f"RKDO qᵗ")
    # MeanTeacher
    draw_graph(axes[0,1], frames_mt_teacher[frame], f"MT Teacher (step {frame})")
    draw_graph(axes[1,1], frames_mt_q[frame], f"MT qᵗ")
    # BYOL
    draw_graph(axes[0,2], frames_byol_teacher[frame], "BYOL Teacher (static)")
    draw_graph(axes[1,2], frames_byol_q[frame], f"BYOL qᵗ (step {frame})")

anim = FuncAnimation(fig, update, frames=steps, interval=250, blit=False)
gif_path = "./data/teacher_student_comparison.gif"
anim.save(gif_path, writer=PillowWriter(fps=5))
gif_path
