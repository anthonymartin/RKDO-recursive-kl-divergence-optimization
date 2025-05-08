import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
from math import sqrt, pi, cos, sin

# #############################################################
#  Kaleidoscopic RKDO animation
#  ------------------------------------------------------------
#  * Full star graph coloured by KL(p||q)
#  * For each node, a "petal" simplex shows (p,q) and arrows
#  #############################################################

# ---------------------- simulation settings ------------------
N      = 12           # number of nodes (keeping it small for clarity)
D      = 3           # embedding dim
STEPS  = 50          # animation frames
ALPHA  = 0.25        # EMA coeff
TAU    = 0.45        # temperature for q
LR     = 2.4         # big learning-rate for vivid motion
JITTER = 0.15        # augment jitter
SEED   = 123
rng    = np.random.default_rng(SEED)

# ---------- helper: L2 normalise rows ------------------------
def l2(v): return v / np.linalg.norm(v, axis=1, keepdims=True)

# ---------- helper: soft neighbourhood ----------------------
def q_field(emb):
    sims = emb @ emb.T / TAU
    np.fill_diagonal(sims, -np.inf)
    exp = np.exp(sims - sims.max(axis=1, keepdims=True))
    q = exp / exp.sum(axis=1, keepdims=True)
    np.fill_diagonal(q, 0.0)
    return q

# ---------- helper: KL divergence per edge ------------------
def edge_KL(p, q):
    # KL row-wise, broadcast to edge weights p_ij log p_ij/q_ij
    mask = (p > 0)
    kl = np.zeros_like(p)
    kl[mask] = p[mask] * (np.log(p[mask] + 1e-12) - np.log(q[mask] + 1e-12))
    return kl

# ---------- helpers for simplex geometry --------------------
SQRT3 = sqrt(3)
verts = np.array([[0, 0], [1, 0], [0.5, SQRT3/2]])   # equilateral triangle
def simplex_xy(prob):
    """Map 3-vector on simplex to 2-D coords"""
    a, b, c = prob
    return b + 0.5*c, (SQRT3/2)*c

def nat_grad(qrow, prow):
    F_inv = np.diag(qrow) - np.outer(qrow, qrow)
    return F_inv @ (prow - qrow)

# ---------- initial embeddings ------------------------------
emb = l2(rng.standard_normal((N, D)))
q_prev = q_field(emb)
p_prev = q_prev.copy()

q_frames, p_frames = [q_prev], [p_prev]

# ---------- simulate RKDO -----------------------------------
for t in range(1, STEPS):
    aug = l2(emb + JITTER * rng.standard_normal(emb.shape))
    q_curr = q_field(aug)
    p_curr = (1-ALPHA)*p_prev + ALPHA*q_prev

    # gradient on emb
    diff = q_curr - p_curr
    grad = ((diff[:,:,None] + diff.T[:,:,None]) * emb[None]).sum(axis=1) / TAU
    emb = l2(emb - LR*grad)

    q_next = q_field(emb)

    q_frames.append(q_next)
    p_frames.append(p_curr)
    q_prev, p_prev = q_next, p_curr

# ---------- star layout -------------------------------------
pos = nx.circular_layout(range(N), scale=4.0)

# --- prepare figure ----
fig, ax = plt.subplots(figsize=(8,8))
ax.set_axis_off()

# offset radial distance for simplex petals
radial_inset = 0.8

def draw_petal(ax, center, angle, qrow, prow):
    """
    Draw a tiny simplex petal near node center.
    angle = angle to node (radians), used to rotate petal.
    """
    # build rotation matrix
    ca, sa = np.cos(angle), np.sin(angle)
    R = np.array([[ca, -sa],[sa, ca]])
    # scale for visibility
    scale = 0.7
    tri = scale * (verts - verts.mean(axis=0)) @ R.T + center

    # outline
    ax.plot(*zip(*(tri.tolist()+[tri[0].tolist()])), color="gray", lw=0.8, alpha=0.6)

    # dots
    # map probabilities qrow (length 3) to coordinates inside petal
    xq, yq = simplex_xy(qrow)
    xp, yp = simplex_xy(prow)
    pt_q = scale * (np.array([xq, yq]) - verts.mean(axis=0)) @ R.T + center
    pt_p = scale * (np.array([xp, yp]) - verts.mean(axis=0)) @ R.T + center
    ax.scatter(*pt_q, color="blue", s=14, zorder=5)
    ax.scatter(*pt_p, color="red",  s=14, zorder=5)
    # geodesic
    ax.plot([pt_q[0], pt_p[0]], [pt_q[1], pt_p[1]], ls="--", lw=0.7, color="gray", alpha=0.6)

    # gradient arrows (natural vs Eucl)
    eucl = prow - qrow
    nat = nat_grad(qrow, prow)
    # scale arrows for visibility
    def to_xy(vec):
        x, y = simplex_xy(qrow + 0.3*vec)
        return scale*(np.array([x,y])-verts.mean(axis=0))@R.T + center
    end_e = to_xy(eucl)
    end_n = to_xy(nat)
    ax.annotate("", xy=end_e, xytext=pt_q,
                arrowprops=dict(arrowstyle="->", color="purple", lw=0.7))
    ax.annotate("", xy=end_n, xytext=pt_q,
                arrowprops=dict(arrowstyle="->", color="limegreen", lw=0.7))

# ------------- animation update func ------------------------
def update(frame):
    ax.clear()
    ax.set_axis_off()
    q = q_frames[frame]
    p = p_frames[frame]
    kl = edge_KL(p, q)
    G = nx.DiGraph()
    for i in range(N):
        for j in range(N):
            if i==j: continue
            G.add_edge(i, j, weight=q[i,j], kl=kl[i,j])

    # draw star
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600,
                           node_color="#111111", edgecolors="#ffffff", linewidths=2)
    for u, v, d in G.edges(data=True):
        # hue = KL divergence
        hue = min(d['kl']*10, 1.0)  # scale KL for colour
        color = plt.cm.hsv(hue)
        width = 0.4 + 5*d['weight']
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u,v)],
                               width=width, edge_color=[color],
                               arrowstyle="->", arrowsize=5, alpha=0.85)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white")

    # draw petals for each node
    for i in range(N):
        # choose top-3 neighbours (excluding self)
        top3 = np.argsort(q[i])[-3:]
        q3 = q[i][top3] / q[i][top3].sum()
        p3 = p[i][top3] / p[i][top3].sum()
        # compute angle of node
        x, y = pos[i]
        angle = np.arctan2(y, x)
        center = np.array([x, y]) * radial_inset
        draw_petal(ax, center, angle, q3, p3)

    ax.set_title(f"RKDO kaleidoscope  •  frame {frame}", color="white")

# ------------- create animation -----------------------------
anim = FuncAnimation(fig, update, frames=STEPS, interval=170)
out_path = "./data/rkdo_kaleidoscope.gif"
anim.save(out_path, writer=PillowWriter(fps=8))
out_path

# ------------ helper: one panel ----------------------------
def draw_star_with_petals(ax, field, teacher):
    ax.clear(); ax.set_axis_off()
    kl = edge_KL(teacher, field)
    G = nx.DiGraph()
    for i in range(N):
        for j in range(N):
            if i!=j:
                G.add_edge(i, j, weight=field[i,j], kl=kl[i,j])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600,
                           node_color="#111", edgecolors="#fff", linewidths=2)
    for u,v,d in G.edges(data=True):
        hue   = min(d['kl']*10, 1.0)
        color = plt.cm.hsv(hue)
        width = 0.4 + 5*d['weight']
        nx.draw_networkx_edges(G,pos,ax=ax,edgelist=[(u,v)],
                               width=width,edge_color=[color],
                               arrowstyle="->",arrowsize=6,alpha=0.9)
    nx.draw_networkx_labels(G,pos,ax=ax,font_color="white")

    # petals
    for i in range(N):
        top3 = np.argsort(field[i])[-3:]
        q3 = field[i][top3]   / field[i][top3].sum()
        p3 = teacher[i][top3] / teacher[i][top3].sum()
        angle  = np.arctan2(pos[i][1], pos[i][0])
        center = np.array(pos[i])*radial_inset
        draw_petal(ax, center, angle, q3, p3)

# ------------ animation update -----------------------------
fig, (ax_q, ax_p) = plt.subplots(1,2,figsize=(14,7))
fig.patch.set_facecolor('black')

def update(frame):
    q, p = q_frames[frame], p_frames[frame]
    draw_star_with_petals(ax_q, q, p)
    draw_star_with_petals(ax_p, p, p)
    ax_q.set_title(f"Student  q  (t={frame})", color="white")
    ax_p.set_title(f"Teacher  p  (t={frame})", color="white")

ani = FuncAnimation(fig, update, frames=STEPS, interval=160)
ani.save("./data/rkdo_kaleidoscope_split.gif", writer=PillowWriter(fps=8))
