import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

# --- helper to map a probability triple (a,b,c) onto 2-D simplex coords -----------
def simplex_xy(p):
    # vertices: A(0,0)=a, B(1,0)=b, C(0.5, √3/2)=c
    a, b, c = p
    x = b + 0.5 * c
    y = (sqrt(3) / 2) * c
    return x, y

# --- simulate one RKDO node with 3-way neighbourhood --------------------------------
rng = np.random.default_rng(0)
# initial q^0 is random point on simplex
q0 = rng.dirichlet([1, 1, 1])
p0 = q0.copy()

alpha = 0.3
steps = 12
q = q0.copy()
p = p0.copy()

for t in range(1, steps + 1):
    # teacher update
    p = (1 - alpha) * p + alpha * q
    # "student" gradient step toward teacher + small jitter
    q = q + 0.7 * (p - q) + rng.normal(scale=0.02, size=3)
    q = np.clip(q, 1e-6, None)
    q = q / q.sum()

    if t == 1:
        q1, p1 = q.copy(), p.copy()  # after one step
    if t == 10:
        q10, p10 = q.copy(), p.copy()

# --- natural vs Euclidean gradient at t=10 ----------------------------------------
# Euclidean gradient direction (raw diff)
grad_eucl = p10 - q10
# Natural gradient direction: project diff with Fisher metric (diag(q) - q q^T)
F_inv = np.diag(q10) - np.outer(q10, q10)
grad_nat = F_inv @ grad_eucl

# rescale arrows for plotting
def scale(vec, length=0.18):
    norm = np.linalg.norm(vec)
    return vec * (length / norm) if norm > 0 else vec

grad_eucl_scaled = scale(grad_eucl)
grad_nat_scaled  = scale(grad_nat)

# --- plotting ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
titles = ["Step 0", "Step 10"]

# draw simplex triangle
verts = np.array([[0, 0], [1, 0], [0.5, sqrt(3)/2]])
for ax in axes:
    ax.plot(*zip(*(verts.tolist() + [verts[0].tolist()])), color="grey", lw=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, sqrt(3)/2 + 0.05)
    ax.set_aspect('equal')
    ax.axis('off')

# plot t=0
xq0, yq0 = simplex_xy(q0)
xp0, yp0 = simplex_xy(p0)
axes[0].scatter([xp0], [yp0], color="red", s=80, label="p^0 = q^0")
axes[0].annotate("", xy=(xp0, yp0), xytext=(xq0, yq0),
                 arrowprops=dict(arrowstyle="->", color="red"))
axes[0].scatter([xq0], [yq0], color="blue", s=80, label="q^0")
axes[0].set_title("Soft Teacher p and Student q at t = 0")
axes[0].legend(loc="upper right", fontsize=8)

# plot t=10
xq10, yq10 = simplex_xy(q10)
xp10, yp10 = simplex_xy(p10)
axes[1].scatter([xp10], [yp10], color="red", s=80, label="Teacher p^{10}")
axes[1].scatter([xq10], [yq10], color="blue", s=80, label="Student q^{10}")

# mixture geodesic (straight line in simplex coords)
axes[1].plot([xq10, xp10], [yq10, yp10], color="gray", ls="--", lw=1, label="mixture geodesic")

# Euclidean vs natural arrows
axes[1].annotate("", xy=(xq10 + grad_eucl_scaled[1] + 0.5*grad_eucl_scaled[2],
                         yq10 + (sqrt(3)/2)*grad_eucl_scaled[2]),
                 xytext=(xq10, yq10),
                 arrowprops=dict(arrowstyle="->", color="purple"),
                 )
axes[1].annotate("", xy=(xq10 + grad_nat_scaled[1] + 0.5*grad_nat_scaled[2],
                         yq10 + (sqrt(3)/2)*grad_nat_scaled[2]),
                 xytext=(xq10, yq10),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 )
axes[1].text(xq10 + 0.02, yq10 - 0.04, "Euclidean grad", color="purple", fontsize=7)
axes[1].text(xq10 + 0.03, yq10 + 0.07, "Natural grad", color="green", fontsize=7)

axes[1].set_title("t = 10   •   natural vs Euclidean directions")
axes[1].legend(loc="upper right", fontsize=8)

plt.tight_layout()
img_path = "./data/mean_teacher_simplex_prototype.png"
plt.savefig(img_path, dpi=160)
img_path
