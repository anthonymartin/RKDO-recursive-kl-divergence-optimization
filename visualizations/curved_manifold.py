import numpy as np
import matplotlib.pyplot as plt
import networkx as network_x  # Renamed to avoid variable name conflicts
from matplotlib.animation import FuncAnimation, PillowWriter
from math import sqrt, pi, cos, sin
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

# #############################################################
#  RKDO Teacher Visualization on Curved Manifold
#  ------------------------------------------------------------
#  * Maps the teacher visualization to a curved manifold
#  * Creates a tessellation pattern with the visualization
#  #############################################################

# ---------------------- simulation settings ------------------
N      = 24           # number of nodes (increased for better tessellation)
D      = 3            # embedding dim
STEPS  = 50           # animation frames
ALPHA  = 0.25         # EMA coeff
TAU    = 0.45         # temperature for q
LR     = 2.4          # big learning-rate for vivid motion
JITTER = 0.15         # augment jitter
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

# -------- Create manifold mapping functions -----------------
def torus_mapping(x, y, R=4, r=1):
    """Map 2D coordinates to a torus in 3D"""
    # x, y in [0, 2π]
    theta = x * 2 * np.pi  # around the circle of the tube
    phi = y * 2 * np.pi    # around the central axis of the torus
    
    # Parametric equations for a torus
    X = (R + r * np.cos(theta)) * np.cos(phi)
    Y = (R + r * np.cos(theta)) * np.sin(phi)
    Z = r * np.sin(theta)
    
    return X, Y, Z

def hyperbolic_paraboloid(x, y, scale=1.0):
    """Map 2D coordinates to a hyperbolic paraboloid (saddle shape)"""
    # x, y in [-1, 1]
    z = scale * (x**2 - y**2)
    return x, y, z

def sinusoidal_surface(x, y, amplitude=0.5, frequency=2):
    """Map 2D coordinates to a sinusoidal surface"""
    # x, y in [-1, 1]
    z = amplitude * np.sin(frequency * np.pi * x) * np.sin(frequency * np.pi * y)
    return x, y, z

# ---------- Create tessellation mapping ---------------------
def create_tessellation_mapping(positions_2d, mapping_func, tessellation_size=(3, 3), 
                                jitter_scale=0.05):
    """
    Create a tessellation by repeating the basic pattern with slight variations.
    
    Parameters:
    - positions_2d: Original 2D positions
    - mapping_func: Function to map from 2D to 3D
    - tessellation_size: Number of tiles in (x, y) directions
    - jitter_scale: Amount of random variation between tiles
    
    Returns:
    - Dictionary of node positions in 3D
    """
    tiles_x, tiles_y = tessellation_size
    n_nodes = len(positions_2d)
    
    # Normalize original positions to [0, 1] range
    pos_array = np.array(list(positions_2d.values()))
    min_x, min_y = pos_array.min(axis=0)
    max_x, max_y = pos_array.max(axis=0)
    norm_pos = {}
    for node, (x, y) in positions_2d.items():
        norm_x = (x - min_x) / (max_x - min_x)
        norm_y = (y - min_y) / (max_y - min_y)
        norm_pos[node] = (norm_x, norm_y)
    
    # Create tessellation
    tessellation_3d = {}
    
    for tile_x in range(tiles_x):
        for tile_y in range(tiles_y):
            # Base offset for this tile (in 0-1 range)
            offset_x = tile_x / tiles_x
            offset_y = tile_y / tiles_y
            
            # Small jitter to make each tile slightly different
            jitter_x = jitter_scale * (rng.random() - 0.5)
            jitter_y = jitter_scale * (rng.random() - 0.5)
            
            # Create nodes for this tile
            for node, (nx, ny) in norm_pos.items():
                # Scale down the positions to fit in the tile
                tile_nx = offset_x + nx / tiles_x + jitter_x
                tile_ny = offset_y + ny / tiles_y + jitter_y
                
                # Apply manifold mapping to get 3D coordinates
                x3d, y3d, z3d = mapping_func(tile_nx, tile_ny)
                
                # Create a new node ID for this tile's copy
                new_node = f"{node}_{tile_x}_{tile_y}"
                tessellation_3d[new_node] = (x3d, y3d, z3d)
    
    return tessellation_3d

# ---------- draw petal function (modified for 3D) -----------
def draw_petal_3d(ax, center, normal_vec, qrow, prow, scale=0.2):
    """
    Draw a tiny simplex petal near node center in 3D.
    normal_vec = local normal vector to the manifold
    """
    # Create a local coordinate system on the manifold
    # normal_vec should be normalized
    normal_vec = normal_vec / np.linalg.norm(normal_vec)
    
    # Find two orthogonal vectors in the tangent plane
    if np.abs(normal_vec[2]) > 0.9:  # If normal is close to z-axis
        u = np.array([1, 0, 0]) - normal_vec[0] * normal_vec
    else:
        u = np.array([0, 0, 1]) - normal_vec[2] * normal_vec
    
    u = u / np.linalg.norm(u)
    v = np.cross(normal_vec, u)
    
    # Local coordinate system (u, v, normal_vec)
    R = np.column_stack((u, v, normal_vec))
    
    # Map the triangle vertices to 3D
    tri_2d = scale * (verts - verts.mean(axis=0))
    tri_3d = np.zeros((3, 3))
    
    for i in range(3):
        # Map to the tangent plane using u and v vectors
        tri_3d[i] = center + tri_2d[i, 0] * u + tri_2d[i, 1] * v
    
    # Draw the triangle outline
    edges = [(0, 1), (1, 2), (2, 0)]
    for i, j in edges:
        ax.plot([tri_3d[i, 0], tri_3d[j, 0]], 
                [tri_3d[i, 1], tri_3d[j, 1]], 
                [tri_3d[i, 2], tri_3d[j, 2]], 
                color="gray", lw=0.8, alpha=0.6)
    
    # Map probabilities to points inside the triangle
    xq, yq = simplex_xy(qrow)
    xp, yp = simplex_xy(prow)
    
    pt_q_2d = scale * (np.array([xq, yq]) - verts.mean(axis=0))
    pt_p_2d = scale * (np.array([xp, yp]) - verts.mean(axis=0))
    
    pt_q_3d = center + pt_q_2d[0] * u + pt_q_2d[1] * v
    pt_p_3d = center + pt_p_2d[0] * u + pt_p_2d[1] * v
    
    # Add points and connecting line
    ax.scatter(pt_q_3d[0], pt_q_3d[1], pt_q_3d[2], color="blue", s=10, zorder=5)
    ax.scatter(pt_p_3d[0], pt_p_3d[1], pt_p_3d[2], color="red", s=10, zorder=5)
    ax.plot([pt_q_3d[0], pt_p_3d[0]], 
            [pt_q_3d[1], pt_p_3d[1]], 
            [pt_q_3d[2], pt_p_3d[2]], 
            ls="--", lw=0.5, color="gray", alpha=0.6)
    
    # Skip gradient arrows in 3D to avoid clutter
    return pt_q_3d, pt_p_3d

# ------------ helper: one panel ----------------------------
def draw_star_with_petals(ax, field, teacher):
    ax.clear(); ax.set_axis_off()
    kl = edge_KL(teacher, field)
    G = network_x.DiGraph()
    for i in range(N):
        for j in range(N):
            if i!=j:
                G.add_edge(i, j, weight=field[i,j], kl=kl[i,j])
    network_x.draw_networkx_nodes(G, pos, ax=ax, node_size=600,
                           node_color="#111", edgecolors="#fff", linewidths=2)
    for u,v,d in G.edges(data=True):
        hue   = min(d['kl']*10, 1.0)
        color = plt.cm.hsv(hue)
        width = 0.4 + 5*d['weight']
        network_x.draw_networkx_edges(G,pos,ax=ax,edgelist=[(u,v)],
                               width=width,edge_color=[color],
                               arrowstyle="->",arrowsize=6,alpha=0.9)
    network_x.draw_networkx_labels(G,pos,ax=ax,font_color="white")

    # petals
    for i in range(N):
        top3 = np.argsort(field[i])[-3:]
        q3 = field[i][top3]   / field[i][top3].sum()
        p3 = teacher[i][top3] / teacher[i][top3].sum()
        angle  = np.arctan2(pos[i][1], pos[i][0])
        center = np.array(pos[i])*radial_inset
        draw_petal(ax, center, angle, q3, p3)

# --- Draw on curved manifold with tessellation --------------
def draw_teacher_on_manifold(frame):
    """Draw the teacher visualization on a curved manifold with tessellation"""
    p = p_frames[frame]  # We only care about the teacher field
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Choose a manifold mapping function
    # mapping_func = torus_mapping  # Option 1: Torus
    # mapping_func = hyperbolic_paraboloid  # Option 2: Saddle
    mapping_func = sinusoidal_surface  # Option 3: Sinusoidal waves
    
    # Create circular layout for original graph
    original_pos = network_x.circular_layout(range(N), scale=0.8)
    
    # Get min/max for normalization
    pos_array = np.array(list(original_pos.values()))
    min_x, min_y = pos_array.min(axis=0)
    max_x, max_y = pos_array.max(axis=0)
    
    # Create tessellated mapping
    tessellation_size = (3, 3)  # 3x3 grid of repeated patterns
    pos_3d = create_tessellation_mapping(original_pos, mapping_func, 
                                        tessellation_size=tessellation_size,
                                        jitter_scale=0.02)
    
    # Create graph instances for each tile
    tiles_x, tiles_y = tessellation_size
    total_tiles = tiles_x * tiles_y
    
    for tile_x in range(tiles_x):
        for tile_y in range(tiles_y):
            G = network_x.DiGraph()
            
            # Add nodes for this tile
            for i in range(N):
                node_id = f"{i}_{tile_x}_{tile_y}"
                G.add_node(node_id)
            
            # Add edges for this tile
            for i in range(N):
                for j in range(N):
                    if i != j:
                        source = f"{i}_{tile_x}_{tile_y}"
                        target = f"{j}_{tile_x}_{tile_y}"
                        weight = p[i, j]
                        G.add_edge(source, target, weight=weight)
            
            # Draw nodes
            node_xyz = np.array([pos_3d[node] for node in G.nodes()])
            ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2],
                      s=50, c='black', edgecolors='white', linewidth=1)
            
            # Draw edges
            for u, v, d in G.edges(data=True):
                weight = d.get('weight', 0.1)
                if weight < 0.05:  # Skip very weak edges for clarity
                    continue
                
                # Get 3D positions
                u_pos = pos_3d[u]
                v_pos = pos_3d[v]
                
                # Edge width and color based on weight
                width = 0.3 + 3 * weight
                color = plt.cm.plasma(min(weight * 5, 1.0))  # More vibrant color map
                
                # Draw the edge
                ax.plot([u_pos[0], v_pos[0]], 
                        [u_pos[1], v_pos[1]], 
                        [u_pos[2], v_pos[2]], 
                        linewidth=width, color=color, alpha=0.8)
            
            # Draw petals for each node
            for i in range(N):
                node_id = f"{i}_{tile_x}_{tile_y}"
                center = np.array(pos_3d[node_id])
                
                # Estimate normal vector using manifold function
                epsilon = 0.01
                x, y = original_pos[i]
                nx = (x - min_x) / (max_x - min_x)
                ny = (y - min_y) / (max_y - min_y)
                
                # Scale to tile
                tile_nx = tile_x / tiles_x + nx / tiles_x
                tile_ny = tile_y / tiles_y + ny / tiles_y
                
                # Get nearby points to compute normal
                p1 = np.array(mapping_func(tile_nx + epsilon, tile_ny))
                p2 = np.array(mapping_func(tile_nx - epsilon, tile_ny))
                p3 = np.array(mapping_func(tile_nx, tile_ny + epsilon))
                p4 = np.array(mapping_func(tile_nx, tile_ny - epsilon))
                
                # Compute tangent vectors
                t1 = p1 - p2
                t2 = p3 - p4
                
                # Normal is cross product of tangent vectors
                normal = np.cross(t1, t2)
                normal = normal / np.linalg.norm(normal)
                
                # Choose top-3 neighbors for the petal
                top3 = np.argsort(p[i])[-3:]
                p3_norm = p[i][top3] / p[i][top3].sum()
                
                # Draw petal (using p as both student and teacher for simplicity)
                draw_petal_3d(ax, center, normal, p3_norm, p3_norm, scale=0.12)
    
    # Set equal aspect ratio and add title/labels
    ax.set_box_aspect([1, 1, 0.4])  # Adjust for the shape of the manifold
    ax.set_title(f"RKDO Teacher on Curved Manifold - Frame {frame}", color="black", fontsize=14)
    ax.set_axis_off()
    
    # Set specific view angle
    ax.view_init(elev=25, azim=frame % 360)
    
    return fig

# -------- Create animation frames ---------------------------
# Define selected frames to render
frames_to_render = range(0, STEPS, 2)  # Just render half of the frames for efficiency

# Create output directory
os.makedirs("./manifold_frames", exist_ok=True)

# Render individual frames
for frame in frames_to_render:
    fig = draw_teacher_on_manifold(frame)
    plt.savefig(f"./manifold_frames/frame_{frame:03d}.png", 
                dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)

print("Rendered frames to ./manifold_frames/")
print("To create animation, you can use:")
print("ffmpeg -framerate 8 -i ./manifold_frames/frame_%03d.png -c:v libx264 -pix_fmt yuv420p curved_tessellation.mp4")