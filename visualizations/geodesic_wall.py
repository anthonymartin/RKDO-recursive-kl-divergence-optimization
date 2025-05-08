import numpy as np
import matplotlib.pyplot as plt
import networkx as network_x
from matplotlib.animation import FuncAnimation, PillowWriter
from math import sqrt, pi, cos, sin
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

# #############################################################
#  RKDO Geodesic Wall Visualization 
#  ------------------------------------------------------------
#  * Creates a wavy wall of connected geodesics from RKDO
#  * Multiple instances connected in a tessellation pattern
#  #############################################################

# ---------------------- simulation settings ------------------
N      = 30           # number of nodes (increased for better geodesic effect)
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

# -------- Create wall mapping functions -----------------
def wavy_wall(x, y, amplitude=0.6, frequency=1.5):
    """Map 2D coordinates to a wavy wall surface"""
    # y controls the wave
    z = amplitude * np.sin(frequency * np.pi * y)
    return x, y, z

def rippled_wall(x, y, amplitude=0.5, frequency_x=1, frequency_y=2):
    """Map 2D coordinates to a rippled wall with varying x and y frequencies"""
    z = amplitude * np.sin(frequency_x * np.pi * x) * np.sin(frequency_y * np.pi * y)
    return x, y, z

def curved_wall(x, y, curvature=0.8):
    """Map 2D coordinates to a curved wall"""
    # x-based curvature
    z = curvature * (x - 0.5)**2
    return x, y, z

# ---------- Create tessellation mapping ---------------------
def create_tessellation_mapping(positions_2d, mapping_func, tessellation_size=(4, 3), 
                               jitter_scale=0.02, scale_factor=2.0):
    """
    Create a tessellation by repeating the basic pattern with slight variations.
    
    Parameters:
    - positions_2d: Original 2D positions
    - mapping_func: Function to map from 2D to 3D
    - tessellation_size: Number of tiles in (x, y) directions
    - jitter_scale: Amount of random variation between tiles
    - scale_factor: How much to scale the positions (stretches out the wall)
    
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
            offset_x = tile_x / tiles_x * scale_factor
            offset_y = tile_y / tiles_y * scale_factor
            
            # Small jitter to make each tile slightly different
            jitter_x = jitter_scale * (rng.random() - 0.5)
            jitter_y = jitter_scale * (rng.random() - 0.5)
            
            # Create nodes for this tile
            for node, (nx, ny) in norm_pos.items():
                # Scale down the positions to fit in the tile
                tile_nx = offset_x + nx / tiles_x * scale_factor + jitter_x
                tile_ny = offset_y + ny / tiles_y * scale_factor + jitter_y
                
                # Apply manifold mapping to get 3D coordinates
                x3d, y3d, z3d = mapping_func(tile_nx, tile_ny)
                
                # Create a new node ID for this tile's copy
                new_node = f"{node}_{tile_x}_{tile_y}"
                tessellation_3d[new_node] = (x3d, y3d, z3d)
    
    return tessellation_3d

# ------------ Create geodesic visualization -----------------
def draw_geodesic_lines(ax, p_matrix, pos_3d, threshold=0.05, 
                        num_lines=40, alpha_scale=2.0, width_scale=2.0,
                        cmap=plt.cm.viridis, node_alpha=0.6):
    """Draw geodesic lines connecting nodes based on probability field p"""
    # Get a list of the strongest connections
    edges = []
    for i in range(N):
        for j in range(N):
            if i != j and p_matrix[i, j] > threshold:
                edges.append((i, j, p_matrix[i, j]))
    
    # Sort edges by strength (strongest first)
    edges.sort(key=lambda x: x[2], reverse=True)
    
    # Limit number of lines for clarity
    if len(edges) > num_lines:
        edges = edges[:num_lines]
    
    # Draw the selected edges
    for tile_x in range(tiles_x):
        for tile_y in range(tiles_y):
            for i, j, strength in edges:
                # Get 3D positions
                source = f"{i}_{tile_x}_{tile_y}"
                target = f"{j}_{tile_x}_{tile_y}"
                
                if source not in pos_3d or target not in pos_3d:
                    continue
                
                source_pos = pos_3d[source]
                target_pos = pos_3d[target]
                
                # Calculate edge properties based on strength
                width = 0.5 + width_scale * strength
                alpha = min(0.2 + alpha_scale * strength, 1.0)
                color = cmap(min(strength * 2, 0.99))
                
                # Draw the edge as a line in 3D
                ax.plot([source_pos[0], target_pos[0]],
                        [source_pos[1], target_pos[1]],
                        [source_pos[2], target_pos[2]],
                        linewidth=width, color=color, alpha=alpha, zorder=5)
            
            # Draw nodes with transparency to emphasize the geodesic lines
            for i in range(N):
                node = f"{i}_{tile_x}_{tile_y}"
                if node in pos_3d:
                    pos = pos_3d[node]
                    ax.scatter(pos[0], pos[1], pos[2], color='black', 
                              alpha=node_alpha, s=20, edgecolors='white', 
                              linewidth=0.5, zorder=10)

# --- Draw geodesic wall with tessellation --------------
def draw_geodesic_wall(frame):
    """Draw the RKDO geodesic wall visualization"""
    p = p_frames[frame]  # Teacher field for geodesics
    
    # Create 3D figure with dark background
    fig = plt.figure(figsize=(14, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Choose a mapping function for the wall
    mapping_func = wavy_wall  # Default
    # mapping_func = rippled_wall  # Alternative
    # mapping_func = curved_wall   # Alternative
    
    # Create geodesic graph layout
    # Using force directed layout for more organic connections
    original_pos = network_x.spring_layout(range(N), seed=SEED)
    
    # Get min/max for normalization
    pos_array = np.array(list(original_pos.values()))
    min_x, min_y = pos_array.min(axis=0)
    max_x, max_y = pos_array.max(axis=0)
    
    # Create tessellated mapping on the wall
    global tiles_x, tiles_y
    tiles_x, tiles_y = 4, 3  # Wall dimensions
    tessellation_size = (tiles_x, tiles_y)
    
    pos_3d = create_tessellation_mapping(
        original_pos, 
        mapping_func,
        tessellation_size=tessellation_size,
        jitter_scale=0.01,  # Small jitter for natural variation
        scale_factor=1.5    # Stretches out the wall
    )
    
    # Draw the geodesic connections
    draw_geodesic_lines(
        ax, p, pos_3d,
        threshold=0.02,     # Minimum connection strength 
        num_lines=80,       # Maximum number of lines per tile
        alpha_scale=3.0,    # How much connection strength affects transparency
        width_scale=3.0,    # How much connection strength affects line width
        cmap=plt.cm.plasma, # Color map for the lines
        node_alpha=0.7      # Node transparency
    )
    
    # Add subtle grid to enhance the 3D effect
    x_min, x_max = -0.2, tiles_x * 1.5 + 0.2
    y_min, y_max = -0.2, tiles_y * 1.5 + 0.2
    z_min, z_max = -0.8, 0.8
    
    # Calculate grid based on the mapping function for better adherence to the surface
    grid_density = 20
    grid_x = np.linspace(x_min, x_max, grid_density)
    grid_y = np.linspace(y_min, y_max, grid_density)
    
    # Draw subtle grid lines on the wall surface
    for x in grid_x[::2]:  # Skip every other line for clarity
        points = [mapping_func(x, y) for y in grid_y]
        ax.plot([p[0] for p in points], 
                [p[1] for p in points], 
                [p[2] for p in points], 
                color='gray', alpha=0.1, linewidth=0.5)
    
    for y in grid_y[::2]:  # Skip every other line for clarity
        points = [mapping_func(x, y) for x in grid_x]
        ax.plot([p[0] for p in points], 
                [p[1] for p in points], 
                [p[2] for p in points], 
                color='gray', alpha=0.1, linewidth=0.5)
    
    # Set view limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Set equal aspect ratio for better 3D visualization
    ax.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min])
    
    # Remove axis labels and ticks for cleaner look
    ax.set_axis_off()
    
    # Set specific view angle - fixed directly in front of the wall
    elevation = 10  # Slight elevation to see the wave shape
    azimuth = 270   # Front view (looking at the wall)
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Add subtle title
    ax.set_title(f"RKDO Geodesic Wall • Frame {frame}", 
                color='white', fontsize=14, pad=20)
    
    return fig

# -------- Create animation directly as GIF ---------------------------
# Create figure first
fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

# Define update function for animation
def update_frame(frame):
    ax.clear()
    ax.set_facecolor('black')
    
    p = p_frames[frame]  # Teacher field for geodesics
    
    # Choose a mapping function for the wall
    mapping_func = wavy_wall  # Default
    # mapping_func = rippled_wall  # Alternative
    # mapping_func = curved_wall   # Alternative
    
    # Create geodesic graph layout
    # Using force directed layout for more organic connections
    original_pos = network_x.spring_layout(range(N), seed=SEED)
    
    # Get min/max for normalization
    pos_array = np.array(list(original_pos.values()))
    min_x, min_y = pos_array.min(axis=0)
    max_x, max_y = pos_array.max(axis=0)
    
    # Create tessellated mapping on the wall
    global tiles_x, tiles_y
    tiles_x, tiles_y = 4, 3  # Wall dimensions
    tessellation_size = (tiles_x, tiles_y)
    
    pos_3d = create_tessellation_mapping(
        original_pos, 
        mapping_func,
        tessellation_size=tessellation_size,
        jitter_scale=0.01,  # Small jitter for natural variation
        scale_factor=1.5    # Stretches out the wall
    )
    
    # Draw the geodesic connections
    draw_geodesic_lines(
        ax, p, pos_3d,
        threshold=0.02,     # Minimum connection strength 
        num_lines=80,       # Maximum number of lines per tile
        alpha_scale=3.0,    # How much connection strength affects transparency
        width_scale=3.0,    # How much connection strength affects line width
        cmap=plt.cm.plasma, # Color map for the lines
        node_alpha=0.7      # Node transparency
    )
    
    # Add subtle grid to enhance the 3D effect
    x_min, x_max = -0.2, tiles_x * 1.5 + 0.2
    y_min, y_max = -0.2, tiles_y * 1.5 + 0.2
    z_min, z_max = -0.8, 0.8
    
    # Draw subtle grid lines on the wall surface
    grid_density = 10  # Reduced for GIF size
    grid_x = np.linspace(x_min, x_max, grid_density)
    grid_y = np.linspace(y_min, y_max, grid_density)
    
    for x in grid_x[::2]:  # Skip every other line for clarity
        points = [mapping_func(x, y) for y in grid_y]
        ax.plot([p[0] for p in points], 
                [p[1] for p in points], 
                [p[2] for p in points], 
                color='gray', alpha=0.1, linewidth=0.5)
    
    for y in grid_y[::2]:  # Skip every other line for clarity
        points = [mapping_func(x, y) for x in grid_x]
        ax.plot([p[0] for p in points], 
                [p[1] for p in points], 
                [p[2] for p in points], 
                color='gray', alpha=0.1, linewidth=0.5)
    
    # Set view limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Set equal aspect ratio for better 3D visualization
    ax.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min])
    
    # Remove axis labels and ticks for cleaner look
    ax.set_axis_off()
    
    # Set specific view angle that showcases the wall
    # Rotate view based on frame for animation
    elevation = 20  # Fixed elevation
    azimuth = 230 + frame % 90  # Rotate 90 degrees over the animation
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Add subtle title
    ax.set_title(f"RKDO Geodesic Wall • Frame {frame}", 
                color='white', fontsize=14, pad=20)
    
    # Make tight layout to ensure proper dimensions for GIF
    fig.tight_layout()

# Create animation directly as GIF
frames_to_animate = list(range(0, STEPS, 4))  # Use fewer frames for smaller GIF
anim = FuncAnimation(fig, update_frame, frames=frames_to_animate, interval=150)

# Save directly as GIF
output_path = "./data/rkdo_geodesic_wall.gif"
anim.save(output_path, writer=PillowWriter(fps=8))

print(f"Animation saved as {output_path}")