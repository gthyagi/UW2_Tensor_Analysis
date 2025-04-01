# ## Plot Stress SHmax

import nest_asyncio
nest_asyncio.apply()

import h5py
import pyvista as pv
import numpy as np
from matplotlib import pyplot as plt
from cmcrameri import cm
from PIL import Image
from scipy.interpolate import griddata
import os
from pyvista import CellType

# +
# input dir

# input_dir = '/Volumes/seagate4_1/spherical_models/sum_sph_128448608_1104_DMesh_30_25_const_coh_UPDen_0.0_SPDen_1.0_LMVisc_30_hden15_LM_mitp08/'
# input_dir = '/Volumes/seagate4_1/spherical_models/sum_sph_128448608_1104_DMesh_30_25_const_coh_UPDen_0.0_SPDen_1.0_LMVisc_30_LM_smean2/'
# input_dir = '/Volumes/seagate4_1/spherical_models/sum_sph_128448608_1104_DMesh_30_25_const_coh_UPDen_0.0_SPDen_1.0_LMVisc_30_LM_mitp08/'
# input_dir = '/Volumes/seagate4_1/spherical_models/sum_sph_128448608_1104_DMesh_30_25_const_coh_UPDen_0.0_SPDen_1.0_LMVisc_30/'
# input_dir = '/Volumes/seagate4_1/spherical_models/sum_sph_80352368_336_DMesh_30_25_const_coh_UPDen_0.0_SPDen_1.0_LMVisc_30/'
input_dir = '/Volumes/seagate4_1/spherical_models/sum_sph_80272320_240_DMesh_30_25_const_coh_UPDen_0.0_SPDen_1.0_LMVisc_30/'

output_dir = input_dir+'stress_tensor_analysis/'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

file_no = '00001'

# dimensions for the analysis
dim = 2
# -

# #### Read mesh data and create pyvista mesh

# +
# Load coordinates and connectivity
with h5py.File(f'{input_dir}mesh.00000.h5', "r") as f:
    coords = f["vertices"][:]       
    connectivity = f["en_map"][:]

# PyVista expects: [npts, pt0, pt1, ..., pt7, npts, pt0, ...]
num_cells = connectivity.shape[0]
npts_per_cell = connectivity.shape[1]  # Should be 8

# Flatten into VTK cell format
vtk_cells = np.hstack([np.full((num_cells, 1), npts_per_cell, dtype=np.int32),  # prepend '8' to each cell
                       connectivity]).ravel()
cell_types = np.full(num_cells, CellType.HEXAHEDRON, dtype=np.uint8) # Cell types: one per cell
grid = pv.UnstructuredGrid(vtk_cells, cell_types, coords) # Build the unstructured grid

# # Plot the mesh
# grid.plot(show_edges=True)
# -

# #### Underworld returns the stress components in the following order:
# $$σ_{xx}, σ_{yy}, σ_{zz}, σ_{xy}, σ_{xz}, σ_{yz}$$

# +
# Load stress components
with h5py.File(f"{input_dir}stressDField.{file_no}.h5", "r") as f:
    stress_D = f["data"][:]

print("stress_D shape:", stress_D.shape)

with h5py.File(f"{input_dir}stressNDField.{file_no}.h5", "r") as f:
    stress_ND = f["data"][:]

print("stress_ND shape:", stress_ND.shape)


# -

def build_tensor(s):
    '''
    Build symmetric 3×3 stress tensors
    s: (6,) → tensor: (3, 3)
    '''
    σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz = s
    return np.array([
        [σ_xx, σ_xy, σ_xz],
        [σ_xy, σ_yy, σ_yz],
        [σ_xz, σ_yz, σ_zz]
    ])


def compute_stress_second_invariant(stress_comps):
    """
    Compute the second invariant J2 of the deviatoric stress tensor
    for a batch of 3x3 stress tensors.

    Parameters:
        stress_tensors: np.ndarray of shape (N, 3, 3)

    Returns:
        j2: np.ndarray of shape (N,), second invariant of deviatoric stress
    """
    # stress tensor
    stress_tensors = np.array([build_tensor(s) for s in stress_comps])  # shape: (N, 3, 3)

    # Compute trace (mean normal stress)
    trace = np.trace(stress_tensors, axis1=1, axis2=2)  # shape (N,)

    # Identity matrix
    I = np.eye(3)

    # Deviatoric stress tensors
    deviatoric = stress_tensors - trace[:, None, None] / 3.0 * I

    # J2 = 0.5 * sum(s_ij * s_ij)
    j2 = 0.5 * np.einsum('nij,nij->n', deviatoric, deviatoric)

    return j2


# Compute radius array
radius_arr = np.linalg.norm(grid.points, axis=1)
surface_mask = np.abs(radius_arr - 1.0) < 1e-6 # tolerance depends on radial resolution
surface_points = grid.points[surface_mask]
surface_stress_comps = np.hstack((stress_D, stress_ND))[surface_mask]
surface = pv.PolyData(surface_points) # Create PolyData (no connectivity, point cloud)
surface_mesh = surface.delaunay_2d() # Reconstruct surface mesh from the point cloud
surface_mesh.point_data["stress"] = surface_stress_comps
surface_mesh["stress_inv"] = compute_stress_second_invariant(surface_stress_comps)
r_hat = surface_mesh.points / np.linalg.norm(surface_mesh.points, axis=1)[:, None]  # compute radial unit vector
surface_mesh["r_hat"] = r_hat

# Plot
surface_mesh.plot(scalars="stress_inv", cmap=cm.hawaii_r.resampled(20), clim=[1e-8, 1e-4], cpos='xy')

# #### Compute eigen values and eigen vectors

# $$
# \begin{array}{lccc}
# \hline
# \text{Regime} & S_1 & S_2 & S_3 \\
# \hline
# \text{Normal}      & S_V       & S_{H\max}  & S_{h\min} \\
# \text{Strike-slip} & S_{H\max} & S_V        & S_{h\min} \\
# \text{Reverse}     & S_{H\max} & S_{h\min}  & S_V       \\
# \hline
# \end{array}
# $$

# arrays to store info
N = surface_stress_comps.shape[0]
tectonic_style_str = np.full(N, "unknown", dtype=object)
principal_stress = np.zeros((N, dim), dtype=float) # eigenvalues
principal_dirs = np.zeros((N, dim, dim), dtype=float) # eigenvectors
SHmax = np.zeros((N, dim), dtype=float)
SHmin = np.zeros((N, dim), dtype=float)


def compute_tectonic_style_str(stress_comps):
    """
    Input: stress components
    Return: Array with tectonic style as strings
    """
    N = stress_comps.shape[0]
    tectonic_style_str = np.full(N, "unknown", dtype=object)

    for i in range(N):    
        G = -build_tensor(stress_comps[i]) # [:dim, :dim] # Reshape the 9-component gradient into a 3x3 matrix
        E = 0.5 * (G + G.T) # Compute the symmetric tensor
        w, v = np.linalg.eig(E) # Compute eigenvalues (w) and eigenvectors (v) of E
        idx = np.argsort(w)[::-1] # Sort eigenvalues (and eigenvectors) in descending order (largest first)
        w = w[idx]
        v = v[:, idx]  # eigenvectors as columns
    
        # Compute the absolute dot product with z_hat for each eigenvector to find the vertical one
        dot_vals = np.abs(np.dot(r_hat[i], v))  # returns an array of 3 values
        vertical_idx = np.argmax(dot_vals)
    
        # define tectonic regime
        if vertical_idx==2:
            tectonic_style_str[i] = 'Thrust'
        elif vertical_idx==0:
            tectonic_style_str[i] = 'Normal'
        else:
            tectonic_style_str[i] = 'Strike-slip'
    return tectonic_style_str


# calculate eigen values and vectors
for i in range(N):    
    G = -build_tensor(surface_stress_comps[i])[:dim, :dim] # Reshape the 9-component gradient into a 3x3 matrix
    E = 0.5 * (G + G.T) # Compute the symmetric tensor
    w, v = np.linalg.eig(E) # Compute eigenvalues (w) and eigenvectors (v) of E
    idx = np.argsort(w)[::-1] # Sort eigenvalues (and eigenvectors) in descending order (largest first)
    w = w[idx]
    v = v[:, idx]  # eigenvectors as columns

    # storing in array
    principal_stress[i] = w
    principal_dirs[i] = v

    if dim==3:
        # Compute the absolute dot product with z_hat for each eigenvector to find the vertical one
        dot_vals = np.abs(np.dot(r_hat[i], v))  # returns an array of 3 values
        vertical_idx = np.argmax(dot_vals)
    
        # define tectonic regime
        # manual way
        if vertical_idx==2:
            SHmax[i] = v[:, 0]
            SHmin[i] = v[:, 1]
        elif vertical_idx==0:
            SHmax[i] = v[:, 1]
            SHmin[i] = v[:, 2]
        else:
            SHmax[i] = v[:, 0]
            SHmin[i] = v[:, 2]

        # # programatic way
        # horizontal_idxs = [j for j in range(3) if j != vertical_idx] # Get the horizontal indices (excluding the vertical one)
        # idx_hmax, idx_hmin = sorted(horizontal_idxs, key=lambda j: w[j], reverse=True) # Sort horizontal indices by eigenvalue (largest first) and unpack them
        # SHmax, SHmin = v[:, idx_hmax], v[:, idx_hmin] # Extract the corresponding eigenvectors
    else:
        SHmax[i] = v[:, 0]
        SHmin[i] = v[:, 1]


def vector_projection_normalise(shmax, shmin, r_hat, projection_only=False, normalize_only=False):
    """
    Projects the input vectors onto the horizontal plane by removing the component 
    along r_hat, and/or normalizes them.
    """
    if projection_only:
        dot_shmax = np.sum(shmax * r_hat, axis=1, keepdims=True)
        dot_shmin = np.sum(shmin * r_hat, axis=1, keepdims=True)
        return shmax - dot_shmax * r_hat, shmin - dot_shmin * r_hat

    elif normalize_only:
        return (
            shmax / np.linalg.norm(shmax, axis=1, keepdims=True),
            shmin / np.linalg.norm(shmin, axis=1, keepdims=True)
        )

    else:
        dot_shmax = np.sum(shmax * r_hat, axis=1, keepdims=True)
        dot_shmin = np.sum(shmin * r_hat, axis=1, keepdims=True)

        shmax_proj = shmax - dot_shmax * r_hat
        shmin_proj = shmin - dot_shmin * r_hat

        shmax_proj /= np.linalg.norm(shmax_proj, axis=1, keepdims=True)
        shmin_proj /= np.linalg.norm(shmin_proj, axis=1, keepdims=True)

        return shmax_proj, shmin_proj


# +
# Create RGB color array (float, 0–1) for each point
tectonic_style_str = compute_tectonic_style_str(surface_stress_comps)

colors = np.zeros((N, 3))  # shape (N, 3)

for i in range(N):
    s = tectonic_style_str[i]

    if s=='Thrust':
        colors[i] = [0.0, 0.0, 1.0]   # Blue for Thrust
    elif s=='Normal':
        colors[i] = [1.0, 0.0, 0.0]   # Red for Normal
    else:
        colors[i] = [0.0, 1.0, 0.0]   # Green for Strike-slip


# -

def make_3d_vec(arr):
    'convert array of (N, 2) to (N, 3)'
    vec2d = arr
    vec3d = np.zeros((vec2d.shape[0], 3))
    vec3d[:, :2] = vec2d  # Pad with zero Z
    return vec3d


# +
# creating data on mesh
surface_mesh["principal_stress_1"] = principal_stress[:, 0]
surface_mesh["principal_stress_2"] = principal_stress[:, 1]
if dim==3:
    surface_mesh["principal_stress_3"] = principal_stress[:, 2]

    surface_mesh["principal_dir_1"] = principal_dirs[:, :, 0]  # direction for largest eigen value
    surface_mesh["principal_dir_2"] = principal_dirs[:, :, 1]  # direction for intermediate eigen value
    surface_mesh["principal_dir_3"] = principal_dirs[:, :, 2]  # direction for smallest eigen value

    surface_mesh['SHmax'], surface_mesh['SHmin'] = vector_projection_normalise(SHmax, SHmin, r_hat)
     
else:
    surface_mesh["principal_dir_1"] = make_3d_vec(principal_dirs[:, :, 0])  # direction for largest eigen value
    surface_mesh["principal_dir_2"] = make_3d_vec(principal_dirs[:, :, 1])  # direction for smallest eigen value

    shmax_shmin = vector_projection_normalise(SHmax, SHmin, r_hat, normalize_only=True)
    surface_mesh['SHmax'] = make_3d_vec(shmax_shmin[0])
    surface_mesh['SHmin'] = make_3d_vec(shmax_shmin[1])

surface_mesh["style_color"] = colors
# -

# Plot
surface_mesh.plot(scalars="principal_stress_1", cmap=plt.cm.coolwarm.resampled(10), clim=[-9e-3, 9e-3], cpos='xy')

# +
# plotting eigen vectors
glyph1 = surface_mesh.glyph(orient="principal_dir_1", scale=False, factor=2e-3)
glyph2 = surface_mesh.glyph(orient="principal_dir_2", scale=False, factor=2e-3)
# glyph3 = surface_mesh.glyph(orient="principal_dir_3", scale=True, factor=50)

pl = pv.Plotter()
# pl.add_mesh(surface_mesh, scalars="stress_inv", cmap=cm.hawaii_r.resampled(20), clim=[1e-7, 1e-4], )# show_edges=True)
pl.add_mesh(glyph1, scalars="style_color", rgb=True)
pl.add_mesh(glyph2, scalars="style_color", rgb=True)
# pl.add_mesh(glyph3, color="green")
pl.show(cpos='xy')

# +
N = 66 #76 # 64
mask = np.arange(surface_mesh.n_points) % N == 0

# Extract sampled points only
sampled_points = surface_mesh.points[mask]
sampled_SHmin = surface_mesh["SHmin"][mask]
sampled_SHmax = surface_mesh["SHmax"][mask]
sampled_stress_inv = surface_mesh["stress_inv"][mask]
sampled_style_color = surface_mesh["style_color"][mask]

# Create a new PolyData
resampled_mesh = pv.PolyData(sampled_points)
resampled_mesh["SHmin"] = sampled_SHmin
resampled_mesh["SHmax"] = sampled_SHmax
resampled_mesh["stress_inv"] = sampled_stress_inv
resampled_mesh["style_color"] = sampled_style_color

# +
# Plot ONLY the coarse mesh
pl = pv.Plotter()
# pl.add_mesh(surface_mesh, scalars="stress_inv", opacity=1.0, show_scalar_bar=False,
#            cmap=cm.hawaii_r.resampled(20), clim=[1e-7, 1e-4], show_edges=False,)

# sbar = pl.add_scalar_bar(title="Stress Invariant", vertical=True, title_font_size=20, 
#                  label_font_size=20, 
#                  width=0.1,        # relative width of the scalar bar
#                  height=0.8,       # relative height of the scalar bar
#                  position_x=0.88,  # x-position (from left) of the scalar bar
#                  position_y=0.1,    # y-position (from bottom) of the scalar bar
#                  n_labels=5
#                  )

# Create an arrow glyph with no tip
custom_arrow = pv.Arrow(tip_length=0.0, tip_radius=0.0, shaft_radius=0.03)
scale_factor = 0.015 # 0.2
scale = False

glyphs_max_pos = resampled_mesh.glyph(orient="SHmax", scale=scale, factor=scale_factor, geom=custom_arrow)
pl.add_mesh(glyphs_max_pos, scalars="style_color", rgb=True)

resampled_mesh["minus_SHmax"] = -np.array(resampled_mesh["SHmax"])
glyphs_max_neg = resampled_mesh.glyph(orient="minus_SHmax", scale=scale, factor=scale_factor, geom=custom_arrow)
pl.add_mesh(glyphs_max_neg, scalars="style_color", rgb=True)

pl.show(cpos='xy')
pl.camera.zoom(1.4)
# # Save a high-resolution screenshot as a PNG file
# filename = f'{output_dir}stress_sh_max_min_{vel_file_no}'
# p.screenshot(f'{filename}.png', scale=6)

# # Convert the PNG to PDF using Pillow
# im = Image.open(f'{filename}.png')
# im.save(f'{filename}.pdf', "PDF", resolution=100.0)

# +
# # this is background data
# slice_coords = slice_mesh.points
# slice_scalar = slice_mesh["dilatation"]

# # Define the grid you want
# x_old, y_old = slice_coords[:,0], slice_coords[:,1]
# x_new = np.linspace(x_old.min(), x_old.max(), 260)
# y_new = np.linspace(y_old.min(), y_old.max(), 260)
# X, Y = np.meshgrid(x_new, y_new)
# Z = griddata(points=(x_old, y_old), values=slice_scalar, xi=(X, Y), method='cubic')

# # this is foreground data
# # Make a coarse grid
# nx, ny, nz = 20, 20, 1
# grid = pv.ImageData()
# grid.dimensions = (nx, ny, nz)
# grid.origin = (x_min, y_min, z_min)

# dx = (x_max - x_min) / (nx - 1)
# dy = (y_max - y_min) / (ny - 1)
# dz = 1.0  # for 2D slice
# grid.spacing = (dx, dy, dz)

# # Resample the slice_mesh onto the coarse grid
# coarse_mesh = grid.sample(masked_mesh)

# X_c = coarse_mesh.points[:, 0].reshape(nx, ny)
# Y_c = coarse_mesh.points[:, 1].reshape(nx, ny)

# SHmax_x = coarse_mesh["SHmax"][:, 0].reshape(nx, ny)  
# SHmax_y = coarse_mesh["SHmax"][:, 1].reshape(nx, ny)
# SHmax_c = [SHmax_x, SHmax_y]

# SHmin_x = coarse_mesh["SHmin"][:, 0].reshape(nx, ny) 
# SHmin_y = coarse_mesh["SHmin"][:, 1].reshape(nx, ny)
# SHmin_c = [SHmin_x, SHmin_y]
# -

def plot_scalar_with_shmax_shmin(
    # Data for contourf and quiver plots
    X, Y, Z,
    X_c, Y_c,
    SHmax_c, SHmin_c,
    # Contour options
    cmap=None,
    levels=50,
    vmin=-0.04,
    vmax=0.04,
    # Quiver options
    scale=0.4,
    # Axis formatting options
    x_axis_label='bottom',  # options: 'off', 'top', 'bottom'
    y_axis_label='left',    # options: 'off', 'left'
    ax_text_size=18,
    xlim=(-512, 0),
    ylim=(0, 512),
    # File saving options
    output_dir=output_dir,
    fileformat='pdf',       # options: 'pdf', 'eps', 'png', etc.
    filename="test",
    # Subplot adjustment (passed as dict)
    subplots_adjust=dict(left=0.125, right=0.9, bottom=0.11, top=0.88, wspace = 0.2, hspace = 0.2),
    # Colorbar separate saving options
    cb_horz_save=False,
    cb_vert_save=False,
    cb_name="colorbar",
    cb_axis_label="Dilatation",
    cb_horz_label_xpos=0.5,
    cb_horz_label_ypos=1.1,
    cb_vert_label_xpos=1.1,
    cb_vert_label_ypos=0.5,
    # Optionally supply a separate colormap for colorbar saving
    colormap=None
):
    """
    Plots a filled contour (Z over X, Y) with overlaid quiver arrows
    for SHmax (blue) and SHmin (red) (and their negatives), formats the axes,
    adjusts subplot spacing, and saves the figure. Optionally, the colorbar
    is saved separately as a PNG.

    Parameters
    ----------
    X, Y, Z : 2D arrays
        Data for the contourf plot.
    X_c, Y_c : 1D or 2D arrays
        Coordinates for the quiver arrows.
    SHmax_c, SHmin_c : tuple or list of two arrays each
        For example, SHmax_c = (Umax, Vmax) and SHmin_c = (Umin, Vmin).
    cmap : matplotlib.colors.Colormap, optional
        Colormap for contourf. If None, uses plt.cm.viridis.
    levels : int, optional
        Number of contour levels.
    vmin, vmax : float, optional
        Color limits for the contour plot.
    scale : float, optional
        Scale factor for the quiver arrows.
    x_axis_label : {'off', 'top', 'bottom'}, optional
        Controls the x-axis labeling.
    y_axis_label : {'off', 'left'}, optional
        Controls the y-axis labeling.
    ax_text_size : int, optional
        Font size for axis labels and ticks.
    xlim : tuple, optional
        x-axis limits.
    ylim : tuple, optional
        y-axis limits.
    output_dir : str, optional
        Directory to save the figure.
    fileformat : str, optional
        Format for saving the figure (e.g., 'pdf', 'eps', 'png').
    savefile : str, optional
        Base name for the saved file.
    subplots_adjust : dict, optional
        Keyword arguments for plt.subplots_adjust.
    cb_save : bool, optional
        If True, save the colorbar separately as a PNG.
    cb_vert_save : bool, optional
        If True, also save a vertical colorbar as a PNG.
    cb_name : str, optional
        Base name for the saved colorbar file.
    cb_axis_label : str, optional
        Title for the colorbar.
    cb_label_xpos, cb_label_ypos : float, optional
        Title position for the horizontal colorbar.
    cb_vert_label_xpos, cb_vert_label_ypos : float, optional
        Title position for the vertical colorbar.
    colormap : matplotlib.colors.Colormap, optional
        Colormap to use for the colorbar if different from cmap.
    
    Returns
    -------
    None
    """

    plt.rc('font', size=ax_text_size)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the filled contour
    cs = ax.contourf(
        X, Y, Z,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    # Plot quiver arrows for SHmax (blue) and SHmin (red)
    ax.quiver(
        X_c, Y_c,
        SHmax_c[0], SHmax_c[1],
        color="blue", scale=scale, headwidth=1, headlength=0
    )
    ax.quiver(
        X_c, Y_c,
        -SHmax_c[0], -SHmax_c[1],
        color="blue", scale=scale, headwidth=1, headlength=0
    )
    ax.quiver(
        X_c, Y_c,
        SHmin_c[0], SHmin_c[1],
        color="red", scale=scale, headwidth=1, headlength=0
    )
    ax.quiver(
        X_c, Y_c,
        -SHmin_c[0], -SHmin_c[1],
        color="red", scale=scale, headwidth=1, headlength=0
    )

    # Format grid and ticks
    ax.grid()  # grid on
    ax.tick_params(axis='both', direction='in')  # turn tickmarks inward
    ax.tick_params(axis='x', which='major', pad=13)  # adjust x-axis tick padding

    # Set axis limits and labels for x-axis
    ax.set_xlim(xlim)
    if x_axis_label == 'off':
        ax.set_xticklabels([])  # turn-off tick labels
        ax.set_xlabel("")        # turn-off axis label
    elif x_axis_label == 'top':
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('x (km)', fontsize=ax_text_size, labelpad=10)
        ax.xaxis.set_tick_params(labelsize=ax_text_size)
    elif x_axis_label == 'bottom':
        ax.set_xlabel('x (km)', fontsize=ax_text_size)
        ax.xaxis.set_tick_params(labelsize=ax_text_size)

    # Set axis limits and labels for y-axis
    ax.set_ylim(ylim)
    if y_axis_label == 'off':
        ax.set_yticklabels([])  # turn-off tick labels
        ax.set_ylabel("")        # turn-off axis label
    elif y_axis_label == 'left':
        ax.set_ylabel('y (km)', fontsize=ax_text_size)
        ax.yaxis.set_tick_params(labelsize=ax_text_size)

    # Adjust subplot spacing
    plt.subplots_adjust(**subplots_adjust)

    # Save the figure using the provided file format and output path
    if fileformat == 'pdf':
        plt.savefig(output_dir + filename + "." + fileformat, format=fileformat, bbox_inches='tight')
    elif fileformat == 'png':
        plt.savefig(output_dir + filename + "." + fileformat, dpi=150)

    # Optionally save the colorbar separately as a horizontal PNG file
    if cb_horz_save:
        a = np.array([[vmin, vmax]])
        plt.figure(figsize=(5, 5))
        plt.imshow(a, cmap=cmap)
        plt.gca().set_visible(False)
        cax = plt.axes([0.1, 0.2, 1.15, 0.06])
        cb = plt.colorbar(orientation='horizontal', cax=cax)
        cb.ax.set_title(cb_axis_label, fontsize=ax_text_size, x=cb_horz_label_xpos, y=cb_horz_label_ypos)

        if fileformat == 'pdf':
            plt.savefig(output_dir + cb_name + "_horz.pdf", format=fileformat, bbox_inches='tight')
        elif fileformat == 'png':
            plt.savefig(output_dir + cb_name + "_horz.png", dpi=150)

    # Optionally save the colorbar separately as a vertical PNG file
    if cb_vert_save:
        a = np.array([[vmin, vmax]])
        plt.figure(figsize=(5, 5))
        plt.imshow(a, cmap=cmap)
        plt.gca().set_visible(False)
        cax = plt.axes([0.1, 0.2, 0.06, 1.15])
        cb = plt.colorbar(orientation='vertical', cax=cax)
        cb.ax.set_title(cb_axis_label, fontsize=ax_text_size, x=cb_vert_label_xpos, y=cb_vert_label_ypos)
        
        if fileformat == 'pdf':
            plt.savefig(output_dir + cb_name + "." + "_vert.pdf", format=fileformat, bbox_inches='tight')
        elif fileformat == 'png':
            plt.savefig(output_dir + cb_name + "." + "_vert.png", dpi=150)

    plt.show()

# +
# # plot
# plot_scalar_with_shmax_shmin(
#     X, Y, Z,
#     X_c, Y_c,
#     SHmax_c,
#     SHmin_c,
#     cmap=cm.roma_r.resampled(20),  
#     levels=50,
#     vmin=-1e-14,
#     vmax=1e-14,
#     scale=1600000000,
#     x_axis_label='bottom',
#     y_axis_label='left',
#     ax_text_size=18,
#     xlim=(-512, 0),
#     ylim=(0, 512),
#     output_dir=output_dir,
#     fileformat='pdf',
#     filename=f"stress_dilatation_shmax_shmin_{vel_file_no}_{analy_space}",
#     # subplots_adjust=dict(left=0.35, right=0.85, bottom=0.1, top=0.85),
#     cb_horz_save=True,
#     cb_vert_save=False,
#     cb_name=f"stress_dilatation_shmax_shmin_{vel_file_no}_{analy_space}",
#     cb_axis_label=r"Dilatation $(1/s)$",
#     cb_horz_label_xpos=0.5,
#     cb_horz_label_ypos=-2.1,
# )
# -




