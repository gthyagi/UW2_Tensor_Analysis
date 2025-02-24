# ## Plot Strainrate SHmax

import nest_asyncio
nest_asyncio.apply()

import h5py
import pyvista as pv
import numpy as np
from matplotlib import pyplot as plt
from cmcrameri import cm
from PIL import Image
from scipy.interpolate import griddata

# +
# input dir
input_dir = './'

output_dir = input_dir

vel_file_no = 16

compute_eigen_3d = not True

if compute_eigen_3d:
    analy_space = '3d'
else:
    analy_space = '2d'

# +
# create mesh
nx, ny, nz = 65, 65, 33

# reading mesh data
with h5py.File('mesh.h5', 'r') as f:
    mesh_coords = f['vertices'][:]

# Initialize the StructuredGrid
mesh = pv.StructuredGrid()
mesh.points = mesh_coords
mesh.dimensions = [nx, ny, nz]

# +
# # Plot the grid, showing cell edges
# mesh.plot(show_edges=True)

# +
# reading velocity data
with h5py.File(f'{input_dir}velocityField-{vel_file_no}.h5', 'r') as f:
    velocity = f['data'][:]  

# accessing velocity data to mesh
mesh.point_data['velocity'] = velocity*3.17e-10/1000 # convert cm/yr to m/s

# # plot velocity magnitude
# mesh.plot(scalars='velocity', clim=[0, 5], cmap=plt.cm.viridis.resampled(12))
# -

# $$
# \mathbf{v} = \begin{pmatrix}
# u_x \\
# u_y \\
# u_z
# \end{pmatrix},
# \quad
# \nabla \mathbf{v} = 
# \begin{pmatrix}
# \frac{\partial u_x}{\partial x} & \frac{\partial u_x}{\partial y} & \frac{\partial u_x}{\partial z} \\
# \frac{\partial u_y}{\partial x} & \frac{\partial u_y}{\partial y} & \frac{\partial u_y}{\partial z} \\
# \frac{\partial u_z}{\partial x} & \frac{\partial u_z}{\partial y} & \frac{\partial u_z}{\partial z}
# \end{pmatrix}
# $$
#

# compute velocity derivates
velocity_derivative = mesh.compute_derivative(scalars="velocity", gradient=True, 
                                              divergence=True, vorticity=True)

# plot velocity gradient component
velocity_gradient = velocity_derivative["gradient"]
# velocity_derivative.plot(scalars=velocity_gradient[:, 0], cmap=cm.roma.resampled(20), 
#                          show_edges=False, clim=[-0.1, 0.1])

# +
N = velocity_gradient.shape[0]

tectonic_style = np.zeros((N, 1), dtype=float)
dilatation = np.zeros((N, 1), dtype=float) # e_xx + e_yy
strainrate_zz = np.zeros((N, 1), dtype=float) # e_zz
for i in range(N):
    G = velocity_gradient[i].reshape((3, 3))
    E = 0.5 * (G + G.T)
    dilatation[i] = E[0,0] + E[1,1]
    strainrate_zz[i] = E[2,2]

# Allocate arrays for principal strains (eigenvalues) and eigenvectors (directions)
if compute_eigen_3d:
    principal_strains = np.zeros((N, 3), dtype=float)
    principal_dirs = np.zeros((N, 3, 3), dtype=float)
else:
    principal_strains = np.zeros((N, 2), dtype=float)
    principal_dirs = np.zeros((N, 2, 2), dtype=float)
    
for i in range(N):
    
    if compute_eigen_3d:
        # Reshape the 9-component gradient into a 3x3 matrix
        G = velocity_gradient[i].reshape((3, 3))
    else:
        G = velocity_gradient[i].reshape((3, 3))[:2, :2]
    
    # Compute the symmetric strain-rate tensor: E = 0.5 * (G + G.T)
    E = 0.5 * (G + G.T)
    
    # Compute eigenvalues (w) and eigenvectors (v) of E
    w, v = np.linalg.eig(E)
    
    # Sort eigenvalues (and eigenvectors) in descending order (largest first)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]  # eigenvectors as columns
    
    principal_strains[i] = w
    principal_dirs[i] = v

# Allocate arrays for horizontal eigenvector directions:
# SHmax: maximum horizontal principal direction
# SHmin: minimum horizontal principal direction
if compute_eigen_3d:
    SHmax = np.zeros((N, 3), dtype=float)
    SHmin = np.zeros((N, 3), dtype=float)
else:
    SHmax = np.zeros((N, 2), dtype=float)
    SHmin = np.zeros((N, 2), dtype=float)

# Define the vertical unit vector
z_hat = np.array([0, 0, 1])

for i in range(N):
    w = principal_strains[i]
    v = principal_dirs[i]  # shape (3,3) with eigenvectors in columns

    if compute_eigen_3d:
        # Compute the absolute dot product with z_hat for each eigenvector to find the vertical one
        dot_vals = np.abs(np.dot(z_hat, v))  # returns an array of 3 values
        vertical_idx = np.argmax(dot_vals)
        
        # The remaining two indices correspond to horizontal eigenvectors
        horizontal_idxs = [j for j in range(3) if j != vertical_idx]
        
        # Among the horizontal eigenvalues, select the one with the maximum value for SHmax
        if w[horizontal_idxs[0]] >= w[horizontal_idxs[1]]:
            idx_hmax = horizontal_idxs[0]
            idx_hmin = horizontal_idxs[1]
        else:
            idx_hmax = horizontal_idxs[1]
            idx_hmin = horizontal_idxs[0]
        
        # Extract the eigenvectors corresponding to SHmax and SHmin
        v_Hmax = v[:, idx_hmax]
        v_Hmin = v[:, idx_hmin]
        
        # Project each eigenvector onto the horizontal plane by removing the vertical component
        v_Hmax_proj = v_Hmax - np.dot(v_Hmax, z_hat) * z_hat
        v_Hmin_proj = v_Hmin - np.dot(v_Hmin, z_hat) * z_hat
        
        # Normalize the projected vectors
        v_Hmax_norm = v_Hmax_proj / np.linalg.norm(v_Hmax_proj)
        v_Hmin_norm = v_Hmin_proj / np.linalg.norm(v_Hmin_proj)
    else:
        v_Hmax_norm = v[:, 0] / np.linalg.norm(v[:, 0])
        v_Hmin_norm = v[:, 1] / np.linalg.norm(v[:, 1])

        v_Hmax_proj = w[0]*v[:, 0]
        v_Hmin_proj = w[-1]*v[:, 1]
    
        
    # Store the horizontal principal directions
    # SHmax[i] = v_Hmax_norm
    # SHmin[i] = v_Hmin_norm
    SHmax[i] = v_Hmax_proj
    SHmin[i] = v_Hmin_proj

    # style
    tectonic_style[i] = (w[0]+w[-1])/np.max([np.abs(w[0]), np.abs(w[-1])])


# +
# creating data on mesh
mesh["principal_strain_1"] = principal_strains[:, 0]
mesh["principal_strain_2"] = principal_strains[:, 1]
if compute_eigen_3d:
    mesh["principal_strain_3"] = principal_strains[:, 2]
    
mesh["principal_dir_1"] = principal_dirs[:, :, 0]  # direction for largest strain
mesh["principal_dir_2"] = principal_dirs[:, :, 1]  # direction for intermediate strain
if compute_eigen_3d:
    mesh["principal_dir_3"] = principal_dirs[:, :, 2]  # direction for smallest strain

mesh['gradient_zz'] = velocity_gradient[:,8]
mesh['dilatation'] = dilatation
mesh['strainrate_zz'] = strainrate_zz

if compute_eigen_3d:
    mesh['SHmax'] = SHmax
    mesh['SHmin'] = SHmin
else:
    mesh['SHmax'] = np.column_stack([SHmax, np.zeros(SHmax.shape[0])])
    mesh['SHmin'] = np.column_stack([SHmin, np.zeros(SHmin.shape[0])])

mesh['style'] = tectonic_style

# +
# # plotting largest eigen value
# p = pv.Plotter()
# p.add_mesh(mesh, scalars="principal_strain_1", cmap=plt.cm.coolwarm.resampled(10), 
#            show_edges=False, clim=[-0.1, 0.1])
# p.show()

# +
# # plotting eigen vectors
# glyph1 = mesh.glyph(orient="principal_dir_1", scale=True, factor=50)
# glyph2 = mesh.glyph(orient="principal_dir_2", scale=True, factor=50)
# glyph3 = mesh.glyph(orient="principal_dir_3", scale=True, factor=50)

# p = pv.Plotter()
# # p.add_mesh(vel_derivs, scalars="principal_strain_1", cmap=plt.cm.coolwarm.resampled(10), show_edges=False, clim=[-0.1, 0.1])
# p.add_mesh(glyph1, color="black")
# p.add_mesh(glyph2, color="blue")
# p.add_mesh(glyph3, color="green")
# p.show()
# -

# This creates a plane normal to the z-axis at z = z_value
z_value = -8.0
plane = pv.Plane(center=(0, 0, z_value), direction=(0, 0, 1),
                 i_size=1e5, j_size=1e5)
slice_mesh = mesh.slice(normal="z", origin=(0, 0, z_value))

# +
p = pv.Plotter()

p.add_mesh(slice_mesh, scalars="dilatation", cmap=plt.cm.coolwarm.resampled(20), 
           clim=[-1e-14, 1e-14],)# show_edges=True)
glyphs_max = slice_mesh.glyph(orient="SHmax", scale=True, factor=2e15)
p.add_mesh(glyphs_max, color="black")

glyphs_min = slice_mesh.glyph(orient="SHmin", scale=True, factor=2e15)
p.add_mesh(glyphs_min, color="red")

p.view_xy()
p.show_axes()
p.show()

# +
x_min, x_max, y_min, y_max, z_min, z_max = slice_mesh.bounds
delta = 24 # Choose a boundary margin (delta)

# Create a mask to exclude points near boundaries
points = slice_mesh.points
mask = ((points[:, 0] > x_min + delta) &
        (points[:, 0] < x_max - delta) &
        (points[:, 1] > y_min + delta) &
        (points[:, 1] < y_max - delta))

# Extract the masked region
masked_mesh = slice_mesh.extract_points(mask, adjacent_cells=True)

# p = pv.Plotter()
# p.add_mesh(slice_mesh, scalars="gradient_zz", cmap=plt.cm.coolwarm.resampled(10), 
#            clim=[-0.05, 0.05],)
# glyphs = masked_mesh.glyph(orient="SHmax", scale=True, factor=1500,)
# p.add_mesh(glyphs, color="black")
# p.view_xy()

# p.show()
# +
# Make a coarse grid
nx, ny, nz = 40, 40, 1
grid = pv.ImageData()
grid.dimensions = (nx, ny, nz)
grid.origin = (x_min, y_min, z_min)

dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
dz = 1.0  # for 2D slice
grid.spacing = (dx, dy, dz)

# Resample the slice_mesh onto the coarse grid
coarse_mesh = grid.sample(masked_mesh)

# Plot ONLY the coarse mesh
p = pv.Plotter()

# p.add_mesh(slice_mesh, scalars="style", opacity=1.0, show_scalar_bar=False,
#            cmap=cm.broc.resampled(20), clim=[-1, 1], show_edges=False,)
p.add_mesh(slice_mesh, scalars="dilatation", opacity=1.0, show_scalar_bar=False,
           cmap=cm.roma_r.resampled(10), clim=[-1e-14, 1e-14], show_edges=False,)

sbar = p.add_scalar_bar(title="Dilatation", vertical=True, title_font_size=20, 
                 label_font_size=20, 
                 width=0.1,        # relative width of the scalar bar
                 height=0.8,       # relative height of the scalar bar
                 position_x=0.88,  # x-position (from left) of the scalar bar
                 position_y=0.1,    # y-position (from bottom) of the scalar bar
                 n_labels=5
                 )

# Create an arrow glyph with no tip
custom_arrow = pv.Arrow(tip_length=0.0, tip_radius=0.0, shaft_radius=0.025)
scale = True
scale_factor = 2e15

glyphs_max_pos = coarse_mesh.glyph(orient="SHmax", scale=scale, factor=scale_factor, geom=custom_arrow)
p.add_mesh(glyphs_max_pos, color="blue")
coarse_mesh["minus_SHmax"] = -np.array(coarse_mesh["SHmax"])
glyphs_max_neg = coarse_mesh.glyph(orient="minus_SHmax", scale=scale, factor=scale_factor, geom=custom_arrow)
p.add_mesh(glyphs_max_neg, color="blue")

glyphs_min_pos = coarse_mesh.glyph(orient="SHmin", scale=scale, factor=scale_factor, geom=custom_arrow)
p.add_mesh(glyphs_min_pos, color="red")
coarse_mesh["minus_SHmin"] = -np.array(coarse_mesh["SHmin"])
glyphs_min_neg = coarse_mesh.glyph(orient="minus_SHmin", scale=scale, factor=scale_factor, geom=custom_arrow)
p.add_mesh(glyphs_min_neg, color="red")

p.show(cpos='xy')

p.camera.zoom(1.4)

# # Save a high-resolution screenshot as a PNG file
# if compute_eigen_3d:
#     filename = f'{output_dir}strain_sh_max_min_{vel_file_no}_3d'
# else:
#     filename = f'{output_dir}strain_sh_max_min_{vel_file_no}_2d'
    
# p.screenshot(f'{filename}.png', scale=6)

# # Convert the PNG to PDF using Pillow
# im = Image.open(f'{filename}.png')
# im.save(f'{filename}.pdf', "PDF", resolution=100.0)
# +
# this is background data
slice_coords = coarse_mesh.points # slice_mesh.points
slice_scalar = coarse_mesh["dilatation"] # slice_mesh["dilatation"]

# Define the grid you want
x_old, y_old = slice_coords[:,0], slice_coords[:,1]
x_new = np.linspace(x_old.min(), x_old.max(), 260)
y_new = np.linspace(y_old.min(), y_old.max(), 260)
X, Y = np.meshgrid(x_new, y_new)
Z = griddata(points=(x_old, y_old), values=slice_scalar, xi=(X, Y), method='cubic')

# this is foreground data
# Make a coarse grid
nx, ny, nz = 20, 20, 1
grid = pv.ImageData()
grid.dimensions = (nx, ny, nz)
grid.origin = (x_min, y_min, z_min)

dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
dz = 1.0  # for 2D slice
grid.spacing = (dx, dy, dz)

# Resample the slice_mesh onto the coarse grid
coarse_mesh = grid.sample(masked_mesh)

X_c = coarse_mesh.points[:, 0].reshape(nx, ny)
Y_c = coarse_mesh.points[:, 1].reshape(nx, ny)

SHmax_x = coarse_mesh["SHmax"][:, 0].reshape(nx, ny)  
SHmax_y = coarse_mesh["SHmax"][:, 1].reshape(nx, ny)
SHmax_c = [SHmax_x, SHmax_y]

SHmin_x = coarse_mesh["SHmin"][:, 0].reshape(nx, ny) 
SHmin_y = coarse_mesh["SHmin"][:, 1].reshape(nx, ny)
SHmin_c = [SHmin_x, SHmin_y]


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



# plot
plot_scalar_with_shmax_shmin(
    X, Y, Z,
    X_c, Y_c,
    SHmax_c,
    SHmin_c,
    cmap=cm.roma_r.resampled(20),  
    levels=50,
    vmin=-2e-14,
    vmax=2e-14,
    scale=16e-14,
    x_axis_label='bottom',
    y_axis_label='left',
    ax_text_size=18,
    xlim=(-512+delta, 0-delta),
    ylim=(0+delta, 512-delta),
    output_dir=output_dir,
    fileformat='pdf',
    filename=f"strain_dilatation_shmax_shmin_{vel_file_no}_{analy_space}",
    # subplots_adjust=dict(left=0.35, right=0.85, bottom=0.1, top=0.85),
    cb_horz_save=True,
    cb_vert_save=False,
    cb_name=f"strain_dilatation_shmax_shmin_{vel_file_no}_{analy_space}",
    cb_axis_label=r"Dilatation $(1/s)$",
    cb_horz_label_xpos=0.5,
    cb_horz_label_ypos=-2.1,
)


