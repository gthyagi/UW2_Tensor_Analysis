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
from pyvista import Spline
from matplotlib.path import Path

import math
from scipy import spatial
from scipy.stats import linregress
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import warnings
import matplotlib.patches as patches
from scipy import interpolate
from scipy.spatial import KDTree
from numpy.polynomial.polynomial import polyfit
import matplotlib.ticker as mticker
import copy
import transform_data_coords as tds
from netCDF4 import Dataset

# +
# Suppress specific warnings
import numpy.polynomial.polyutils as pu

warnings.simplefilter('ignore', pu.RankWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# +
# input dir

input_dir = '/Volumes/seagate4_1/spherical_models/sum_sph_128448608_1104_DMesh_30_25_const_coh_UPDen_0.0_SPDen_1.0_LMVisc_30_hden15_LM_mitp08/'
# input_dir = '/Volumes/seagate4_1/spherical_models/sum_sph_128448608_1104_DMesh_30_25_const_coh_UPDen_0.0_SPDen_1.0_LMVisc_30_LM_smean2/'
# input_dir = '/Volumes/seagate4_1/spherical_models/sum_sph_128448608_1104_DMesh_30_25_const_coh_UPDen_0.0_SPDen_1.0_LMVisc_30_LM_mitp08/'
# input_dir = '/Volumes/seagate4_1/spherical_models/sum_sph_128448608_1104_DMesh_30_25_const_coh_UPDen_0.0_SPDen_1.0_LMVisc_30/'
# input_dir = '/Volumes/seagate4_1/spherical_models/sum_sph_80352368_336_DMesh_30_25_const_coh_UPDen_0.0_SPDen_1.0_LMVisc_30/'
# input_dir = '/Volumes/seagate4_1/spherical_models/sum_sph_80272320_240_DMesh_30_25_const_coh_UPDen_0.0_SPDen_1.0_LMVisc_30/'

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
        colors[i] = [0.0, 0.4, 0.0]   # dark Green for Strike-slip
# -

if s=='Thrust':
    colors[i] = [0.0, 0.0, 1.0]   # Blue for Thrust
elif s=='Normal':
    colors[i] = [1.0, 0.0, 0.0]   # Red for Normal
else:
    colors[i] = [0.0, 0.4, 0.0]   # dark Green for Strike-slip


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
def _convert_cubedsphere_xyz_to_llr(c_xyz: np.ndarray) -> np.ndarray:
    """
    Converts cubedsphere (x, y, z) coordinates to cubedsphere (longitude, latitude, radius).
    """
    if np.any(np.isclose(c_xyz[:, 2], 0)):
        raise ValueError("z coordinate is zero for one or more points; cannot perform conversion.")

    tan_lon = c_xyz[:, 0] / c_xyz[:, 2]
    tan_lat = c_xyz[:, 1] / c_xyz[:, 2]
    factor = np.sqrt(tan_lon**2 + tan_lat**2 + 1)

    lon = np.degrees(np.arctan(tan_lon))
    lat = np.degrees(np.arctan(tan_lat))
    radius = c_xyz[:, 2] * factor
    return np.column_stack((lon, lat, radius))

def _convert_cubedsphere_llr_to_xyz(c_llr: np.ndarray) -> np.ndarray:
		"""
		Converts cubedsphere coordinates from (longitude, latitude, radius) to (x, y, z).

		Calculations:
		  - Compute the tangent of the longitude and latitude (in radians).
		  - Compute d = radius / sqrt(tan(lon)^2 + tan(lat)^2 + 1).
		  - Compute:
			  x = d * tan(lon)
			  y = d * tan(lat)
			  z = d
		"""
		# Compute tangent values for longitude and latitude (converted from degrees to radians)
		tan_lon = np.tan(np.deg2rad(c_llr[:, 0]))
		tan_lat = np.tan(np.deg2rad(c_llr[:, 1]))
		denom = np.sqrt(tan_lon**2 + tan_lat**2 + 1)
		d = c_llr[:, 2] / denom
		return np.column_stack((d * tan_lon, d * tan_lat, d))

# +
# resample surface points
surface_mesh_llr = np.round(_convert_cubedsphere_xyz_to_llr(surface_mesh.points), 6)
lon_min, lon_max = np.ceil(surface_mesh_llr[:,0].min()), np.floor(surface_mesh_llr[:,0].max())
lat_min, lat_max = np.ceil(surface_mesh_llr[:,1].min()), np.floor(surface_mesh_llr[:,1].max())

freq=1
lon_new = np.linspace(lon_min, lon_max, num=np.int32((lon_max-lon_min)/freq)+1, endpoint=True)
lat_new = np.linspace(lat_min, lat_max, num=np.int32((lat_max-lat_min)/freq)+1, endpoint=True)
print(lon_new, lat_new)

lon_new_grid, lat_new_grid = np.meshgrid(lon_new, lat_new)
new_grid_points = np.column_stack((lon_new_grid.ravel(), lat_new_grid.ravel()))


# -

def interpolate_vector_field(points, vectors, new_grid_points, method='linear'):
    """
    Interpolate a 3D vector field onto a new grid.
    """
    u, v, w = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    
    u_interp = griddata(points, u, new_grid_points, method=method)
    v_interp = griddata(points, v, new_grid_points, method=method)
    w_interp = griddata(points, w, new_grid_points, method=method)

    vec_interp = np.stack([u_interp, v_interp, w_interp], axis=1)
    return vec_interp


# interpolating new data
SHmax_interp = interpolate_vector_field(surface_mesh_llr[:, 0:2], surface_mesh["SHmax"], new_grid_points)
color_interp = interpolate_vector_field(surface_mesh_llr[:, 0:2], surface_mesh["style_color"], new_grid_points, method='nearest')

# Create a new PolyData
new_grid_points_3d = np.hstack([new_grid_points, np.ones((new_grid_points.shape[0], 1))])
resampled_mesh = pv.PolyData(_convert_cubedsphere_llr_to_xyz(new_grid_points_3d))
resampled_mesh["SHmax"] = SHmax_interp
resampled_mesh["style_color"] = color_interp

# load boundary vtk file
path = '/Users/tgol0006/phd_tg/phd_b2023/create_vtk_rot_models/model_61_120_-45_35_rot_vtk/'
# List of VTK file names
filenames = [
    'uw_ind_aus_pb.vtk',
    'uw_ind_aus_cont.vtk',
    'uw_model_bbox.vtk',
    'sum_subduction_symbols.vtk',
    'him_subduction_symbols.vtk',
]

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

# Loop through and add each mesh
for fname in filenames:
    mesh = pv.read(path + fname)
    pl.add_mesh(mesh, color='k', opacity=1.0)

pl.show(cpos='xy')
pl.camera.zoom(1.4)
# # Save a high-resolution screenshot as a PNG file
# filename = f'{output_dir}stress_sh_max_min_{vel_file_no}'
# p.screenshot(f'{filename}.png', scale=6)

# # Convert the PNG to PDF using Pillow
# im = Image.open(f'{filename}.png')
# im.save(f'{filename}.pdf', "PDF", resolution=100.0)
# -

def get_t_pts_m_angle(_trench_pt, _nn_pt=4, _n_pts=10):
    """
    input: trench points, marker frequency and how many nearneighbours to find
    output: marker coords, angle
    """
    indx = np.linspace(10, len(_trench_pt)-10, num=_n_pts, endpoint=True, dtype=int)
    t_pts_kdtree = spatial.KDTree(_trench_pt)
    dist_arr, indx_arr = t_pts_kdtree.query(_trench_pt, k=_nn_pt)
    marker_angle = np.zeros((_trench_pt.shape[0], 1))
    for i, nn in enumerate(indx_arr):
        nn_sort = np.sort(nn)
        b, m = polyfit(_trench_pt[nn_sort][:,0], _trench_pt[nn_sort][:,1], 1)
        marker_angle[i] = (math.atan(m)*180/math.pi)
        
    return (_trench_pt[indx], marker_angle[indx])


def rotate_arr(_data, _angle, _tc=''):
    """
    Inputs: data array, rotation angle
    Returns: Rotated coordinates
    """
    
    def rotate(_origin, _point, _angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        """
        ox, oy = _origin
        px, py = _point
        qx = ox + math.cos(_angle) * (px - ox) - math.sin(_angle) * (py - oy)
        qy = oy + math.sin(_angle) * (px - ox) + math.cos(_angle) * (py - oy)
        return qx, qy

    if _data.shape[1]==2:
        _data_mod = np.zeros((_data.shape[0], 3))
        _data_mod[:,0:2] = _data
        _data_mod[:,2] = 0
        _data_sphllr = _tc.translonlatr2sphlonlatr(_tc.geolonlat2translonlat(_data_mod))
        rotated_data_sphllr = copy.deepcopy(_data_sphllr)
    else:
        _data_sphllr = _tc.translonlatr2sphlonlatr(_tc.geolonlat2translonlat(_data))
        rotated_data_sphllr = copy.deepcopy(_data_sphllr)
        
    for count, coords in enumerate(_data_sphllr[:,0:2]):
        rotated_data_sphllr[count][0], rotated_data_sphllr[count][1] = rotate([0, 0], coords, math.radians(_angle))
    
    rotated_data_coords = _tc.translonlat2geolonlat(_tc.sphlonlatr2translonlatr(rotated_data_sphllr))
    
    return rotated_data_coords


def rotate_vec_arr(_data, _angle):
    """
    Inputs: data array, rotation angle
    Returns: Rotated coordinates
    """
    
    def rotate(_origin, _point, _angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        """
        ox, oy = _origin
        px, py = _point
        qx = ox + math.cos(_angle) * (px - ox) - math.sin(_angle) * (py - oy)
        qy = oy + math.sin(_angle) * (px - ox) + math.cos(_angle) * (py - oy)
        return qx, qy
    
    rotated_data = copy.deepcopy(_data)
    for count, vector in enumerate(_data[:,0:2]):
        rotated_data[count][0], rotated_data[count][1] = rotate([0, 0], vector, math.radians(_angle))
    
    return rotated_data


def plot_field_data_bmrot(_rotated_crs=True, _p_lon='', _p_lat='', _ax_set_extent='', _left_labels='', _bottom_labels='', _xlabel_size=18, _ylabel_size=18,
                          _xlocator=[90, 100, 110, 120], _ylocator=[-10, 0, 10], _xlocator_mod='', _plot_ind_aus_pb='', _tlinewidth=2, _sum_trench_coords='', 
                          _sum_tline_color='C4', _trench_marker='square', _tmarkersize=18, _markerwidth=2, _him_trench_coords='', _layer_coords_vel_list='', 
                          _rotate_angle_list='', _lvec_freq='', _lvec_scale='', _lvec_width='', _lvec_color_list='', _lvec_label_name='', _regrid_num='', 
                          _ref_vec_patch_loc='', _lvec_legend_loc='', _lvec_legend_col='', _lvec_legend_title='', _model_bbox_list='', _bbox_color_list='',
                          _parameter_patch_loc='', _parameter='', _fig_label='', _fig_label_size='', _output_path='', _fname='', _fformat='', _dpi=150,
                          _font_size=21):
    """
    plot field data
    """
    # fig settings
    primary_fs = _font_size # primary fontsize
    secondary_fs = 0.85*_font_size # secondary fontsize
    plt.rc('font', size=primary_fs) # controls default text sizes
    
    if _rotated_crs:
        fig = plt.figure(figsize=(9, 18))
        proj = ccrs.RotatedPole(pole_latitude=_p_lat, pole_longitude=_p_lon)
    else:
        fig = plt.figure(figsize=(20, 10))
        proj = ccrs.PlateCarree()
    
    # axes settings
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent(_ax_set_extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=1)
    
    # grid settings
    gl=ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.2, draw_labels=False, x_inline=False, y_inline=False,)
    gl.left_labels = _left_labels
    gl.bottom_labels = _bottom_labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': primary_fs}
    gl.ylabel_style = {'size': primary_fs}
    if _rotated_crs:
        if _left_labels:
            gl.xlocator = mticker.FixedLocator(_xlocator)
            if _xlocator_mod==[90, 100]:
                gl.bottom_labels = not _bottom_labels
                gl.xlocator = mticker.FixedLocator(_xlocator_mod) #[90, 100]
                ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.2, xlocs=_xlocator)
        else:
            if _bottom_labels:
                if _left_labels:
                    gl.xlocator = mticker.FixedLocator(_xlocator)
                    ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.2, xlocs=_xlocator)
                else:
                    gl.xlocator = mticker.FixedLocator(_xlocator_mod)
                    ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.2, xlocs=_xlocator)
            else:
                gl.xlocator = mticker.FixedLocator(_xlocator)
            
        gl.ylocator = mticker.FixedLocator(_ylocator)
    
     
    # plotting ind au plate boundary
    if len(_plot_ind_aus_pb)!=0:
        ax.plot(_plot_ind_aus_pb[:,0], _plot_ind_aus_pb[:,1], color='brown', transform=ccrs.PlateCarree(), linewidth=_tlinewidth)
    
    # sum trench line and symbols
    if len(_sum_trench_coords)!=0:
        plt.plot(_sum_trench_coords[:,0], _sum_trench_coords[:,1], color=_sum_tline_color, transform=ccrs.PlateCarree(), linewidth=_tlinewidth)
        sum_marker_coords, sum_marker_angle = get_t_pts_m_angle(_sum_trench_coords)
        for i, coords in enumerate(sum_marker_coords):
            if sum_marker_angle[i]<=0:
                if _trench_marker=='triangle':
                    plt.plot(coords[0]+0.2, coords[1]+0.5, '>', marker=(3, 0, sum_marker_angle[i]), markersize=_tmarkersize, color=_sum_tline_color, 
                             transform=ccrs.PlateCarree())
                elif _trench_marker=='square':
                    plt.plot(coords[0]+0.2, coords[1]+0.5, 's', marker=(4, 0, sum_marker_angle[i]+60), markersize=_tmarkersize, markerfacecolor='none', 
                             color=_sum_tline_color, transform=ccrs.PlateCarree(), markeredgewidth=_markerwidth,)
            elif sum_marker_angle[i]>55:
                if _trench_marker=='triangle':
                    plt.plot(coords[0]+0.6, coords[1]+1, '>', marker=(3, 0, sum_marker_angle[i]-45), markersize=_tmarkersize, color=_sum_tline_color, 
                             transform=ccrs.PlateCarree())
                elif _trench_marker=='square':
                    plt.plot(coords[0]+1.0, coords[1]+1, 's', marker=(4, 0, sum_marker_angle[i]+60), markersize=_tmarkersize, markerfacecolor='none', 
                             color=_sum_tline_color, transform=ccrs.PlateCarree(), markeredgewidth=_markerwidth,)
            elif sum_marker_angle[i]>=0 and sum_marker_angle[i]<=55:
                if _trench_marker=='triangle':
                    plt.plot(coords[0]+0.2, coords[1]+1, '>', marker=(3, 0, sum_marker_angle[i]+135), markersize=_tmarkersize, color=_sum_tline_color, 
                             transform=ccrs.PlateCarree())
                elif _trench_marker=='square':
                    plt.plot(coords[0]+0.2, coords[1]+1, 's', marker=(4, 0, sum_marker_angle[i]+60), markersize=_tmarkersize, markerfacecolor='none', 
                             color=_sum_tline_color, transform=ccrs.PlateCarree(), markeredgewidth=_markerwidth,)
                    
    # him trench line and symbols
    if len(_him_trench_coords)!=0:
        plt.plot(_him_trench_coords[:,0], _him_trench_coords[:,1], color=_sum_tline_color, transform=ccrs.PlateCarree(), linewidth=_tlinewidth)
        him_marker_coords, him_marker_angle = get_t_pts_m_angle(_him_trench_coords, _n_pts=5)
        for i, coords in enumerate(him_marker_coords):
            if him_marker_angle[i]<=0:
                if _trench_marker=='triangle':
                    plt.plot(coords[0]+0.2, coords[1]+0.5, '>', marker=(3, 0, him_marker_angle[i]), markersize=_tmarkersize, color=_sum_tline_color, 
                             transform=ccrs.PlateCarree())
                elif _trench_marker=='square':
                    plt.plot(coords[0]+0.2, coords[1]+0.5, 's', marker=(4, 0, him_marker_angle[i]+60), markersize=_tmarkersize, markerfacecolor='none', 
                             color=_sum_tline_color, transform=ccrs.PlateCarree(), markeredgewidth=_markerwidth,)
    
    # plotting velocity vectors in layer 
    if len(_layer_coords_vel_list)!=0:
        for i, layer_coords_vel in enumerate(_layer_coords_vel_list):
            layer_coords = layer_coords_vel[0]
            layer_vel = rotate_vec_arr(layer_coords_vel[1][:,0:2], _rotate_angle_list[i])
            if _regrid_num==0:
                Q1 = ax.quiver(layer_coords[:,0][::_lvec_freq], layer_coords[:,1][::_lvec_freq], 
                               layer_vel[:,0][::_lvec_freq], layer_vel[:,1][::_lvec_freq], 
                               scale=_lvec_scale, zorder=3, transform=ccrs.PlateCarree(), 
                               color=_lvec_color_list[i], headwidth=1, headlength=0)
                Q2 = ax.quiver(layer_coords[:,0][::_lvec_freq], layer_coords[:,1][::_lvec_freq], 
                               -layer_vel[:,0][::_lvec_freq], -layer_vel[:,1][::_lvec_freq], 
                               scale=_lvec_scale, zorder=3, transform=ccrs.PlateCarree(), 
                               color=_lvec_color_list[i], headwidth=1, headlength=0)
                
            else:
                Q1 = ax.quiver(layer_coords[:,0][::_lvec_freq], layer_coords[:,1][::_lvec_freq], 
                               layer_vel[:,0][::_lvec_freq], layer_vel[:,1][::_lvec_freq], 
                               scale=_lvec_scale, width=_lvec_width, zorder=3, transform=ccrs.PlateCarree(), 
                               color=_lvec_color_list[i], label=_lvec_label_name[i], regrid_shape=_regrid_num)
        # qk1 = ax.quiverkey(Q1, 0.28, 0.275, 50, "50 mm/yr", coordinates='figure', color='k', zorder=10)
        
        # # adding patch around quiverkey
        # rect1 = patches.Rectangle((_ref_vec_patch_loc[0], _ref_vec_patch_loc[1]), _ref_vec_patch_loc[2], 
        #                           _ref_vec_patch_loc[3], alpha=0.5, ec='k', fc="white", linewidth=2, zorder=3, 
        #                           transform=ax.transAxes,)
        # ax.add_patch(rect1)
        
        # # legend location
        # if type(_lvec_legend_loc)!=int: 
        #     ax.legend(fontsize=primary_fs, ncol=_lvec_legend_col, loc='best', title=_lvec_legend_title,
        #                bbox_to_anchor=(_lvec_legend_loc[0], _lvec_legend_loc[1]))
        # else:
        #     ax.legend(fontsize=primary_fs, ncol=_lvec_legend_col, loc=_lvec_legend_loc, title=_lvec_legend_title)
    
    # plotting models bbox
    if len(_model_bbox_list)!=0:
        for i, model_bbox in enumerate(_model_bbox_list):
            ax.plot(model_bbox[:,0], model_bbox[:,1], color=_bbox_color_list[i])
    
    # parameter value display
    if len(_parameter_patch_loc)!=0: 
        ax.text(_parameter_patch_loc[0], _parameter_patch_loc[1], _parameter, horizontalalignment='center', 
                fontsize=primary_fs, transform=ax.transAxes,
                bbox=dict(facecolor='none', edgecolor='k', boxstyle='round, pad=0.2', alpha=0.5, fc="white"))
    
    # figure label
    if len(_fig_label)!=0: 
        ax.text(-0.045, 1.02, _fig_label, color='k', fontsize=primary_fs, transform=ax.transAxes,
                bbox=dict(facecolor='none', edgecolor='k', boxstyle='round, pad=0.2', alpha=0.5, fc="white"))
    
    # This make sure all fig are same size     
    plt.subplots_adjust(left=0.175, right=0.9, bottom=0.12, top=0.88, wspace=None, hspace=None) 
    
    # saving the plot
    if _fformat=='eps':
        fig.savefig(_output_path+_fname+"."+_fformat, format=_fformat, bbox_inches='tight')
    elif _fformat=='png':
        fig.savefig(_output_path+_fname+"."+_fformat, dpi=_dpi, bbox_inches='tight')
    elif _fformat=='pdf':
        fig.savefig(_output_path+_fname+"."+_fformat, format=_fformat, bbox_inches='tight')
    elif _fformat=='ps':
        fig.savefig(_output_path+_fname+"."+_fformat, format=_fformat, bbox_inches='tight', transparent=True)
        
    return

# loading data
pb_path = '/Users/tgol0006/phd_tg/phd_b2023/model_shapes_3d/make_boundary_pts/output_dir/'
sum_trench_coords = np.loadtxt(pb_path+'sum_trench_coords.txt', delimiter=',')
him_trench_coords = np.loadtxt(pb_path+'him_trench_coords.txt', delimiter=',')
ind_aus_pb_coords = np.loadtxt(pb_path+'ind_aus_pb2002_slab2.txt', delimiter=',')

# +
# plot settings
ax_extent_rot = [70.7, 111.6, -41.4, 31.6]

# rotation pole coordinates
p_lat=74 
p_lon=182.5

left_labels, bottom_labels = True, True

color_list = ['C2', 'k']

# loading box coordinates
model_bbox_list = []

model_shape_path = '/Users/tgol0006/phd_tg/phd_b2023/model_shapes_3d/'
model = 'model_61_120_-45_35_rot'
bbox_path = model_shape_path+model+'/qgis_files/'

bbox_coords = np.loadtxt(bbox_path+'box_coords.txt', delimiter=',')
model_bbox_list += [bbox_coords]

rotate_angle = -16
# -

tc = tds.transform_coords(61, 120, -45, 35)
resam_surface_rot_glld = np.round(tc.uw_xyz2geo_lonlatr(resampled_mesh.points), decimals=2)
print(resam_surface_rot_glld)
resam_surface_glld = rotate_arr(resam_surface_rot_glld, rotate_angle, tc)
print(resam_surface_glld)

ref_vec_patch_loc2 = [0.02, 0.02, 0.25, 0.07]
parameter_patch_loc=[0.86, 0.94]

# preparing data for the plot
layer_coords_vel_list = [(resam_surface_glld[:,0:2], resampled_mesh['SHmax'][:,0:2])]
color_list = [resampled_mesh['style_color']]

plot_field_data_bmrot(_rotated_crs=True, _p_lon=p_lon, _p_lat=p_lat, _ax_set_extent=ax_extent_rot, _left_labels=left_labels, _bottom_labels=bottom_labels, 
                      _xlabel_size=18, _ylabel_size=18, _xlocator=[60, 80, 100, 120], _ylocator=[-20, 0, 20, 40], _xlocator_mod='', _plot_ind_aus_pb=ind_aus_pb_coords, 
                      _tlinewidth=3, _sum_trench_coords=sum_trench_coords, _sum_tline_color='C4', _trench_marker='square', _tmarkersize=18, _markerwidth=2, 
                      _him_trench_coords=him_trench_coords, _layer_coords_vel_list=layer_coords_vel_list, _rotate_angle_list=[rotate_angle], _lvec_freq=1, 
                      _lvec_scale=75, _lvec_width=0.007, _lvec_color_list=color_list, _lvec_label_name='', _regrid_num=0, _ref_vec_patch_loc='', 
                      _lvec_legend_loc=1, _lvec_legend_col=1, _lvec_legend_title='', _model_bbox_list=model_bbox_list, _bbox_color_list=['cyan', 'k'],
                      _parameter_patch_loc='', _parameter='', _fig_label='', _fig_label_size=18, _output_path=output_dir, _fname='model3c_shmax', 
                      _fformat='pdf', _dpi=150,)


def load_nc_file(file_path):
    """
    By loading nc file creates new arrays from data within the dataset
    """
    dataset = Dataset(file_path)
    values = dataset.variables['z'][:] 
    lonsG = dataset.variables['x'][:] # Lons in degrees
    latsG = dataset.variables['y'][:] # Lats in degrees
    dataset.close()
    data_set = {}
    data_set['lon'] = lonsG
    data_set['lat'] = latsG
    data_set['data'] = values
    return data_set


def plot_field_data(_layer_coords='', _layer_data='', _ldata_freq='', _layer_vel='', _lvec_freq=1, _lvec_scale=400,
                    _plot_trench_vel=False, _trench_coords_vel='', _trench_coords_vel_orig='', _tvec_freq=15, 
                    _tvec_scale=150, _regrid_num=20, _colormap='', _vmin='', _vmax='', _plot_markersize='', 
                    _ax_set_extent='', _ax_set_xticks='', _rotated_crs=False, _p_lon='', _p_lat='', _cb_axis_label='',
                    _cb_pos=[0.36, 0.05, 0.3, 0.027], _cb_label_xpos=1.16, _cb_label_ypos=-0.2,  _sum_trench_coords='', 
                    _fig_label='', _fig_label_x='', _fig_label_y='', _parameter_patch_loc='', _parameter='', 
                    _fig_label_size=18, _cb_ticklabels='', _ref_vec_patch_loc='', _cb_display=True, 
                    _quiverkey_loc=[0.4, 0.15], _sum_tline_color='C4', _plot_model_box='', _plot_ind_aus_pb='', 
                    _him_trench_coords='', _cb_bounds='', _left_labels = True, _bottom_labels = True, 
                    _cb_save=False, _ax_text_size=18, _cb_name='', _para_text_size=18, _cb_text_size=22, 
                    _figsize_cb=(10/2, 10/2), _xlocator=[90, 100, 110, 120], _ylocator=[-10, 0, 10], 
                    _xlabel_size=18, _ylabel_size=18, _output_path='', _fname='', _fformat='', _trench_marker='',
                    _dpi=150, _cb_orient='vertical', _sum_tcoords_option=2, _markerwidth=2, _tmarkersize='', 
                    _tlinewidth=2, _xlocator_mod='', _par_color='', _contour_data='', _contour_levels='', _contour_cmap='', 
                    _ctr_vmin='', _ctr_vmax='', _up_pf_list='', _tip_indx_list='', _up_pf_color_list='', _up_pf_name='',
                    _up_pf_name_pos='', _font_size=18, _lvec_color=''):
    """
    plot field data or trench velocities
    """
    # fig settings
    primary_fs = _font_size # primary fontsize
    secondary_fs = 0.85*_font_size # secondary fontsize
    plt.rc('font', size=primary_fs) # controls default text sizes
    
    if _rotated_crs:
        fig = plt.figure(figsize=(7, 14))
        proj = ccrs.RotatedPole(pole_latitude=_p_lat, pole_longitude=_p_lon)
    else:
        fig = plt.figure(figsize=(20, 10))
        proj = ccrs.PlateCarree()
    
    # axes settings
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent(_ax_set_extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=1)
    
    # grid settings
    gl=ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.2, draw_labels=False, x_inline=False, y_inline=False,)
    gl.left_labels = _left_labels
    gl.bottom_labels = _bottom_labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': primary_fs} # font_size
    gl.ylabel_style = {'size': primary_fs} # font_size
    if _rotated_crs:
        if _left_labels:
            gl.xlocator = mticker.FixedLocator(_xlocator)
            if _xlocator_mod==[90, 100]:
                gl.bottom_labels = not _bottom_labels
                gl.xlocator = mticker.FixedLocator(_xlocator_mod) #[90, 100]
                ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.2, xlocs=_xlocator)
        else:
            if _bottom_labels:
                if _left_labels:
                    gl.xlocator = mticker.FixedLocator(_xlocator)
                    ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.2, xlocs=_xlocator)
                else:
                    gl.xlocator = mticker.FixedLocator(_xlocator_mod)
                    ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.2, xlocs=_xlocator)
            else:
                gl.xlocator = mticker.FixedLocator(_xlocator)
            
        gl.ylocator = mticker.FixedLocator(_ylocator)
        
            
    # plotting ind au plate boundary
    if len(_plot_ind_aus_pb)!=0:
        ax.plot(_plot_ind_aus_pb[:,0], _plot_ind_aus_pb[:,1], color='brown', transform=ccrs.PlateCarree(), linewidth=_tlinewidth)
        
    # sum trench line and symbols
    if len(_sum_trench_coords)!=0:
        ax.plot(_sum_trench_coords[:,0], _sum_trench_coords[:,1], color=_sum_tline_color, transform=ccrs.PlateCarree(), 
                 linewidth=_tlinewidth)
        if _sum_tcoords_option==1:
            sum_marker_coords, sum_marker_angle = get_t_pts_m_angle(_sum_trench_coords)
            for i, coords in enumerate(sum_marker_coords):
                if sum_marker_angle[i]<=0:
                    if _trench_marker=='triangle':
                        ax.plot(coords[0]+0.2, coords[1]+0.5, '>', marker=(3, 0, sum_marker_angle[i]), markersize=_tmarkersize,
                                 color=_sum_tline_color, transform=ccrs.PlateCarree())
                    elif _trench_marker=='square':
                        ax.plot(coords[0]+0.0, coords[1]+0.5, 's', marker=(4, 0, sum_marker_angle[i]+20), markersize=_tmarkersize, 
                                markerfacecolor='none', color=_sum_tline_color, transform=ccrs.PlateCarree(),
                                markeredgewidth=_markerwidth, zorder=10)
                elif sum_marker_angle[i]>55:
                    if _trench_marker=='triangle':
                        ax.plot(coords[0]+0.6, coords[1]+1, '>', marker=(3, 0, sum_marker_angle[i]-45), 
                                 markersize=_tmarkersize, color=_sum_tline_color, transform=ccrs.PlateCarree())
                    elif _trench_marker=='square':
                        ax.plot(coords[0]+0.4, coords[1]+1, 's', marker=(4, 0, sum_marker_angle[i]+20), markersize=_tmarkersize, 
                                markerfacecolor='none', color=_sum_tline_color, transform=ccrs.PlateCarree(),
                                markeredgewidth=_markerwidth, zorder=10)
                elif sum_marker_angle[i]>=0 and sum_marker_angle[i]<=55:
                    if _trench_marker=='triangle':
                        ax.plot(coords[0]+0.2, coords[1]+1, '>', marker=(3, 0, sum_marker_angle[i]+135), 
                                markersize=_tmarkersize, color=_sum_tline_color, transform=ccrs.PlateCarree())
                    elif _trench_marker=='square':
                        ax.plot(coords[0]-0.0, coords[1]+1, 's', marker=(4, 0, sum_marker_angle[i]+20), markersize=_tmarkersize, 
                                markerfacecolor='none', color=_sum_tline_color, transform=ccrs.PlateCarree(), 
                                markeredgewidth=_markerwidth, zorder=10)
        # sum trench line symbols
        if _sum_tcoords_option==2:
            indx = np.linspace(10, len(_sum_trench_coords)-10, num=10, endpoint=True, dtype=int)
            tree = spatial.KDTree(_sum_trench_coords)
            for pts in _sum_trench_coords[indx]:
                query_data = tree.query(pts, k=4)
                near_pts = _sum_trench_coords[query_data[1]]
                slope, intercept, r_value, p_value, std_err = linregress(near_pts[:,0], near_pts[:,1])
                slope_deg = math.degrees(math.atan(slope))
                if _trench_marker=='square':
                    if slope_deg < 2:
                        ax.plot(pts[0]+0.33, pts[1]+0.4, 's', marker=(4, 0, slope_deg-45), markersize=_tmarkersize, 
                                color=_sum_tline_color, transform=ccrs.PlateCarree(), markerfacecolor='none', 
                                markeredgewidth=_markerwidth, zorder=10)
                    elif slope_deg >= 2:
                        ax.plot(pts[0]+0.33, pts[1]+0.2, 's', marker=(4, 0, slope_deg-45), markersize=_tmarkersize, 
                                color=_sum_tline_color, transform=ccrs.PlateCarree(), markerfacecolor='none', 
                                markeredgewidth=_markerwidth, zorder=10)
    
    # him trench line and symbols
    if len(_him_trench_coords)!=0:
        ax.plot(_him_trench_coords[:,0], _him_trench_coords[:,1], color=_sum_tline_color, transform=ccrs.PlateCarree(), 
                 linewidth=_tlinewidth)
        him_marker_coords, him_marker_angle = get_t_pts_m_angle(_him_trench_coords, _n_pts=5)
        for i, coords in enumerate(him_marker_coords):
            if him_marker_angle[i]<=0:
                if _trench_marker=='triangle':
                    ax.plot(coords[0]+0.2, coords[1]+0.5, '>', marker=(3, 0, him_marker_angle[i]), 
                            markersize=_tmarkersize, color=_sum_tline_color, transform=ccrs.PlateCarree())
                elif _trench_marker=='square':
                    ax.plot(coords[0]+0.2, coords[1]+0.5, 's', marker=(4, 0, him_marker_angle[i]+45), markersize=_tmarkersize, 
                            markerfacecolor='none', color=_sum_tline_color, transform=ccrs.PlateCarree(), zorder=10,
                            markeredgewidth=_markerwidth)
                    
    # # plotting model boundary box
    # if len(_plot_model_box)!=0:
    #     ax.plot(_plot_model_box[:,0], _plot_model_box[:,1], color='k', transform=ccrs.PlateCarree(), linewidth=2)
        
    # # plotting layer or trench data
    # if _plot_trench_vel:
    #     # plotting trench velocities
    #     Q1 = ax.quiver(_trench_coords_vel[:,0][::_tvec_freq], _trench_coords_vel[:,1][::_tvec_freq], 
    #                    _trench_coords_vel[:,2][::_tvec_freq], _trench_coords_vel[:,3][::_tvec_freq], 
    #                    scale=_tvec_scale, width=0.004, zorder=3, transform=ccrs.PlateCarree(), )
    #     qk1 = ax.quiverkey(Q1, _quiverkey_loc[0], _quiverkey_loc[1], 10, "10 mm/yr", coordinates='figure', color='k', 
    #                        zorder=10)
        
    #     # adding patch around quiverkey
    #     rect1 = patches.Rectangle((_ref_vec_patch_loc[0], _ref_vec_patch_loc[1]), _ref_vec_patch_loc[2], 
    #                               _ref_vec_patch_loc[3], alpha=0.5, ec='k', fc="white", linewidth=2, zorder=3, 
    #                               transform=ax.transAxes,)
    #     ax.add_patch(rect1)
        
    #     # plotting trench velocities original data
    #     if len(_trench_coords_vel_orig)!=0:
    #         Q2 = ax.quiver(_trench_coords_vel_orig[:,0][::_tvec_freq], _trench_coords_vel_orig[:,1][::_tvec_freq], 
    #                        _trench_coords_vel_orig[:,3][::_tvec_freq], _trench_coords_vel_orig[:,4][::_tvec_freq], 
    #                        scale=_tvec_scale, width=0.004, zorder=3, transform=ccrs.PlateCarree(), color='b')
    # else:
    #     # plotting layer data
    #     if len(_cb_bounds)!=0:
    #         bounds = _cb_bounds # uneven bounds changes the colormapping
    #         norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    #         im1 = ax.scatter(_layer_coords[:,0][::_ldata_freq], _layer_coords[:,1][::_ldata_freq], s=_plot_markersize, 
    #                          c=_layer_data[::_ldata_freq], marker='H', cmap=_colormap, #vmin=_vmin, vmax=_vmax, 
    #                          zorder=1, transform=ccrs.PlateCarree(), norm=norm)
    #     else:
    #         im1 = ax.scatter(_layer_coords[:,0][::_ldata_freq], _layer_coords[:,1][::_ldata_freq], s=_plot_markersize, 
    #                          c=_layer_data[::_ldata_freq], marker='H', cmap=_colormap, vmin=_vmin, vmax=_vmax, 
    #                          zorder=1, transform=ccrs.PlateCarree())
        
    #     # colorbar settings
    #     if _cb_display:
    #         cax = plt.axes(_cb_pos) # [left, bottom, width, height]
    #         cb = plt.colorbar(im1, cax, orientation='horizontal')
    #         if len(_cb_ticklabels)!=0:
    #             cb.ax.set_xticklabels(_cb_ticklabels) # set ticks of your format
    #         cb.ax.set_title(_cb_axis_label, fontsize=primary_fs, x=_cb_label_xpos, y=_cb_label_ypos) # font_size
    
    # plotting velocity vectors in layer 
    if len(_layer_vel)!=0:
        Q1 = ax.quiver(_layer_coords[:,0][::_lvec_freq], _layer_coords[:,1][::_lvec_freq], 
                       _layer_vel[:,0][::_lvec_freq], _layer_vel[:,1][::_lvec_freq], 
                       scale=_lvec_scale, zorder=10, transform=ccrs.PlateCarree(), 
                       color=_lvec_color, headwidth=1, headlength=0)
        Q2 = ax.quiver(_layer_coords[:,0][::_lvec_freq], _layer_coords[:,1][::_lvec_freq], 
                       -_layer_vel[:,0][::_lvec_freq], -_layer_vel[:,1][::_lvec_freq], 
                       scale=_lvec_scale, zorder=10, transform=ccrs.PlateCarree(), 
                       color=_lvec_color, headwidth=1, headlength=0)
    #     qk1 = ax.quiverkey(Q1, _quiverkey_loc[0], _quiverkey_loc[1], 50, "50 mm/yr", coordinates='figure', color='k', 
    #                        zorder=10)
    #     # adding patch around quiverkey
    #     rect1 = patches.Rectangle((_ref_vec_patch_loc[0], _ref_vec_patch_loc[1]), _ref_vec_patch_loc[2], 
    #                               _ref_vec_patch_loc[3], alpha=0.5, ec='k', fc="white", linewidth=2, zorder=3, 
    #                               transform=ax.transAxes,)
    #     ax.add_patch(rect1)
    
    # plotting contours
    if len(_contour_data)!=0:    
        im2 = ax.contour(_contour_data['lon'], _contour_data['lat'], -_contour_data['data'], levels=_contour_levels, 
                         transform=ccrs.PlateCarree(), cmap=_contour_cmap, vmin=_ctr_vmin, vmax=_ctr_vmax, linewidths=3)
        ax.clabel(im2, inline=True, fontsize=primary_fs, fmt='%3d', inline_spacing=-1) # font_size
        
    # parameter value display
    if len(_parameter_patch_loc)!=0: 
        ax.text(_parameter_patch_loc[0], _parameter_patch_loc[1], _parameter, horizontalalignment='center', 
                fontsize=primary_fs, transform=ax.transAxes, color=_par_color,
                bbox=dict(facecolor='none', edgecolor='k', boxstyle='round, pad=0.2', alpha=1.0, fc="white")) # font_size
    
    # # figure label
    # if len(_fig_label)!=0: 
    #     ax.text(_fig_label_x, _fig_label_y, _fig_label, color='k', fontsize=primary_fs, transform=ax.transAxes,
    #             bbox=dict(facecolor='none', edgecolor='k', boxstyle='round, pad=0.2', alpha=1.0, fc="white")) # font_size
         
    # ax.tick_params(axis='both', direction='in',) # turn tickmarks inward
    # plt.subplots_adjust(left=0.175, right=0.9, bottom=0.12, top=0.88, wspace=None, hspace=None) # make sure all fig are same size               
    
    # # Plot UP pf coords
    # if len(_up_pf_list)!=0:
    #     for i, pf_coords in enumerate(_up_pf_list):
    #         ax.plot(pf_coords[:,0], pf_coords[:,1], color=_up_pf_color_list[i], linewidth=2, transform=ccrs.PlateCarree())
    #         ax.plot(pf_coords[0][0], pf_coords[0][1], color='g', marker='o', markersize=12, transform=ccrs.PlateCarree())
    #         ax.plot(pf_coords[-1][0], pf_coords[-1][1], color='r', marker='o', markersize=12, transform=ccrs.PlateCarree())
    #         ax.plot(pf_coords[_tip_indx_list[i][0]][0], pf_coords[_tip_indx_list[i][0]][1], color='b', marker='X', markersize=12, transform=ccrs.PlateCarree())
    #         ax.plot(pf_coords[_tip_indx_list[i][1]][0], pf_coords[_tip_indx_list[i][1]][1], color='b', marker='X', markersize=12, transform=ccrs.PlateCarree())
    #         ax.text(pf_coords[0][0]+_up_pf_name_pos[i][0], pf_coords[0][1]+_up_pf_name_pos[i][1], _up_pf_name[i], color=_up_pf_color_list[i], fontsize=primary_fs, 
    #                 transform=ccrs.PlateCarree(), bbox=dict(facecolor='none', edgecolor='k', boxstyle='round, pad=0.2', alpha=1.0, fc="white")) # font_size
    # saving the plot
    if _fformat=='eps':
        fig.savefig(_output_path+_fname+"."+_fformat, format=_fformat, bbox_inches='tight')
    elif _fformat=='png':
        fig.savefig(_output_path+_fname+"."+_fformat, dpi=_dpi, bbox_inches='tight')
    elif _fformat=='pdf':
        fig.savefig(_output_path+_fname+"."+_fformat, format=_fformat, bbox_inches='tight')
    elif _fformat=='ps':
        fig.savefig(_output_path+_fname+"."+_fformat, format=_fformat, bbox_inches='tight', transparent=True)
        
    # save the colorbar separately
    if _cb_save: # vertical colorbar
        plt.figure(figsize=_figsize_cb)
        plt.rc('font', size=primary_fs) # font_size
        if len(_cb_bounds)!=0:
            a = np.array([bounds])
            img = plt.imshow(a, cmap=_colormap, norm=norm)
        else:
            a = np.array([[_vmin,_vmax]])
            img = plt.imshow(a, cmap=_colormap)
            
        plt.gca().set_visible(False)
        if _cb_orient=='vertical':
            cax = plt.axes([0.1, 0.2, 0.06, 1.15])
            cb = plt.colorbar(orientation='vertical', cax=cax)
            cb.ax.set_title(_cb_axis_label, fontsize=primary_fs, x=_cb_label_xpos, y=_cb_label_ypos, rotation=90) # font_size
            if _fformat=='png':
                plt.savefig(_output_path+_fname+'_cbvert.'+_fformat, dpi=150, bbox_inches='tight')
            elif _fformat=='pdf':
                plt.savefig(_output_path+_fname+"_cbvert."+_fformat, format=_fformat, bbox_inches='tight')
        if _cb_orient=='horizontal':
            cax = plt.axes([0.1, 0.2, 1.15, 0.06])
            cb = plt.colorbar(orientation='horizontal', cax=cax)
            cb.ax.set_title(_cb_axis_label, fontsize=primary_fs, x=_cb_label_xpos, y=_cb_label_ypos) # font_size
            if _fformat=='png':
                plt.savefig(_output_path+_fname+'_cbhorz.'+_fformat, dpi=150, bbox_inches='tight')
            elif _fformat=='pdf':
                plt.savefig(_output_path+_fname+"_cbhorz."+_fformat, format=_fformat, bbox_inches='tight')
    return

# +
# plot settings
ax_extent_rot = [100, 107, -17.5, 19.4]

# rotation pole coordinates
p_lat=115 
p_lon=180

left_labels, bottom_labels = True, True

xlocator_mod = [110, 120]
# -

# contour data and settings
slab_dep_path = '/Users/tgol0006/phd_tg/PhD_TG/data_sets/Slab2_AComprehe/Slab2Distribute_Mar2018/'
sum_slab_dep = load_nc_file(slab_dep_path+'sum_slab2_dep_02.23.18.grd')
contour_levels = [90, 200, 400, 600]
ctr_cb_ticks = [90, 200, 400, 600]
contour_cmap = mpl.cm.viridis # scm.turku
ctr_vmin=0
ctr_vmax=660

# plate boundary dataset
pb_path = '/Users/tgol0006/phd_tg/phd_b2023/model_shapes_3d/make_boundary_pts/output_dir/'
sum_tcoords_orig = np.loadtxt(pb_path+'sum_trench_coords.txt', delimiter=',')

# base plot
plot_field_data(_p_lat=p_lat, _p_lon=p_lon, _ax_set_extent=ax_extent_rot, _rotated_crs=True,
                _left_labels=left_labels, _bottom_labels=bottom_labels, _xlocator_mod=xlocator_mod,
                _sum_trench_coords=sum_tcoords_orig, _trench_marker='square', _markerwidth=3, _tmarkersize=18, 
                _sum_tcoords_option=1, _tlinewidth=4,
                _contour_data='', _contour_levels=contour_levels, _contour_cmap=contour_cmap, _ctr_vmin=ctr_vmin, 
                _ctr_vmax=ctr_vmax,
                _output_path=output_dir, _fname='model3c_shmax_along_trench', _fformat='pdf', 
                _layer_coords=resam_surface_glld[:,0:2], _layer_vel=rotate_vec_arr(resampled_mesh['SHmax'][:,0:2], rotate_angle), 
                _lvec_color=resampled_mesh['style_color'], _lvec_scale=75)

# +
# Convert to 3D (z = 0)
points = np.hstack([sum_trench_coords, np.zeros((sum_trench_coords.shape[0], 1))])

# Create line
n_points = points.shape[0]
lines = np.hstack([[n_points], np.arange(n_points)])

polyline = pv.PolyData()
polyline.points = points
polyline.lines = lines

# Create a spline to smooth it (optional)
spline = Spline(polyline.points, n_points=90)

# Compute tangents
spline_tangents = np.gradient(spline.points, axis=0)

# Compute normals in the XY plane (2D cross product with Z unit vector)
# 2D normal = (-dy, dx)
tangents_xy = spline_tangents[:, :2]
normals_xy = np.zeros_like(tangents_xy)
normals_xy[:, 0] = -tangents_xy[:, 1]
normals_xy[:, 1] = tangents_xy[:, 0]

# Normalize
norms = np.linalg.norm(normals_xy, axis=1, keepdims=True)
normals_xy = normals_xy / norms

# Add Z = 0 back
normals = np.hstack([normals_xy, np.zeros((normals_xy.shape[0], 1))])

# Offset the spline points by some distance along normals
distance = 4  # degrees (lon/lat) or km if projected
offset_points = spline.points + distance * normals

# Create offset line
offset_line = pv.PolyData()
offset_line.points = offset_points
offset_line.lines = np.hstack([[offset_points.shape[0]], np.arange(offset_points.shape[0])])

# # Plot original and offset lines
# pl = pv.Plotter()
# pl.add_mesh(spline, color='k', line_width=3, label='Original Line')
# pl.add_mesh(offset_line, color='darkgreen', line_width=3, label='Offset Line')
# pl.add_legend()
# pl.show(cpos='xy')
# -

# Get only the XY (2D) part for polygon creation
poly_points = np.vstack([points[:, :2], offset_points[::-1, :2]])
polygon_path = Path(poly_points) # Create polygon path
inside_poly = polygon_path.contains_points(resam_surface_glld[:, 0:2])  # boolean array

parameter_patch_loc=[0.86, 0.96]
parameter = 'Model3c'

# base plot
plot_field_data(_p_lat=p_lat, _p_lon=p_lon, _ax_set_extent=ax_extent_rot, _rotated_crs=True,
                _left_labels=left_labels, _bottom_labels=bottom_labels, _xlocator_mod=xlocator_mod,
                _sum_trench_coords=sum_tcoords_orig, _trench_marker='square', _markerwidth=3, _tmarkersize=18, 
                _sum_tcoords_option=1, _tlinewidth=4,
                _contour_data=sum_slab_dep, _contour_levels=contour_levels, _contour_cmap=contour_cmap, _ctr_vmin=ctr_vmin, _ctr_vmax=ctr_vmax,
                _output_path=output_dir, _fname='model3c_shmax_along_trench_arc', _fformat='pdf', 
                _layer_coords=resam_surface_glld[:,0:2][inside_poly], _layer_vel=rotate_vec_arr(resampled_mesh['SHmax'][:,0:2][inside_poly], rotate_angle), 
                _lvec_color=resampled_mesh['style_color'][inside_poly], _lvec_scale=50, 
                _parameter=parameter, _parameter_patch_loc=parameter_patch_loc, _par_color='C2')



