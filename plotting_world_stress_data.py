import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

path='./wsm_database/Smoothed_global_stress_maps/'
pd_df = pd.read_csv(f'{path}mean_SHmax_r500_2.dat', delimiter='\t')
print(pd_df.head())
print(pd_df.info())

# +
# Longitude and latitude of grid points
lon = pd_df['LON'].to_numpy()
lat = pd_df['LAT'].to_numpy()
# Mean SHmax orientation (in degrees from North; values between 0 and 180)
orientation = pd_df['SHmax'].to_numpy()

# Convert orientation from degrees to radians
theta = np.deg2rad(orientation)
# Compute arrow components (0° points North so u = sin(theta), v = cos(theta))
u = np.sin(theta)
v = np.cos(theta)

# +
# Create a figure and set a PlateCarree projection (lon/lat)
fig = plt.figure(figsize=(14, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
# Set map extent slightly larger than your data bounds
ax.set_extent([0, 359, -90, 90], crs=ccrs.PlateCarree())

# Add some basic map features
ax.add_feature(cfeature.COASTLINE)

# # Plot the grid points as markers (optional)
# ax.scatter(lon, lat, color='blue', s=50, transform=ccrs.PlateCarree(), zorder=5)

# Plot arrows to represent the mean SHmax orientation at each grid point.
# Adjust the scale parameter to control arrow length.
q = ax.quiver(lon, lat, u, v, scale=150, color='red', transform=ccrs.PlateCarree(), zorder=6, 
              headlength=0, headwidth=0, headaxislength=0)
q2 = ax.quiver(lon, lat, -u, -v, scale=150, color='red', transform=ccrs.PlateCarree(), zorder=6, 
              headlength=0, headwidth=0, headaxislength=0)
ax.quiverkey(q, 0.9, 0.95, 1, "1 unit", labelpos='E', coordinates='figure')

# Add a title and show the plot
plt.title('Mean SHmax Orientation')
plt.show()

# -


