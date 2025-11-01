# generate_heatmap_data.py
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

# Deep Ellum bounding box (approximate)
min_lat, max_lat = 32.778, 32.786
min_lon, max_lon = -96.790, -96.775

# Generate 1,500 random points within Deep Ellum
np.random.seed(42)  # for reproducibility
n_points = 1500

lats = np.random.uniform(min_lat, max_lat, n_points)
lons = np.random.uniform(min_lon, max_lon, n_points)

# Create geometry
geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]

# Create DataFrame
df = pd.DataFrame({
    "latitude": lats,
    "longitude": lons,
    "crime_score": np.random.gamma(shape=2.0, scale=1.5, size=n_points).round(2),  # skewed positive scores
    "cluster": np.random.choice([1, 2, 3, 4, 5], size=n_points),
})

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Optional: simplify geometry (not needed for points, but good practice)
# gdf.geometry = gdf.simplify(tolerance=0.0001)

# Ensure output directory exists
import os
os.makedirs("data", exist_ok=True)

# Save to GeoJSON
output_path = "data/heatmap_points.geojson"
gdf.to_file(output_path, driver="GeoJSON")

print(f"✅ Successfully created {output_path} with {len(gdf)} points.")