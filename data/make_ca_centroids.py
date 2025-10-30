from pathlib import Path
import geopandas as gpd
import pandas as pd

# Input: reprojected GeoJSON
IN_GJ = Path("data/ca_regions.geojson")
OUT_CSV = Path("data/ca_centroids.csv")

gdf = gpd.read_file(IN_GJ)

# Calculate centroids in an equal-area projection, then convert to WGS84
gdf_proj = gdf.to_crs("EPSG:3310")  # California Albers
centroids = gdf_proj.centroid.to_crs("EPSG:4326")

# Build DataFrame
df = pd.DataFrame({
    "region_id": gdf["region_id"],
    "lat": centroids.y,
    "lon": centroids.x
}).sort_values("region_id")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV} with {len(df)} centroids.")