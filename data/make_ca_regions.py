from pathlib import Path
import geopandas as gpd

IN_GJ = Path("data/ca_regions_3857.geojson") 
OUT_GJ = Path("data/ca_regions.geojson")

gdf = gpd.read_file(IN_GJ)
gdf.set_crs("EPSG:3857", inplace=True)
gdf = gdf.to_crs("EPSG:4326")

gdf["region_id"] = gdf["CountyName"]

gdf = gdf[["region_id", "geometry"]]

OUT_GJ.parent.mkdir(parents=True, exist_ok=True)
gdf.to_file(OUT_GJ, driver="GeoJSON")
print(f"Wrote {OUT_GJ} with {len(gdf)} counties (EPSG:4326).")