#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a static ban mask (GeoTIFF) from an AOI GeoJSON using OpenStreetMap data.
- Water (polygons + buffered waterways/coastlines)
- Roads (buffered by given meters)
Output: single-band GeoTIFF where 255 = banned, 0 = free (inside AOI only).

Example:
  python build_osm_mask.py \
    --aoi aoi_izmit.geojson \
    --out static_ban_izmit.tif \
    --res 2.5 \
    --road-buffer 8 \
    --water-line-buffer 12 \
    --osm-buffer 200

Dependencies:
  pip install osmnx geopandas shapely pyproj rasterio numpy
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import osmnx as ox

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------
# Helpers
# ------------------------
def utm_crs_from_lonlat(lon: float, lat: float) -> str:
    zone = int((lon + 180) / 6) + 1
    south = lat < 0
    epsg = 32700 + zone if south else 32600 + zone
    return f"EPSG:{epsg}"

def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf

def osm_geoms_in_polygon(poly_wgs, tags: dict) -> gpd.GeoDataFrame:
    # OSMnx settings for reliability
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 180
    ox.settings.overpass_rate_limit = True
    return ox.geometries_from_polygon(poly_wgs, tags)

def dissolve_and_buffer(lines_gdf: gpd.GeoDataFrame, buffer_m: float, to_crs: str) -> gpd.GeoSeries:
    if lines_gdf.empty:
        return gpd.GeoSeries([], crs=to_crs)
    lines = lines_gdf.to_crs(to_crs)
    geom = unary_union(lines.geometry)
    if geom is None or geom.is_empty:
        return gpd.GeoSeries([], crs=to_crs)
    buf = gpd.GeoSeries([geom], crs=to_crs).buffer(buffer_m, cap_style=2, join_style=2)
    return buf

def dissolve_polygons(poly_gdf: gpd.GeoDataFrame, to_crs: str) -> gpd.GeoSeries:
    if poly_gdf.empty:
        return gpd.GeoSeries([], crs=to_crs)
    polys = poly_gdf.to_crs(to_crs)
    geom = unary_union(polys.geometry)
    if geom is None or geom.is_empty:
        return gpd.GeoSeries([], crs=to_crs)
    return gpd.GeoSeries([geom], crs=to_crs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True, type=str, help="AOI polygon GeoJSON (single or multi polygon).")
    ap.add_argument("--out", default="static_ban.tif", type=str, help="Output GeoTIFF path.")
    ap.add_argument("--res", default=2.5, type=float, help="Resolution in meters per pixel (suggest 2~5).")
    ap.add_argument("--road-buffer", default=8.0, type=float, help="Road buffer (meters).")
    ap.add_argument("--water-line-buffer", default=12.0, type=float, help="Waterway/coastline line buffer (meters).")
    ap.add_argument("--osm-buffer", default=200.0, type=float, help="Expand AOI by X meters before querying OSM (to avoid edge cuts).")
    ap.add_argument("--road-keys", default="motorway,trunk,primary,secondary,tertiary,residential,service,unclassified,living_street",
                    type=str, help="Comma-separated highway types to include.")
    args = ap.parse_args()

    aoi_path = Path(args.aoi)
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI not found: {aoi_path}")

    # 1) Load AOI
    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise RuntimeError("AOI is empty.")
    aoi = ensure_wgs84(aoi)
    aoi_union = unary_union(aoi.geometry)
    if aoi_union.geom_type not in ("Polygon", "MultiPolygon"):
        raise RuntimeError(f"AOI must be polygonal, got: {aoi_union.geom_type}")

    # 2) Pick a metric CRS (UTM by centroid)
    cen = aoi_union.centroid
    utm = utm_crs_from_lonlat(cen.x, cen.y)

    # 3) Buffer AOI for OSM query (to avoid cutting features at the edge)
    aoi_utm = gpd.GeoSeries([aoi_union], crs="EPSG:4326").to_crs(utm)
    aoi_utm_buff = aoi_utm.buffer(args.osm_buffer)
    aoi_wgs_for_osm = aoi_utm_buff.to_crs("EPSG:4326").iloc[0]

    # 4) OSM downloads
    # Roads
    road_keys = [k.strip() for k in args.road_keys.split(",") if k.strip()]
    road_tags = {"highway": True}
    gdf_roads = osm_geoms_in_polygon(aoi_wgs_for_osm, road_tags)
    if not gdf_roads.empty and "highway" in gdf_roads.columns:
        # Keep permitted highway classes
        mask = gdf_roads["highway"].astype(str).isin(road_keys)
        gdf_roads = gdf_roads[mask]
    # Keep only linework
    gdf_roads = gdf_roads[gdf_roads.geom_type.isin(["LineString", "MultiLineString"])].copy()

    # Water (polygons + lines)
    water_tags = {
        "natural": ["water", "sea", "coastline"],
        "water": True,                 # lake, reservoir, river, sea, lagoon...
        "waterway": True,              # rivers/streams as lines
        "landuse": ["reservoir", "basin"]
    }
    gdf_water_all = osm_geoms_in_polygon(aoi_wgs_for_osm, water_tags)
    gdf_water_poly = gdf_water_all[gdf_water_all.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf_water_line = gdf_water_all[gdf_water_all.geom_type.isin(["LineString", "MultiLineString"])].copy()

    # 5) Dissolve/buffer
    #   Roads -> buffer
    road_buf = dissolve_and_buffer(gdf_roads, args.road_buffer, utm)
    #   Water polygons -> dissolve
    water_poly = dissolve_polygons(gdf_water_poly, utm)
    #   Water lines (rivers, coastline) -> buffer
    water_line_buf = dissolve_and_buffer(gdf_water_line, args.water_line_buffer, utm)

    # 6) Merge all bans
    geoms = []
    for gs in (road_buf, water_poly, water_line_buf):
        if gs is not None and not gs.empty:
            geoms.append(gs.iloc[0])
    if not geoms:
        raise RuntimeError("No OSM geometries found for given AOI/tags. Try increasing --osm-buffer or loosening tags.")

    merged = unary_union(geoms)

    # 7) Clip to AOI (projected)
    aoi_proj = gpd.GeoSeries([aoi_union], crs="EPSG:4326").to_crs(utm)
    merged = merged.intersection(aoi_proj.iloc[0])

    if merged.is_empty:
        raise RuntimeError("Merged ban geometry empty after clipping to AOI.")

    # 8) Rasterize
    minx, miny, maxx, maxy = aoi_proj.total_bounds
    res = float(args.res)
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    transform = from_origin(minx, maxy, res, res)

    # Ban mask: 255 inside merged, 0 elsewhere
    ban = rasterize(
        [(merged, 255)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    # Ensure outside AOI is 0
    aoi_mask = rasterize(
        [(aoi_proj.iloc[0], 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    ban[aoi_mask == 0] = 0

    # 9) Save GeoTIFF
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        outp,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint8,
        crs=utm,
        transform=transform,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(ban, 1)

    # 10) Quick preview as PNG (optional)
    try:
        import imageio.v2 as iio
        preview = (ban > 0).astype(np.uint8) * 255
        iio.imwrite(str(outp.with_suffix(".png")), preview)
    except Exception:
        pass

    print(f"✔ Saved ban mask: {outp}")
    print(f"   - CRS: {utm}")
    print(f"   - Size: {width} x {height} px  (res={res} m/px)")
    print(f"   - Preview (optional): {outp.with_suffix('.png')}")

if __name__ == "__main__":
    main()
