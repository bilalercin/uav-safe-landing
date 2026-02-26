# check_mask_quick.py
import json, rasterio
from shapely.geometry import shape, box
import matplotlib.pyplot as plt

AOI = "aoi_izmit.geojson"
TIF = "static_ban_izmit.tif"

# AOI oku (WGS84)
aoi = shape(json.load(open(AOI))["features"][0]["geometry"])

# TIF bbox (maskenin koordinat sistemi)
with rasterio.open(TIF) as ds:
    left, bottom, right, top = ds.bounds
    crs = ds.crs

print("Mask CRS:", crs)
print("Mask bounds:", (left, bottom, right, top))

# Sadece görsel kontrol (bbox)
fig, ax = plt.subplots(1,1, figsize=(6,6))
ax.set_title("Mask BBOX (proj. CRS) – AOI WGS84 değildir")
ax.plot([left,right,right,left,left],[bottom,bottom,top,top,bottom], label="mask bbox")
ax.legend(); plt.tight_layout(); plt.show()
