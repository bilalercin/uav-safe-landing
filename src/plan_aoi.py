# src/plan_aoi.py
import json, math, csv, argparse
from pathlib import Path
from shapely.geometry import shape, Polygon, LineString
from shapely.ops import unary_union
from pyproj import CRS, Transformer

def load_aoi(path):
    gj = json.load(open(path))
    geom = shape(gj["features"][0]["geometry"])
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return unary_union([g for g in geom.geoms])
    raise ValueError("AOI polygon bekleniyor")

def ground_footprint(alt, hfov_deg, vfov_deg):
    # basit dik/nadir model: zemin dikdörtgen ayak izi
    w = 2.0 * alt * math.tan(math.radians(hfov_deg/2.0))
    h = 2.0 * alt * math.tan(math.radians(vfov_deg/2.0))
    return w, h

def utm_crs_for(lon, lat):
    zone = int((lon + 180)/6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def sweep_lines_in_polygon(poly_utm: Polygon, step_x: float):
    minx, miny, maxx, maxy = poly_utm.bounds
    x = minx
    lines = []
    flip = False
    while x <= maxx:
        line = LineString([(x, miny), (x, maxy)])
        segs = poly_utm.intersection(line)
        if segs.is_empty:
            x += step_x; continue
        # split into segments list
        parts = list(segs.geoms) if hasattr(segs, "geoms") else [segs]
        # zigzag yönü
        if flip: parts = parts[::-1]
        lines.extend(parts)
        flip = not flip
        x += step_x
    return lines

def write_csv(path, pts_llh):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat","lon","alt"])
        for lat, lon, alt in pts_llh:
            w.writerow([f"{lat:.7f}", f"{lon:.7f}", f"{alt:.2f}"])

def try_write_kml(path, pts_llh):
    try:
        import simplekml
    except ImportError:
        return
    kml = simplekml.Kml()
    ls = kml.newlinestring(name="Lawnmower")
    ls.coords = [(lon, lat, alt) for lat,lon,alt in pts_llh]
    ls.altitudemode = simplekml.AltitudeMode.relativetoground
    ls.style.linestyle.width = 3
    kml.save(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--alt", type=float, required=True, help="AGL meters")
    ap.add_argument("--hfov", type=float, default=84)
    ap.add_argument("--vfov", type=float, default=63)
    ap.add_argument("--overlap-x", type=float, default=0.30)
    ap.add_argument("--overlap-y", type=float, default=0.30)
    ap.add_argument("--speed", type=float, default=7.0)
    ap.add_argument("--out", required=True, help="CSV or KML path")
    args = ap.parse_args()

    aoi_ll = load_aoi(args.aoi)
    # AOI merkezine göre UTM seç
    lon0, lat0 = aoi_ll.centroid.x, aoi_ll.centroid.y
    crs_ll = CRS.from_epsg(4326)
    crs_utm = utm_crs_for(lon0, lat0)
    to_utm = Transformer.from_crs(crs_ll, crs_utm, always_xy=True).transform
    to_ll  = Transformer.from_crs(crs_utm, crs_ll, always_xy=True).transform

    aoi_utm = Polygon([to_utm(x,y) for x,y in aoi_ll.exterior.coords])
    # Ayak izi ve adımlar
    gw, gh = ground_footprint(args.alt, args.hfov, args.vfov)
    step_x = gw * (1.0 - args.overlap_x)
    # step_y kullanılmıyor (şerit içi numune); waypoint aralığı ~ gh*(1-overlap_y) alınabilir

    lines = sweep_lines_in_polygon(aoi_utm, step_x=step_x)

    # Waypoint üret (her hat baş-son)
    pts_utm = []
    for i, seg in enumerate(lines):
        if seg.length < 1.0:
            continue
        xs, ys = list(seg.coords[0]), list(seg.coords[-1])
        if i % 2 == 0:
            seq = [tuple(xs), tuple(ys)]
        else:
            seq = [tuple(ys), tuple(xs)]
        pts_utm.extend(seq)

    # UTM -> latlon
    pts_llh = []
    for x,y in pts_utm:
        lon, lat = to_ll(x,y)
        pts_llh.append((lat, lon, args.alt))

    outp = Path(args.out)
    if outp.suffix.lower() == ".kml":
        try_write_kml(str(outp), pts_llh)
        # CSV de bırak
        write_csv(str(outp.with_suffix(".csv")), pts_llh)
    else:
        write_csv(str(outp), pts_llh)
        try_write_kml(str(outp.with_suffix(".kml")), pts_llh)

    print(f"Waypoints: {len(pts_llh)}")
    print(f"Saved: {outp} (+paired CSV/KML)")
if __name__ == "__main__":
    main()
