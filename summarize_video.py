# summarize_video.py
import json, sys, csv, statistics as st, re, pathlib
import matplotlib.pyplot as plt

def _norm_entry(e):
    if not isinstance(e, dict): return None
    fr = e.get("frame") or e.get("idx") or e.get("i")
    if fr is None:
        for k in ("file","path","name","im","img"):
            s = e.get(k)
            if isinstance(s, str):
                m = re.search(r"(\d{3,})", s)
                if m: fr = int(m.group(1)); break
    r  = e.get("radius")
    if r is None:
        for k in ("r","safe_radius","rad","radius_px"):
            if k in e: r = e[k]; break
    ok = e.get("ok")
    if ok is None:
        for k in ("safe","is_safe","flag"):
            if k in e: ok = bool(e[k]); break
    if ok is None and isinstance(r,(int,float)): ok = r > 0
    if r is None: r = 0
    if fr is None: return None
    try:
        fr = int(fr); r = int(r); ok = bool(ok)
    except Exception:
        return None
    return {"frame": fr, "radius": r, "ok": ok}

def _scan_lists(obj):
    """JSON içinde derine gidip dict listeleri topla."""
    found = []
    if isinstance(obj, list):
        found.append(obj)
        for x in obj: found += _scan_lists(x)
    elif isinstance(obj, dict):
        for v in obj.values(): found += _scan_lists(v)
    return found

def load_entries(path):
    txt = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    # 1) JSON (liste/dict)
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            cand_lists = [data]
        else:
            cand_lists = []
            # yaygın anahtarlar
            for k in ("frames","items","entries","data","results","rows","per_frame"):
                if isinstance(data.get(k), list):
                    cand_lists.append(data[k])
            # yine yoksa, derin tarama
            if not cand_lists:
                cand_lists = [lst for lst in _scan_lists(data) if isinstance(lst, list)]
        # en makul listeyi seç (içi dict olan listeler)
        best = None
        for lst in cand_lists:
            if lst and isinstance(lst[0], dict):
                # radius/ok içerene öncelik
                score = 0
                if "radius" in lst[0] or "r" in lst[0]: score += 1
                if "ok"     in lst[0]: score += 1
                best = (score, lst) if (best is None or score > best[0]) else best
        if best:
            entries = [_norm_entry(e) for e in best[1]]
            entries = [e for e in entries if e]
            entries.sort(key=lambda x: x["frame"])
            return entries, data
        else:
            return [], data
    except Exception:
        # 2) JSON Lines
        entries = []
        for line in txt.splitlines():
            line=line.strip()
            if not line: continue
            try:
                e = json.loads(line)
                e = _norm_entry(e)
                if e: entries.append(e)
            except Exception:
                pass
        entries.sort(key=lambda x: x["frame"])
        return entries, None

def main():
    if len(sys.argv) < 2:
        print("Kullanım: python summarize_video.py pred_safe/summary_video.json")
        sys.exit(1)
    jp = sys.argv[1]
    entries, root = load_entries(jp)

    if not entries:
        # Per-frame yoksa en azından metadata göster
        meta = {}
        if isinstance(root, dict):
            for k in ("video","out","frames_written","fps_in","imgsz","conf","sky",
                      "ban_road_lanes","static_ban","obstacle_dilation_kernel_size"):
                if k in root: meta[k] = root[k]
        print("[WARN] Per-frame istatistik bulunamadı. Metadata:")
        print(meta if meta else "(bulunamadı)")
        print("İpucu: infer_safe_video.py'da per-frame log yazdır (aşağıdaki patch).")
        return

    frames = [e["frame"] for e in entries]
    r      = [e["radius"] for e in entries]
    ok     = [e["ok"] for e in entries]

    # CSV
    csvp = jp.replace(".json", ".csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame","radius_px","ok"])
        for e in entries: w.writerow([e["frame"], e["radius"], int(e["ok"])])

    # istatistik
    safe_pct = 100.0 * sum(ok) / len(ok)
    r_nonzero = [rr for rr in r if rr>0]
    stats = dict(
        frames=len(frames),
        safe_pct=round(safe_pct,1),
        r_min=min(r) if r else 0,
        r_max=max(r) if r else 0,
        r_med=int(st.median(r_nonzero)) if r_nonzero else 0
    )
    print("Stats:", stats)

    # plot
    plt.figure(figsize=(10,3))
    plt.plot(frames, r, linewidth=1)
    plt.title("Safe Radius over Frames (px)")
    plt.xlabel("frame"); plt.ylabel("radius (px)")
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    outp = jp.replace(".json", "_radius_plot.png")
    plt.savefig(outp, dpi=150)
    print("Wrote:", outp, csvp)

if __name__ == "__main__":
    main()
