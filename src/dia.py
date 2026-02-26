# 4-Fazlı Yatay Süreç Diyagramı (Düzeltilmiş Yerleşim)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import patheffects as pe

FIG_W, FIG_H = 16, 6
MARGIN_TOP = 0.6
TITLE_FS = 16
PHASE_W, PHASE_H = 3.4, 6.0     # daha geniş kutular
STROKE = 2                      # 2px benzeri
GREY = "#f2f2f2"

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

def add_text(*args, **kw):
    t = ax.text(*args, **kw)
    # beyaz kontur ile okunurluk
    t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
    return t

def phase_box(x, y, w, h, title, items, chip=None):
    """
    Faz kutusu: üst başlık, altında 3 alt kutu, iç sağ üstte küçük chip.
    Tümü gri dolgu + 2px sınır.
    """
    # Başlık
    ax.add_patch(Rectangle((x, y + h - 1.2), w, 1.2, facecolor=GREY,
                           edgecolor="black", linewidth=STROKE))
    add_text(x + w/2, y + h - 0.6, title, ha="center", va="center",
             fontsize=12, fontweight="bold")

    # 3 alt kutu
    sh = (h - 1.6) / 3
    for i, it in enumerate(items[:3]):
        ry = y + (2 - i) * sh
        ax.add_patch(Rectangle((x, ry), w, sh - 0.2, facecolor=GREY,
                               edgecolor="black", linewidth=STROKE))
        add_text(x + w/2, ry + (sh - 0.2)/2, it, ha="center", va="center",
                 fontsize=11, wrap=True)

    # İç sağ üst chip (taşma yok)
    if chip:
        chip_w, chip_h = 2.3, 0.55
        cx = x + w - chip_w - 0.15
        cy = y + h - 0.95
        ax.add_patch(Rectangle((cx, cy), chip_w, chip_h, facecolor="white",
                               edgecolor="black", linewidth=STROKE))
        ax.text(cx + chip_w/2, cy + chip_h/2, chip, ha="center", va="center",
                fontsize=9)

def arrow_between(x1, x2, y_mid):
    ax.add_patch(FancyArrowPatch((x1, y_mid), (x2, y_mid),
                                 arrowstyle="->", mutation_scale=16,
                                 linewidth=STROKE))

# Başlık
add_text(8, 8.6 - MARGIN_TOP, "Proje Yöntemi — Bootstrapping + Federated Learning",
         ha="center", va="center", fontsize=TITLE_FS)

# Faz konumları
xs = [1.0, 5.2, 9.4, 13.6]
y = 1.4  # biraz aşağı aldık

# Fazlar
phase_box(xs[0], y, PHASE_W, PHASE_H, "Faz 1 — Öğretmen",
          ["AirSim+Unreal", "Etiketli LiDAR", "PointNet++\nöğretmen"],
          chip="Öğretmen Model")

phase_box(xs[1], y, PHASE_W, PHASE_H, "Faz 2 — Bootstr.",
          ["openTopo LiDAR", "Sözde etiket", "Eğitim verisi"],
          chip="Sözde gerçek veri")

phase_box(xs[2], y, PHASE_W, PHASE_H, "Faz 3 — FL",
          ["Non-IID dağıtım", "Yerel eğitim", "FedAvg global"],
          chip="TLS / DP-SGD")

phase_box(xs[3], y, PHASE_W, PHASE_H, "Faz 4 — Ağ & Doğr.",
          ["Niceleme adaptif", "Kesinti toleransı", "Ölçüm ve test"],
          chip="AirSim testi")

# Fazlar arası oklar (merkezden merkeze)
mid_y = y + PHASE_H/2 - 0.2
for i in range(3):
    arrow_between(xs[i] + PHASE_W + 0.1, xs[i+1] - 0.1, mid_y)

# Faz 3 içi mini FL şeması — orta alt alana, çakışma yok
mini_y = y + 2.1
server_x = xs[2] + 1.45
ax.add_patch(Rectangle((server_x, mini_y + 0.9), 0.9, 0.55,
                       facecolor="white", edgecolor="black", linewidth=STROKE))
ax.text(server_x + 0.45, mini_y + 1.175, "Sunucu\nFedAvg",
        ha="center", va="center", fontsize=9)

for cx in [xs[2] + 0.35, xs[2] + 2.0]:
    ax.add_patch(Rectangle((cx, mini_y), 0.9, 0.55,
                           facecolor="white", edgecolor="black", linewidth=STROKE))
    ax.text(cx + 0.45, mini_y + 0.275, "İHA\nİstemci",
            ha="center", va="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((cx + 0.9, mini_y + 0.55),
                                 (server_x, mini_y + 1.45),
                                 arrowstyle="->", mutation_scale=12, linewidth=STROKE))
    ax.add_patch(FancyArrowPatch((server_x + 0.9, mini_y + 0.9),
                                 (cx, mini_y),
                                 arrowstyle="->", mutation_scale=12, linewidth=STROKE))

# Alt açıklama
ax.text(8, 0.5, "Amaç: 3B nokta bulutuyla gizlilik korumalı acil iniş; sentetik → gerçek bootstrapping.",
        ha="center", fontsize=11)

plt.tight_layout()
plt.savefig("dort_faz_ultra_kompakt.png", dpi=200, bbox_inches="tight")
