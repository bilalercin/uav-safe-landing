# make_video.py
import cv2, glob, os
imgs = sorted(glob.glob("modified_uavid_dataset/test_data/Images/seq29_*.jpg"))
assert imgs, "Frame bulunamadı: desen kontrol et"
os.makedirs("pred_safe", exist_ok=True)
h, w = cv2.imread(imgs[0]).shape[:2]
# MP4 için boyutları çift yapalım
w2, h2 = (w//2)*2, (h//2)*2
vw = cv2.VideoWriter("pred_safe/seq29.mp4",
                     cv2.VideoWriter_fourcc(*"mp4v"), 15, (w2, h2))
for p in imgs:
    fr = cv2.imread(p)
    if fr is None: continue
    if fr.shape[1] != w2 or fr.shape[0] != h2:
        fr = cv2.resize(fr, (w2, h2))
    vw.write(fr)
vw.release()
print("OK -> pred_safe/seq29.mp4")
