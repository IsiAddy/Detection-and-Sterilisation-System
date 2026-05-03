
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import blob_log
import re

image_dir = "/kaggle/working/image_dir"
mask_dir  = "/kaggle/working/masks"
os.makedirs(mask_dir, exist_ok=True)


MIN_SIGMA           = 0.5
MAX_SIGMA           = 10.0   # raised — accommodates high zoom large dots
NUM_SIGMA           = 20     # more steps for wide sigma range
BLOB_THRESHOLD      = 0.02
OVERLAP             = 0.3
TOPHAT_KERNEL       = 41
IMAGEJ_THRESH_16BIT = 95


ADAPTIVE_RATIO = 0.3    # blobs must be at least 30% of the median blob size
ABSOLUTE_MIN   = 10     # hard floor — never count anything below 10px²


GT_IMAGE_IDS = {
    7849,6408,6689,6452,6453,6457,6460,6616,6713,
    6732,6677,6879,6599,6456,6455,6865,6891,6904
}


def load_as_bgr8(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    bit_depth = 16 if img.dtype == np.uint16 else 8
    if img.ndim == 2:
        img8 = (img / 256).astype(np.uint8) if bit_depth == 16 else img
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR), bit_depth
    img_bgr = (img / 256).astype(np.uint8) if bit_depth == 16 else img.copy()
    return img_bgr, bit_depth


def build_red_gate(path, img_bgr, bit_depth):
   
    if bit_depth == 16:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is not None and raw.dtype == np.uint16 and raw.ndim == 3:
            r_raw = raw[:, :, 2]
            r_raw = cv2.resize(r_raw,
                               (img_bgr.shape[1], img_bgr.shape[0]),
                               interpolation=cv2.INTER_AREA)
            return (r_raw >= IMAGEJ_THRESH_16BIT).astype(np.uint8) * 255
    thresh_8bit = max(1, int(round(IMAGEJ_THRESH_16BIT / 65535.0 * 255.0)))
    return (img_bgr[:, :, 2] >= thresh_8bit).astype(np.uint8) * 255


def extract_yellow_dots(img_bgr):
   
    f = img_bgr.astype(np.float32)
    r, g, b = f[:,:,2], f[:,:,1], f[:,:,0]
    yellow = np.minimum(r, g)
    yellow = np.clip(yellow - b * 0.5, 0, 255)
    return yellow.astype(np.uint8)


def get_tophat(img_bgr, red_gate):
   
    yellow = extract_yellow_dots(img_bgr)
    yellow = cv2.bitwise_and(yellow, yellow, mask=red_gate)

    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (TOPHAT_KERNEL, TOPHAT_KERNEL))
    tophat = cv2.morphologyEx(yellow, cv2.MORPH_TOPHAT, k)
    tophat = cv2.GaussianBlur(tophat, (3, 3), 0)

    if np.any(tophat > 0):
        otsu_val, _ = cv2.threshold(tophat, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bright_mask = cv2.threshold(tophat, max(otsu_val, 10),
                                       255, cv2.THRESH_BINARY)
    else:
        bright_mask = np.zeros_like(tophat)

    return cv2.bitwise_and(tophat, tophat, mask=bright_mask)


def detect_blobs_adaptive(img_bgr, red_gate):
  
    tophat = get_tophat(img_bgr, red_gate)

    gray_norm = tophat.astype(np.float32) / 255.0
    blobs = blob_log(
        gray_norm,
        min_sigma=MIN_SIGMA,
        max_sigma=MAX_SIGMA,
        num_sigma=NUM_SIGMA,
        threshold=BLOB_THRESHOLD,
        overlap=OVERLAP,
        exclude_border=2
    )

    if len(blobs) == 0:
        return [], tophat, 0.0, 0.0

    # Compute area for every candidate blob
    all_areas = []
    candidates = []
    for blob in blobs:
        y, x, sigma = blob
        area   = np.pi * (sigma * np.sqrt(2)) ** 2
        yi, xi = int(y), int(x)
        if not (0 <= yi < tophat.shape[0] and
                0 <= xi < tophat.shape[1]):
            continue
        if tophat[yi, xi] == 0:
            continue
        if area >= ABSOLUTE_MIN:        # hard floor only
            candidates.append((blob, area))
            all_areas.append(area)

    if len(all_areas) == 0:
        return [], tophat, 0.0, 0.0

    # Adaptive threshold based on this image's blob size distribution
    median_area = np.median(all_areas)
    adaptive_min = max(ADAPTIVE_RATIO * median_area, ABSOLUTE_MIN)

    valid = [blob for blob, area in candidates if area >= adaptive_min]

    return valid, tophat, median_area, adaptive_min


def extract_image_number(name):
    numbers = re.findall(r'\d{4}', str(name))
    return int(numbers[-1]) if numbers else None

image_paths = []
for ext in ("*.jpg","*.JPG","*.jpeg","*.png","*.tif","*.tiff"):
    image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
image_paths = sorted(set(image_paths))
print(f"Images found: {len(image_paths)}")

csv_rows = []

for i, path in enumerate(image_paths):
    base   = os.path.splitext(os.path.basename(path))[0]
    img_id = extract_image_number(base)
    is_gt  = img_id in GT_IMAGE_IDS

    img_bgr, bit_depth = load_as_bgr8(path)
    if img_bgr is None:
        print(f"SKIP: {base}")
        continue
    img_bgr = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_AREA)

    red_gate = build_red_gate(path, img_bgr, bit_depth)
    valid_blobs, tophat, median_area, adaptive_min = detect_blobs_adaptive(
        img_bgr, red_gate
    )
    count = len(valid_blobs)

    csv_rows.append({
        "image":        base,
        "count":        count,
        "median_area":  round(median_area, 1),
        "adaptive_min": round(adaptive_min, 1),
        "bit_depth":    bit_depth
    })
    print(f"[{i+1}/{len(image_paths)}]  {base}  →  {count}  "
          f"(median_area={median_area:.0f}px²  adaptive_min={adaptive_min:.0f}px²  {bit_depth}-bit)")

   
    viz = img_bgr.copy()
    for blob in valid_blobs:
        y, x, r = blob
        cv2.circle(viz, (int(x), int(y)), max(int(r * 1.4), 4), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(mask_dir, base + "_bbox.png"), viz)

   
    if is_gt:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original ({bit_depth}-bit)")
        axes[0].axis("off")

        axes[1].imshow(tophat, cmap="hot")
        axes[1].set_title(f"Signal  |  adaptive_min={adaptive_min:.0f}px²")
        axes[1].axis("off")

        axes[2].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        for blob in valid_blobs:
            y, x, r = blob
            axes[2].add_patch(plt.Circle((x, y), r * 1.4,
                              color="lime", linewidth=1.5, fill=False))
        axes[2].set_title(f"Detected: {count}")
        axes[2].axis("off")

        plt.suptitle(f"{base}  |  median={median_area:.0f}px²  "
                     f"min={adaptive_min:.0f}px²")
        plt.tight_layout()
        plt.show()

# Save
df_blob = pd.DataFrame(csv_rows)
df_blob.to_csv("/kaggle/working/bacteria_counts.csv", index=False)
print(f"\nDone. {len(csv_rows)} images processed.")
print(df_blob["count"].describe())
print("\nMedian blob area distribution across images:")
print(df_blob["median_area"].describe())
