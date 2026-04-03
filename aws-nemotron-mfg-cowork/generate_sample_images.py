"""
Generate synthetic manufacturing sample images for defect detection demo.
Run this once to create sample images in the sample_images/ directory.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)


def add_noise(img, intensity=15):
    arr = np.array(img).astype(np.int32)
    noise = np.random.randint(-intensity, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── 1. PCB Board – solder bridge defect ──────────────────────────────────────
def make_pcb_defect():
    W, H = 512, 512
    img = Image.new("RGB", (W, H), color=(34, 85, 34))  # PCB green
    draw = ImageDraw.Draw(img)

    # Copper traces (horizontal)
    for y in range(60, H - 60, 50):
        draw.rectangle([30, y, W - 30, y + 8], fill=(184, 115, 51))

    # Vertical traces
    for x in range(60, W - 60, 60):
        draw.rectangle([x, 30, x + 8, H - 30], fill=(184, 115, 51))

    # Solder pads
    pad_positions = [(80, 80), (180, 80), (280, 80), (380, 80),
                     (80, 180), (180, 180), (380, 180),
                     (80, 280), (280, 280), (380, 280)]
    for px, py in pad_positions:
        draw.ellipse([px - 14, py - 14, px + 14, py + 14], fill=(212, 175, 55))
        draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(180, 180, 180))

    # Solder bridge defect (between two adjacent pads)
    draw.ellipse([166, 66, 194, 94], fill=(212, 175, 55))
    draw.ellipse([160, 72, 200, 88], fill=(212, 175, 55))  # bridge blob

    # Label
    draw.rectangle([0, H - 36, W, H], fill=(0, 0, 0, 180))
    draw.text((10, H - 28), "PCB Inspection – Solder Bridge Defect", fill="white")

    img = add_noise(img, 8)
    img = img.filter(ImageFilter.SHARPEN)
    path = os.path.join(OUTPUT_DIR, "pcb_solder_bridge_defect.jpg")
    img.save(path, quality=92)
    print(f"  Saved: {path}")
    return path


# ── 2. Metal surface – surface crack ─────────────────────────────────────────
def make_metal_crack():
    W, H = 512, 512
    # Brushed-metal base
    arr = np.ones((H, W, 3), dtype=np.uint8) * 160
    for i in range(H):
        v = int(150 + 20 * np.sin(i * 0.3) + random.randint(-10, 10))
        arr[i, :] = np.clip([v, v, v + 5], 0, 255)
    img = Image.fromarray(arr)
    img = add_noise(img, 12)
    draw = ImageDraw.Draw(img)

    # Main crack path
    cx, cy = 100, 150
    pts = [(cx, cy)]
    for _ in range(25):
        cx += random.randint(8, 18)
        cy += random.randint(-10, 10)
        pts.append((cx, cy))
    draw.line(pts, fill=(20, 20, 20), width=3)

    # Branch crack
    bx, by = pts[10]
    branch = [(bx, by)]
    for _ in range(10):
        bx += random.randint(4, 12)
        by += random.randint(5, 15)
        branch.append((bx, by))
    draw.line(branch, fill=(30, 30, 30), width=2)

    # Shadow/highlight around crack
    for px, py in pts[::3]:
        draw.ellipse([px - 3, py - 1, px + 3, py + 1], fill=(50, 50, 50))

    draw.rectangle([0, H - 36, W, H], fill=(0, 0, 0))
    draw.text((10, H - 28), "Metal Surface – Surface Crack Detected", fill="white")

    path = os.path.join(OUTPUT_DIR, "metal_surface_crack.jpg")
    img.save(path, quality=92)
    print(f"  Saved: {path}")
    return path


# ── 3. Weld bead – porosity defect ───────────────────────────────────────────
def make_weld_porosity():
    W, H = 512, 512
    # Background steel plate
    arr = np.random.randint(90, 130, (H, W, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # Weld bead (center horizontal stripe)
    mid = H // 2
    for offset in range(-28, 29):
        shade = int(200 - abs(offset) * 4)
        draw.line([(20, mid + offset), (W - 20, mid + offset)],
                  fill=(shade, int(shade * 0.6), int(shade * 0.3)))

    # Ripple marks on weld
    for x in range(30, W - 30, 18):
        draw.arc([x - 10, mid - 22, x + 10, mid + 22],
                 start=200, end=340, fill=(140, 80, 40), width=1)

    # Porosity pits (defects)
    pit_centers = [(160, mid - 5), (240, mid + 8), (310, mid - 3), (390, mid + 2)]
    for px, py in pit_centers:
        r = random.randint(4, 8)
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(30, 20, 15))
        draw.ellipse([px - r + 1, py - r + 1, px, py], fill=(80, 60, 40))

    draw.rectangle([0, H - 36, W, H], fill=(0, 0, 0))
    draw.text((10, H - 28), "Weld Inspection – Porosity Defects", fill="white")

    img = add_noise(img, 6)
    path = os.path.join(OUTPUT_DIR, "weld_porosity_defects.jpg")
    img.save(path, quality=92)
    print(f"  Saved: {path}")
    return path


# ── 4. Casting – good part (no defect) ───────────────────────────────────────
def make_casting_ok():
    W, H = 512, 512
    arr = np.random.randint(170, 210, (H, W, 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(arr[:, :, 0] - 20, 0, 255)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # Cast part outline
    draw.rectangle([80, 80, W - 80, H - 80], outline=(60, 60, 60), width=4)
    draw.rectangle([120, 120, W - 120, H - 120], fill=(190, 185, 180), outline=(80, 80, 80), width=2)

    # Bolt holes (uniform, good)
    for bx, by in [(100, 100), (W - 100, 100), (100, H - 100), (W - 100, H - 100)]:
        draw.ellipse([bx - 14, by - 14, bx + 14, by + 14], fill=(100, 100, 100), outline=(60, 60, 60))
        draw.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=(50, 50, 50))

    # Surface machining lines
    for y in range(130, H - 130, 12):
        draw.line([(122, y), (W - 122, y)], fill=(175, 170, 165), width=1)

    draw.rectangle([0, H - 36, W, H], fill=(0, 60, 0))
    draw.text((10, H - 28), "Cast Part – No Defects Found (OK)", fill="white")

    img = add_noise(img, 5)
    path = os.path.join(OUTPUT_DIR, "casting_no_defect_ok.jpg")
    img.save(path, quality=92)
    print(f"  Saved: {path}")
    return path


if __name__ == "__main__":
    print("Generating sample manufacturing images...")
    make_pcb_defect()
    make_metal_crack()
    make_weld_porosity()
    make_casting_ok()
    print("\nDone! 4 sample images created in sample_images/")
