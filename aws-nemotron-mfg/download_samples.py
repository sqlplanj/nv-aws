"""
Download a small set of public-domain manufacturing / defect images into samples/.

Sources: Wikimedia Commons (CC0 / public domain) and other open datasets.
Run once: python download_samples.py
"""

import urllib.request
from pathlib import Path

SAMPLES = {
    # Surface crack on metal
    "metal_crack.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/"
        "Cracked_steel.jpg/640px-Cracked_steel.jpg"
    ),
    # Weld bead (good reference)
    "weld_bead.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/"
        "Mig_weld_-_single_pass.jpg/640px-Mig_weld_-_single_pass.jpg"
    ),
    # PCB board (electronics inspection)
    "pcb_board.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/"
        "PCB_JTAG_interface_on_a_Turris_Omnia.jpg/"
        "640px-PCB_JTAG_interface_on_a_Turris_Omnia.jpg"
    ),
    # Corroded metal surface
    "corrosion.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/"
        "Rust_and_dirt.jpg/640px-Rust_and_dirt.jpg"
    ),
}

def main():
    out = Path("samples")
    out.mkdir(exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (sample downloader)"}

    for name, url in SAMPLES.items():
        dest = out / name
        if dest.exists():
            print(f"  already exists: {name}")
            continue
        print(f"  downloading {name} …", end=" ", flush=True)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as r, open(dest, "wb") as f:
                f.write(r.read())
            print("OK")
        except Exception as e:
            print(f"FAILED ({e})")

    print(f"\nSamples saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
