#!/usr/bin/env python3
"""
Quarter pie (sector) with smooth angle-based (conic) gradient:
0° -> white, 90° -> #A9C4EB

Black outline around the sector (two radii + arc).

Output:
  quarter_pie_black_outline.png  (transparent background)
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------- settings --------------------
OUT = "quarter_pie_black_outline.png"
DPI = 300

N = 900          # final resolution
SS = 3           # supersampling factor (2–4)

R = 1.0          # radius
FEATHER_PX = 2.5 # soft edge width in final pixels

TARGET_HEX = "8FBF9F"

# Outline styling (BLACK)
OUTLINE_LW = 3
# --------------------------------------------------


def hex_to_rgb01(h: str):
    h = h.strip().lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def srgb_to_linear(c):
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)


def linear_to_srgb(c):
    a = 0.055
    return np.where(c <= 0.0031308, 12.92 * c, (1 + a) * (c ** (1 / 2.4)) - a)


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def downsample_mean(img, ss):
    H, W, C = img.shape
    img = img.reshape(H // ss, ss, W // ss, ss, C)
    return img.mean(axis=(1, 3))


def main():
    n_hi = N * SS
    x = np.linspace(-R, R, n_hi)
    y = np.linspace(-R, R, n_hi)
    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X * X + Y * Y)
    theta = np.arctan2(Y, X)  # [-pi, pi]

    # Conic parameter in first quadrant: 0..1 for theta in [0, pi/2]
    t = np.clip(theta / (np.pi / 2), 0.0, 1.0)

    # Feather widths
    px_size = (2 * R) / N
    w_r = FEATHER_PX * px_size
    w_theta = FEATHER_PX * (np.pi / 2) / N

    # Smooth alpha for quarter sector boundaries
    d_arc = R - r
    d_left = theta - 0.0
    d_top = (np.pi / 2) - theta

    alpha = (
        smoothstep(0.0, w_r, d_arc)
        * smoothstep(0.0, w_theta, d_left)
        * smoothstep(0.0, w_theta, d_top)
    )
    alpha = np.where(r <= R + 2 * w_r, alpha, 0.0)

    # Gamma-correct interpolation: white -> target
    c0 = np.array([1.0, 1.0, 1.0])
    c1 = hex_to_rgb01(TARGET_HEX)
    col_lin = (1 - t)[..., None] * srgb_to_linear(c0) + t[..., None] * srgb_to_linear(c1)
    col = linear_to_srgb(np.clip(col_lin, 0.0, 1.0))

    img_hi = np.zeros((n_hi, n_hi, 4), dtype=float)
    img_hi[..., :3] = col
    img_hi[..., 3] = alpha

    img = downsample_mean(img_hi, SS)

    # ---- Plot + BLACK outline ----
    fig = plt.figure(figsize=(4, 4), dpi=DPI)
    ax = plt.axes([0, 0, 1, 1])

    ax.imshow(img, origin="lower", extent=[-R, R, -R, R], interpolation="bilinear")

    # Outline: two radii + arc
    ax.plot([0, R], [0, 0], linewidth=OUTLINE_LW, color="black")
    ax.plot([0, 0], [0, R], linewidth=OUTLINE_LW, color="black")
    ang = np.linspace(0, np.pi / 2, 600)
    ax.plot(R * np.cos(ang), R * np.sin(ang), linewidth=OUTLINE_LW, color="black")

    pad = 0.06 * R
    ax.set_xlim(-pad, R + pad)
    ax.set_ylim(-pad, R + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    plt.savefig(OUT, dpi=DPI, transparent=True)
    plt.close(fig)

    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()