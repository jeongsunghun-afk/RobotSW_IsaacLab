"""Go2 Parkour terrain-motion latent framework diagram → SVG + PNG."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "/home/lgb/IsaacLab/_workspace/terrain_motion_latent/framework_diagram"

# canvas: x 0-100, y 0-152 (y down)
fig, ax = plt.subplots(figsize=(14, 19.5))
ax.set_xlim(0, 100)
ax.set_ylim(152, 0)
ax.axis("off")

C_STAGE0 = "#FDEBD0"
C_STAGE1 = "#D6EAF8"
C_TRACK1 = "#D5F5E3"
C_TRACK2 = "#FADBD8"
C_STAGE3 = "#F4ECF7"
C_GATE = "#34495E"
C_CONSTR = "#FEF9E7"
C_EDGE = "#5D6D7E"

BODY_FS = 10.5
TITLE_FS = 12.5


def box(x, y, w, h, color, ls="-"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor=color,
            edgecolor=C_EDGE,
            linewidth=1.4,
            linestyle=ls,
            mutation_aspect=0.66,
        )
    )


def header(x, y, text):
    ax.text(x, y, text, fontsize=TITLE_FS, fontweight="bold", color="#1B2631", va="top", ha="left")


def body(x, y, lines, fs=BODY_FS, dy=2.45):
    for i, ln in enumerate(lines):
        ax.text(x, y + i * dy, ln, fontsize=fs, color="#212F3D", va="top", ha="left")
    return y + len(lines) * dy


def gate(x, y, w, text):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            3.4,
            boxstyle="round,pad=0.25,rounding_size=0.8",
            facecolor=C_GATE,
            edgecolor="none",
            mutation_aspect=0.66,
        )
    )
    ax.text(x + w / 2, y + 1.7, text, fontsize=10, fontweight="bold", color="white", va="center", ha="center")


def arrow(x0, y0, x1, y1, style="-|>", color="#2C3E50", lw=2.2, rad=0.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            mutation_scale=22,
            color=color,
            linewidth=lw,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


# ---------------- title ----------------
ax.text(50, 2.4, "Go2 Parkour — Terrain-Conditioned Motion Latent Framework",
        fontsize=19, fontweight="bold", ha="center", va="top", color="#17202A")
ax.text(50, 6.2, "Natural terrain traversal via skill embedding + dual-track RL  (2026-06-12, team go2-latent-parkour)",
        fontsize=11, ha="center", va="top", color="#566573")

# ---------------- STAGE 0 ----------------
s0_y = 10.5
box(7, s0_y, 86, 18.5, C_STAGE0)
header(9, s0_y + 1.8, "STAGE 0  —  Paired (Terrain, Motion) Dataset Construction")
body(9, s0_y + 5.6, [
    "Assets:  flat walk / run / trot / pace mocap  +  jump motion (no terrain info)",
    "Manual obstacle placement  →  height-map paired with jump motion   [HIL, arXiv 2505.12619]",
    "Retargeting to Go2 dynamics:  kino-dynamic IK+MPC  /  STMR   [arXiv 2507.00677 / 2404.11557]",
])
gate(9, s0_y + 13.7, 82, "GATE-0   physics QC: joint limits · penetration · foot-slip · impact / CoT")

arrow(50, s0_y + 19.6, 50, s0_y + 23.2)

# ---------------- STAGE 1 ----------------
s1_y = 34.5
box(7, s1_y, 86, 21, C_STAGE1)
header(9, s1_y + 1.8, "STAGE 1  —  Terrain-Conditioned Motion AE  (skill embedding, single AE)")
body(9, s1_y + 5.6, [
    "Encoder E(motion, terrain) → z        Decoder D(z, terrain, proprio) → motion   (frozen for Track 2)",
    "Factorization:  command / terrain / phase  =  conditioning inputs  —  z encodes skill mode only",
    "Latent:  1) RVQ or discrete+continuous hybrid    2) VQ codebook  [Lifelike Agility, Nat. MI 24]",
    "            3) Gaussian + learnable conditioned prior  [PULSE, ICLR 24]  (fallback)",
])
gate(9, s1_y + 16.2, 82, "GATE-1   reconstruction · codebook perplexity / utilization · flat-coverage reproduction")

arrow(50, s1_y + 22.1, 50, s1_y + 25.7)

# ---------------- STAGE 2 ----------------
s2_y = 61.5
box(7, s2_y, 86, 44, "#FBFCFC")
header(9, s2_y + 1.8, "STAGE 2  —  Dual-Track Joint Training   (parameter-separated, round-synced)")

# track 1
t_y = s2_y + 6.2
box(9.5, t_y, 39, 26.5, C_TRACK1)
header(11, t_y + 1.6, "TRACK 1 — In-distribution terrain")
body(11, t_y + 5.2, [
    "flat · hurdle · low step",
    "Imitation: motion tracking +",
    "terrain-conditioned AMP   [HIL]",
    "Updates: encoder · decoder · prior",
    "Style enforced by reference data",
], dy=2.6)
ax.text(11, t_y + 19.4, "IL objective only on AE params", fontsize=9.5, style="italic", color="#1E8449", va="top")

# track 2
box(51.5, t_y, 39, 26.5, C_TRACK2)
header(53, t_y + 1.6, "TRACK 2 — OOD terrain")
body(53, t_y + 5.2, [
    "stair · gap · crawl (terrain added first)",
    "High-level  π(z | proprio, terrain):",
    "skill sampling in prior space",
    "[Lifelike / Motion Priors Reimagined]",
    "New-skill capacity: new codes / adapters",
    "(old skills frozen)   [AdaptNet / PHC*]",
    "Task reward + KL-to-prior · decoder frozen",
], dy=2.6)

# sync arrows between tracks
arrow(48.9, t_y + 12.0, 51.1, t_y + 12.0, style="-|>", color="#7D3C98", lw=1.8)
arrow(51.1, t_y + 15.0, 48.9, t_y + 15.0, style="-|>", color="#7D3C98", lw=1.8)
ax.text(50.0, t_y + 13.5, "decoder round-sync (EMA)", fontsize=8.2, color="#7D3C98", ha="center", va="center",
        rotation=90)

gate(9.5, s2_y + 38.3, 81, "GATE-2   flat style retention · obstacle success rate · no codebook collapse")

arrow(50, s2_y + 45.1, 50, s2_y + 48.7)

# ---------------- STAGE 3 ----------------
s3_y = 111.5
box(7, s3_y, 86, 15, C_STAGE3, ls="--")
header(9, s3_y + 1.8, "STAGE 3 (optional)  —  Self-Imitation Refeed")
body(9, s3_y + 5.6, [
    "Stabilized OOD skills → promote rollouts to dataset → AE re-train  (loop to STAGE 1)",
    "Demoted from core mechanism: physics gate + AE-manifold / gait-statistics style metrics",
])

# loop arrow stage3 -> stage1 (routed in left margin, outside all boxes)
ax.plot([6.4, 3.2], [s3_y + 8, s3_y + 8], color="#8E44AD", lw=1.8, ls="--")
ax.plot([3.2, 3.2], [s3_y + 8, s1_y + 10.5], color="#8E44AD", lw=1.8, ls="--")
arrow(3.2, s1_y + 10.5, 6.4, s1_y + 10.5, color="#8E44AD", lw=1.8, ls="--")
ax.text(1.6, 88, "coverage expansion loop", fontsize=9.5, color="#8E44AD", rotation=90, va="center", ha="center")

# ---------------- constraints ----------------
c_y = 131.5
box(7, c_y, 86, 16.5, C_CONSTR)
header(9, c_y + 1.8, "HARD CONSTRAINTS  (all satisfied by construction)")
body(9, c_y + 5.4, [
    "additive on existing Go2-ParkourImitation-v0  ·  policy obs 46-dim unchanged  ·  no env rewrite",
    "no contact-sensor obs (sim-to-real)  ·  latent acts via decoder / discriminator side only",
    "(no latent-action substitution)  ·  action-space residual = opt-in, default OFF  [arXiv 2505.16084]",
])

ax.text(93, 150.5, "* PHC progressive primitives: pending one-pass verification", fontsize=9,
        color="#7B7D7D", ha="right", va="top", style="italic")

fig.savefig(OUT + ".svg", bbox_inches="tight", facecolor="white")
fig.savefig(OUT + ".png", dpi=160, bbox_inches="tight", facecolor="white")
print("saved", OUT + ".svg / .png")
