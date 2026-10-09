"""Render docs/banner.png (1280x640: GitHub social preview / LinkedIn). Numbers are the measured ones in runs/."""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG, CARD, INK, MUTED, ACCENT, GREY = "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#58a6ff", "#30363d"

fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1280)
ax.set_ylim(0, 640)
ax.axis("off")


def card(x, y, w, h, edge=GREY, face=CARD, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                                fc=face, ec=edge, lw=lw))


# title
ax.text(64, 560, "LLMtoSystemOne", color=INK, fontsize=44, fontweight="bold", va="center")
ax.text(64, 505, "Turn any Hugging Face LLM into a calibrated decision model",
        color=INK, fontsize=21, va="center")
ax.text(64, 468, "TypeSafe /v1/systemone compatible  ·  choice · yes/no · score  ·  no free text, always valid",
        color=MUTED, fontsize=14.5, va="center")

# left: the request
card(64, 120, 520, 300)
ax.text(88, 392, "state", color=MUTED, fontsize=13, va="center")
ax.text(88, 362, "Refunds need a receipt and < 30 days.", color=INK, fontsize=14.5, va="center")
ax.text(88, 336, "Bought 12 days ago, no receipt, wants a refund.", color=INK, fontsize=14.5, va="center")
ax.text(88, 290, "question  (choice)", color=MUTED, fontsize=13, va="center")
for i, opt in enumerate(["[1] refund", "[2] decline", "[3] escalate"]):
    ax.text(88, 258 - 34 * i, opt, color=INK, fontsize=15, family="monospace", va="center")
ax.text(88, 148, "Best answer: [", color=ACCENT, fontsize=15, family="monospace", va="center")

# arrow
ax.annotate("", xy=(650, 270), xytext=(596, 270),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2, mutation_scale=22))

# right: probabilities read from next-token logits (real output, Qwen3.5-4B, no training)
card(662, 120, 554, 300, edge=ACCENT, lw=2)
ax.text(686, 392, "P(next token) on the option ids", color=MUTED, fontsize=13, va="center")
probs = [("refund", 0.015), ("decline", 0.977), ("escalate", 0.008)]
for i, (lab, p) in enumerate(probs):
    y = 330 - 62 * i
    ax.text(686, y, lab, color=INK, fontsize=15, va="center")
    ax.add_patch(FancyBboxPatch((800, y - 13), 300, 26, boxstyle="round,pad=0,rounding_size=6", fc=GREY, ec="none"))
    ax.add_patch(FancyBboxPatch((800, y - 13), max(300 * p, 6), 26, boxstyle="round,pad=0,rounding_size=6",
                                fc=ACCENT if p > 0.5 else MUTED, ec="none"))
    ax.text(1112, y, f"{p:.1%}", color=INK if p > 0.5 else MUTED, fontsize=14, va="center",
            fontweight="bold" if p > 0.5 else "normal")
ax.text(686, 148, "Qwen3.5-4B, no fine-tuning", color=MUTED, fontsize=13, va="center")

# bottom stats
stats = ["78.4% on JevBench-231 (Qwen3.5-4B, training-free)", "ECE 0.041", "LoRA training: LLM-as-Jev (arXiv 2610.02076)"]
renderer = fig.canvas.get_renderer()
x = 64
for s in stats:
    t = ax.text(x + 20, 65, s, color=INK, fontsize=13, ha="left", va="center", zorder=3)
    bb = t.get_window_extent(renderer)  # pixels == data units here (1280x640 axes at 100 dpi)
    w = bb.width + 40
    card(x, 42, w, 46, face=BG)
    x += w + 16

os.makedirs("docs", exist_ok=True)
fig.savefig("docs/banner.png", facecolor=BG)
print("docs/banner.png")
