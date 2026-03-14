"""Generate recap charts for Hopper SDXL inference optimization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

OUT_DIR = Path("docs/perf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Semantic color palette
COLOR_BASELINE = "#b8a08c"
COLOR_SNAPSHOT = "#9aab8e"
COLOR_COMPILE = "#5b8a8f"
COLOR_BF16 = "#7b8fa1"
COLOR_FINAL = "#1a6b5a"


def save_bar_chart(
    filename: str,
    title: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    ylabel: str,
    annotations: list[str] | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="#2d2d2d", linewidth=0.8)
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.28)

    for i, (bar, value) in enumerate(zip(bars, values)):
        # Value label above bar
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.03,
            f"{value:.2f}s",
            ha="center",
            va="bottom",
            fontsize=11,
        )
        # "What changed" annotation below bar label
        if annotations and i < len(annotations):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                -max(values) * 0.06,
                annotations[i],
                ha="center",
                va="top",
                fontsize=7.5,
                color="#555555",
                fontstyle="italic",
            )

    fig.subplots_adjust(bottom=0.18)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def save_tradeoff_scatter():
    labels = [
        "No snapshot baseline",
        "Snapshot base model",
        "fp16 + compile_repeated_blocks",
        "bf16 + channels_last + compile_repeated_blocks",
        "Final: snapshotted Hopper LoRA",
    ]
    cold = [49.2, 33.0, 33.0, 99.45, 31.03]
    warm = [21.0, 21.0, 17.7, 18.9, 19.53]
    colors = [COLOR_BASELINE, COLOR_SNAPSHOT, COLOR_COMPILE, COLOR_BF16, COLOR_FINAL]

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, x, y, color in zip(labels, cold, warm, colors):
        ax.scatter(x, y, s=120, color=color, edgecolor="#2d2d2d", linewidth=0.8)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=10)

    ax.set_title("Cold vs Warm Tradeoff", fontsize=16, pad=14)
    ax.set_xlabel("Cold / first-request total time (s)")
    ax.set_ylabel("Warm request total time (s)")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "tradeoff_scatter.png", dpi=180)
    plt.close(fig)


def save_final_warm_variance():
    labels = ["Warm 1", "Warm 2", "Warm 3"]
    values = [19.40, 19.53, 19.65]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(labels, values, marker="o", markersize=8, linewidth=2.5, color=COLOR_FINAL)
    ax.set_title("Final Path Warm-Run Consistency", fontsize=16, pad=14)
    ax.set_ylabel("Total time (s)")
    ax.set_axisbelow(True)
    ax.set_ylim(18.5, 20.2)

    for label, value in zip(labels, values):
        ax.text(label, value + 0.05, f"{value:.2f}s", ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_warm_variance.png", dpi=180)
    plt.close(fig)


def save_final_path_summary():
    labels = [
        "After deploy\n(first request)",
        "After restore\n(cold)",
        "Warm\n(mean)",
    ]
    values = [28.37, 31.03, 19.53]
    colors = [COLOR_FINAL, COLOR_FINAL, COLOR_FINAL]
    annotations = [
        "fresh container, no snapshot",
        "GPU memory restored from disk",
        "avg of 3 consecutive runs",
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="#2d2d2d", linewidth=0.8)
    ax.set_title("Final Production Path Summary", fontsize=16, pad=14)
    ax.set_ylabel("Total time (s)")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.28)

    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.03,
            f"{value:.2f}s",
            ha="center",
            va="bottom",
            fontsize=11,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            -max(values) * 0.06,
            annotations[i],
            ha="center",
            va="top",
            fontsize=7.5,
            color="#555555",
            fontstyle="italic",
        )

    fig.subplots_adjust(bottom=0.18)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_path_summary.png", dpi=180)
    plt.close(fig)


def main():
    save_bar_chart(
        filename="cold_start_progression.png",
        title="Cold / First-Request Latency Progression",
        labels=[
            "No snapshot\nbaseline",
            "Snapshot\nbase model",
            "Current final\nsnapshot LoRA",
        ],
        values=[49.2, 33.0, 31.03],
        colors=[COLOR_BASELINE, COLOR_SNAPSHOT, COLOR_FINAL],
        ylabel="Total time (s)",
        annotations=[
            "@app.function, reload every call",
            "GPU memory serialized",
            "LoRA merged + snapshotted",
        ],
    )

    save_bar_chart(
        filename="warm_latency_progression.png",
        title="Warm Latency Tradeoff Across Optimization Paths",
        labels=[
            "No snapshot\nbaseline",
            "Snapshot\nbase model",
            "fp16 + compile\nrepeated blocks",
            "bf16 + channels_last\n+ compile",
            "Current final\nsnapshot LoRA",
        ],
        values=[21.0, 21.0, 17.7, 18.9, 19.53],
        colors=[COLOR_BASELINE, COLOR_SNAPSHOT, COLOR_COMPILE, COLOR_BF16, COLOR_FINAL],
        ylabel="Warm total time (s)",
        annotations=[
            "@app.function, reload every call",
            "GPU memory serialized",
            "torch.compile on UNet blocks",
            "bfloat16 + memory layout",
            "LoRA merged + snapshotted",
        ],
    )

    save_tradeoff_scatter()
    save_final_warm_variance()
    save_final_path_summary()


if __name__ == "__main__":
    main()
