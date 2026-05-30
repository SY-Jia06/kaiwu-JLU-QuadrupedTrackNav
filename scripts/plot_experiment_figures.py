#!/usr/bin/env python3
"""Generate README figures from the curated experiment log numbers."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "assets" / "figures"


def setup_style():
    font_candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            break
    else:
        font_name = "DejaVu Sans"

    sns.set_theme(
        context="talk",
        style="whitegrid",
        font=font_name,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 180,
        },
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(ROOT))


def plot_p9_summary():
    data = pd.DataFrame(
        [
            ("P8-1_12000", "总分", 47.51),
            ("P8-1_12000", "时间分", 48.87),
            ("P8-1_12000", "姿态分", 55.16),
            ("P8-1_12000", "能耗分", 56.00),
            ("Plast_12150", "总分", 48.10),
            ("Plast_12150", "时间分", 50.09),
            ("Plast_12150", "姿态分", 54.31),
            ("Plast_12150", "能耗分", 56.32),
            ("Plast_12200", "总分", 45.55),
            ("Plast_12200", "时间分", 48.63),
            ("Plast_12200", "姿态分", 54.37),
            ("Plast_12200", "能耗分", 55.15),
        ],
        columns=["检查点", "指标", "分数"],
    )
    completion = pd.DataFrame(
        [
            ("P8-1_12000", 0.87),
            ("Plast_12150", 0.88),
            ("Plast_12200", 0.84),
        ],
        columns=["检查点", "完成率"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [2.2, 1]})
    palette = ["#7A90A4", "#2E7D6B", "#C46A4A"]
    sns.barplot(data=data, x="指标", y="分数", hue="检查点", palette=palette, ax=axes[0])
    axes[0].set_title("P9 甜点区：得分指标")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("得分")
    axes[0].set_ylim(40, 60)
    axes[0].legend(title="检查点", loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False)

    sns.barplot(
        data=completion,
        x="检查点",
        y="完成率",
        hue="检查点",
        palette=palette,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("完成率")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("完成率")
    axes[1].set_ylim(0.78, 0.91)
    axes[1].tick_params(axis="x", rotation=28)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f", padding=3, fontsize=11)

    fig.suptitle("最终巩固：12150 超过 P8 基线，12200 开始过训", y=1.04)
    save(fig, "p9_checkpoint_summary.png")


def plot_level_tradeoff_heatmap():
    levels = ["L0", "L1", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]
    score_rows = {
        "P7.3-11750": [54.94, 60.35, 57.30, 53.16, 50.47, 50.47, 42.22, 1.43, 5.59],
        "P8-A-11930": [52.79, 55.37, 54.56, 50.89, 36.53, 51.93, 46.44, 12.10, 23.40],
        "P8-1_12000": [57.71, 52.96, 57.73, 52.03, 47.50, 51.76, 50.17, 23.36, 34.07],
        "Plast_12150": [58.02, 55.27, 53.75, 54.93, 53.64, 51.53, 45.76, 34.27, 25.39],
        "Plast_12200": [33.04, 56.88, 42.18, 55.96, 53.68, 50.46, 43.15, 45.20, 29.25],
    }
    comp_rows = {
        "P7.3-11750": [0.88, 0.99, 1.00, 0.95, 0.93, 0.96, 0.84, 0.03, 0.13],
        "P8-A-11930": [0.87, 0.93, 0.96, 0.91, 0.68, 0.99, 0.92, 0.25, 0.52],
        "P8-1_12000": [0.92, 0.88, 0.99, 0.93, 0.87, 0.98, 0.98, 0.48, 0.75],
        "Plast_12150": [0.95, 0.92, 0.94, 0.98, 0.98, 0.99, 0.91, 0.71, 0.56],
        "Plast_12200": [0.55, 0.93, 0.73, 0.98, 0.97, 0.95, 0.86, 0.92, 0.64],
    }
    score_df = pd.DataFrame.from_dict(score_rows, orient="index", columns=levels)
    comp_df = pd.DataFrame.from_dict(comp_rows, orient="index", columns=levels)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)
    sns.heatmap(
        score_df,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "得分"},
        ax=axes[0],
    )
    axes[0].set_title("分难度得分：高难救援 vs 分布均衡")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.heatmap(
        comp_df,
        cmap="Greens",
        annot=True,
        fmt=".2f",
        linewidths=0.8,
        linecolor="white",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "完成率"},
        ax=axes[1],
    )
    axes[1].set_title("分难度完成率")
    axes[1].set_xlabel("决赛 track 难度档")
    axes[1].set_ylabel("")
    fig.suptitle("决赛问题本质是分布权衡，不是单个奖励技巧", y=1.02)
    save(fig, "level_tradeoff_heatmap.png")


def plot_standard_curriculum():
    data = pd.DataFrame(
        [
            ("P1-500", "P1 低难步态", 500, 27.94, 13.00, 0.00),
            ("P1-1000", "P1 低难步态", 1000, 54.08, 61.31, 22.27),
            ("P1-1500", "P1 低难步态", 1500, 55.90, 72.91, 26.87),
            ("P1-2000", "P1 低难步态", 2000, 57.30, 67.92, 29.53),
            ("P1-2500", "P1 低难步态", 2500, 57.86, 68.68, 16.59),
            ("P2-3000", "P2 地形课程", 3000, 61.38, 77.76, 21.98),
            ("P2-3500", "P2 地形课程", 3500, 65.12, 83.26, 29.35),
            ("P2-4000", "P2 地形课程", 4000, 65.78, 85.38, 25.92),
            ("P2-4500", "P2 地形课程", 4500, 64.94, 85.84, 33.32),
            ("P3-5000", "P3 过载课程", 5000, 57.26, 74.79, 26.52),
            ("P3-6000", "P3 过载课程", 6000, 48.38, 56.55, 12.74),
            ("P3-7000", "P3 过载课程", 7000, 23.61, 13.84, 0.00),
            ("P3+-4500", "P3+ 单变量全难度", 4500, 69.94, 91.45, 39.82),
            ("P3+-5000", "P3+ 单变量全难度", 5000, 70.10, 89.97, 38.87),
            ("P3+-6000", "P3+ 单变量全难度", 6000, 69.40, 90.09, 36.61),
            ("P3+-7000", "P3+ 单变量全难度", 7000, 69.75, 91.17, 44.87),
        ],
        columns=["检查点", "阶段", "训练步数", "总分", "前进分", "时间分"],
    )
    long = data.melt(
        id_vars=["检查点", "阶段", "训练步数"],
        value_vars=["总分", "前进分", "时间分"],
        var_name="指标",
        value_name="得分",
    )

    fig, ax = plt.subplots(figsize=(15, 6.4))
    sns.lineplot(
        data=long,
        x="训练步数",
        y="得分",
        hue="指标",
        style="阶段",
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=8,
        ax=ax,
    )
    ax.set_title("标准步态训练课程：分阶段加难度优于一次性过载训练")
    ax.set_xlabel("训练检查点步数")
    ax.set_ylabel("标准评测得分")
    ax.set_ylim(0, 100)
    ax.axvspan(5000, 7000, color="#E8B4A0", alpha=0.18, label="P3 过载崩溃")
    ax.text(6100, 8, "P3 过载\n能力崩掉", ha="center", va="bottom", fontsize=11, color="#8A3E2F")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    save(fig, "standard_curriculum_curve.png")


def plot_track_progression():
    data = pd.DataFrame(
        [
            ("P6.1-500", "只接接口", "早期 track [0,1,3,4]", 0.04, 0.00),
            ("P6.2-10700", "导航 reward", "早期 track [0,1,3,4]", 19.02, 0.34),
            ("P6.2-11100", "导航 reward", "早期 track [0,1,3,4]", 43.60, 0.75),
            ("P6.4.1-11300", "command 槽导航", "早期 track [0,1,3,4]", 53.64, 0.91),
            ("P7.3-11750", "决赛口径基线", "决赛 track [0-9]", 41.84, 0.75),
            ("P8-A-11930", "level8/9 暴露", "决赛 track [0-9]", 42.71, 0.79),
            ("P8-1_12000", "中高难回填", "决赛 track [0-9]", 47.51, 0.87),
            ("Plast_12150", "全量低 lr 巩固", "决赛 track [0-9]", 48.10, 0.88),
            ("Plast_12200", "过训", "决赛 track [0-9]", 45.55, 0.84),
        ],
        columns=["检查点", "阶段", "评测口径", "总分", "完成率"],
    )

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    palette = {"早期 track [0,1,3,4]": "#5B8FB9", "决赛 track [0-9]": "#2E7D6B"}
    sns.lineplot(
        data=data,
        x="检查点",
        y="总分",
        hue="评测口径",
        marker="o",
        linewidth=2.8,
        markersize=9,
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_title("赛道导航里程碑：先算法跃迁，再决赛分布调优")
    axes[0].set_ylabel("总分")
    axes[0].set_xlabel("")
    axes[0].legend(title="", frameon=False, loc="lower right")

    sns.lineplot(
        data=data,
        x="检查点",
        y="完成率",
        hue="评测口径",
        marker="o",
        linewidth=2.8,
        markersize=9,
        palette=palette,
        ax=axes[1],
    )
    axes[1].set_ylabel("完成率")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend_.remove()
    axes[1].tick_params(axis="x", rotation=24)
    axes[1].axvline(3.5, color="#777777", linestyle="--", linewidth=1.4)
    axes[1].text(3.55, 0.08, "评测口径切换", fontsize=11, color="#555555")
    save(fig, "track_progression_curve.png")


def main():
    setup_style()
    plot_p9_summary()
    plot_level_tradeoff_heatmap()
    plot_standard_curriculum()
    plot_track_progression()


if __name__ == "__main__":
    main()
