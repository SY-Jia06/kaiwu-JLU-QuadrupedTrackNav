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


def plot_curriculum_modes_data_curve():
    """A data-first view of curriculum phases and reward/command navigation modes."""
    standard = pd.DataFrame(
        [
            ("P1-500", "command 引导步态", 1, 27.94, 13.00, 0.00),
            ("P1-1000", "command 引导步态", 2, 54.08, 61.31, 22.27),
            ("P1-1500", "command 引导步态", 3, 55.90, 72.91, 26.87),
            ("P1-2000", "command 引导步态", 4, 57.30, 67.92, 29.53),
            ("P1-2500", "command 引导步态", 5, 57.86, 68.68, 16.59),
            ("P2-3000", "地形课程", 6, 61.38, 77.76, 21.98),
            ("P2-3500", "地形课程", 7, 65.12, 83.26, 29.35),
            ("P2-4000", "地形课程", 8, 65.78, 85.38, 25.92),
            ("P2-4500", "地形课程", 9, 64.94, 85.84, 33.32),
            ("P3-5000", "过载课程反例", 10, 57.26, 74.79, 26.52),
            ("P3-6000", "过载课程反例", 11, 48.38, 56.55, 12.74),
            ("P3-7000", "过载课程反例", 12, 23.61, 13.84, 0.00),
            ("P3+-4500", "单变量全难度", 13, 69.94, 91.45, 39.82),
            ("P3+-5000", "单变量全难度", 14, 70.10, 89.97, 38.87),
            ("P3+-6000", "单变量全难度", 15, 69.40, 90.09, 36.61),
            ("P3+-7000", "单变量全难度", 16, 69.75, 91.17, 44.87),
        ],
        columns=["检查点", "阶段", "x", "总分", "前进分", "时间分"],
    )
    track = pd.DataFrame(
        [
            ("P6.1-500", "reward 模式前", 1, 0.04, 0.00),
            ("P6.2-10700", "reward 模式", 2, 19.02, 0.34),
            ("P6.2-11100", "reward 模式", 3, 43.60, 0.75),
            ("P6.4.1-11300", "command 模式", 4, 53.64, 0.91),
            ("P7.3-11750", "决赛口径基线", 5, 41.84, 0.75),
            ("P8-A-11930", "高难暴露", 6, 42.71, 0.79),
            ("P8-1_12000", "中高难回填", 7, 47.51, 0.87),
            ("Plast_12150", "全量低 lr 巩固", 8, 48.10, 0.88),
            ("Plast_12200", "过训反例", 9, 45.55, 0.84),
        ],
        columns=["检查点", "阶段", "x", "总分", "完成率"],
    )
    levels = pd.DataFrame(
        [
            ("P7.3-11750", 1, 50.47, 1.43, 5.59),
            ("P8-A-11930", 2, 36.53, 12.10, 23.40),
            ("P8-1_12000", 3, 47.50, 23.36, 34.07),
            ("Plast_12150", 4, 53.64, 34.27, 25.39),
            ("Plast_12200", 5, 53.68, 45.20, 29.25),
        ],
        columns=["检查点", "x", "level5", "level8", "level9"],
    )

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [1.05, 1.05, 0.9]})
    fig.suptitle("课程与模式选择：现有数据支持的训练路径", fontsize=24, weight="bold", y=0.995)

    # Panel 1: standard locomotion curriculum
    std_long = standard.melt(
        id_vars=["检查点", "阶段", "x"],
        value_vars=["总分", "前进分", "时间分"],
        var_name="指标",
        value_name="得分",
    )
    sns.lineplot(
        data=std_long,
        x="x",
        y="得分",
        hue="指标",
        marker="o",
        linewidth=2.4,
        markersize=7,
        ax=axes[0],
    )
    axes[0].axvspan(0.5, 5.5, color="#EAF2FF", alpha=0.65)
    axes[0].axvspan(5.5, 9.5, color="#EEF8EA", alpha=0.65)
    axes[0].axvspan(9.5, 12.5, color="#FCE9DF", alpha=0.75)
    axes[0].axvspan(12.5, 16.5, color="#EEF8EA", alpha=0.40)
    axes[0].text(3.0, 94, "指令引导步态", ha="center", color="#2E5FA7", fontsize=12, weight="bold")
    axes[0].text(7.5, 94, "地形课程", ha="center", color="#3A7D44", fontsize=12, weight="bold")
    axes[0].text(11.0, 94, "过载课程反例", ha="center", color="#A34B32", fontsize=12, weight="bold")
    axes[0].text(14.5, 94, "单变量全难度", ha="center", color="#3A7D44", fontsize=12, weight="bold")
    axes[0].set_title("A. 标准步态阶段：先用指令打破不走，再用地形课程扩展能力")
    axes[0].set_ylabel("标准步态评测得分")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 100)
    axes[0].set_xticks(standard["x"])
    axes[0].set_xticklabels(standard["检查点"], rotation=28, ha="right", fontsize=9)
    axes[0].legend(title="指标", ncol=3, loc="lower right", frameon=False)

    # Panel 2: reward mode vs command mode track navigation
    ax2 = axes[1]
    sns.lineplot(data=track, x="x", y="总分", marker="o", color="#2F6DA3", linewidth=2.8, label="总分", ax=ax2)
    ax2b = ax2.twinx()
    sns.lineplot(data=track, x="x", y="完成率", marker="o", color="#2E7D6B", linewidth=2.8, label="完成率", ax=ax2b)
    ax2.axvspan(0.5, 1.5, color="#F2F2F2", alpha=0.7)
    ax2.axvspan(1.5, 3.5, color="#FFF2DF", alpha=0.75)
    ax2.axvspan(3.5, 4.5, color="#E9F7F3", alpha=0.75)
    ax2.axvspan(4.5, 9.5, color="#EAF6EF", alpha=0.55)
    ax2.text(1.0, 54, "只接接口", ha="center", color="#555555", fontsize=11, weight="bold")
    ax2.text(2.5, 54, "奖励模式", ha="center", color="#B46B1E", fontsize=11, weight="bold")
    ax2.text(4.0, 54, "指令模式跃迁", ha="center", color="#2E7D6B", fontsize=11, weight="bold")
    ax2.text(7.0, 54, "决赛分布调优", ha="center", color="#2E7D6B", fontsize=11, weight="bold")
    ax2.set_title("B. 赛道导航阶段：奖励模式拉起完成率，指令模式产生结构性跃迁")
    ax2.set_ylabel("总分")
    ax2b.set_ylabel("完成率")
    ax2.set_ylim(0, 58)
    ax2b.set_ylim(0, 1.02)
    ax2.set_xticks(track["x"])
    ax2.set_xticklabels(track["检查点"], rotation=24, ha="right", fontsize=10)
    ax2.set_xlabel("")
    lines, labels = ax2.get_legend_handles_labels()
    lines_b, labels_b = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines_b, labels + labels_b, loc="lower right", frameon=False)
    ax2b.legend_.remove()

    # Panel 3: final distribution trade-off
    lev_long = levels.melt(
        id_vars=["检查点", "x"],
        value_vars=["level5", "level8", "level9"],
        var_name="难度档",
        value_name="得分",
    )
    lev_long["难度档"] = lev_long["难度档"].map(
        {
            "level5": "难度5",
            "level8": "难度8",
            "level9": "难度9",
        }
    )
    sns.lineplot(
        data=lev_long,
        x="x",
        y="得分",
        hue="难度档",
        marker="o",
        linewidth=2.6,
        markersize=8,
        ax=axes[2],
    )
    axes[2].axvspan(1.5, 2.5, color="#FFF2DF", alpha=0.75)
    axes[2].axvspan(2.5, 4.5, color="#EAF6EF", alpha=0.65)
    axes[2].axvspan(4.5, 5.5, color="#FCE9DF", alpha=0.75)
    axes[2].text(2.0, 52, "高难暴露\n抬难度9、伤难度5", ha="center", color="#A34B32", fontsize=10)
    axes[2].text(3.55, 52, "回填+巩固\n选 12150", ha="center", color="#2E7D6B", fontsize=10, weight="bold")
    axes[2].text(5.0, 52, "12200\n过训反例", ha="center", color="#A34B32", fontsize=10)
    axes[2].set_title("C. 决赛分布：最终不是选单项最强，而是选难度5/8/9 的均衡点")
    axes[2].set_ylabel("分难度得分")
    axes[2].set_xlabel("")
    axes[2].set_ylim(0, 60)
    axes[2].set_xticks(levels["x"])
    axes[2].set_xticklabels(levels["检查点"], rotation=20, ha="right", fontsize=10)
    axes[2].set_xlabel("")
    axes[2].legend(title="难度档", ncol=3, loc="lower right", frameon=False)

    save(fig, "curriculum_modes_data_curve.png")


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

    early = data[data["评测口径"] == "早期 track [0,1,3,4]"].copy()
    final = data[data["评测口径"] == "决赛 track [0-9]"].copy()

    def plot_split_track(df, title, note, name, color):
        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax2 = ax.twinx()

        sns.lineplot(
            data=df,
            x="检查点",
            y="总分",
            marker="o",
            linewidth=3.0,
            markersize=9,
            color=color,
            ax=ax,
        )
        sns.lineplot(
            data=df,
            x="检查点",
            y="完成率",
            marker="s",
            linewidth=2.5,
            markersize=8,
            color="#2E7D6B",
            ax=ax2,
        )

        ax.set_title(title)
        ax.set_ylabel("总分")
        ax2.set_ylabel("完成率")
        ax2.set_ylim(0, 1.02)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=18)
        ax.text(
            0.02,
            0.92,
            note,
            transform=ax.transAxes,
            fontsize=11,
            color="#374151",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F9FB", edgecolor="#D9DEE6"),
        )
        save(fig, name)

    plot_split_track(
        early,
        "早期 track：从 goal obs 到 command 槽导航",
        "同一早期 track 口径：[0,1,3,4]",
        "track_early_progression_curve.png",
        "#2F6DA3",
    )
    plot_split_track(
        final,
        "决赛 track：从高难暴露到全量巩固",
        "决赛口径：[0,1,3,4,5,6,7,8,9]",
        "track_final_progression_curve.png",
        "#8A5A44",
    )


def main():
    setup_style()
    plot_curriculum_modes_data_curve()
    plot_p9_summary()
    plot_level_tradeoff_heatmap()
    plot_standard_curriculum()
    plot_track_progression()


if __name__ == "__main__":
    main()
