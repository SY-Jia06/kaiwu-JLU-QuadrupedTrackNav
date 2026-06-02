# Git 开发记录

本文记录两条 Git 线索：

1. **实际代码开发 worktree**：`/Users/j/.config/superpowers/worktrees/code-legged_robot_competition_26-public-22.0.3-comp-normal-lite.saas.sim/e6-curriculum-from-scratch`
2. **当前展示仓库**：`kaiwu-JLU-QuadrupedTrackNav`

注意：实际开发 worktree 当前仍有未提交改动，包括 `agent_ppo/agent.py`、track-nav 配置、goal observation、reward 与监控等最终决赛代码。因此下面的 Git 提交历史主要覆盖从 baseline 到 P6 track/nav base 的开发过程；P6 之后的完整实验演化以 [`EXPERIMENTS.md`](../EXPERIMENTS.md) 为准。

## 实际代码开发提交

| 日期 | commit | 提交信息 | 对应阶段 |
| --- | --- | --- | --- |
| 2026-05-07 | `48724fc` | Initial Kaiwu regional finals legged robot package | 初始代码包 |
| 2026-05-08 | `19ea1d9` | Stabilize PPO locomotion cold start | 冷启动稳定性 |
| 2026-05-08 | `45f9f2c` | Add PPO experiment log | 实验记录框架 |
| 2026-05-09 | `dbb834d` | E4 add feet air time locomotion reward | standard locomotion reward |
| 2026-05-09 | `515683d` | E5 focus stairs inv training distribution | 反台阶训练分布 |
| 2026-05-09 | `d888f5a` | E5-1 raise stairs inv curriculum level | 反台阶课程提升 |
| 2026-05-09 | `8c527a3` | E6-P1 start curriculum gait training | P1 从零课程步态 |
| 2026-05-09 | `1c9ceb8` | C0-P2 transfer gait to stairs | P2 地形课程迁移 |
| 2026-05-09 | `f83949a` | C0-P3 focus full terrain stairs inv | P3 过载课程反例 |
| 2026-05-10 | `167c2ab` | C0-P3-1 isolate full difficulty | P3+ 单变量全难度 |
| 2026-05-10 | `3f7b540` | C0-P4 add maze ten percent | P4 加入 10% maze |
| 2026-05-10 | `7d1b2b7` | C0-P4 increase maze exposure | P4 增加 maze exposure |
| 2026-05-12 | `90d4c0f` | C0-P5 light reward alignment | P5 reward 对齐 |
| 2026-05-12 | `8b8817e` | C0-P6 add track nav base stage | P6 track-nav base |
| 2026-05-12 | `fddd166` | C0-P6 add track config fallback | track 配置 fallback |
| 2026-05-12 | `2e3713b` | C0-P6 inline full track fallback | track fallback 内联 |
| 2026-05-13 | `9512178` | C0-P6 refresh track goal state | track goal state 刷新 |

## 从 Git 历史能讲出的开发线索

这条 Git 线能支撑一个清晰的早期故事：先解决 PPO 冷启动和“不走”的问题，再通过 command curriculum 与 terrain curriculum 得到可迁移步态，随后把 standard locomotion 迁移到 track-nav 框架。

关键转折包括：

- `8c527a3`：正式从零开 P1 课程步态线，开始把早期探索收束成可解释主线。
- `1c9ceb8`：P2 把直行步态迁移到坡和台阶，说明能力不是只会平地前进。
- `f83949a` 与 `167c2ab`：P3 一次性过载失败，P3+ 拆变量恢复。这是“每次只改一个主变量”的重要证据。
- `90d4c0f`：P5 reward 对齐得到更强 standard 基础，为后续 track 迁移提供 checkpoint。
- `8b8817e` 到 `9512178`：开始将 goal/track 相关状态接入训练代码，进入决赛 track navigation 阶段。

## 当前展示仓库提交

当前仓库主要用于沉淀最终方案、图表、故事线和可展示材料。它不是训练代码的原始开发仓库，而是把实验结论整理成可读 repo。

| 日期 | commit | 提交信息 |
| --- | --- | --- |
| 2026-05-30 | `e016f8b` | Initial track navigation solution |
| 2026-05-30 | `0944e01` | Add experiment figures to README |
| 2026-05-31 | `599fa40` | Add checkpoint selection tree |
| 2026-05-31 | `dd70169` | Add local story deck prototype |
| 2026-05-31 | `b49d92c` | Focus standard story on P-series curriculum |
| 2026-05-31 | `8fb46b8` | Add data evidence to standard curriculum slides |
| 2026-05-31 | `1f3f250` | Split track progression figures |
| 2026-05-31 | `f80f460` | Redesign waypoint command slide |
| 2026-06-01 | `d57a215` | Add same-setting standard scaling curve |

## 和完整实验记录的关系

Git 记录适合证明“代码是逐步演化的”，但不适合替代实验记录。比赛阶段的大量 checkpoint sweep、平台 eval 结果、压力测和分 level 权衡都记录在 [`EXPERIMENTS.md`](../EXPERIMENTS.md)。

答辩时更稳的说法是：

> Git 历史展示了代码从 standard locomotion 到 track-nav 框架的演化；`EXPERIMENTS.md` 展示了平台评测驱动的 checkpoint 选择与分布调优过程。我们不是一次性调出最终模型，而是在每个阶段根据失败现象做单变量修改，再用同口径 eval 决定是否继承 checkpoint。
