#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os

import toml


# Valid task types (Isaac Lab native config format)
# 有效任务类型（Isaac Lab 原生配置格式）
_VALID_TASKS = {"standard", "track"}


class StageConfig:
    """
    Base class for training stage configuration.
    训练阶段配置基类。

    Subclass this and override fields to define a new training stage.
    继承此类并覆盖字段来定义新的训练阶段。
    """

    # --- Stage identity
    # 阶段标识 ---
    name = ""
    task_type = "standard"

    # --- Model architecture dimensions (Isaac Lab Unitree-Go2-Velocity constants)
    # These are fixed by the Isaac Lab task definition and the network structure;
    # users are not expected to change them. Do NOT move them into user TOML.
    # 模型架构维度（Isaac Lab Unitree-Go2-Velocity 常量）
    # 由 Isaac Lab 任务定义与网络结构决定，用户不应修改；也不应放进用户 TOML。
    num_actions = 12  # Go2 joint action dim / Go2 关节动作维度
    num_proprio_obs = 45  # proprioceptive obs dim / 本体感知观测维度
    num_scan = 256  # 16x16 height-scan dim / 16x16 高度扫描维度
    num_goal_obs = 0  # goal obs dim appended in track stages / track 阶段追加的目标观测维度
    num_policy_observations = 301  # proprio(45) + scan(256)
    num_critic_observations = 316  # critic_proprio(60) + scan(256)

    # --- Model architecture
    # 模型架构 ---
    model_class = "ActorCritic"
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"

    # --- Training hyperparameters
    # 训练超参数 ---
    lr = 3e-4
    num_learning_epochs = 5
    num_mini_batches = 4
    num_steps_per_env = 48
    min_normalized_std = [0.05, 0.02, 0.05] * 4

    # --- Saving
    # 保存 ---
    model_save_interval = 10


class CustomConfig(StageConfig):
    # TODO: you can refer to LocomotionConfig to design your own track-terrain
    # navigation training stage. The following items need to be specified:
    # 1. stage name;
    # 2. task_type;
    # 3. whether to use hierarchical training;
    # 4. semantics and dimension of the policy action;
    # 5. obs dimension (whether to concatenate goal information);
    # 6. training hyperparameters.
    #
    # After adding a new training stage, a corresponding training config file
    # must be created in the same directory.
    # Filename convention: train_env_conf_<task_type>_<stage.name>.toml
    # Refer to train_env_conf_standard_locomotion.toml as an example.
    #
    # TODO：可参考 LocomotionConfig 自行设计 track 地形导航训练阶段。
    # 需要明确：
    # 1. stage 名称；
    # 2. task_type；
    # 3. 是否采用分层训练；
    # 4. policy action 的语义和维度；
    # 5. obs 维度（是否拼接 goal 信息）；
    # 6. 训练超参。
    #
    # 新增训练阶段后，需在同目录创建对应训练配置文件。
    # 文件命名规则：train_env_conf_<task_type>_<stage.name>.toml
    # 可参考 train_env_conf_standard_locomotion.toml。
    pass


class LocomotionConfig(StageConfig):
    """
    Stage: locomotion — learn stable walking on mixed terrain.
    阶段：locomotion —— 在混合地形上学习稳定行走。
    """

    name = "locomotion"
    task_type = "standard"


class TrackNavConfig(StageConfig):
    """
    Stage: nav — transfer the standard locomotion policy to track traversal.
    阶段：nav —— 将 standard locomotion 策略迁移到 track 赛道通行。

    This appends a compact goal observation to make navigation learnable while
    relying on partial checkpoint loading to preserve standard locomotion
    weights.
    这里追加紧凑目标观测，让导航有可学习的信息；同时依靠部分 checkpoint
    加载保留 standard locomotion 权重。
    """

    name = "nav"
    task_type = "track"
    num_goal_obs = 4
    num_policy_observations = 305  # proprio(45) + scan(256) + goal(4)
    num_critic_observations = 320  # critic_proprio(60) + scan(256) + goal(4)

    # P9 final consolidation starts from a strong candidate; use a lower
    # learning rate to polish the full difficulty band without overwriting
    # level8/9 behavior.
    # P9 最终巩固从强候选出发；继续降低学习率，避免全量分布洗掉 level8/9。
    lr = 5e-5


class Config:
    """
    Unified config entry point.
    统一配置入口。

    Set ``Config.CURRENT`` to a StageConfig subclass, then read
    hyperparameters via ``Config.CURRENT.lr``, ``Config.CURRENT.num_mini_batches``, etc.

    设置 ``Config.CURRENT`` 为某个 StageConfig 子类，然后通过
    ``Config.CURRENT.lr``、``Config.CURRENT.num_mini_batches`` 等读取超参数。
    """

    # Switch stage by changing CURRENT
    # 通过修改 CURRENT 切换阶段
    CURRENT = TrackNavConfig

    @staticmethod
    def load_conf(logger):
        """
        Load user configuration file based on current stage.
        根据当前阶段加载用户配置文件。

        Args:
            logger: logger instance | 日志实例

        Returns:
            tuple: (usr_conf, usr_conf_file, is_eval, stage)
        """
        from common_python.config.config_control import CONFIG
        from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

        stage = Config.CURRENT
        task_type = stage.task_type

        if task_type not in _VALID_TASKS:
            raise ValueError(
                f"Invalid task_type '{task_type}' in stage '{stage.name}'. " f"Only {_VALID_TASKS} are supported."
            )

        # Determine if it's evaluation mode
        # 判断是否为评估模式
        is_eval = False
        if hasattr(CONFIG, "run_mode"):
            is_eval = CONFIG.run_mode in [
                KaiwuDRLDefine.RUN_MODE_EVAL,
                KaiwuDRLDefine.RUN_MODE_EXAM,
            ]

        if is_eval:
            usr_conf_file = f"tools/eval/conf/eval_env_conf.toml"
        else:
            usr_conf_file = f"agent_ppo/conf/train_env_conf_{task_type}_{stage.name}.toml"

        if not is_eval and stage is TrackNavConfig and not os.path.exists(usr_conf_file):
            logger.warning(
                f"Config file not found: {usr_conf_file}; "
                "using built-in full P7 TrackNavConfig fallback. "
                "This path only requires the matching reward_process.py updates."
            )
            usr_conf = _track_nav_fallback_conf()
        else:
            usr_conf = _load_conf(usr_conf_file, logger)

        if usr_conf is None:
            error_msg = f"usr_conf is None, please check {usr_conf_file}"
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info(f"Stage: {stage.name}, task_type: {task_type}, model: {stage.model_class}")

        return usr_conf, usr_conf_file, is_eval, stage


def _deep_merge(base, override):
    """
    Recursively merge override dict into base dict.
    递归将 override 字典合并到 base 字典中（override 优先）。

    Args:
        base: Base config dictionary | 基础配置字典
        override: Override config dictionary | 覆盖配置字典

    Returns:
        dict: Merged config dictionary
    """
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_conf(conf_file, logger):
    """
    Load config: first load base TOML, then deep-merge user TOML on top.
    加载配置：先加载 base TOML，再用用户 TOML 覆盖合并。

    Base files provide model architecture dimensions (num_actions, num_proprio_obs, etc.)
    so user configs only need business-tunable parameters.
    Base 文件提供模型架构维度参数，用户配置只需保留业务可调参数。

    Args:
        conf_file: Path to the user TOML config file | 用户配置文件路径
        logger: Logger instance | 日志实例

    Returns:
        dict: Merged config dictionary, or None on failure
    """
    if not os.path.exists(conf_file):
        logger.error(f"Config file not found: {conf_file}")
        return None

    # Determine base file by mode (eval or train)
    # 根据模式选择 base 文件（eval 或 train）
    mode = "eval" if "eval" in conf_file else "train"
    base_file = os.path.join("tools", "conf", "base", f"{mode}_env_base.toml")

    # Load base config (optional — missing base is not fatal)
    # 加载 base 配置（可选 — base 缺失不致命）
    base_config = {}
    if os.path.exists(base_file):
        try:
            with open(base_file, "r", encoding="utf-8") as f:
                base_config = toml.load(f)
            logger.info(f"Loaded base config: {base_file}")
        except Exception as e:
            logger.warning(f"Cannot load base config: {base_file}. Error: {e}")

    # Load user config
    # 加载用户配置
    try:
        with open(conf_file, "r", encoding="utf-8") as f:
            user_config = toml.load(f)
        logger.info(f"Loaded user config: {conf_file}")
    except Exception as e:
        logger.error(f"Cannot load config file: {conf_file}. Error: {e}")
        return None

    # Deep merge: base ← user (user wins)
    # 深度合并：base ← user（用户配置优先）
    if base_config:
        config = _deep_merge(base_config, user_config)
    else:
        config = user_config

    return config


def _track_nav_fallback_conf():
    """
    Built-in P7 track config for platforms that fail to package newly added
    TOML files. This mirrors train_env_conf_track_nav.toml and expects the
    matching reward_process.py updates to be present.

    当平台没有打包新增 TOML 文件时使用的内置 P7 track 配置。
    内容与 train_env_conf_track_nav.toml 对齐，并假设同步复制了 reward_process.py。
    """
    return {
        "env": {
            "num_envs": 2048,
            "episode_length_s": 120.0,
        },
        "env_conf": {
            "seed": 0,
        },
        "terrain": {
            "mode": "track",
            "num_rows": 10,
            "num_cols": 10,
            "difficulty_range": [0.0, 1.0],
            "curriculum": True,
            "max_init_terrain_level": 9,
            "track": {
                "track_length": 5,
                "sub_terrains": [
                    "pyramid_slope",
                    "pyramid_slope_inv",
                    "pyramid_stairs",
                    "pyramid_stairs_inv",
                    "open_entry_maze",
                ],
            },
        },
        "domain_rand": {
            "enable_domain_rand": True,
            "randomize_friction": True,
            "friction_range": [0.5, 1.3],
            "push_robots": False,
            "push_interval_s": 15,
            "max_push_vel_xy": 0.5,
        },
        "noise": {
            "add_noise": True,
        },
        "init_state": {
            "pos": [0.0, 0.0, 0.35],
        },
        "commands": {
            "resampling_time": [5.0, 5.0],
            "limit": {
                "lin_vel_x": [0.0, 0.8],
                "lin_vel_y": [-0.2, 0.2],
                "ang_vel_z": [-1.2, 1.2],
            },
            "ranges": {
                "lin_vel_x": [0.2, 0.6],
                "lin_vel_y": [-0.1, 0.1],
                "ang_vel_yaw": [-0.8, 0.8],
            },
        },
        "rewards": {
            "track_lin_vel_xy": {
                "weight": 0.6,
                "params": {"std": 0.35, "command_name": "base_velocity"},
            },
            "track_ang_vel_z": {
                "weight": 0.15,
                "params": {"std": 0.4, "command_name": "base_velocity"},
            },
            "feet_air_time": {
                "weight": 0.3,
                "params": {"command_name": "base_velocity", "threshold": 0.2},
            },
            "approach_goal": {"weight": 2.0},
            "reach_goal": {
                "weight": 50.0,
                "params": {"threshold": 0.6},
            },
            "navigation_time": {"weight": -0.01},
            "heuristic_navigation": {
                "weight": 2.0,
                "params": {
                    "obstacle_threshold": -0.3,
                    "near_x_end": 10,
                    "body_y_start": 3,
                    "body_y_end": 13,
                    "speed_std": 0.6,
                    "blocked_scale": 0.6,
                },
            },
            "deadend_escape": {
                "weight": 0.8,
                "params": {
                    "obstacle_threshold": -0.3,
                    "near_x_end": 10,
                    "body_y_start": 3,
                    "body_y_end": 13,
                    "trapped_threshold": 0.3,
                    "progress_threshold": 0.05,
                    "turn_std": 0.6,
                },
            },
            "wall_proximity_brake": {
                "weight": -1.0,
                "params": {
                    "obstacle_threshold": -0.3,
                    "near_x_end": 8,
                    "body_y_start": 3,
                    "body_y_end": 13,
                    "slow_speed": 0.15,
                    "fast_speed": 0.8,
                },
            },
            "obstacle_evasion": {
                "weight": -1.5,
                "params": {
                    "command_name": "base_velocity",
                    "obstacle_threshold": -0.3,
                    "near_x_end": 10,
                    "body_y_start": 3,
                    "body_y_end": 13,
                    "turn_std": 0.5,
                },
            },
            "termination": {"weight": -8.0},
            "lin_vel_z": {"weight": -3.0},
            "ang_vel_xy": {"weight": -0.08},
            "joint_acc": {"weight": -2.5e-7},
            "joint_torques": {"weight": -2.0e-4},
            "dof_pos_limits": {"weight": -2.0},
            "action_rate": {"weight": -0.015},
            "undesired_contacts": {
                "weight": -2.5,
                "params": {"threshold": 1},
            },
            "feet_stumble": {"weight": -0.1},
            "flat_orientation": {"weight": -3.0},
        },
    }
