# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
RewardProcess — custom reward processor (lite baseline).
RewardProcess — 自定义奖励处理器（lite baseline）。

This file only ships two example rewards:
    1. _reward_reach_goal       — goal-reaching judgment (0.6 m)
    2. _reward_forward_velocity — forward velocity reward (dense, demonstrates reward writing style)
本文件仅预置两个示例 reward：
    1. _reward_reach_goal       — 赛题到达判定（0.6 m）
    2. _reward_forward_velocity — 前向速度奖励（dense，展示 reward 写法）

Other generic locomotion rewards (track_lin_vel_xy / joint_acc / action_rate, etc.)
are inherited from RewardProcessBase (see tools/base_env/base_reward.py).
Players only need to activate them in the TOML; no need to re-implement them here.
其余通用 locomotion reward（track_lin_vel_xy / joint_acc / action_rate 等）
继承自 RewardProcessBase（见 tools/base_env/base_reward.py），
选手在 TOML 中激活即可，无需在此重复实现。

If players need to train a navigation policy, please add more rewards in this file.
选手若需训练导航策略，请在本文件自行添加更多 reward。
"""

import torch

from agent_ppo.feature.goal_observation import build_track_navigation_command
from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):
    def _zero_reward(self):
        return torch.zeros(self.env.num_envs, device=self.env.device)

    def _front_obstacle_density(
        self,
        obstacle_threshold: float = -0.3,
        near_x_end: int = 10,
        body_y_start: int = 3,
        body_y_end: int = 13,
    ):
        sensors = getattr(self.env.scene, "sensors", {})
        if "height_scanner" not in sensors:
            return self._zero_reward()

        sensor = sensors["height_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.reshape(self.env.num_envs, 16, 16)
        window = grid[:, body_y_start:body_y_end, :near_x_end]
        blocked_cols = (window < obstacle_threshold).any(dim=-1).float()
        return blocked_cols.mean(dim=-1)

    def _goal_progress_speed(self):
        goal_positions = getattr(self.env, "goal_positions", None)
        if goal_positions is None:
            return self._zero_reward()

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = goal_positions[:, :2]
        goal_delta = goal_pos - robot_pos
        goal_dist = torch.norm(goal_delta, dim=1, keepdim=True).clamp(min=1e-6)
        goal_dir = goal_delta / goal_dist

        lin_vel_w = getattr(robot.data, "root_lin_vel_w", None)
        if lin_vel_w is None:
            return self._zero_reward()

        return torch.sum(lin_vel_w[:, :2] * goal_dir, dim=1)

    def _navigation_command(self):
        robot = self._get_robot_asset()
        return build_track_navigation_command(self.env, robot.data.root_pos_w)

    def _reward_reach_goal(self, threshold: float = 0.6):
        """Reward for reaching the maze exit (returns 1.0 when distance < 0.6 m).
        到达迷宫出口奖励（distance < 0.6 m 时返回 1.0）。

        Note:
            The threshold must match the threshold of _goal_reached_termination
            in tools/unitree_rl_lab/.../velocity_env_cfg.py (currently 0.6 m),
            otherwise a "termination-reward dead zone" will appear.
            threshold 必须与 tools/unitree_rl_lab/.../velocity_env_cfg.py 中
            _goal_reached_termination 的 threshold 一致（当前 0.6 m），
            否则会产生"终止-奖励死区"。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()

    def _reward_forward_velocity(self):
        """Forward velocity reward: x-direction velocity in the robot body frame (the larger the better).
        前向速度奖励：机器人本体坐标系下 x 方向速度（越大越好）。

        This is an example reward that demonstrates how to read the robot state and
        build a dense signal.
        示例性 reward，展示如何读取机器人状态并构造 dense signal。
        """
        robot = self._get_robot_asset()
        return robot.data.root_lin_vel_b[:, 0]

    def _reward_feet_air_time(self, command_name: str = "base_velocity", threshold: float = 0.2):
        """Reward long enough foot air time when a forward command is active.
        当前进命令存在时，奖励足端有足够滞空时间，用来鼓励形成步态。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")

        first_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] == 0.0
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

        nav_cmd = self._navigation_command()
        if nav_cmd is None:
            nav_cmd = self.env.command_manager.get_command(command_name)
        is_moving = torch.norm(nav_cmd[:, :2], dim=1) > 0.1
        return reward * is_moving.float()

    def _reward_track_lin_vel_xy(self, std: float = 0.35, command_name: str = "base_velocity"):
        """Track the waypoint-derived navigation command instead of random env command."""
        nav_cmd = self._navigation_command()
        if nav_cmd is None:
            nav_cmd = self.env.command_manager.get_command(command_name)

        robot = self._get_robot_asset()
        vel_error = torch.sum(torch.square(nav_cmd[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1)
        return torch.exp(-vel_error / (std * std))

    def _reward_track_ang_vel_z(self, std: float = 0.4, command_name: str = "base_velocity"):
        """Track the waypoint-derived yaw command instead of random env command."""
        nav_cmd = self._navigation_command()
        if nav_cmd is None:
            nav_cmd = self.env.command_manager.get_command(command_name)

        robot = self._get_robot_asset()
        yaw_error = torch.square(nav_cmd[:, 2] - robot.data.root_ang_vel_b[:, 2])
        return torch.exp(-yaw_error / (std * std))

    def _reward_approach_goal(self):
        """Reward reducing distance to the track/maze exit.
        距离 track/maze 出口变近时给正奖励。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        current_dist = torch.norm(goal_pos - robot_pos, dim=1)

        if not hasattr(self.env, "_previous_goal_dist") or self.env._previous_goal_dist is None:
            self.env._previous_goal_dist = current_dist.clone()
            return torch.zeros(self.env.num_envs, device=self.env.device)

        delta_dist = current_dist - self.env._previous_goal_dist

        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        delta_dist[reset_mask] = 0.0

        self.env._previous_goal_dist = current_dist.clone()
        return -delta_dist

    def _reward_navigation_time(self):
        """Constant per-step signal, weighted negatively in TOML to prefer faster traversal.
        每步固定信号，在 TOML 中配负权重，用来鼓励更快通过。
        """
        return torch.ones(self.env.num_envs, device=self.env.device)

    def _reward_heuristic_navigation(
        self,
        obstacle_threshold: float = -0.3,
        near_x_end: int = 10,
        body_y_start: int = 3,
        body_y_end: int = 13,
        speed_std: float = 0.6,
        blocked_scale: float = 0.6,
    ):
        """Reward velocity projected toward the track goal, damped when blocked ahead.
        奖励朝目标方向的速度；前方被遮挡时降低该奖励，避免学会冲墙。
        """
        progress_speed = self._goal_progress_speed()
        toward_goal = torch.tanh(progress_speed / speed_std)
        blocked = self._front_obstacle_density(
            obstacle_threshold=obstacle_threshold,
            near_x_end=near_x_end,
            body_y_start=body_y_start,
            body_y_end=body_y_end,
        )
        clear_factor = (1.0 - blocked_scale * blocked).clamp(min=0.0, max=1.0)
        return toward_goal * clear_factor

    def _reward_deadend_escape(
        self,
        obstacle_threshold: float = -0.3,
        near_x_end: int = 10,
        body_y_start: int = 3,
        body_y_end: int = 13,
        trapped_threshold: float = 0.3,
        progress_threshold: float = 0.05,
        turn_std: float = 0.6,
    ):
        """Reward turning when the front scanner says blocked and goal progress stalls.
        前方堵住且朝目标推进停滞时，奖励主动转向脱困。
        """
        blocked = self._front_obstacle_density(
            obstacle_threshold=obstacle_threshold,
            near_x_end=near_x_end,
            body_y_start=body_y_start,
            body_y_end=body_y_end,
        )
        progress_speed = self._goal_progress_speed()
        robot = self._get_robot_asset()
        yaw_rate = torch.abs(robot.data.root_ang_vel_b[:, 2])

        trapped = (blocked > trapped_threshold) & (torch.abs(progress_speed) < progress_threshold)
        turning = torch.tanh(yaw_rate / turn_std)
        return trapped.float() * turning

    def _reward_wall_proximity_brake(
        self,
        obstacle_threshold: float = -0.3,
        near_x_end: int = 8,
        body_y_start: int = 3,
        body_y_end: int = 13,
        slow_speed: float = 0.15,
        fast_speed: float = 0.8,
    ):
        """Penalize high forward body velocity when obstacles are close ahead.
        前方近距离有墙/障碍时惩罚高速前冲。
        """
        blocked = self._front_obstacle_density(
            obstacle_threshold=obstacle_threshold,
            near_x_end=near_x_end,
            body_y_start=body_y_start,
            body_y_end=body_y_end,
        )
        robot = self._get_robot_asset()
        forward_speed = torch.relu(robot.data.root_lin_vel_b[:, 0] - slow_speed)
        speed_ratio = (forward_speed / max(fast_speed - slow_speed, 1e-6)).clamp(max=1.0)
        return blocked * speed_ratio

    def _reward_termination(self):
        """Penalize real failures while excluding successful goal-reaching terminations.
        惩罚真实失败，同时排除到达目标导致的成功终止。
        """
        term_mgr = self.env.termination_manager
        failure = term_mgr.terminated & ~term_mgr.time_outs

        active_terms = getattr(term_mgr, "active_terms", [])
        if "goal_reached" in active_terms:
            failure = failure & ~term_mgr.get_term("goal_reached")

        return failure.float()

    def _reward_obstacle_evasion(
        self,
        command_name: str = "base_velocity",
        obstacle_threshold: float = -0.3,
        near_x_end: int = 10,
        body_y_start: int = 3,
        body_y_end: int = 13,
        turn_std: float = 0.5,
    ):
        """Penalize driving straight into nearby obstacles.
        惩罚前方近距离有障碍时仍然直冲。

        This uses the existing height scanner, so it does not change the policy
        observation dimension.
        该项只使用已有 height scanner，不改变 policy observation 维度。
        """
        sensors = self.env.scene.sensors
        if "height_scanner" not in sensors:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        asset = self._get_robot_asset()
        sensor = sensors["height_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.view(self.env.num_envs, 16, 16)

        window = grid[:, body_y_start:body_y_end, :near_x_end]
        col_blocked = (window < obstacle_threshold).any(dim=-1).float()
        blocked = col_blocked.mean(dim=-1)

        yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
        not_evading = torch.exp(-yaw_rate / turn_std)

        cmd = self._navigation_command()
        if cmd is None:
            cmd = self.env.command_manager.get_command(command_name)
        has_fwd_cmd = (cmd[:, 0] > 0.05).float()
        return blocked * not_evading * has_fwd_cmd

    def _reward_feet_stumble(self):
        """Penalize feet hitting vertical surfaces such as stair edges or walls.
        惩罚足端撞到台阶边缘或墙面等垂直面。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]

        forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
        forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        return torch.any(forces_xy > 5 * forces_z, dim=1).float()
