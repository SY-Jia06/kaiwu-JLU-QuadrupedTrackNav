#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Compact waypoint and command helpers for track navigation."""

import torch


def build_track_goal_observation(process, obs):
    """Return [target_x_body, target_y_body, goal_distance, target_yaw_error].

    During Isaac Lab observation-manager shape probing, track goal state or the
    robot asset may not be fully initialized yet. Return zeros in that case so
    the declared 4-dim contract is stable through reset.
    """
    num_envs = obs.shape[0]
    zeros = torch.zeros(num_envs, 4, device=obs.device, dtype=obs.dtype)

    target = build_track_navigation_target(process.env, obs)
    if target is None:
        return zeros

    target_x_body, target_y_body, goal_distance, _use_local_waypoint = target
    yaw_error = torch.atan2(target_y_body, target_x_body)

    # Keep features in the same compact numeric range as P6.2.
    return torch.stack(
        [
            target_x_body.clamp(min=-20.0, max=20.0) / 20.0,
            target_y_body.clamp(min=-20.0, max=20.0) / 20.0,
            goal_distance.clamp(max=20.0) / 20.0,
            yaw_error / torch.pi,
        ],
        dim=1,
    )


def apply_track_navigation_command(process, obs):
    """Overwrite obs[6:9] velocity command with a waypoint-derived command.

    The pretrained locomotion policy already knows how to react to the
    base_velocity command slot. Feeding navigation through that slot is a much
    stronger interface than asking the policy to discover new appended features
    from scratch.
    """
    if obs.shape[1] < 9:
        return obs

    nav_cmd = build_track_navigation_command(process.env, obs)
    if nav_cmd is None:
        return obs

    obs = obs.clone()
    obs[:, 6:9] = nav_cmd.to(device=obs.device, dtype=obs.dtype)
    return obs


def build_track_navigation_command(
    env,
    reference,
    max_forward: float = 0.65,
    min_forward: float = 0.45,
    lateral_gain: float = 0.25,
    yaw_gain: float = 0.65,
    max_yaw_rate: float = 0.8,
):
    """Build [vx, vy, wz] command from the current navigation target."""
    target = build_track_navigation_target(env, reference)
    if target is None:
        return None

    target_x_body, target_y_body, goal_distance, use_local_waypoint = target
    target_angle = torch.atan2(target_y_body, target_x_body)
    abs_angle = torch.abs(target_angle)

    # Do not let large yaw error turn into "stop and rotate". The transferred
    # locomotion policy is better at walking while turning than at in-place
    # navigation, and stopping kills approach_goal.
    angle_ratio = (abs_angle / (0.5 * torch.pi)).clamp(min=0.0, max=1.0)
    turn_slowdown = 1.0 - 0.35 * angle_ratio
    distance_scale = (goal_distance / 3.0).clamp(min=0.0, max=1.0)
    local_slowdown = torch.where(use_local_waypoint > 0.5, 0.85, 1.0)
    vx = max_forward * turn_slowdown * distance_scale * local_slowdown
    vx = torch.where(goal_distance > 0.8, vx.clamp(min=min_forward), torch.zeros_like(vx))

    vy = (lateral_gain * torch.sin(target_angle)).clamp(min=-0.15, max=0.15)
    wz = (yaw_gain * target_angle).clamp(min=-max_yaw_rate, max=max_yaw_rate)
    return torch.stack([vx, vy, wz], dim=1)


def build_track_navigation_target(
    env,
    reference,
    obstacle_threshold: float = -0.3,
    blocked_threshold: float = 0.25,
    waypoint_forward_m: float = 3.0,
    waypoint_lateral_m: float = 1.2,
    side_obstacle_margin: float = 0.08,
):
    """Return body-frame navigation target with a local waypoint near obstacles.

    The final goal is still used when the front corridor is open. When the
    height scanner says the front corridor is blocked, choose the less blocked
    side only when the side evidence is clear. In symmetric corridors, keep the
    waypoint on the final-goal side so stairs stay straight and maze ties do not
    drift into a fixed detour. The final-goal distance is returned separately so
    the policy keeps long-horizon context while receiving local steering.
    """
    goal_positions = getattr(env, "goal_positions", None)
    if goal_positions is None:
        return None

    robot = _get_robot_asset(env)
    if robot is None or not hasattr(robot, "data") or not hasattr(robot.data, "root_pos_w"):
        return None

    robot_pos = robot.data.root_pos_w[:, :2].to(device=reference.device, dtype=reference.dtype)
    goal_pos = goal_positions[:, :2].to(device=reference.device, dtype=reference.dtype)
    delta_w = goal_pos - robot_pos

    yaw = _get_robot_yaw(robot, reference)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    final_x_body = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
    final_y_body = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]
    goal_distance = torch.norm(delta_w, dim=1)

    blocked, right_blocked, left_blocked = _height_scanner_navigation_hint(
        env,
        reference,
        obstacle_threshold=obstacle_threshold,
    )
    use_local_waypoint = blocked > blocked_threshold

    local_x = torch.full_like(final_x_body, waypoint_forward_m)
    local_y = _choose_local_waypoint_y(
        final_y_body,
        right_blocked,
        left_blocked,
        waypoint_lateral_m=waypoint_lateral_m,
        side_obstacle_margin=side_obstacle_margin,
    )
    target_x_body = torch.where(use_local_waypoint, local_x, final_x_body)
    target_y_body = torch.where(use_local_waypoint, local_y, final_y_body)

    return target_x_body, target_y_body, goal_distance, use_local_waypoint.float()


def _choose_local_waypoint_y(
    final_y_body,
    right_blocked,
    left_blocked,
    waypoint_lateral_m: float,
    side_obstacle_margin: float,
):
    side_delta = torch.abs(left_blocked - right_blocked)
    symmetric_sides = side_delta <= side_obstacle_margin

    ones = torch.ones_like(final_y_body)
    obstacle_direction = torch.where(left_blocked <= right_blocked, ones, -ones)
    goal_side_sign = torch.sign(final_y_body)
    goal_side_sign = torch.where(goal_side_sign == 0.0, obstacle_direction, goal_side_sign)

    symmetric_lateral = final_y_body.clamp(min=-waypoint_lateral_m, max=waypoint_lateral_m)
    symmetric_lateral = goal_side_sign * torch.abs(symmetric_lateral)
    obstacle_lateral = obstacle_direction * waypoint_lateral_m
    return torch.where(symmetric_sides, symmetric_lateral, obstacle_lateral)


def _height_scanner_navigation_hint(env, reference, obstacle_threshold: float = -0.3):
    sensors = getattr(getattr(env, "scene", None), "sensors", {})
    zeros = torch.zeros(reference.shape[0], device=reference.device, dtype=reference.dtype)

    if "height_scanner" not in sensors:
        return zeros, zeros, zeros

    sensor = sensors["height_scanner"]
    if not hasattr(sensor, "data") or not hasattr(sensor.data, "ray_hits_w"):
        return zeros, zeros, zeros

    scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
    grid = scan.reshape(reference.shape[0], 16, 16).to(device=reference.device, dtype=reference.dtype)

    near = grid[:, 3:13, :10]
    front_blocked = (near < obstacle_threshold).any(dim=-1).float().mean(dim=-1)

    right = grid[:, :8, :12]
    left = grid[:, 8:, :12]
    right_blocked = (right < obstacle_threshold).float().mean(dim=(1, 2))
    left_blocked = (left < obstacle_threshold).float().mean(dim=(1, 2))

    return front_blocked, right_blocked, left_blocked


def _get_robot_asset(env):
    scene = getattr(env, "scene", None)
    if scene is None:
        return None

    for name in ("robot", "unitree_go2", "go2"):
        try:
            return scene[name]
        except (KeyError, TypeError):
            pass

    for attr in ("articulations", "assets"):
        container = getattr(scene, attr, None)
        if isinstance(container, dict):
            for asset in container.values():
                if hasattr(getattr(asset, "data", None), "root_pos_w"):
                    return asset

    return None


def _get_robot_yaw(robot, obs):
    quat = getattr(robot.data, "root_quat_w", None)
    if quat is None:
        return torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)

    quat = quat.to(device=obs.device, dtype=obs.dtype)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_to_pi(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))
