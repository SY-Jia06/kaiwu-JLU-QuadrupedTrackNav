# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PolicyObservationProcess — custom policy observation processor.
PolicyObservationProcess — 自定义 policy 观测处理器。

obs layout:
  - standard: [proprio(45) | height_scan(256)] → 301 dim
  - track nav: [proprio(45) | height_scan(256) | goal(4)] → 305 dim
观测布局：
  - standard：[proprio(45) | height_scan(256)] → 301 维
  - track nav：[proprio(45) | height_scan(256) | goal(4)] → 305 维

Extending to track terrain (optional):
    In track terrain the environment additionally provides the following
    read-only attributes (not available in standard terrain):
      - env.goal_positions  (num_envs, 3)  — exit position in world frame
      - env.goal_yaw        (num_envs,)    — exit heading in world frame
    The environment always exposes these scene sensors (available in both
    standard and track terrains, accessed via env.scene.sensors["<name>"]):
      - "height_scanner"  — default forward ground-clearance scan
      - "nav_scanner"     — forward-looking occlusion scan (wider range,
                             suited for obstacle avoidance / turning)
    Players can construct their own obs from these inputs. After appending,
    update the Stage config (observation dim) and model input dim accordingly.

扩展到 track 地形时（可选）：
    track 地形下，环境会额外提供以下只读属性（standard 地形没有）：
      - env.goal_positions  (num_envs, 3)  — 出口在世界坐标系下的 3D 位置
      - env.goal_yaw        (num_envs,)    — 出口在世界坐标系下的朝向
    环境在两种地形下都会通过 env.scene.sensors["<name>"] 提供以下传感器：
      - "height_scanner"  — 默认前方地面高度扫描
      - "nav_scanner"     — 前瞻遮挡扫描（范围更大，适合避障 / 转向判断）
    选手可从这些属性和传感器自行构造 obs。
    拼接后需同步修改 Stage 的观测维度和 model 输入维度。
"""

from agent_ppo.conf.conf import Config
from agent_ppo.feature.goal_observation import apply_track_navigation_command, build_track_goal_observation
from tools.base_env.observation_process import ObservationProcess


class PolicyObservationProcess(ObservationProcess):
    target_group = "policy"

    def process(self):
        obs = self.default_observation()
        return self._append_track_goal_obs(obs)

    def _append_track_goal_obs(self, obs):
        """Append compact track goal features when the track API is available.

        Track goal state may be absent during Isaac Lab's observation-manager
        shape probing; the helper returns zeros until env.goal_positions and the
        robot asset are available.
        """
        if Config.CURRENT.num_goal_obs <= 0:
            return obs

        obs = apply_track_navigation_command(self, obs)
        goal_obs = build_track_goal_observation(self, obs)
        obs = self.concatenate_terms(obs, goal_obs)
        return obs
