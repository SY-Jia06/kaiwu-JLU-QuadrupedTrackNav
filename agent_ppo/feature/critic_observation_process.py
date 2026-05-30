# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CriticObservationProcess — custom critic observation processor.
CriticObservationProcess — 自定义 critic 观测处理器。

critic obs layout:
  - standard: [critic_proprio(60) | height_scan(256)] → 316 dim
  - track nav: [critic_proprio(60) | height_scan(256) | goal(4)] → 320 dim
critic 观测布局：
  - standard：[critic_proprio(60) | height_scan(256)] → 316 维
  - track nav：[critic_proprio(60) | height_scan(256) | goal(4)] → 320 维

When extending to track terrain, please refer to the extension guide in
policy_observation_process.py; the critic observation must stay in sync
with the policy on the task-information convention.
扩展到 track 地形时，请参考 policy_observation_process.py 的扩展指引；
critic 观测需保持与 policy 同步的任务信息约定。
"""

from agent_ppo.conf.conf import Config
from agent_ppo.feature.goal_observation import apply_track_navigation_command, build_track_goal_observation
from tools.base_env.observation_process import ObservationProcess


class CriticObservationProcess(ObservationProcess):
    target_group = "critic"

    def process(self):
        obs = self.default_observation()
        if Config.CURRENT.num_goal_obs <= 0:
            return obs

        obs = apply_track_navigation_command(self, obs)
        goal_obs = build_track_goal_observation(self, obs)
        obs = self.concatenate_terms(obs, goal_obs)
        return obs
