from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


class P6GoalObsStaticTests(unittest.TestCase):
    def test_track_nav_stage_declares_goal_obs_and_save_interval(self):
        conf = read("agent_ppo/conf/conf.py")
        app_conf = read("conf/configure_app.toml")

        self.assertIn("num_goal_obs = 4", conf)
        self.assertIn("num_policy_observations = 305", conf)
        self.assertIn("num_critic_observations = 320", conf)
        self.assertIn("model_save_interval = 10", conf)
        self.assertIn("dump_model_freq = 10", app_conf)
        self.assertIn("max_init_terrain_level = 9", read("agent_ppo/conf/train_env_conf_track_nav.toml"))

    def test_agent_uses_stage_declared_obs_components(self):
        agent = read("agent_ppo/agent.py")

        self.assertIn("self.num_obs = stage.num_proprio_obs + stage.num_scan + stage.num_goal_obs", agent)
        self.assertIn("actor_obs_shape=(self.num_obs,)", agent)

    def test_policy_and_critic_append_goal_observation(self):
        policy = read("agent_ppo/feature/policy_observation_process.py")
        critic = read("agent_ppo/feature/critic_observation_process.py")

        for source in (policy, critic):
            self.assertIn("build_track_goal_observation(self, obs)", source)
            self.assertIn("obs = self.concatenate_terms(obs, goal_obs)", source)

    def test_goal_observation_does_not_depend_on_missing_platform_helper(self):
        goal_obs = read("agent_ppo/feature/goal_observation.py")
        policy = read("agent_ppo/feature/policy_observation_process.py")
        critic = read("agent_ppo/feature/critic_observation_process.py")

        self.assertIn('getattr(env, "goal_positions", None)', goal_obs)
        self.assertNotIn("goal_position_in_robot_frame()", policy)
        self.assertNotIn("goal_position_in_robot_frame()", critic)

    def test_p63_goal_observation_uses_local_waypoint_target(self):
        goal_obs = read("agent_ppo/feature/goal_observation.py")
        reward_process = read("agent_ppo/feature/reward_process.py")
        policy = read("agent_ppo/feature/policy_observation_process.py")
        critic = read("agent_ppo/feature/critic_observation_process.py")

        self.assertIn("def build_track_navigation_target", goal_obs)
        self.assertIn("def build_track_navigation_command", goal_obs)
        self.assertIn("def apply_track_navigation_command", goal_obs)
        self.assertIn("obs[:, 6:9] = nav_cmd", goal_obs)
        self.assertIn("turn_slowdown = 1.0 - 0.35 * angle_ratio", goal_obs)
        self.assertIn("max_yaw_rate: float = 0.8", goal_obs)
        self.assertIn("blocked_threshold: float = 0.25", goal_obs)
        self.assertIn("waypoint_lateral_m: float = 1.2", goal_obs)
        self.assertNotIn("side_imbalance_threshold", goal_obs)
        self.assertIn("_height_scanner_navigation_hint", goal_obs)
        self.assertIn("use_local_waypoint", goal_obs)
        self.assertIn("build_track_navigation_command", reward_process)
        self.assertNotIn("_waypoint_progress_speed", reward_process)
        self.assertIn("progress_speed = self._goal_progress_speed()", reward_process)
        self.assertIn("def _reward_track_lin_vel_xy", reward_process)
        self.assertIn("def _reward_track_ang_vel_z", reward_process)
        self.assertIn("apply_track_navigation_command(self, obs)", policy)
        self.assertIn("apply_track_navigation_command(self, obs)", critic)

    def test_p73_local_waypoint_uses_goal_side_when_corridor_is_symmetric(self):
        goal_obs = read("agent_ppo/feature/goal_observation.py")

        self.assertIn("side_obstacle_margin: float = 0.08", goal_obs)
        self.assertIn("blocked, right_blocked, left_blocked", goal_obs)
        self.assertIn("def _choose_local_waypoint_y", goal_obs)
        self.assertIn("symmetric_sides = side_delta <= side_obstacle_margin", goal_obs)
        self.assertIn("goal_side_sign = torch.sign(final_y_body)", goal_obs)
        self.assertIn("symmetric_lateral = final_y_body.clamp", goal_obs)
        self.assertIn("return front_blocked, right_blocked, left_blocked", goal_obs)
        self.assertNotIn("side_sign = torch.where(left_blocked <= right_blocked, ones, -ones)", goal_obs)

    def test_partial_checkpoint_loader_still_allows_shape_mismatch(self):
        agent = read("agent_ppo/agent.py")

        self.assertIn("def _load_model_partial", agent)
        self.assertIn("new_param.zero_()", agent)
        self.assertIn("new_param[slices] = old_param[slices]", agent)

    def test_p6_monitor_declares_navigation_reward_panels(self):
        monitor = read("agent_ppo/conf/monitor_builder.py")

        for metric in (
            "reward_approach_goal",
            "reward_heuristic_navigation",
            "reward_deadend_escape",
            "reward_wall_proximity_brake",
        ):
            self.assertIn(f'metrics_name="{metric}"', monitor)
            self.assertIn(f"avg({metric}{{}})", monitor)

        for metric in (
            "nav_cmd_vx_mean",
            "nav_cmd_vy_abs_mean",
            "nav_cmd_wz_abs_mean",
            "nav_goal_dist_mean",
            "nav_target_yaw_abs_mean",
        ):
            self.assertIn(f'metrics_name="{metric}"', monitor)
            self.assertIn(f"avg({metric}{{}})", monitor)

    def test_p64_workflow_reports_navigation_diagnostics(self):
        workflow = read("agent_ppo/workflow/train_workflow.py")

        for metric in (
            "nav_cmd_vx_mean",
            "nav_cmd_vy_abs_mean",
            "nav_cmd_wz_abs_mean",
            "nav_goal_dist_mean",
            "nav_target_yaw_abs_mean",
        ):
            self.assertIn(f'"{metric}"', workflow)

    def test_p6_track_nav_configs_activate_navigation_rewards(self):
        toml_conf = tomllib.loads(read("agent_ppo/conf/train_env_conf_track_nav.toml"))
        fallback = read("agent_ppo/conf/conf.py")

        for reward_name in (
            "approach_goal",
            "heuristic_navigation",
            "deadend_escape",
            "wall_proximity_brake",
        ):
            self.assertIn(reward_name, toml_conf["rewards"])
            self.assertIn(f'"{reward_name}":', fallback)

    def test_p9_full_band_consolidation_keeps_hard_level_available(self):
        toml_conf = tomllib.loads(read("agent_ppo/conf/train_env_conf_track_nav.toml"))
        fallback = read("agent_ppo/conf/conf.py")

        self.assertEqual([0.0, 1.0], toml_conf["terrain"]["difficulty_range"])
        self.assertEqual(9, toml_conf["terrain"]["max_init_terrain_level"])
        self.assertIn('"difficulty_range": [0.0, 1.0]', fallback)
        self.assertIn('"max_init_terrain_level": 9', fallback)

    def test_p6_reward_process_implements_official_navigation_slots(self):
        reward_process = read("agent_ppo/feature/reward_process.py")

        for method_name in (
            "_reward_heuristic_navigation",
            "_reward_deadend_escape",
            "_reward_wall_proximity_brake",
        ):
            self.assertIn(f"def {method_name}", reward_process)


if __name__ == "__main__":
    unittest.main()
