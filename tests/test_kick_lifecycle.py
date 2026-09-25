from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from booster_mjlab.tasks.kick.mdp._kick_state import advance_kick_progress
from booster_mjlab.tasks.kick.mdp.commands import KickCommand, KickCommandCfg
from booster_mjlab.tasks.kick.mdp.observations import noisy_ball_state_relative_b
from booster_mjlab.tasks.kick.mdp.rewards import (
    accepted_launch,
    launch_quality,
    stable_completion,
    useful_contact,
)


class _CommandManager:
    def __init__(self, command: KickCommand):
        self.command = command

    def get_term(self, _name: str) -> KickCommand:
        return self.command


class _TerminationManager:
    def __init__(self, num_envs: int):
        self.active_terms = ["fell_over", "illegal_contact", "mis_kick"]
        self.values = {
            name: torch.zeros(num_envs, dtype=torch.bool) for name in self.active_terms
        }

    def get_term(self, name: str) -> torch.Tensor:
        return self.values[name]


def _make_env(num_envs: int = 2):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
            root_link_pos_w=torch.zeros(num_envs, 3),
            root_link_lin_vel_w=torch.zeros(num_envs, 3),
            root_link_ang_vel_w=torch.zeros(num_envs, 3),
        )
    )
    ball = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.zeros(num_envs, 3),
            root_link_lin_vel_w=torch.zeros(num_envs, 3),
        )
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(found=torch.zeros(num_envs, 1, dtype=torch.bool))
    )

    class _Scene:
        sensors = {"foot_ball": sensor}

        def __getitem__(self, name: str):
            return {"robot": robot, "ball": ball}[name]

    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene=_Scene())
    cfg = KickCommandCfg(
        entity_name="robot",
        ball_entity_name="ball",
        resampling_time_range=(1.0e9, 1.0e9),
        ranges=KickCommandCfg.Ranges(
            speed=(2.0, 2.0),
            yaw=(0.0, 0.0),
            chip_angle=(0.0, 0.0),
            speed_tolerance_fraction=(0.25, 0.25),
            direction_tolerance=(math.radians(10.0), math.radians(10.0)),
            elevation_tolerance=(math.radians(5.0), math.radians(5.0)),
            urgency=(0.5, 0.5),
        ),
    )
    command = KickCommand(cfg, env)
    command.target_vel_w[:] = torch.tensor([2.0, 0.0, 0.0])
    command.speed_min[:] = 1.5
    command.speed_max[:] = 2.5
    command.direction_tolerance[:] = math.radians(10.0)
    command.elevation_tolerance[:] = math.radians(5.0)
    env.command_manager = _CommandManager(command)
    env.termination_manager = _TerminationManager(num_envs)
    return env, command, sensor, ball


def _advance(env, delay: int, recovery: int) -> torch.Tensor:
    return advance_kick_progress(
        env,
        "foot_ball",
        "kick",
        SceneEntityCfg("ball"),
        delay,
        recovery,
        ("fell_over", "illegal_contact", "mis_kick"),
    )


class KickLifecycleTest(unittest.TestCase):
    def test_event_rewards_keep_whole_event_units_under_dt_scaling(self) -> None:
        env, command, _sensor, _ball = _make_env()
        env.step_dt = 0.02
        env.cfg = SimpleNamespace(scale_rewards_by_dt=True)
        command.just_contacted[0] = True
        command.just_measured[0] = True
        command.accepted_launch[0] = True
        command.just_completed[0] = True
        command.achieved_vel_w[0] = torch.tensor([2.0, 0.0, 0.0])

        # RewardManager later multiplies each value by weight * step_dt.
        contact_return = useful_contact(env, "kick")[0] * 1.0 * env.step_dt
        quality_return = launch_quality(env, "kick")[0] * 2.0 * env.step_dt
        accepted_return = accepted_launch(env, "kick")[0] * 7.0 * env.step_dt
        recovery_return = stable_completion(env, "kick")[0] * 8.0 * env.step_dt
        self.assertAlmostEqual(contact_return.item(), 1.0)
        self.assertAlmostEqual(quality_return.item(), 2.0)
        self.assertAlmostEqual(accepted_return.item(), 7.0)
        self.assertAlmostEqual(recovery_return.item(), 8.0)

    def test_partial_observation_reset_does_not_advance_other_envs(self) -> None:
        env, _command, _sensor, ball = _make_env()
        env.step_dt = 0.02
        env.common_step_counter = 0
        cfg = SimpleNamespace(
            params={
                "pos_noise_std": 0.0,
                "vel_noise_std": 0.0,
                "latency_steps": 2,
                "dropout_prob": 0.0,
            }
        )
        observation = noisy_ball_state_relative_b(cfg, env)
        observation(env)
        env.common_step_counter = 1
        ball.data.root_link_pos_w[:, 0] = torch.tensor([1.0, 2.0])
        observation(env)
        unaffected_buffer = observation.buffer[1].clone()

        observation.reset(torch.tensor([0]))
        ball.data.root_link_pos_w[:, 0] = torch.tensor([3.0, 4.0])
        observation(env)
        self.assertTrue(torch.equal(observation.buffer[1], unaffected_buffer))

    def test_delay_counts_subsequent_steps_and_recovery(self) -> None:
        env, command, sensor, ball = _make_env()
        ball.data.root_link_lin_vel_w[0] = torch.tensor([2.0, 0.0, 0.0])
        sensor.data.found[0] = True

        self.assertFalse(_advance(env, delay=3, recovery=2)[0])
        self.assertTrue(command.just_contacted[0])
        self.assertFalse(command.has_kicked[0])
        self.assertEqual(command.measure_countdown[0], 3)

        for expected in (2, 1):
            self.assertFalse(_advance(env, delay=3, recovery=2)[0])
            self.assertEqual(command.measure_countdown[0], expected)
            self.assertFalse(command.has_kicked[0])

        self.assertFalse(_advance(env, delay=3, recovery=2)[0])
        self.assertTrue(command.has_kicked[0])
        self.assertTrue(command.accepted_launch[0])
        self.assertTrue(command.just_measured[0])
        self.assertEqual(command.recovery_countdown[0], 2)

        self.assertFalse(_advance(env, delay=3, recovery=2)[0])
        done = _advance(env, delay=3, recovery=2)
        self.assertTrue(done[0])
        self.assertTrue(command.stable_success[0])
        self.assertTrue(command.just_completed[0])
        self.assertEqual(command.metrics["stable_success"][0], 1.0)

    def test_zero_delay_measures_on_contact_step(self) -> None:
        env, command, sensor, ball = _make_env()
        ball.data.root_link_lin_vel_w[0] = torch.tensor([0.0, 2.0, 0.0])
        sensor.data.found[0] = True

        done = _advance(env, delay=0, recovery=4)
        self.assertTrue(done[0])
        self.assertTrue(command.has_kicked[0])
        self.assertFalse(command.accepted_launch[0])
        self.assertTrue(command.attempt_finished[0])

    def test_invalid_recovery_never_becomes_success(self) -> None:
        env, command, sensor, ball = _make_env()
        ball.data.root_link_lin_vel_w[0] = torch.tensor([2.0, 0.0, 0.0])
        sensor.data.found[0] = True
        self.assertFalse(_advance(env, delay=0, recovery=1)[0])
        env.termination_manager.values["fell_over"][0] = True

        done = _advance(env, delay=0, recovery=1)
        self.assertFalse(done[0])
        self.assertFalse(command.stable_success[0])

    def test_recovery_waits_for_low_torso_rotation(self) -> None:
        env, command, sensor, ball = _make_env()
        ball.data.root_link_lin_vel_w[0] = torch.tensor([2.0, 0.0, 0.0])
        sensor.data.found[0] = True
        self.assertFalse(_advance(env, delay=0, recovery=1)[0])

        # The recovery timer has elapsed, but hand-off remains unsafe while the
        # torso is still rotating. It completes once the robot settles.
        env.scene["robot"].data.root_link_ang_vel_w[0, 0] = 2.0
        self.assertFalse(_advance(env, delay=0, recovery=1)[0])
        self.assertFalse(command.stable_success[0])

        env.scene["robot"].data.root_link_ang_vel_w[0, 0] = 0.0
        done = _advance(env, delay=0, recovery=1)
        self.assertTrue(done[0])
        self.assertTrue(command.stable_success[0])

    def test_resampling_explicitly_clears_episode_state(self) -> None:
        env, command, _sensor, _ball = _make_env()
        ids = torch.arange(env.num_envs)
        command.has_contacted[:] = True
        command.has_kicked[:] = True
        command.stable_success[:] = True
        command.measure_countdown[:] = 2
        command.metrics["contacted"][:] = 1.0

        command._resample_command(ids)
        self.assertFalse(command.has_contacted.any())
        self.assertFalse(command.has_kicked.any())
        self.assertFalse(command.stable_success.any())
        self.assertTrue(torch.all(command.measure_countdown == -1))
        self.assertFalse(command.episode_active.any())

        command._resample_command(ids)
        self.assertTrue(command.episode_active.all())


if __name__ == "__main__":
    unittest.main()
