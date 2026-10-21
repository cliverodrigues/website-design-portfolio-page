"""Tests for the F1 track environment. Run with ``pytest`` from projects/f1-gym."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import f1_gym
from f1_gym import F1TrackEnv, gap_follower


def test_passes_gymnasium_checker():
    check_env(gym.make("F1Track-v0").unwrapped, skip_render_check=True)


def test_reset_is_reproducible_with_seed():
    a, _ = F1TrackEnv().reset(seed=42)
    b, _ = F1TrackEnv().reset(seed=42)
    np.testing.assert_array_equal(a, b)


def test_leaving_track_terminates_with_penalty():
    env = F1TrackEnv()
    env.reset(seed=0)
    # Full throttle, full left lock: the car runs off the inside.
    for _ in range(500):
        _, reward, terminated, _, _ = env.step(np.array([1.0, 1.0]))
        if terminated:
            break
    assert terminated
    assert reward < -5


def test_baseline_driver_completes_a_lap():
    env = F1TrackEnv(n_laps=1)
    obs, _ = env.reset(seed=0)
    while True:
        obs, _, terminated, truncated, info = env.step(gap_follower(obs))
        if terminated or truncated:
            break
    assert info["laps"] == 1
    assert 20.0 < info["lap_times"][0] < 120.0


def test_rays_see_edges_on_straight_start():
    env = F1TrackEnv()
    obs, _ = env.reset(seed=0)
    side_rays = obs[[0, 6]] * env.RAY_RANGE
    # The car starts near the middle, so the side rays hit the edges about
    # half a track width away.
    assert np.all(side_rays < env.TRACK_WIDTH)


@pytest.mark.parametrize("kwargs", [{"n_laps": 0}, {"max_steps": 0}, {"render_mode": "x"}])
def test_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        F1TrackEnv(**kwargs)
