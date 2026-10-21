"""F1-style racing environment for Gymnasium.

Importing this package registers ``"F1Track-v0"`` so it can be created with
``gymnasium.make`` just like the built-in environments.
"""

from gymnasium.envs.registration import register

from .env import F1TrackEnv, gap_follower, make_track

register(id="F1Track-v0", entry_point="f1_gym.env:F1TrackEnv")

__all__ = ["F1TrackEnv", "gap_follower", "make_track"]
