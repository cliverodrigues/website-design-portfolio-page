"""A lightweight F1-style racing environment for Gymnasium.

The car drives around a procedurally shaped closed circuit. The agent sees the
track the way a simple LIDAR would (distance rays), plus its own speed and
alignment with the track, and controls steering and throttle/brake.

Why it is built this way:

* **Pure NumPy, no physics engine** - it runs thousands of steps per second on
  a laptop CPU, so a PPO agent trains in minutes without needing the GPU.
* **Kinematic bicycle model with a grip limit** - simple enough to read, but
  the car still understeers if it enters a corner too fast, so the agent has
  to learn to brake before turns (the core skill of racing).
* **Progress-based reward** - the agent is paid for metres travelled along the
  centre line, not for raw speed, so driving in circles or backwards does not
  pay.

Example:
    >>> import gymnasium as gym
    >>> import f1_gym  # registers "F1Track-v0"
    >>> env = gym.make("F1Track-v0")
    >>> obs, info = env.reset(seed=0)
    >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Ray angles relative to the car's heading, in radians. Positive = left.
RAY_ANGLES = np.deg2rad([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])


def make_track(n_points: int = 400, scale: float = 220.0) -> np.ndarray:
    """Build the centre line of a closed, twisty circuit.

    A circle whose radius is wobbled by a few sine/cosine terms gives a mix of
    fast sweepers and tighter hairpins while staying free of self-crossings.

    Args:
        n_points: Number of points sampled along the centre line.
        scale: Base radius of the circuit in metres.

    Returns:
        Array of shape ``(n_points, 2)`` with x/y coordinates in metres,
        ordered counter-clockwise.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    radius = scale * (
        1.0
        + 0.25 * np.sin(2.0 * theta)
        + 0.15 * np.cos(3.0 * theta)
        + 0.05 * np.sin(5.0 * theta)
    )
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)


class F1TrackEnv(gym.Env):
    """Single-car time-trial environment.

    Observation (``Box(10,)``, all roughly in ``[-1, 1]``):
        * ``0-6``  distance to the track edge along 7 rays, divided by range
        * ``7``    speed divided by top speed
        * ``8``    heading error vs. track direction, divided by pi
        * ``9``    lateral offset from the centre line, divided by half width

    Action (``Box(2,)`` in ``[-1, 1]``):
        * ``0``    steering (positive = left)
        * ``1``    throttle (positive) / brake (negative)

    Reward:
        ``+0.1`` per metre of progress along the centre line, ``-0.01`` per
        step (time penalty), ``-10`` for leaving the track and ``+50`` per
        completed lap.

    Episode ends when the car leaves the track or finishes ``n_laps`` laps
    (terminated), or after ``max_steps`` steps (truncated).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # --- Car / physics constants (SI units) -------------------------------
    DT = 0.1  # seconds per step
    WHEELBASE = 3.6  # metres
    MAX_STEER = 0.4  # radians (~23 degrees)
    MAX_ACCEL = 12.0  # m/s^2 at full throttle
    MAX_BRAKE = 30.0  # m/s^2 at full brake (F1 brakes are brutal)
    DRAG = 0.0012  # quadratic drag coefficient, gives a natural top speed
    MAX_SPEED = 90.0  # m/s (~324 km/h)
    GRIP = 35.0  # max lateral acceleration in m/s^2 (~3.5 g)

    # --- Track / sensor constants ----------------------------------------
    TRACK_WIDTH = 14.0  # metres
    RAY_RANGE = 120.0  # metres

    def __init__(
        self,
        render_mode: str | None = None,
        n_laps: int = 1,
        max_steps: int = 3000,
    ) -> None:
        """Create the environment.

        Args:
            render_mode: ``"human"`` opens a matplotlib window, ``"rgb_array"``
                returns frames as NumPy arrays, ``None`` disables rendering.
            n_laps: Laps needed to finish the episode.
            max_steps: Step limit before the episode is truncated.

        Raises:
            ValueError: If ``render_mode`` is not supported or the lap/step
                counts are not positive.
        """
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}, "
                f"got {render_mode!r}"
            )
        if n_laps < 1 or max_steps < 1:
            raise ValueError("n_laps and max_steps must both be >= 1")

        self.render_mode = render_mode
        self.n_laps = n_laps
        self.max_steps = max_steps

        # Pre-compute everything about the track once; each step then only
        # needs cheap vector maths.
        self.track = make_track()
        self._seg_start = self.track
        self._seg_vec = np.roll(self.track, -1, axis=0) - self.track
        self._seg_len = np.linalg.norm(self._seg_vec, axis=1)
        self._seg_len_sq = self._seg_len**2
        self._cum_len = np.concatenate([[0.0], np.cumsum(self._seg_len)[:-1]])
        self.track_length = float(self._seg_len.sum())
        self._half_width = self.TRACK_WIDTH / 2.0

        # Track edges: offset the centre line sideways by half the width. The
        # normal at each point averages its two neighbouring segments so the
        # edges stay smooth through corners.
        tangent = self._seg_vec / self._seg_len[:, None]
        tangent = tangent + np.roll(tangent, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)  # points left
        self._left = self.track + normal * self._half_width
        self._right = self.track - normal * self._half_width
        self._left_vec = np.roll(self._left, -1, axis=0) - self._left
        self._right_vec = np.roll(self._right, -1, axis=0) - self._right

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._fig = None  # matplotlib figure, created lazily on first render
        self._last_arc: float | None = None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _project(
        self, points: np.ndarray, segs: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project points onto the centre line.

        This is the workhorse of the environment: for every point it finds the
        closest centre-line segment using broadcasting, i.e. a
        ``(n_points, n_segments)`` distance table built in one shot instead of
        a Python loop.

        Args:
            points: Array of shape ``(n, 2)``.
            segs: Optional indices of the segments to test. Restricting the
                search to the stretch of track around the car is what keeps
                the environment fast; ``None`` tests every segment.

        Returns:
            ``(distance, arc_position)``: distance from each point to the centre
            line, and how far along the track (in metres) the closest spot is.
        """
        if segs is None:
            segs = np.arange(len(self._seg_len))
        start, vec = self._seg_start[segs], self._seg_vec[segs]
        rel = points[:, None, :] - start[None, :, :]  # (n, segs, 2)
        t = np.einsum("nsk,sk->ns", rel, vec) / self._seg_len_sq[segs]
        t = np.clip(t, 0.0, 1.0)
        diff = rel - t[..., None] * vec[None]
        dist_sq = np.einsum("nsk,nsk->ns", diff, diff)
        best = np.argmin(dist_sq, axis=1)
        rows = np.arange(len(points))
        idx = segs[best]
        arc = self._cum_len[idx] + t[rows, best] * self._seg_len[idx]
        return np.sqrt(dist_sq[rows, best]), arc

    def _nearby_segments(self, arc: float, reach: float) -> np.ndarray:
        """Indices of centre-line segments within ``reach`` metres of ``arc``."""
        n = len(self._seg_len)
        centre = int(np.searchsorted(self._cum_len, arc, side="right") - 1)
        half = int(np.ceil(reach / self._seg_len.min())) + 1
        if 2 * half + 1 >= n:
            return np.arange(n)
        return np.arange(centre - half, centre + half + 1) % n

    def _signed_offset(self, point: np.ndarray) -> tuple[float, float, float]:
        """Return (lateral offset, track heading, arc position) for one point.

        The offset is positive when the car is left of the centre line.
        """
        segs = None if self._last_arc is None else self._nearby_segments(
            self._last_arc, self.MAX_SPEED * self.DT + self.TRACK_WIDTH
        )
        dist, arc = self._project(point[None], segs)
        idx = int(np.searchsorted(self._cum_len, arc[0], side="right") - 1)
        tangent = self._seg_vec[idx] / self._seg_len[idx]
        rel = point - self._seg_start[idx]
        side = np.sign(tangent[0] * rel[1] - tangent[1] * rel[0]) or 1.0
        track_heading = float(np.arctan2(tangent[1], tangent[0]))
        return float(side * dist[0]), track_heading, float(arc[0])

    def _cast_rays(self) -> np.ndarray:
        """Measure distance to the track edge along each sensor ray.

        Every ray is intersected exactly with the nearby left/right edge
        segments. For a ray ``O + t*D`` and a segment ``P + u*E`` the crossing
        point solves a 2x2 linear system, which 2D cross products give in
        closed form:

            t = cross(P - O, E) / cross(D, E)
            u = cross(P - O, D) / cross(D, E)

        A hit is valid when ``t > 0`` (in front of the car) and ``0 <= u <= 1``
        (inside the segment); the smallest valid ``t`` is the distance.
        """
        angles = self.heading + RAY_ANGLES  # (rays,)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (rays, 2)

        # A ray can only reach track within RAY_RANGE of the car, so only
        # those edge segments need testing.
        segs = self._nearby_segments(self._last_arc, self.RAY_RANGE * 1.5)
        p = np.concatenate([self._left[segs], self._right[segs]])  # (m, 2)
        e = np.concatenate([self._left_vec[segs], self._right_vec[segs]])

        def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        po = p - self.pos  # (m, 2)
        denom = cross(dirs[:, None, :], e[None, :, :])  # (rays, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cross(po[None], e[None]) / denom
            u = cross(po[None], dirs[:, None, :]) / denom
        hit = (np.abs(denom) > 1e-9) & (t > 0.0) & (u >= 0.0) & (u <= 1.0)
        t = np.where(hit, t, np.inf).min(axis=1)
        return np.minimum(t, self.RAY_RANGE)

    def _get_obs(self) -> np.ndarray:
        rays = self._cast_rays() / self.RAY_RANGE
        offset, track_heading, _ = self._signed_offset(self.pos)
        heading_err = (self.heading - track_heading + np.pi) % (2.0 * np.pi) - np.pi
        obs = np.concatenate(
            [
                rays,
                [
                    self.speed / self.MAX_SPEED,
                    heading_err / np.pi,
                    offset / self._half_width,
                ],
            ]
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Place the car on the start line, pointing down the track.

        A small random lateral offset and heading jitter (seeded via
        ``self.np_random``) stop the agent from memorising a single line.
        """
        super().reset(seed=seed)
        start, nxt = self.track[0], self.track[1]
        tangent = (nxt - start) / np.linalg.norm(nxt - start)
        normal = np.array([-tangent[1], tangent[0]])
        jitter = self.np_random.uniform(-0.3, 0.3) * self._half_width

        self.pos = start + normal * jitter
        self.heading = float(np.arctan2(tangent[1], tangent[0]))
        self.heading += float(self.np_random.uniform(-0.05, 0.05))
        self.speed = 0.0
        self.steps = 0
        self.progress = 0.0  # metres travelled along the centre line
        self.laps = 0
        self.lap_times: list[float] = []
        self._lap_start_step = 0
        self._last_arc = None  # forces a full-track search on the first lookup
        _, _, self._last_arc = self._signed_offset(self.pos)
        self._trail = [self.pos.copy()]

        return self._get_obs(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the simulation by ``DT`` seconds."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        steer = float(action[0]) * self.MAX_STEER
        pedal = float(action[1])

        # Longitudinal: throttle or brake, minus aerodynamic drag.
        accel = pedal * (self.MAX_ACCEL if pedal >= 0 else self.MAX_BRAKE)
        accel -= self.DRAG * self.speed**2
        self.speed = float(np.clip(self.speed + accel * self.DT, 0.0, self.MAX_SPEED))

        # Lateral: kinematic bicycle yaw rate, capped by tyre grip. If the
        # requested turn needs more grip than available the car understeers -
        # it simply turns less than asked, so going too fast into a corner
        # pushes it wide and off the track.
        yaw_rate = self.speed / self.WHEELBASE * np.tan(steer)
        if self.speed > 1e-3:
            max_yaw = self.GRIP / self.speed
            yaw_rate = float(np.clip(yaw_rate, -max_yaw, max_yaw))
        self.heading += yaw_rate * self.DT
        self.pos = self.pos + self.speed * self.DT * np.array(
            [np.cos(self.heading), np.sin(self.heading)]
        )
        self.steps += 1
        self._trail.append(self.pos.copy())

        # Progress along the track, handling the wrap-around at the start line.
        offset, _, arc = self._signed_offset(self.pos)
        delta = (arc - self._last_arc + self.track_length / 2) % self.track_length
        delta -= self.track_length / 2
        self._last_arc = arc
        self.progress += delta

        reward = 0.1 * delta - 0.01
        terminated = False

        if abs(offset) > self._half_width:
            reward -= 10.0
            terminated = True
        elif self.progress >= (self.laps + 1) * self.track_length:
            self.laps += 1
            self.lap_times.append((self.steps - self._lap_start_step) * self.DT)
            self._lap_start_step = self.steps
            reward += 50.0
            terminated = self.laps >= self.n_laps

        truncated = not terminated and self.steps >= self.max_steps

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, self._info()

    def _info(self) -> dict[str, Any]:
        return {
            "speed_kph": self.speed * 3.6,
            "progress_m": self.progress,
            "lap_fraction": self.progress / self.track_length,
            "laps": self.laps,
            "lap_times": list(self.lap_times),
        }

    def render(self) -> np.ndarray | None:
        """Draw the track, the car and its racing line with matplotlib.

        matplotlib is imported here rather than at module level so that
        training (which never renders) does not need it installed.
        """
        if self.render_mode is None:
            return None
        try:
            import matplotlib

            if self.render_mode == "rgb_array":
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "Rendering needs matplotlib: pip install matplotlib"
            ) from exc

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            if self.render_mode == "human":
                plt.ion()
        ax = self._ax
        ax.clear()
        loop = np.vstack([self.track, self.track[:1]])
        ax.plot(loop[:, 0], loop[:, 1], color="#555", lw=self.TRACK_WIDTH * 0.9)
        ax.plot(loop[:, 0], loop[:, 1], color="#ddd", lw=0.6, ls="--")
        trail = np.array(self._trail)
        ax.plot(trail[:, 0], trail[:, 1], color="#e10600", lw=1.2)
        ax.plot(*self.pos, marker="o", color="#e10600", ms=6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{self.speed * 3.6:5.0f} km/h   lap {self.laps + 1}")

        self._fig.canvas.draw()
        if self.render_mode == "human":
            plt.pause(1.0 / self.metadata["render_fps"])
            return None
        frame = np.asarray(self._fig.canvas.buffer_rgba())[..., :3]
        return frame.copy()

    def close(self) -> None:
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None


def gap_follower(obs: np.ndarray) -> np.ndarray:
    """A hand-written baseline driver, no learning involved.

    Steers toward whichever side has more open track and picks a target speed
    from how much straight is visible ahead. Useful to sanity-check the
    environment and as a score to beat with RL.

    Args:
        obs: Observation from :class:`F1TrackEnv`.

    Returns:
        Action ``[steer, pedal]`` in ``[-1, 1]``.
    """
    rays = obs[:7]
    speed, heading_err, offset = obs[7], obs[8], obs[9]
    left = rays[4:].mean()  # rays at +30, +60, +90 deg
    right = rays[:3].mean()  # rays at -90, -60, -30 deg
    steer = 3.0 * (left - right) - 1.5 * heading_err - 0.3 * offset

    # Visible straight-line distance sets the speed we can carry.
    ahead = max(rays[2], rays[3], rays[4])
    target = 0.25 + 0.75 * ahead**1.5
    pedal = 6.0 * (target - speed)
    return np.clip(np.array([steer, pedal], dtype=np.float32), -1.0, 1.0)
