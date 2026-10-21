"""Drive laps with the baseline or a trained agent and report lap times.

Usage:
    python play.py                                  # hand-written gap follower
    python play.py --model models/best_model.zip    # trained PPO agent
    python play.py --render                         # watch it (needs matplotlib)
    python play.py --export ../../assets/f1-lap.json  # save a lap for the website
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from f1_gym import F1TrackEnv, gap_follower


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, help="PPO .zip; omit for the baseline")
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--export", type=Path, help="write the run as JSON")
    return parser.parse_args()


def load_driver(model_path: Path | None):
    """Return a function mapping observation -> action."""
    if model_path is None:
        return gap_follower
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path} - run train.py first")
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    # deterministic=True uses the mean action, i.e. no exploration noise.
    return lambda obs: model.predict(obs, deterministic=True)[0]


def main() -> int:
    args = parse_args()
    try:
        driver = load_driver(args.model)
    except (FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}")
        return 1

    env = F1TrackEnv(render_mode="human" if args.render else None, n_laps=args.laps)
    obs, info = env.reset(seed=args.seed)
    trajectory = [env.pos.round(2).tolist() + [round(info["speed_kph"], 1)]]
    total_reward = 0.0
    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(driver(obs))
            total_reward += reward
            trajectory.append(env.pos.round(2).tolist() + [round(info["speed_kph"], 1)])
            if terminated or truncated:
                break
    finally:
        env.close()

    print(f"Driver:       {'PPO ' + str(args.model) if args.model else 'gap follower'}")
    print(f"Laps:         {info['laps']}/{args.laps}")
    for i, lap in enumerate(info["lap_times"], start=1):
        print(f"  Lap {i}:      {lap:6.1f} s")
    print(f"Top speed:    {max(p[2] for p in trajectory):6.0f} km/h")
    print(f"Total reward: {total_reward:8.1f}")
    if info["laps"] < args.laps:
        print(f"Crashed / timed out at {info['lap_fraction'] * 100:.0f}% of a lap")

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(
            json.dumps(
                {
                    "dt": env.DT,
                    "track": np.round(env.track, 2).tolist(),
                    "track_width": env.TRACK_WIDTH,
                    "lap_times": info["lap_times"],
                    "frames": trajectory,  # [x, y, speed_kph] per step
                }
            )
        )
        print(f"Exported {len(trajectory)} frames to {args.export}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
