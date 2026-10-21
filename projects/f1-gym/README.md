# F1 Gym: a racing environment for reinforcement learning

A custom [Gymnasium](https://gymnasium.farama.org/) environment where an F1-style
car learns to lap a circuit. It's pure NumPy, so it runs at about 5,000 steps/sec on
one CPU core and a PPO agent trains on a laptop with no GPU.

The live replay is on the portfolio page at `f1-racing.html`.

## Quick start

```bash
cd projects/f1-gym
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest                                   # run the tests
python play.py                           # baseline driver, prints lap times
python train.py                          # train PPO (~300k steps)
python play.py --model models/best_model.zip --render
python play.py --model models/best_model.zip --export ../../assets/f1-lap.json
```

The last command replaces the replay shown on the website with your agent's run.

## Results

| Driver | Best lap | Top speed |
|---|---|---|
| `gap_follower` baseline | 37.3 s | 217 km/h |
| PPO, 300k steps (under 3 min on CPU) | **23.9 s** | 273 km/h |

## The environment

| | |
|---|---|
| **Observation** `Box(10,)` | 7 distance rays (−90°…+90°) to the track edge, speed, heading error, lateral offset |
| **Action** `Box(2,)` | steering (−1 right … +1 left), pedal (−1 brake … +1 throttle) |
| **Reward** | +0.1 per metre of progress, −0.01 per step, −10 for leaving the track, +50 per lap |
| **Ends** | leaving the track or finishing `n_laps` (terminated), `max_steps` (truncated) |

## How it works

- **Car physics.** A kinematic bicycle model, so heading changes by
  `speed / wheelbase * tan(steer)`. The yaw rate is capped so lateral
  acceleration stays under a 3.5 g grip limit. Arrive at a corner too fast and
  the car understeers off the track, which makes the agent learn to brake early.
- **Sensors.** Each ray is intersected exactly with the left and right edge
  polylines using 2D cross products, vectorised over all rays and segments at once.
- **Speed tricks.** Each step only searches the track segments near the car,
  so the cost per step stays the same however long the circuit is.
- **Reward design.** The reward pays for progress along the centre line rather
  than raw speed, so spinning in circles or reversing can't game it.

## Files

```
f1_gym/env.py      environment + gap_follower baseline
f1_gym/__init__.py registers "F1Track-v0"
train.py           PPO training with Stable Baselines3 (CPU)
play.py            run a driver, print lap times, export a replay JSON
tests/             pytest suite (includes Gymnasium's check_env)
```
