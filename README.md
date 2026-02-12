Snake

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To watch GA play:

```bash
python -m ui.watch_ga --model models/ga_best.npy --seed 0
```

To train GA:

```bash
python -m training.train_ga --generations 300 --pop_size 100 --episodes_per_genome 3 --seed0 0
```

**Feature set** (11 input features):

- Immediate danger in 3 directions (straight, left, right)
- Direction encoding (4-way one-hot)
- Food direction (binary in 4 directions)

To watch A\* agent:

```bash
python -m ui.watch_agent --seed 0
```

To watch DQN (non imitation)

```bash
python -m ui.watch_dqn_baseline --model models/dqn_baseline_best.npz --seed 0
```

To play yourself:

````bash
python -m ui.play_human

## Imitation Learning (Behavioral Cloning from A\*)

For fastest path to high scores, train via imitation learning from A\* trajectories:

1. **Collect A\* trajectories** (one-time):

```bash
python -m training.collect_trajectories --episodes 200 --seed0 0
```

2. **Train imitation model** (supervised learning to replicate A\*):

```bash
python -m training.train_imitation --epochs 100 --batch_size 128 --lr 1e-2
```

3. **Watch the trained imitation model**:

```bash
python -m ui.watch_rl --model models/rl_best.npz --seed 0
```

Notes:

- Fast training (~10 min for 100 epochs)
- Reaches 25-27 mean score (vs A\*'s ~60, but 3x RL's ~9)
- Limited by feature representation (11 local features vs A\*'s global board access)

## RL (DQN) Usage

To train the DQN-based RL agent from scratch (saves best model to `models/rl_best.npz`):

```bash
python -m training.train_rl --episodes 5000 --lr 1e-3 --epsilon_start 0.7 --epsilon_end 0.05 --batch_size 64 --warmup 300 --target_update_steps 1000 --seed0 0
```

To watch a trained RL model:

```bash
python -m ui.watch_rl --model models/rl_best.npz --seed 0
```

```

```
````
