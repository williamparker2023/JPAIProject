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

**GA Algorithm Improvements**:

- **Multi-point crossover**: Splits genomes at 2-4 random points, inheriting segments from each parent (more realistic genetic recombination than uniform averaging)
- **Adaptive mutation**: Higher mutation rate early in training for exploration, decays over time for refinement
- **Conservative initialization**: Weights initialized to smaller scale (0.5 std) to stabilize early learning

**Feature set** (11 input features):

- Immediate danger in 3 directions (straight, left, right)
- Direction encoding (4-way one-hot)
- Food direction (binary in 4 directions)

**Training notes**:

- Simpler features + smarter GA operators work better than complex feature engineering
- Default mutation rate (0.1) works well; increase to 0.2 for more exploration in early generations
- `--init_scale 0.5` (conservative) is better than 1.0 for stable learning
- Tighter truncation (`--stall_multiplier 50`, `--hard_cap_multiplier 30`) discourages infinite loops

**Watching notes**: When watching GA play, even tighter truncation is used by default (`--stall_multiplier 30`, `--hard_cap_multiplier 20`) to prevent visual infinite loops. You can adjust these if needed:

- Higher values = more patient (longer time between fruits before auto-stopping)
- Lower values = stricter (stops sooner if no progress)

To watch A\* agent:

```bash
python -m ui.watch_agent --seed 0
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

This generates `models/astar_trajectories.pkl` with 138K+ (state, action) pairs.

2. **Train imitation model** (supervised learning to replicate A\*):

```bash
python -m training.train_imitation --epochs 100 --batch_size 128 --lr 1e-2
```

Saves best model to `models/rl_best.npz`. Expected performance: ~25-27 mean score on unseen seeds.

3. **Watch the trained model**:

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

Notes:

- The RL agent is a small MLP DQN saved as a `.npz` (arrays: `W1`, `b1`, `W2`, `b2`).
- Much slower than imitation learning; expect ~9-15 mean score after 5000 episodes
- Imitation learning recommended for faster convergence to baseline performance

```

```
````
