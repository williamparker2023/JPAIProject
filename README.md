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

```bash
python -m ui.play_human
```
