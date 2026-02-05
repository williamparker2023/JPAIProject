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
python -m training.train_ga --generations 100 --pop_size 100 --episodes_per_genome 3 --save_dir models --seed0 0
```

To watch A\* agent:

```bash
python -m ui.watch_agent --seed 0
```

To play yourself:

```bash
python -m ui.play_human
```
