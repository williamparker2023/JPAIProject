Snake

to watch ga
python -m ui.watch_ga --model models/ga_best.npy --seed 0

to make ga
python -m training.train_ga --generations 100 --pop_size 100 --episodes_per_genome 3 --save_dir models --seed0 0
