"""Debug extended features to find NaNs or invalid values."""
from env.snake_env import SnakeEnv
from env.features_extended import extended_features
from env.features import compact_features
import numpy as np

env = SnakeEnv()
obs, _ = env.reset()

print("=== Extended Features Debug ===\n")

for step in range(10):
    obs['legal_actions'] = env.legal_actions()
    
    # Get both representations
    ext_feat = extended_features(obs)
    comp_feat = compact_features(obs)
    
    print(f"Step {step}:")
    print(f"  Head: {obs['snake'][-1]}, Food: {obs['food']}, Len: {len(obs['snake'])}")
    print(f"  Grid: {obs['width']}x{obs['height']}")
    print(f"  Compact (11): {comp_feat}")
    print(f"  Extended (21): {ext_feat}")
    print(f"  Has NaN: {np.any(np.isnan(ext_feat))}")
    print(f"  Has Inf: {np.any(np.isinf(ext_feat))}")
    print(f"  Min/Max: {np.min(ext_feat):.4f} / {np.max(ext_feat):.4f}")
    print()
    
    # Take random action
    a = env.legal_actions()[0] if env.legal_actions() else 0
    res = env.step(a)
    obs = res.obs
    
    if res.terminated or res.truncated:
        obs, _ = env.reset()
