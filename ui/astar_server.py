#!/usr/bin/env python3
from __future__ import annotations

import random
import os
from typing import Optional

import numpy as np
from flask import Flask, jsonify, request, make_response

from agents.astar_agent import AStarAgent
from agents.ga_agent import GAAgent
from agents.rl_agent import RLAgent
from env.snake_env import SnakeEnv

app = Flask(__name__)


def run_astar_once(seed: Optional[int] = None) -> int:
    env = SnakeEnv()
    if seed is not None:
        env.seed(seed)
    obs, _ = env.reset(seed=seed)
    agent = AStarAgent()
    while True:
        action = agent.act(obs)
        res = env.step(action)
        obs = res.obs
        if res.terminated or res.truncated:
            return env.score


def run_agent_once(agent, seed: Optional[int] = None) -> int:
    env = SnakeEnv()
    if seed is not None:
        env.seed(seed)
    obs, _ = env.reset(seed=seed)
    while True:
        # provide legal actions if agent expects them
        try:
            obs['legal_actions'] = env.legal_actions()
        except Exception:
            pass
        try:
            action = agent.act(obs)
        except TypeError:
            action = agent.act(obs, epsilon=0.0)
        res = env.step(action)
        obs = res.obs
        if res.terminated or res.truncated:
            return env.score


@app.route('/astar_avg')
def astar_avg():
    try:
        runs = int(request.args.get('runs', 10))
    except Exception:
        runs = 10

    seeds_param = request.args.get('seeds')
    scores = []
    for i in range(runs):
        if seeds_param:
            try:
                seeds = [int(s) for s in seeds_param.split(',') if s.strip()]
                seed = seeds[i % len(seeds)] if seeds else None
            except Exception:
                seed = None
        else:
            seed = random.randrange(2**30)
        s = run_astar_once(seed=seed)
        scores.append(int(s))

    avg = float(sum(scores)) / len(scores) if scores else 0.0
    max_score = max(scores) if scores else 0
    resp = make_response(jsonify({'runs': runs, 'scores': scores, 'avg': avg, 'max': max_score, 'best': max_score}))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/models_best')
def models_best():
    try:
        runs = int(request.args.get('runs', 10))
    except Exception:
        runs = 10

    seeds_param = request.args.get('seeds')
    seeds = None
    if seeds_param:
        try:
            seeds = [int(s) for s in seeds_param.split(',') if s.strip()]
        except Exception:
            seeds = None

    results = {}

    # A*
    astar_scores = []
    for i in range(runs):
        seed = seeds[i % len(seeds)] if seeds else random.randrange(2**30)
        astar_scores.append(run_astar_once(seed=seed))
    results['astar'] = {'scores': astar_scores, 'max': int(max(astar_scores)), 'mean': float(np.mean(astar_scores))}

    # GA
    ga_path = os.path.join('models', 'ga_best.npy')
    if os.path.exists(ga_path):
        genome = np.load(ga_path)
        ga_agent = GAAgent(genome)
        ga_scores = []
        for i in range(runs):
            seed = seeds[i % len(seeds)] if seeds else random.randrange(2**30)
            ga_scores.append(run_agent_once(ga_agent, seed=seed))
        results['ga'] = {'scores': ga_scores, 'max': int(max(ga_scores)), 'mean': float(np.mean(ga_scores))}
    else:
        results['ga'] = {'error': 'model file not found', 'path': ga_path}

    # RL / Imitation
    rl_path = os.path.join('models', 'rl_best.npz')
    if os.path.exists(rl_path):
        data = np.load(rl_path)
        rl_agent = None
        if 'W1' in data and 'W2' in data:
            rl_agent = RLAgent(W1=data['W1'], b1=data['b1'], W2=data['W2'], b2=data['b2'])
        elif 'W' in data and 'b' in data:
            rl_agent = RLAgent()
            rl_agent.W2 = data['W'].astype(np.float32)
            rl_agent.b2 = data['b'].astype(np.float32)

        if rl_agent is not None:
            rl_scores = []
            for i in range(runs):
                seed = seeds[i % len(seeds)] if seeds else random.randrange(2**30)
                rl_scores.append(run_agent_once(rl_agent, seed=seed))
            results['rl'] = {'scores': rl_scores, 'max': int(max(rl_scores)), 'mean': float(np.mean(rl_scores))}
        else:
            results['rl'] = {'error': 'unsupported model format', 'path': rl_path}
    else:
        results['rl'] = {'error': 'model file not found', 'path': rl_path}

    resp = make_response(jsonify({'runs': runs, 'results': results}))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
