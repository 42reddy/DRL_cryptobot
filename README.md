# PPO Crypto Trading Agent

Simple PPO implementation for cryptocurrency trading with ensemble model management.

## What it does

Trains a trading agent using Proximal Policy Optimization (PPO) to make buy/sell decisions. Includes model ensemble functionality to save and manage multiple versions based on performance.

## Core features

- PPO with clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Gradient clipping and entropy regularization
- Ensemble model saving based on performance thresholds

## Quick usage

```python
# Initialize agent
agent = PPOEnsemble(
    model=your_trading_model,
    lr=5e-6,
    clip_epsilon=0.1,
    ensemble_size=5
)

# Train on trading data
loss = agent.update(
    states=market_states,
    actions=trading_actions, 
    old_log_probs=action_probs,
    rewards=trading_rewards,
    dones=episode_ends,
    next_states=next_market_states
)
```

## Key parameters

- `clip_epsilon`: PPO clipping parameter (0.1)
- `value_coef`/`entropy_coef`: Loss weighting coefficients
- `ensemble_size`: Number of models to keep in ensemble
- `save_threshold`: Performance threshold for saving models

## Requirements

```
torch
```

Built for financial RL where you want stable policy updates and multiple model versions for robustness.
