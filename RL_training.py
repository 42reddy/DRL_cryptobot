import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from sklearn.decomposition import PCA
from ta.momentum import RSIIndicator
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import ta
from PPO import PPOEnsemble
from transformer import Transformer
from torch.distributions import Normal
import math

import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv("combined_5crypto_ohlcv.csv", parse_dates=['time'])

prices = df.iloc[0]
weights = np.array([1, prices['BTC_close'] / prices['ETH_close'], prices['BTC_close'] / prices['BCH_close'],
                         prices['BTC_close'] / prices['XRP_close'], prices['BTC_close'] / prices['LTC_close']]) / 70


raw_prices = df[['time','BTC_close', 'ETH_close', 'BCH_close', 'XRP_close', 'LTC_close']].copy()
df_copy = df.copy()

symbols = ['BTC', 'ETH', 'BCH', 'XRP', 'LTC']
feature_columns = []

# Step 1: Detrend OHLCV via percentage change
for symbol in symbols:
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_copy[f"{symbol}_{col}"] = df_copy[f"{symbol}_{col}"].pct_change()

# Step 2: Compute technical indicators on detrended series
for symbol in symbols:
    prefix = f"{symbol}_"

    df_copy[f"{prefix}sma30"] = df_copy[f"{prefix}close"].rolling(window=30).mean()
    df_copy[f"{prefix}sma60"] = df_copy[f"{prefix}close"].rolling(window=60).mean()

    delta = df_copy[f"{prefix}close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df_copy[f"{prefix}rsi"] = 100 - (100 / (1 + rs))

    tp = (df[f"{prefix}high"] + df_copy[f"{prefix}low"] + df_copy[f"{prefix}close"]) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df_copy[f"{prefix}cci"] = (tp - sma_tp) / (0.015 * mad)

    ema12 = df_copy[f"{prefix}close"].ewm(span=12).mean()
    ema26 = df_copy[f"{prefix}close"].ewm(span=26).mean()
    df_copy[f"{prefix}macd"] = ema12 - ema26

    high = df_copy[f"{prefix}high"]
    low = df_copy[f"{prefix}low"]
    close = df_copy[f"{prefix}close"]
    tr_components = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1)
    tr = tr_components.max(axis=1)
    df_copy[f"{prefix}atr"] = tr.rolling(14).mean()

    from ta.trend import adx
    df_copy[f"{prefix}adx"] = adx(high, low, close, window=14)

    feature_columns += [f"{prefix}{col}" for col in ['open', 'high', 'low', 'close', 'volume',
                                                     'sma30', 'sma60', 'rsi', 'cci', 'macd', 'atr', 'adx']]


df_copy[feature_columns] = df_copy[feature_columns].replace([np.inf, -np.inf], np.nan)
df_copy.dropna(subset=feature_columns, inplace=True)
raw_prices = raw_prices.loc[df_copy.index]

# Drop timestamp and serial number columns
df_copy.drop(columns=['time'], inplace=True, errors='ignore')
raw_prices.drop(columns=['time'], inplace=True, errors='ignore')

for symbol in symbols:
    df_copy[f"{symbol}_alloc"] = 0.0

df_copy['usd_balance'] = 1_000_000.0

scaler = StandardScaler()
df_copy[feature_columns] = scaler.fit_transform(df_copy[feature_columns])
df_copy = df_copy.to_numpy()
raw_prices = raw_prices.to_numpy()


def split_data_evenly(array, num_parts=60):
    length = len(array)
    chunk_size = length // num_parts
    splits = []

    for i in range(num_parts):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_parts - 1 else length
        splits.append(array[start:end])

    return splits


monthly_periods = split_data_evenly(df_copy, num_parts=60)
monthly_raw_periods = split_data_evenly(raw_prices, 60)


print(f"Total splits: {len(monthly_periods)}")
print(f"Shape of split 0: {monthly_periods[0].shape}")
print(f"Shape of split -1: {monthly_periods[-1].shape}")


def get_training_data(monthly_periods, test_month_index, train_months=6):
    start_idx = max(0, test_month_index - train_months)
    training_chunks = monthly_periods[start_idx:test_month_index]
    return np.concatenate(training_chunks, axis=0)


def get_validation_indices(training_data, num_weeks=9, hours_per_week=168):
    total_weeks = len(training_data) // hours_per_week
    if total_weeks < num_weeks:
        raise ValueError("Training data too short to extract 9 weeks.")

    selected = np.random.choice(total_weeks, size=num_weeks, replace=False)
    selected.sort()

    val_indices = []
    for w in selected:
        start = w * hours_per_week
        end = start + hours_per_week
        val_indices.append((start, end))

    return val_indices


def get_test_data(monthly_periods, test_month_index):
    return monthly_periods[test_month_index]


i = 12

train_df = get_training_data(monthly_periods, i)
val_indices = get_validation_indices(train_df)
test_df = get_test_data(monthly_periods, i)


def build_sequences(df, seq_len=12, step_size=1):

    states = []
    next_states = []

    data = df
    total_steps = len(data)

    for i in range(0, total_steps - seq_len - 1, step_size):
        state_seq = data[i : i + seq_len]
        next_state_seq = data[i + 1 : i + 1 + seq_len]

        states.append(state_seq)
        next_states.append(next_state_seq)

    states = torch.tensor(states)
    next_states = torch.tensor(next_states)

    return states, next_states

states , next_states = build_sequences(train_df, 12, 1)

SEQ_LEN = 12

update_epochs = 4
total_updates = 2000

# Instantiate model and PPO agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    n_layers=4,
    n_heads=4,
    d_model=32,
    input_dim=66,
    ff_dim=64,
    seq_len=12,
    action_dim=5
).to(device)

agent = PPOEnsemble(model)

episode_length = 252
batch_size = 4
transaction_cost = 0.01



def train_agent(raw_prices, total_updates, model, agent):
    for i in range(total_updates):
        episode_returns = []
        all_states, all_actions, all_log_probs, all_rewards, all_dones, all_next_states, all_values = [], [], [], [], [], [], []

        for env_id in range(4):
            train_df = get_training_data(monthly_periods, env_id+8)
            raw_prices_df = get_training_data(monthly_raw_periods, env_id+8)

            data, next_data = build_sequences(train_df, 12, 1)
            current_prices, next_prices = build_sequences(raw_prices_df, 12, 1)
            for episode in range(16):
                idx = np.random.randint(0, len(data) - episode_length - 1)

                # Initialize portfolio
                usd_balance = 1_000_000.0
                allocations = torch.zeros(5)
                reward_norm = 1

                for t in range(episode_length):
                    state_data = data[idx + t]
                    next_state_data = next_data[idx + t]

                    current_price_data = current_prices[idx + t]
                    next_price_data = next_prices[idx + t]

                    state = torch.tensor(state_data, dtype=torch.float32).to(agent.device)
                    next_state = torch.tensor(next_state_data, dtype=torch.float32).to(agent.device)

                    current_price = torch.tensor(current_price_data, dtype=torch.float32).to(agent.device)
                    next_price = torch.tensor(next_price_data, dtype=torch.float32).to(agent.device)

                    with torch.no_grad():
                        mu, sigma, value = model(state)
                        sigma = torch.clamp(sigma, min=1e-6)
                        dist = Normal(mu, sigma)
                        action = dist.sample()
                        log_prob = dist.log_prob(action).sum()

                    action = action.squeeze()

                    current_holdings = state[-1, -6:-1]
                    usd_balance = state[-1, -1].item()
                    old_prices = current_price[-1]
                    new_prices = next_price[-1]

                    action_scale = torch.tensor(weights, dtype=torch.float32, device=agent.device)
                    delta_alloc = action * action_scale
                    new_holdings = torch.clamp(current_holdings + delta_alloc, min=0)
                    data[idx + t + 1][-1, -6:-1] = new_holdings

                    old_value = (current_holdings * old_prices).sum() + usd_balance
                    new_usd_balance = usd_balance - torch.dot(old_prices, delta_alloc)
                    new_value = (new_holdings * new_prices).sum() + new_usd_balance
                    data[idx + t + 1][-1, -1] = new_usd_balance

                    pct_return = (new_value - old_value) / old_value
                    reward = pct_return.item()
                    reward_norm = max(reward_norm, abs(pct_return.item()))
                    reward /= reward_norm

                    all_states.append(state.cpu())
                    all_actions.append(action.cpu())
                    all_log_probs.append(log_prob.cpu())
                    all_rewards.append(reward)
                    all_next_states.append(next_state.cpu())
                    all_dones.append(t == episode_length - 1)
                    all_values.append(value.cpu().squeeze())

                episode_returns.append(np.sum(all_rewards[-episode_length:]))

        # Stack after collecting from all 4 envs
        states = torch.stack(all_states)
        actions = torch.stack(all_actions)
        log_probs = torch.stack(all_log_probs)
        rewards = torch.tensor(all_rewards, dtype=torch.float32)
        dones = torch.tensor(all_dones, dtype=torch.bool)
        next_states = torch.stack(all_next_states)
        values = torch.tensor(all_values, dtype=torch.float32)

        # PPO update
        loss = agent.update(states, actions, log_probs, rewards, dones, next_states)

        print(f"Update {i + 1}/{total_updates} - Avg Episode Return: {np.mean(episode_returns):.4f} - Loss: {loss:.4f}")


    print(f"\nTraining Complete!")



train_agent(raw_prices, total_updates, model, agent)





