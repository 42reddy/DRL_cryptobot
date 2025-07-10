import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class PPOEnsemble:
    def __init__(self,
                 model,
                 lr=5e-6,
                 clip_epsilon=0.1,
                 value_coef=-1,
                 entropy_coef=-1,
                 max_grad_norm=0.5,
                 ensemble_size=5,
                 save_threshold=0.1):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # Debug print

        # Main model
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # PPO hyperparameters
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Ensemble management
        self.ensemble_size = ensemble_size
        self.save_threshold = save_threshold
        self.ensemble_models = []
        self.ensemble_scores = []
        self.best_score = float('-inf')

    def update(self, states, actions, old_log_probs, rewards, dones, next_states, epochs=5):
        def ensure_tensor(x, dtype=torch.float32):
            if isinstance(x, torch.Tensor):
                return x.to(dtype=dtype, device=self.device)
            return torch.tensor(x, dtype=dtype, device=self.device)

        # Move all to device and correct types
        states = ensure_tensor(states)
        actions = ensure_tensor(actions)
        old_log_probs = ensure_tensor(old_log_probs)
        rewards = ensure_tensor(rewards)
        dones = ensure_tensor(dones, dtype=torch.bool)
        next_states = ensure_tensor(next_states)

        # GAE with fixed implementation
        advantages, returns = self._compute_gae(states, rewards, dones, next_states)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for epoch in range(epochs):
            mu, sigma, values = self.model(states)
            # Remove excessive action clamping and add sigma clamping instead
            sigma = torch.clamp(sigma, min=1e-6)  # Prevent sigma from becoming too small
            dist = Normal(mu, sigma)

            # Calculate log probabilities
            new_log_probs = dist.log_prob(actions)
            # Handle multidimensional actions properly
            if new_log_probs.dim() > 1:
                new_log_probs = new_log_probs.sum(dim=-1)

            entropy = dist.entropy()
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)

            # PPO ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(-1), returns)

            # Entropy regularization
            entropy_loss = -entropy.mean()

            # Total loss
            loss = 100* policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            # Optional: print loss components for debugging
            if epoch == epochs - 1:
                print(f"Policy Loss: {policy_loss.item():.4f}, "
                      f"Value Loss: {value_loss.item():.4f}, "
                      f"Entropy Loss: {entropy_loss.item():.4f}")

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / epochs

        return avg_loss

    def _compute_gae(self, states, rewards, dones, next_states, gamma=0.99, lambda_=0.95):

        with torch.no_grad():
            _, _, values = self.model(states)
            _, _, next_values = self.model(next_states)

            # Remove unnecessary dimensions
            values = values.squeeze(-1)
            next_values = next_values.squeeze(-1)

            # If episode ends (done == True), there is no next state value
            next_values = next_values * (~dones)

            # TD residuals (delta_t = r_t + γ * V(s_{t+1}) - V(s_t))
            deltas = rewards + gamma * next_values - values

            # Initialize advantage buffer
            advantages = torch.zeros_like(rewards)
            advantage = 0

            # Compute advantage recursively from the end (backward in time)
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    advantage = 0  # reset at episode boundary
                # GAE formula: A_t = δ_t + γλ * A_{t+1}
                advantage = deltas[t] + gamma * lambda_ * advantage
                advantages[t] = advantage

            # The target for the value function is: return = advantage + V(s_t)
            returns = advantages + values

            # Clip advantages and returns to prevent explosions
            advantages = advantages.clamp(-50, 50)
            returns = returns.clamp(-100, 100)

        return advantages, returns
