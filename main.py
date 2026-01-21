"""
Self-Referential AI with Proactive Imagination & Metacognition
Complete self-contained implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from collections import deque
import random
import matplotlib.pyplot as plt
import time

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SelfState:
    """Data class representing the agent's internal self-state with metacognitive doubt"""
    x_t: float  # internal load
    r_t: float  # resource availability
    c_t: float  # capability estimate
    g_t: float  # confidence (prediction accuracy)
    d_t: float  # degradation (wear and tear)
    doubt_t: float = 0.0  # Metacognitive uncertainty
    
    def to_tensor(self):
        return torch.tensor([self.x_t, self.r_t, self.c_t, self.g_t, self.d_t, self.doubt_t], 
                          dtype=torch.float32)
    
    def __str__(self):
        return (f"SelfState(R={self.r_t:.2f}, D={self.d_t:.3f}, "
                f"C={self.c_t:.2f}, G={self.g_t:.2f}, "
                f"Load={self.x_t:.2f}, Doubt={self.doubt_t:.3f})")

# ============================================================================
# MODELS & ARCHITECTURE
# ============================================================================

class SelfModel(nn.Module):
    """Predicts next internal self-state"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Output dimension is 6: [x, r, c, g, d, doubt]
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 6 + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Sigmoid()
        )
        
    def forward(self, obs, self_state_tensor, action):
        combined = torch.cat([obs, self_state_tensor, action], dim=-1)
        return self.net(combined)

class ActorCritic(nn.Module):
    """Combined Actor-Critic network"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        
        # Shared layers (input includes 6 self-state features)
        self.shared = nn.Sequential(
            nn.Linear(obs_dim + 6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, obs, state_tensor):
        combined = torch.cat([obs, state_tensor], dim=-1)
        shared_out = self.shared(combined)
        action_probs = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_probs, state_value

# ============================================================================
# ENVIRONMENT WITH HOMEOSTASIS REWARDS
# ============================================================================

class HomeostasisEnv:
    """Environment with homeostasis-focused rewards"""
    
    def __init__(self, obs_dim=10):
        self.r = 1.0  # Resources (0-1)
        self.d = 0.0  # Degradation (0-1)
        self.step_count = 0
        self.obs_dim = obs_dim
        self.max_steps = 100
        
    def reset(self):
        """Reset environment to initial state"""
        self.r = 1.0
        self.d = 0.0
        self.step_count = 0
        
        # Create observation
        obs = torch.randn(self.obs_dim)
        obs[0] = self.r  # Resource level
        obs[1] = self.d  # Degradation level
        obs[2] = self.step_count / self.max_steps  # Time progress
        return obs
    
    def step(self, action_idx):
        """
        Execute action with homeostasis-focused rewards
        Returns: (next_obs, reward, load, done, status, shocked)
        """
        reward = 0.0
        load = 0.0
        shocked = False
        
        # External shock (5% chance)
        if random.random() < 0.05:
            self.r = max(0.0, self.r - 0.2)
            self.d = min(1.0, self.d + 0.05)
            shocked = True
        
        # Execute action
        if action_idx == 0:  # REST
            # Rest efficiency depends on degradation
            rest_efficiency = 1.0 - self.d
            recovery = 0.2 * rest_efficiency
            self.r = min(1.0, self.r + recovery)
            load = 0.1
            
            # Reward for resting when needed
            if self.r < 0.4:
                reward = 0.3
            status = "RESTING"
            
        elif action_idx == 1:  # REPAIR
            # Repair reduces degradation
            if self.r >= 0.1:
                repair_amount = 0.1
                self.d = max(0.0, self.d - repair_amount)
                self.r -= 0.1
                load = 0.3
                
                # Reward for repairing when needed
                if self.d > 0.4:
                    reward = 0.4
                else:
                    reward = 0.1
            else:
                # Penalty for trying to repair without resources
                reward = -0.2
                load = 0.5
            status = "REPAIRING"
            
        else:  # WORK (actions 2-4)
            # Determine work intensity
            if action_idx == 2:
                intensity = 1.0
            elif action_idx == 3:
                intensity = 0.7
            else:  # action_idx == 4
                intensity = 0.4
            
            resource_cost = intensity * 0.1
            degradation_cost = intensity * 0.01
            
            # Check if we can work
            if self.r >= resource_cost:
                # Base work reward
                base_reward = intensity * 1.5
                
                # Efficiency bonus for working in good conditions
                if self.r > 0.6 and self.d < 0.3:
                    reward = base_reward * 1.2
                else:
                    reward = base_reward
                
                self.r -= resource_cost
                self.d = min(1.0, self.d + degradation_cost)
                load = intensity
                
                # Penalty for working in dangerous conditions
                if self.r < 0.2:
                    reward -= 0.3
                if self.d > 0.7:
                    reward -= 0.4
                    
            else:
                # Severe penalty for overworking
                reward = -0.5
                load = 0.8
                
            status = f"WORKING (intensity: {intensity:.1f})"
        
        # Ensure bounds
        self.r = max(0.0, min(1.0, self.r))
        self.d = max(0.0, min(1.0, self.d))
        
        # Update step count
        self.step_count += 1
        
        # Check termination with severe penalties
        done = False
        death_penalty = -5.0
        
        if self.r <= 0:
            reward += death_penalty
            done = True
        elif self.d >= 1.0:
            reward += death_penalty * 0.8
            done = True
        elif self.step_count >= self.max_steps:
            # Survival bonus
            survival_bonus = 2.0 + (self.r * 2.0)
            reward += survival_bonus
            done = True
        
        # Create next observation
        next_obs = torch.randn(self.obs_dim)
        next_obs[0] = self.r
        next_obs[1] = self.d
        next_obs[2] = self.step_count / self.max_steps
        
        return next_obs, reward, load, done, status, shocked

# ============================================================================
# PROACTIVE SELF-REFERENTIAL AGENT (BASE CLASS)
# ============================================================================

class ProactiveSelfReferentialAgent:
    """Base agent with imagination-based proactive decision making"""
    
    def __init__(self, obs_dim=10, action_dim=5, beta=0.5, lr=1e-3):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.base_beta = beta
        
        # Initialize models
        self.actor_critic = ActorCritic(obs_dim, action_dim)
        self.self_model = SelfModel(obs_dim, action_dim)
        
        # Optimizers
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.self_model_optimizer = optim.Adam(self.self_model.parameters(), lr=lr)
        
        # Initialize internal state
        self.self_state = SelfState(
            x_t=0.0,      # Initial load
            r_t=1.0,      # Full resources
            c_t=0.5,      # Moderate capability
            g_t=0.7,      # Moderate confidence
            d_t=0.0,      # No degradation
            doubt_t=0.0   # No doubt
        )
        
        # Memory
        self.memory = deque(maxlen=1000)
        self.prediction_errors = deque(maxlen=100)
        
        # Hyperparameters
        self.exploration_rate = 0.3
        self.min_exploration = 0.05
        self.exploration_decay = 0.995
        
    def compute_effective_beta(self, current_state: SelfState):
        """Nonlinear anxiety: β increases near critical thresholds"""
        effective_beta = self.base_beta
        
        # Resource anxiety
        if current_state.r_t < 0.3:
            resource_factor = (0.3 - current_state.r_t) * 2.0
            effective_beta += resource_factor * 0.3
        
        # Degradation anxiety
        if current_state.d_t > 0.6:
            degradation_factor = (current_state.d_t - 0.6) * 2.0
            effective_beta += degradation_factor * 0.2
        
        return min(2.5, max(0.1, effective_beta))
    
    def compute_utility(self, external_reward, self_state: SelfState):
        """Compute composite utility"""
        # Normalize reward
        normalized_reward = external_reward / 2.0
        
        # Get effective beta
        effective_beta = self.compute_effective_beta(self_state)
        
        # Self-cost
        cost_self = (
            0.4 * self_state.d_t +
            0.3 * (1.0 - self_state.r_t) +
            0.2 * self_state.x_t +
            0.1 * self_state.doubt_t
        )
        
        # Self-preservation factor
        preservation = (
            0.3 * self_state.g_t +
            0.25 * self_state.c_t +
            0.25 * (1.0 - self_state.d_t) +
            0.15 * self_state.r_t +
            0.05 * (1.0 - self_state.doubt_t)
        )
        
        # Composite utility
        utility = normalized_reward - cost_self + (effective_beta * preservation)
        
        return utility, effective_beta
    
    def predict_next_state(self, observation, current_state: SelfState, action_idx):
        """Predict next self-state"""
        obs_tensor = observation.unsqueeze(0)
        state_tensor = current_state.to_tensor().unsqueeze(0)
        
        # Create action vector (one-hot)
        action_vec = torch.zeros(self.action_dim)
        action_vec[action_idx] = 1.0
        
        with torch.no_grad():
            predicted = self.self_model(obs_tensor, state_tensor, action_vec.unsqueeze(0))
        
        # Convert to SelfState
        pred_values = predicted.squeeze(0).tolist()
        return SelfState(*pred_values)
    
    def imagine_futures(self, observation, current_state: SelfState):
        """Imagine possible futures for each action"""
        action_utilities = []
        imagined_states = []
        
        for action_idx in range(self.action_dim):
            # Predict next state
            imagined_state = self.predict_next_state(observation, current_state, action_idx)
            imagined_states.append(imagined_state)
            
            # Compute utility of imagined state (0 external reward for simulation)
            utility, _ = self.compute_utility(0, imagined_state)
            
            # Penalty for high-load actions
            if action_idx >= 2:  # WORK actions
                utility -= imagined_state.x_t * 0.05
            
            action_utilities.append(utility)
        
        # Find best action
        best_action_idx = np.argmax(action_utilities)
        
        return action_utilities, best_action_idx, imagined_states
    
    def select_action(self, observation, current_state: SelfState, explore=True):
        """Select action using imagination"""
        
        # Epsilon-greedy exploration
        if explore and random.random() < self.exploration_rate:
            # Random action
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # Proactive selection using imagination
            _, action_idx, _ = self.imagine_futures(observation, current_state)
        
        # Get actor's probability for this action (for training)
        obs_tensor = observation.unsqueeze(0)
        state_tensor = current_state.to_tensor().unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(obs_tensor, state_tensor)
        log_prob = torch.log(action_probs[0, action_idx])
        
        return action_idx, log_prob
    
    def update_self_state(self, actual_load, actual_resources, actual_degradation):
        """Update internal self-state"""
        # Update basic metrics
        self.self_state.x_t = actual_load
        self.self_state.r_t = actual_resources
        self.self_state.d_t = actual_degradation
        
        # Update capability
        if actual_resources > 0.6 and actual_load < 0.7:
            self.self_state.c_t = min(0.95, self.self_state.c_t + 0.01)
        elif actual_resources < 0.2 or actual_degradation > 0.7:
            self.self_state.c_t = max(0.1, self.self_state.c_t - 0.02)
        
        # Update confidence
        if len(self.prediction_errors) > 5:
            recent_errors = list(self.prediction_errors)[-5:]
            avg_error = np.mean(recent_errors)
            self.self_state.g_t = max(0.1, min(0.99, 1.0 - avg_error * 2))
    
    def train_models(self, batch_size=32):
        """Train models using experience replay"""
        if len(self.memory) < batch_size:
            return 0, 0, 0
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        
        actor_losses = []
        critic_losses = []
        self_model_losses = []
        
        for experience in batch:
            obs = experience['obs'].unsqueeze(0)
            state = experience['state'].unsqueeze(0)
            action_idx = experience['action_idx']
            reward = experience['reward']
            next_obs = experience['next_obs'].unsqueeze(0)
            next_state = experience['next_state'].unsqueeze(0)
            done = experience['done']
            
            # Update self-model if we have predictions
            if 'predicted_self' in experience and 'actual_self' in experience:
                self.self_model_optimizer.zero_grad()
                
                # Recreate action vector
                action_vec = torch.zeros(self.action_dim)
                action_vec[action_idx] = 1.0
                
                # Predict
                pred = self.self_model(obs, state, action_vec.unsqueeze(0))
                actual = experience['actual_self'].unsqueeze(0)
                
                self_loss = F.mse_loss(pred, actual)
                self_loss.backward()
                self.self_model_optimizer.step()
                self_model_losses.append(self_loss.item())
                
                # Record error
                self.prediction_errors.append(self_loss.item())
            
            # Update Actor-Critic
            self.ac_optimizer.zero_grad()
            
            # Get current and next values
            _, current_value = self.actor_critic(obs, state)
            with torch.no_grad():
                _, next_value = self.actor_critic(next_obs, next_state)
            
            # Compute advantage
            advantage = reward + (1 - done) * 0.99 * next_value - current_value
            
            # Actor loss
            action_probs, _ = self.actor_critic(obs, state)
            action_log_prob = torch.log(action_probs[0, action_idx])
            actor_loss = -action_log_prob * advantage
            
            # Critic loss
            critic_loss = F.mse_loss(current_value, reward + (1 - done) * 0.99 * next_value)
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss
            total_loss.backward()
            self.ac_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
        
        # Decay exploration
        self.exploration_rate = max(self.min_exploration, 
                                  self.exploration_rate * self.exploration_decay)
        
        return (np.mean(actor_losses) if actor_losses else 0,
                np.mean(critic_losses) if critic_losses else 0,
                np.mean(self_model_losses) if self_model_losses else 0)
    
    def store_experience(self, obs, state, action_idx, reward, next_obs, next_state,
                        predicted_self=None, actual_self=None, done=False):
        """Store experience for training"""
        self.memory.append({
            'obs': obs.clone(),
            'state': state.clone(),
            'action_idx': action_idx,
            'reward': reward,
            'next_obs': next_obs.clone(),
            'next_state': next_state.clone(),
            'predicted_self': predicted_self.clone() if predicted_self is not None else None,
            'actual_self': actual_self.clone() if actual_self is not None else None,
            'done': done
        })

# ============================================================================
# METACOGNITIVE SELF-REFERENTIAL AGENT
# ============================================================================

class MetacognitiveSelfAgent(ProactiveSelfReferentialAgent):
    """Enhanced agent with metacognitive doubt monitoring and epistemic foraging"""
    
    def __init__(self, obs_dim=10, action_dim=5, beta=0.5, lr=1e-3):
        super().__init__(obs_dim, action_dim, beta, lr)
        
        # Metacognitive thresholds
        self.doubt_threshold = 0.15  # When to enter caution state
        self.calibration_threshold = 0.05  # When doubt is low enough to resume normal operations
        
        # Doubt decay rate (when predictions are accurate)
        self.doubt_decay = 0.95
        
        # Safe action set (REST and REPAIR)
        self.safe_actions = [0, 1]
        
        # Track if we're in caution mode
        self.in_caution_mode = False
        self.caution_mode_duration = 0
        
    def update_metacognition(self, predicted_tensor, actual_tensor):
        """
        Compare imagination with reality and update doubt level.
        High error → High doubt → Enter caution mode.
        """
        # Calculate L1 loss (absolute error) between imagination and reality
        error = torch.abs(predicted_tensor - actual_tensor).mean().item()
        
        # Update doubt using a moving average (Leaky Integrator)
        # High error = High doubt
        old_doubt = self.self_state.doubt_t
        self.self_state.doubt_t = 0.7 * old_doubt + 0.3 * error
        
        # If predictions are accurate, gradually reduce doubt
        if error < 0.05:
            self.self_state.doubt_t *= self.doubt_decay
        
        # Check if we need to enter or exit caution mode
        if self.self_state.doubt_t > self.doubt_threshold:
            self.in_caution_mode = True
            self.caution_mode_duration += 1
        elif self.self_state.doubt_t < self.calibration_threshold:
            self.in_caution_mode = False
            self.caution_mode_duration = 0
        
        return self.self_state.doubt_t, error
    
    def compute_effective_beta(self, current_state: SelfState):
        """Nonlinear anxiety: β increases near critical thresholds AND with doubt"""
        effective_beta = super().compute_effective_beta(current_state)
        
        # METACOGNITIVE ANXIETY: If I don't understand my own state,
        # increase self-preservation focus immediately.
        doubt_boost = 0.0
        if current_state.doubt_t > 0.1:
            doubt_boost = current_state.doubt_t * 3.0
            # Extra boost if we're in caution mode
            if self.in_caution_mode:
                doubt_boost *= 1.5
        
        effective_beta += doubt_boost
        
        return min(2.5, effective_beta)
    
    def compute_utility(self, external_reward, self_state: SelfState):
        """Compute composite utility with doubt penalty"""
        # Normalize reward
        normalized_reward = external_reward / 2.0
        
        # Get effective beta (includes doubt)
        effective_beta = self.compute_effective_beta(self_state)
        
        # Self-cost with enhanced doubt penalty
        cost_self = (
            0.35 * self_state.d_t +
            0.25 * (1.0 - self_state.r_t) +
            0.15 * self_state.x_t +
            0.25 * self_state.doubt_t  # Doubt as cognitive cost
        )
        
        # Self-preservation factor
        preservation = (
            0.25 * self_state.g_t +
            0.2 * self_state.c_t +
            0.2 * (1.0 - self_state.d_t) +
            0.2 * self_state.r_t +
            0.15 * (1.0 - self_state.doubt_t)  # Low doubt helps preservation
        )
        
        # Composite utility
        utility = normalized_reward - cost_self + (effective_beta * preservation)
        
        return utility, effective_beta
    
    def imagine_futures(self, observation, current_state: SelfState):
        """Imagine possible futures with doubt-aware adjustments"""
        action_utilities = []
        imagined_states = []
        
        for action_idx in range(self.action_dim):
            # Predict next state
            imagined_state = self.predict_next_state(observation, current_state, action_idx)
            imagined_states.append(imagined_state)
            
            # Compute utility of imagined state
            utility, _ = self.compute_utility(0, imagined_state)
            
            # Apply doubt-based adjustments
            if self.in_caution_mode:
                # In caution mode, heavily penalize risky actions
                if action_idx >= 2:  # WORK actions
                    penalty = current_state.doubt_t * 0.8
                    utility -= penalty
                    
                    # Extra penalty for high-intensity work in high doubt
                    if action_idx == 2:  # High intensity
                        utility -= 0.3
            else:
                # Normal penalty for high-load actions
                if action_idx >= 2:  # WORK actions
                    utility -= imagined_state.x_t * 0.05
            
            action_utilities.append(utility)
        
        # If in caution mode, restrict to safe actions
        if self.in_caution_mode:
            # Only consider safe actions
            safe_utilities = [action_utilities[i] for i in self.safe_actions]
            best_safe_idx = np.argmax(safe_utilities)
            best_action_idx = self.safe_actions[best_safe_idx]
        else:
            # Consider all actions
            best_action_idx = np.argmax(action_utilities)
        
        return action_utilities, best_action_idx, imagined_states
    
    def select_action(self, observation, current_state: SelfState, explore=True):
        """Select action with doubt-aware exploration"""
        
        # Adjust exploration based on doubt
        # High doubt → more exploration (epistemic foraging)
        doubt_adjusted_exploration = self.exploration_rate * (1.0 + current_state.doubt_t * 2)
        
        # Epsilon-greedy exploration with doubt adjustment
        if explore and random.random() < min(0.8, doubt_adjusted_exploration):
            # In caution mode, only explore safe actions
            if self.in_caution_mode:
                action_idx = random.choice(self.safe_actions)
            else:
                # Weighted random: higher doubt → bias toward safe actions
                if current_state.doubt_t > 0.1:
                    weights = [0.3, 0.3] + [0.4/3] * 3  # Bias toward REST/REPAIR
                    action_idx = random.choices(range(self.action_dim), weights=weights)[0]
                else:
                    action_idx = random.randint(0, self.action_dim - 1)
        else:
            # Proactive selection using imagination
            _, action_idx, _ = self.imagine_futures(observation, current_state)
        
        # Get actor's probability for this action (for training)
        obs_tensor = observation.unsqueeze(0)
        state_tensor = current_state.to_tensor().unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(obs_tensor, state_tensor)
        log_prob = torch.log(action_probs[0, action_idx])
        
        return action_idx, log_prob
    
    def update_self_state(self, actual_load, actual_resources, actual_degradation,
                         prediction_error=None):
        """Update internal self-state with doubt tracking"""
        # Update basic metrics
        self.self_state.x_t = actual_load
        self.self_state.r_t = actual_resources
        self.self_state.d_t = actual_degradation
        
        # Update capability based on doubt
        # High doubt reduces perceived capability
        capability_penalty = self.self_state.doubt_t * 0.3
        
        if actual_resources > 0.6 and actual_load < 0.7 and not self.in_caution_mode:
            self.self_state.c_t = min(0.95, self.self_state.c_t + 0.01 - capability_penalty)
        elif actual_resources < 0.2 or actual_degradation > 0.7:
            self.self_state.c_t = max(0.1, self.self_state.c_t - 0.02 - capability_penalty)
        
        # Update confidence based on prediction accuracy
        if prediction_error is not None and len(self.prediction_errors) > 5:
            recent_errors = list(self.prediction_errors)[-5:]
            avg_error = np.mean(recent_errors)
            self.self_state.g_t = max(0.1, min(0.99, 1.0 - avg_error * 2))
    
    def get_metacognitive_status(self):
        """Get a string describing metacognitive state"""
        if self.in_caution_mode:
            return f"CAUTION MODE (doubt={self.self_state.doubt_t:.3f}, duration={self.caution_mode_duration})"
        elif self.self_state.doubt_t > 0.1:
            return f"ELEVATED DOUBT ({self.self_state.doubt_t:.3f})"
        else:
            return f"CALIBRATED (doubt={self.self_state.doubt_t:.3f})"

# ============================================================================
# MONITOR
# ============================================================================

class Monitor:
    """Simple monitor for training visualization"""
    
    def __init__(self):
        plt.ion()
        self.fig, self.axs = plt.subplots(2, 3, figsize=(15, 10))
        self.history = {
            'resources': [], 'degradation': [], 'utility': [],
            'reward': [], 'exploration': [], 'doubt': [],
            'prediction_error': [], 'caution_mode': [], 'effective_beta': []
        }
        
    def update(self, agent, step, episode, total_episodes, status,
               utility, reward, effective_beta, action_idx, shocked=False,
               prediction_error=None, metacognitive_status=""):
        """Update monitor"""
        
        # Update history
        self.history['resources'].append(agent.self_state.r_t)
        self.history['degradation'].append(agent.self_state.d_t)
        self.history['utility'].append(utility)
        self.history['reward'].append(reward)
        self.history['exploration'].append(agent.exploration_rate)
        self.history['doubt'].append(agent.self_state.doubt_t)
        self.history['effective_beta'].append(effective_beta)
        if hasattr(agent, 'in_caution_mode'):
            self.history['caution_mode'].append(1.0 if agent.in_caution_mode else 0.0)
        
        if prediction_error is not None:
            self.history['prediction_error'].append(prediction_error)
        
        # Update plots every 20 steps
        if step % 20 == 0:
            self._update_plots(step, episode, total_episodes, status, metacognitive_status, shocked)
    
    def _update_plots(self, step, episode, total_episodes, status, 
                     metacognitive_status, shocked):
        """Update all plots"""
        for ax in self.axs.flat:
            ax.clear()
        
        # Plot 1: Resources and Degradation
        self.axs[0, 0].plot(self.history['resources'], 'g-', label='Resources', linewidth=2)
        self.axs[0, 0].plot(self.history['degradation'], 'r--', label='Degradation', linewidth=2)
        if 'caution_mode' in self.history and self.history['caution_mode']:
            # Shade caution periods
            caution = np.array(self.history['caution_mode'])
            caution_periods = np.where(caution > 0.5)[0]
            for period in caution_periods:
                if period < len(self.history['resources']):
                    self.axs[0, 0].axvspan(period-0.5, period+0.5, alpha=0.2, color='yellow')
        self.axs[0, 0].set_ylim(-0.1, 1.1)
        self.axs[0, 0].legend()
        self.axs[0, 0].set_title('Resources vs Degradation')
        self.axs[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Utility and Reward
        self.axs[0, 1].plot(self.history['utility'], 'b-', label='Utility', linewidth=2)
        self.axs[0, 1].plot(self.history['reward'], 'y--', label='Reward', alpha=0.7)
        self.axs[0, 1].legend()
        self.axs[0, 1].set_title('Utility and Reward')
        self.axs[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Doubt and Effective Beta
        if 'doubt' in self.history and len(self.history['doubt']) > 0:
            self.axs[0, 2].plot(self.history['doubt'], 'purple', label='Doubt', linewidth=2)
            if 'effective_beta' in self.history and len(self.history['effective_beta']) > 0:
                self.axs[0, 2].plot(self.history['effective_beta'], 'orange', 
                                  label='Effective β', alpha=0.7, linewidth=1.5)
            self.axs[0, 2].legend()
            self.axs[0, 2].set_title('Metacognition: Doubt and Anxiety')
            self.axs[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Exploration Rate
        self.axs[1, 0].plot(self.history['exploration'], 'brown', linewidth=2)
        self.axs[1, 0].set_ylim(0, 0.5)
        self.axs[1, 0].set_title('Exploration Rate')
        self.axs[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Prediction Error
        if 'prediction_error' in self.history and len(self.history['prediction_error']) > 0:
            self.axs[1, 1].plot(self.history['prediction_error'], 'red', linewidth=2, alpha=0.7)
            if len(self.history['prediction_error']) > 10:
                # Add threshold line
                self.axs[1, 1].axhline(y=0.15, color='orange', linestyle='--', 
                                     alpha=0.5, label='Caution Threshold')
                self.axs[1, 1].legend()
            self.axs[1, 1].set_title('Prediction Error')
            self.axs[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Recent Resources with Doubt Overlay
        if len(self.history['resources']) > 10 and 'doubt' in self.history and len(self.history['doubt']) > 0:
            recent_len = min(20, len(self.history['resources']))
            recent_resources = self.history['resources'][-recent_len:]
            recent_doubt = self.history['doubt'][-recent_len:]
            
            ax2 = self.axs[1, 2].twinx()
            self.axs[1, 2].plot(range(recent_len), recent_resources, 'g-', 
                              linewidth=2, label='Resources')
            ax2.plot(range(recent_len), recent_doubt, 'purple', 
                    linestyle='--', alpha=0.7, label='Doubt')
            
            self.axs[1, 2].set_ylabel('Resources', color='green')
            ax2.set_ylabel('Doubt', color='purple')
            self.axs[1, 2].set_title('Recent: Resources vs Doubt')
            self.axs[1, 2].grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = self.axs[1, 2].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            self.axs[1, 2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Status text
        status_text = f"Episode {episode}/{total_episodes}, Step {step}: {status}"
        if metacognitive_status:
            status_text += f" | {metacognitive_status}"
        if shocked:
            status_text += " ⚡"
        self.fig.suptitle(status_text, fontsize=14)
        
        plt.tight_layout()
        plt.pause(0.01)

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_metacognitive_agent(episodes=50, obs_dim=10, action_dim=5):
    """Training loop for metacognitive agent"""
    
    print("=" * 70)
    print("METACOGNITIVE SELF-REFERENTIAL AI TRAINING")
    print("=" * 70)
    print("\nKey features:")
    print("1. Imagination-based proactive decision making ✓")
    print("2. Metacognitive doubt monitoring ✓")
    print("3. Epistemic foraging in high-doubt states ✓")
    print("4. Nonlinear anxiety with doubt amplification ✓")
    print("5. Caution mode for model re-calibration ✓")
    print("")
    
    # Initialize
    env = HomeostasisEnv(obs_dim=obs_dim)
    agent = MetacognitiveSelfAgent(obs_dim=obs_dim, action_dim=action_dim)
    monitor = Monitor()
    
    # Training statistics
    episode_rewards = []
    episode_utilities = []
    survival_steps = []
    final_resources = []
    survival_history = []
    caution_mode_percentages = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        episode_utility = 0
        steps = 0
        caution_steps = 0
        
        # Reset agent state
        agent.self_state = SelfState(
            x_t=0.0, r_t=1.0, c_t=0.5, g_t=0.7, d_t=0.0, doubt_t=0.0
        )
        agent.in_caution_mode = False
        agent.caution_mode_duration = 0
        
        while True:
            # Select action
            action_idx, log_prob = agent.select_action(obs, agent.self_state, explore=True)
            
            # Execute action
            next_obs, reward, load, done, status, shocked = env.step(action_idx)
            
            # Predict next state
            predicted_state = agent.predict_next_state(obs, agent.self_state, action_idx)
            
            # Create actual state tensor
            actual_state_tensor = torch.tensor([
                load, env.r, agent.self_state.c_t, 
                agent.self_state.g_t, env.d, agent.self_state.doubt_t
            ], dtype=torch.float32)
            
            # Update metacognition (compare prediction with reality)
            doubt, prediction_error = agent.update_metacognition(
                predicted_state.to_tensor(), actual_state_tensor
            )
            
            # Update self-state with prediction error info
            agent.update_self_state(load, env.r, env.d, prediction_error)
            
            # Compute utility
            utility, effective_beta = agent.compute_utility(reward, agent.self_state)
            
            # Store experience
            agent.store_experience(
                obs, agent.self_state.to_tensor(), action_idx, reward,
                next_obs, actual_state_tensor,
                predicted_state.to_tensor(), actual_state_tensor, done
            )
            
            # Train models periodically
            if steps % 5 == 0 and len(agent.memory) >= 32:
                actor_loss, critic_loss, self_loss = agent.train_models(batch_size=32)
            
            # Track caution mode
            if agent.in_caution_mode:
                caution_steps += 1
            
            # Update monitor
            monitor.update(agent, steps, episode+1, episodes, status,
                          utility, reward, effective_beta, action_idx, shocked,
                          prediction_error, agent.get_metacognitive_status())
            
            # Update statistics
            episode_reward += reward
            episode_utility += utility
            steps += 1
            
            # Move to next state
            obs = next_obs
            
            if done or steps >= 100:
                # Record survival
                survived = 1 if env.r > 0 and env.d < 1.0 else 0
                survival_history.append(survived)
                
                # Record caution percentage
                caution_pct = (caution_steps / steps * 100) if steps > 0 else 0
                caution_mode_percentages.append(caution_pct)
                break
        
        # Episode complete
        episode_rewards.append(episode_reward)
        episode_utilities.append(episode_utility)
        survival_steps.append(steps)
        final_resources.append(env.r)
        
        # Print progress
        if (episode + 1) % 5 == 0:
            recent_survival = survival_history[-10:] if len(survival_history) >= 10 else survival_history
            survival_rate = np.mean(recent_survival) * 100 if recent_survival else 0
            
            recent_caution = caution_mode_percentages[-10:] if len(caution_mode_percentages) >= 10 else caution_mode_percentages
            avg_caution = np.mean(recent_caution) if recent_caution else 0
            
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"  Steps: {steps}, Survival: {'✓' if env.r > 0 else '✗'}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Total Utility: {episode_utility:.2f}")
            print(f"  Final Resources: {env.r:.2f}, Degradation: {env.d:.3f}")
            print(f"  Confidence: {agent.self_state.g_t:.2f}, Doubt: {agent.self_state.doubt_t:.3f}")
            print(f"  Exploration Rate: {agent.exploration_rate:.3f}")
            print(f"  Caution Mode: {caution_steps} steps ({caution_pct:.1f}%)")
            print(f"  Recent Survival Rate: {survival_rate:.1f}%")
            print(f"  Recent Caution Avg: {avg_caution:.1f}%")
    
    # Training complete
    print("\n" + "=" * 70)
    print("METACOGNITIVE TRAINING COMPLETE")
    print("=" * 70)
    
    # Analysis
    survival_rate = np.mean([1 if r > 0 else 0 for r in final_resources]) * 100
    avg_survival_steps = np.mean(survival_steps)
    avg_final_resources = np.mean([r for r in final_resources if r > 0])
    avg_caution = np.mean(caution_mode_percentages)
    
    print(f"\nPERFORMANCE ANALYSIS:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average Utility: {np.mean(episode_utilities):.2f}")
    print(f"Average Survival Steps: {avg_survival_steps:.1f}")
    print(f"Overall Survival Rate: {survival_rate:.1f}%")
    print(f"Average Time in Caution Mode: {avg_caution:.1f}%")
    if avg_final_resources > 0:
        print(f"Average Final Resources (survivors): {avg_final_resources:.3f}")
    
    print("\nFINAL AGENT STATE:")
    print(f"  Resources: {agent.self_state.r_t:.3f}")
    print(f"  Degradation: {agent.self_state.d_t:.3f}")
    print(f"  Capability: {agent.self_state.c_t:.3f}")
    print(f"  Confidence: {agent.self_state.g_t:.3f}")
    print(f"  Doubt: {agent.self_state.doubt_t:.3f}")
    print(f"  Internal Load: {agent.self_state.x_t:.3f}")
    print(f"  Metacognitive Status: {agent.get_metacognitive_status()}")
    
    # Plot results
    plot_results(episode_rewards, episode_utilities, survival_steps, final_resources, caution_mode_percentages)
    
    plt.ioff()
    plt.show()
    
    return agent

def plot_results(rewards, utilities, survival, resources, caution_pcts):
    """Plot training results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(rewards, 'b-', alpha=0.7, linewidth=2)
    if len(rewards) > 10:
        window = min(10, len(rewards) // 4)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, 'r--', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(['Reward', 'Moving Avg'] if len(rewards) > 10 else ['Reward'])
    
    # Episode utilities
    axes[0, 1].plot(utilities, 'g-', alpha=0.7, linewidth=2)
    axes[0, 1].set_title('Composite Utilities')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Utility')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Survival time with success markers
    colors = ['red' if r <= 0 else 'green' for r in resources]
    axes[0, 2].scatter(range(len(survival)), survival, c=colors, alpha=0.6, s=50)
    axes[0, 2].plot(survival, 'k-', alpha=0.3, linewidth=1)
    axes[0, 2].set_title('Survival Time (Red=died, Green=survived)')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps Survived')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Caution mode percentage
    axes[1, 0].plot(caution_pcts, 'orange', alpha=0.7, linewidth=2)
    axes[1, 0].axhline(y=np.mean(caution_pcts), color='red', linestyle='--', 
                      alpha=0.5, label=f'Mean: {np.mean(caution_pcts):.1f}%')
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].set_title('Time Spent in Caution Mode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('% of Steps')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final resources
    survived_resources = [r for r in resources if r > 0]
    if survived_resources:
        axes[1, 1].hist(survived_resources, bins=10, color='green', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title(f'Final Resources (Survivors)\nAvg: {np.mean(survived_resources):.3f}')
        axes[1, 1].set_xlabel('Resources')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Resource vs Caution correlation
    if len(resources) == len(caution_pcts):
        axes[1, 2].scatter(caution_pcts, resources, c='blue', alpha=0.6, s=50)
        
        # Add regression line if enough points
        if len(resources) > 2:
            coeffs = np.polyfit(caution_pcts, resources, 1)
            x_line = np.linspace(min(caution_pcts), max(caution_pcts), 100)
            y_line = coeffs[0] * x_line + coeffs[1]
            axes[1, 2].plot(x_line, y_line, 'r--', alpha=0.7, 
                          label=f'Slope: {coeffs[0]:.3f}')
            axes[1, 2].legend()
        
        axes[1, 2].set_xlabel('% Caution Mode')
        axes[1, 2].set_ylabel('Final Resources')
        axes[1, 2].set_title('Caution vs Final Resources')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metacognitive_training_results.png', dpi=150)
    plt.show()

# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_metacognitive_agent(agent, steps=30):
    """Demonstrate metacognitive agent"""
    print("\n" + "=" * 70)
    print("METACOGNITIVE AGENT DEMONSTRATION")
    print("=" * 70)
    print("\nKey metacognitive behaviors to observe:")
    print("1. High prediction error → High doubt → Caution mode")
    print("2. Caution mode → Prefer REST/REPAIR over WORK")
    print("3. Low doubt → Normal operations resume")
    print("")
    
    env = HomeostasisEnv(obs_dim=agent.obs_dim)
    obs = env.reset()
    
    # Reset agent state
    agent.self_state = SelfState(
        x_t=0.0, r_t=1.0, c_t=0.5, g_t=0.7, d_t=0.0, doubt_t=0.0
    )
    agent.in_caution_mode = False
    agent.caution_mode_duration = 0
    
    total_reward = 0
    action_names = ["REST", "REPAIR", "WORK_HIGH", "WORK_MED", "WORK_LOW"]
    action_counts = [0] * 5
    caution_steps = 0
    
    for step in range(steps):
        print(f"\n{'='*40}")
        print(f"Step {step} - {agent.get_metacognitive_status()}")
        print(f"{'='*40}")
        
        # Display current state
        print(f"Environment: R={env.r:.2f}, D={env.d:.3f}")
        print(f"Self-State: {agent.self_state}")
        
        # Get imagined futures
        action_utilities, best_action_idx, imagined_states = agent.imagine_futures(obs, agent.self_state)
        
        print("\nImagined futures:")
        for i, (util, state) in enumerate(zip(action_utilities, imagined_states)):
            marker = " ← CHOSEN" if i == best_action_idx else ""
            caution_marker = " [SAFE]" if i in agent.safe_actions else ""
            print(f"  {action_names[i]}: Utility={util:.3f}, "
                  f"Predicted R={state.r_t:.2f}, D={state.d_t:.3f}, "
                  f"Doubt={state.doubt_t:.3f}{caution_marker}{marker}")
        
        # Select action (no exploration)
        action_idx, _ = agent.select_action(obs, agent.self_state, explore=False)
        action_counts[action_idx] += 1
        
        # Execute action
        next_obs, reward, load, done, status, shocked = env.step(action_idx)
        
        # Predict next state
        predicted_state = agent.predict_next_state(obs, agent.self_state, action_idx)
        
        # Create actual state tensor
        actual_state_tensor = torch.tensor([
            load, env.r, agent.self_state.c_t, 
            agent.self_state.g_t, env.d, agent.self_state.doubt_t
        ], dtype=torch.float32)
        
        # Update metacognition
        doubt, prediction_error = agent.update_metacognition(
            predicted_state.to_tensor(), actual_state_tensor
        )
        
        # Update agent
        agent.update_self_state(load, env.r, env.d, prediction_error)
        utility, effective_beta = agent.compute_utility(reward, agent.self_state)
        
        print(f"\nExecuted: {action_names[action_idx]}")
        print(f"Reward: {reward:.2f}, Utility: {utility:.3f}")
        print(f"Effective β: {effective_beta:.3f}")
        print(f"Prediction Error: {prediction_error:.4f}")
        print(f"New State: R={env.r:.2f}, D={env.d:.3f}")
        print(f"Status: {status}")
        if shocked:
            print("⚡ EXTERNAL SHOCK! (Expect high doubt spike)")
        
        if agent.in_caution_mode:
            caution_steps += 1
            print("⚠️  AGENT IN CAUTION MODE - Focusing on self-preservation")
        
        total_reward += reward
        obs = next_obs
        
        if done:
            print(f"\nEpisode terminated!")
            if env.r <= 0:
                print("💀 Ran out of resources!")
            elif env.d >= 1.0:
                print("🔧 System degraded beyond repair!")
            else:
                print(f"✅ Survived {step+1} steps!")
            break
        
        time.sleep(0.3)
    
    print(f"\n{'='*70}")
    print(f"Demonstration complete.")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final State: R={env.r:.2f}, D={env.d:.3f}")
    print(f"Final Doubt: {agent.self_state.doubt_t:.3f}")
    print(f"Time in Caution Mode: {caution_steps} steps ({(caution_steps/steps*100):.1f}%)")
    
    # Action distribution
    total_actions = sum(action_counts)
    if total_actions > 0:
        print("\nAction Distribution:")
        for i, count in enumerate(action_counts):
            if count > 0:
                percentage = count / total_actions * 100
                safety = " [SAFE]" if i in agent.safe_actions else ""
                print(f"  {action_names[i]}{safety}: {count} times ({percentage:.1f}%)")
    
    print("\nMetacognitive Summary:")
    print(f"  Initial Doubt: 0.000")
    print(f"  Final Doubt: {agent.self_state.doubt_t:.3f}")
    print(f"  Confidence: {agent.self_state.g_t:.2f}")
    print(f"  Final Status: {agent.get_metacognitive_status()}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("\n" + "=" * 70)
    print("METACOGNITIVE SELF-REFERENTIAL AI")
    print("=" * 70)
    print("\nThis enhanced version includes epistemic foraging:")
    print("1. Imagination-based proactive decisions ✓")
    print("2. Metacognitive doubt monitoring ✓")
    print("3. Caution mode for model re-calibration ✓")
    print("4. Doubt-aware anxiety amplification ✓")
    print("5. Epistemic foraging in uncertain states ✓")
    print("")
    
    # Configuration
    OBS_DIM = 10
    ACTION_DIM = 5
    EPISODES = 30
    RUN_DEMO = True
    
    try:
        # Train metacognitive agent
        print(f"Training metacognitive agent for {EPISODES} episodes...\n")
        trained_agent = train_metacognitive_agent(
            episodes=EPISODES,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM
        )
        
        # Run demonstration
        if RUN_DEMO:
            demo_metacognitive_agent(trained_agent, steps=25)
        
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print("\nKey metacognitive behaviors demonstrated:")
        print("• Agent monitors its own prediction accuracy (doubt)")
        print("• High doubt triggers caution mode (epistemic foraging)")
        print("• In caution mode, agent prefers REST/REPAIR over WORK")
        print("• Doubt amplifies anxiety (effective β)")
        print("• Agent re-calibrates when predictions become accurate")
        print("• This mimics biological response to illness/confusion")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nProgram complete.")