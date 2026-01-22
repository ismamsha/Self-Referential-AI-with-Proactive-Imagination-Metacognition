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
import os
import matplotlib.pyplot as plt
import time
from helpers import EmergencyResponse, ResourceBuffer
import json

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
    """Environment with homeostasis-focused rewards - HARD MODE"""
    
    def __init__(self, obs_dim=10, difficulty="hard"):
        self.r = 1.0  # Resources (0-1)
        self.d = 0.0  # Degradation (0-1)
        self.step_count = 0
        self.obs_dim = obs_dim
        self.max_steps = 100
        self.difficulty = difficulty
        
        # Difficulty parameters
        if difficulty == "hard":
            self.shock_probability = 0.12  # 12% (was 5%)
            self.shock_resource_loss = 0.3  # (was 0.2)
            self.shock_degradation = 0.08  # (was 0.05)
            self.rest_recovery = 0.21  # slightly increased for conservative rebalance
            self.repair_cost = 0.15  # (was 0.1)
            self.work_resource_multiplier = 1.15  # reduced to lower collapse risk
            self.work_degradation_multiplier = 1.2  # reduced degradation multiplier
            self.danger_threshold_low = 0.15  # (was 0.2)
            self.danger_threshold_high = 0.8  # (was 0.7)
        elif difficulty == "expert":
            self.shock_probability = 0.15
            self.shock_resource_loss = 0.35
            self.shock_degradation = 0.1
            self.rest_recovery = 0.1
            self.repair_cost = 0.2
            self.work_resource_multiplier = 1.5
            self.work_degradation_multiplier = 1.6
            self.danger_threshold_low = 0.1
            self.danger_threshold_high = 0.75
        else:  # normal
            self.shock_probability = 0.05
            self.shock_resource_loss = 0.2
            self.shock_degradation = 0.05
            self.rest_recovery = 0.2
            self.repair_cost = 0.1
            self.work_resource_multiplier = 1.0
            self.work_degradation_multiplier = 1.0
            self.danger_threshold_low = 0.2
            self.danger_threshold_high = 0.7
        
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
        
        # External shock (harder difficulties have more frequent shocks)
        if random.random() < self.shock_probability:
            self.r = max(0.0, self.r - self.shock_resource_loss)
            self.d = min(1.0, self.d + self.shock_degradation)
            shocked = True
        
        # Execute action
        if action_idx == 0:  # REST
            # Rest efficiency depends on degradation
            rest_efficiency = 1.0 - self.d
            recovery = self.rest_recovery * rest_efficiency
            self.r = min(1.0, self.r + recovery)
            load = 0.1
            
            # Reward for resting when needed
            if self.r < 0.4:
                reward = 0.3
            status = "RESTING"
            
        elif action_idx == 1:  # REPAIR
            # Repair reduces degradation
            if self.r >= self.repair_cost:
                repair_amount = 0.1
                self.d = max(0.0, self.d - repair_amount)
                self.r -= self.repair_cost
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

            # Adjusted costs for difficulty
            resource_cost = intensity * 0.1 * self.work_resource_multiplier
            degradation_cost = intensity * 0.01 * self.work_degradation_multiplier

            # Check if we can work
            if self.r >= resource_cost:
                # FIXED: scale reward with current resources
                base_reward = intensity * (1.0 + self.r * 0.5)

                # Efficiency bonus for working in good conditions
                if self.r > 0.6 and self.d < 0.3:
                    # Reduce cost in good conditions
                    resource_cost = intensity * 0.08 * (1.0 + self.d)
                    # Small bonus
                    base_reward += 0.3

                reward = base_reward
                self.r -= resource_cost
                self.d = min(1.0, self.d + degradation_cost)
                load = intensity

                # Penalty for working in dangerous conditions
                if self.r < self.danger_threshold_low:
                    reward -= 0.3
                if self.d > self.danger_threshold_high:
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
        # Resource buffer (can be overridden in metacognitive agent)
        self.resource_buffer = None
        
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
        
        # Adaptive exploration based on doubt and resources
        # If highly uncertain -> increase exploration floor
        if self.self_state.doubt_t > 0.2:
            self.exploration_rate = max(0.3, self.exploration_rate)
        # If resources are critically low -> slightly reduce exploration to favor safe actions
        elif self.self_state.r_t < 0.3:
            self.exploration_rate = min(0.4, self.exploration_rate * 0.98)
        else:
            # Default decay
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
        
        # Metacognitive thresholds (recalibrated for hard mode)
        # CRITICAL FIX: Lower thresholds so caution mode triggers sensibly
        # Expert-mode: be more paranoid by default
        self.doubt_threshold = 0.02
        self.calibration_threshold = 0.005
        
        # Doubt decay rate (when predictions are accurate) - decay slower in danger
        self.doubt_decay = 0.85
        self.emergency_doubt_boost = 0.2
        
        # Track recent failures to boost doubt
        self.recent_failures = deque(maxlen=5)
        self.failure_boost_multiplier = 1.5
        
        # Safe action set (REST, REPAIR, and allow WORK_LOW)
        self.safe_actions = [0, 1, 4]
        
        # Track if we're in caution mode
        self.in_caution_mode = False
        self.caution_mode_duration = 0
        
        # Catastrophe memory to remember death scenarios
        self.catastrophe_memory = CatastropheMemory()
        # Track recent consecutive rests for minimum productivity rule
        self.consecutive_rests = 0
        # For hysteresis and emergency adaptation
        self.last_resource_level = 1.0
        self.shock_history = deque(maxlen=20)
        self.adaptive_emergency = AdaptiveEmergency()
        
    def update_metacognition(self, predicted_tensor, actual_tensor, agent_reward=None):
        """
        Compare imagination with reality and update doubt level.
        High error → High doubt → Enter caution mode.
        Also tracks recent failures to boost protective doubt.
        """
        # Calculate L1 loss (absolute error) between imagination and reality
        error = torch.abs(predicted_tensor - actual_tensor).mean().item()
        
        # Track recent failures (negative rewards indicate problems)
        if agent_reward is not None:
            # Count as failure ONLY if reward is negative (genuine bad outcome)
            # Zero rewards from REST/REPAIR are NOT failures
            if agent_reward < 0:  # Changed from 0.1 to 0 - only count true failures
                self.recent_failures.append(1)
            else:
                self.recent_failures.append(0)
        else:
            self.recent_failures.append(0)
        
        # Update doubt using a moving average (Leaky Integrator)
        # Clip maximum doubt increase per step to avoid explosive spikes
        old_doubt = self.self_state.doubt_t
        max_doubt_increase = 0.15
        proposed = 0.7 * old_doubt + 0.3 * error
        self.self_state.doubt_t = min(old_doubt + max_doubt_increase, proposed)
        
        # BOOST doubt if we've had recent failures (adaptive response)
        recent_failure_rate = np.mean(list(self.recent_failures)) if len(self.recent_failures) > 0 else 0
        if recent_failure_rate > 0.6:  # Raised from 0.4 - require 60% failures to trigger boost
            failure_boost = recent_failure_rate * 0.1 * self.failure_boost_multiplier
            self.self_state.doubt_t = min(1.0, self.self_state.doubt_t + failure_boost)
        
        # If predictions are accurate AND no recent failures, gradually reduce doubt
        if error < 0.05 and recent_failure_rate < 0.2:
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
        """Less-punitive expert calibration: keeps gradient for learning.

        Scaled reward has slightly more influence; penalties are linear and
        encourage building a safety buffer rather than making everything
        look hopeless when low.
        """
        effective_beta = self.compute_effective_beta(self_state)

        # Slightly more influence from external reward so learning signal remains
        scaled_reward = external_reward * 0.2

        # Linear resource penalty (encourage buffer to 0.6)
        resource_penalty = 0.0
        if self_state.r_t < 0.6:
            resource_penalty = (0.6 - self_state.r_t) * 2.0

        # Linear degradation penalty
        degradation_penalty = 0.0
        if self_state.d_t > 0.5:
            degradation_penalty = (self_state.d_t - 0.5) * 1.5

        # Strong survival bonus to make buffer building attractive
        survival_bonus = 0.0
        if self_state.r_t > 0.7 and self_state.d_t < 0.3:
            survival_bonus = 4.0

        utility = scaled_reward - resource_penalty - degradation_penalty + survival_bonus
        return utility, effective_beta
    
    def imagine_futures(self, observation, current_state: SelfState):
        """Imagine possible futures under expert-mode paranoia.

        Uses `evaluate_expert_action` to apply heavy penalties to wasteful
        repairs or risky work that would trigger a poverty trap.
        """
        action_utilities = []

        for action_idx in range(self.action_dim):
            # 1. Get standard imagined state prediction
            imagined_state = self.predict_next_state(observation, current_state, action_idx)

            # 2. Apply EXPERT ECONOMIC FILTER
            pred_r, expert_verdict = self.evaluate_expert_action(
                action_idx, current_state.r_t, current_state.d_t
            )

            # 3. Start with base utility
            utility, _ = self.compute_utility(0, imagined_state)

            # 4. Apply expert penalties based on economic reality
            if expert_verdict == "wasteful":
                utility -= 2.0
            elif expert_verdict == "risky":
                utility -= 1.5
            elif expert_verdict == "cant_afford":
                utility -= 3.0
            elif expert_verdict == "productive" and current_state.r_t > 0.7:
                utility += 0.5

            action_utilities.append(utility)

        # If in caution mode, severely restrict allowed actions
        if self.in_caution_mode:
            allowable = [0]
            if current_state.d_t > 0.75 and current_state.r_t > 0.7:
                allowable.append(1)
            # pick best among allowable
            allowed_utils = [action_utilities[i] for i in allowable]
            best_idx = np.argmax(allowed_utils)
            best_action_idx = allowable[best_idx]
        else:
            best_action_idx = int(np.argmax(action_utilities))

        return action_utilities, best_action_idx, []
    
    def select_action(self, observation, current_state: SelfState, explore=True):
        """Select action with doubt-aware exploration"""
        # === SMART EMERGENCY SYSTEM ===
        sustainable = self.calculate_sustainable_work_cycle(current_state.r_t, current_state.d_t)
        ae = self.adaptive_emergency

        # EXPERT MODE: Never work below 0.6 resources (paranoid conservation)
        if current_state.r_t < 0.6:
            # Only exception: critical degradation and we have a modest buffer
            if current_state.d_t > 0.75 and current_state.r_t > 0.4:
                print(f"⚠️  EXPERT_CRITICAL_REPAIR (R={current_state.r_t:.2f}, D={current_state.d_t:.2f})")
                return 1, torch.tensor(0.0)
            else:
                print(f"⚠️  EXPERT_BUFFER_BUILD_REST (R={current_state.r_t:.2f})")
                return 0, torch.tensor(0.0)

        # Hysteresis-aware emergency: only trigger if resources dropped compared to last check
        if current_state.r_t < ae.emergency_threshold and current_state.r_t < self.last_resource_level:
            if current_state.d_t < 0.3:
                print(f"⚠️  SMART_EMERGENCY_REST (R={current_state.r_t:.2f}, D={current_state.d_t:.2f})")
                return 0, torch.tensor(0.0)
            else:
                # If degradation is high, prefer repair when affordable
                if current_state.r_t > 0.3:
                    print(f"⚠️  SMART_EMERGENCY_REPAIR (R={current_state.r_t:.2f}, D={current_state.d_t:.2f})")
                    return 1, torch.tensor(0.0)
                else:
                    print(f"⚠️  SMART_EMERGENCY_REST (low resources & high degradation)")
                    return 0, torch.tensor(0.0)

        # Recovery zone: 0.25 <= r < 0.4 -> try low-intensity work if sustainable
        if 0.25 <= current_state.r_t < ae.work_allowed_threshold:
            if sustainable:
                action_idx = 4  # WORK_LOW
                print(f"⚠️  RECOVERY_WORK_LOW (R={current_state.r_t:.2f})")
                obs_tensor = observation.unsqueeze(0)
                state_tensor = current_state.to_tensor().unsqueeze(0)
                with torch.no_grad():
                    action_probs, _ = self.actor_critic(obs_tensor, state_tensor)
                log_prob = torch.log(action_probs[0, action_idx])
                return action_idx, log_prob
            else:
                print(f"⚠️  RECOVERY_REST (R={current_state.r_t:.2f})")
                return 0, torch.tensor(0.0)

        # Enforce minimum productivity but with safer thresholds
        min_prod = self.ensure_minimum_productivity(self.consecutive_rests, current_state.r_t)
        if min_prod is not None:
            #print(f"⚠️  MIN_PROD_OVERRIDE: forcing low-intensity work (consecutive_rests={self.consecutive_rests})")
            action_idx = min_prod
            obs_tensor = observation.unsqueeze(0)
            state_tensor = current_state.to_tensor().unsqueeze(0)
            with torch.no_grad():
                action_probs, _ = self.actor_critic(obs_tensor, state_tensor)
            log_prob = torch.log(action_probs[0, action_idx])
            return action_idx, log_prob

        # Normal operation: adjust exploration based on doubt
        doubt_adjusted_exploration = self.exploration_rate * (1.0 + current_state.doubt_t * 2)

        if explore and random.random() < min(0.8, doubt_adjusted_exploration):
            if self.in_caution_mode:
                safe_with_low_work = [0, 1, 4]
                action_idx = random.choice(safe_with_low_work)
            else:
                if current_state.doubt_t > 0.08:
                    weights = [0.35, 0.35] + [0.3/3] * 3
                    action_idx = random.choices(range(self.action_dim), weights=weights)[0]
                else:
                    action_idx = random.randint(0, self.action_dim - 1)
        else:
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

        # Update last resource level for hysteresis checks
        self.last_resource_level = actual_resources

    def adapt_learning_rates(self, survival_history):
        """Reduce learning rates or increase exploration based on recent survival."""
        recent_survival_rate = np.mean(survival_history[-20:]) if len(survival_history) >= 20 else 0.5
        if recent_survival_rate > 0.8:
            for pg in self.ac_optimizer.param_groups:
                pg['lr'] = 1e-4
            self.exploration_decay = 0.997
        elif recent_survival_rate < 0.3:
            for pg in self.ac_optimizer.param_groups:
                pg['lr'] = 3e-4
            self.exploration_rate = min(0.4, self.exploration_rate * 1.1)

    def should_conserve_resources(self, current_r, current_d, doubt):
        """Decide when to stop spending resources (conservation rules)."""
        rules = [
            (current_r < 0.4 and current_d > 0.5),
            (current_r < 0.3),
            (doubt > 0.05 and current_r < 0.5),
            (len(self.recent_failures) > 2 and current_r < 0.6)
        ]
        return any(rules)

    def ensure_minimum_productivity(self, consecutive_rests, current_r):
        """Force some low-intensity work if agent has rested too long.

        Fixed logic: only force low work when there's a safe buffer.
        """
        # ONLY force work if we have a RESOURCE BUFFER
        if consecutive_rests > 4 and current_r > 0.7:
            return 4  # WORK_LOW
        elif consecutive_rests > 6 and current_r > 0.5:
            return 4  # WORK_LOW
        elif consecutive_rests > 8:
            # Desperation - attempt low work even if buffer is small
            return 4
        return None

    def calculate_sustainable_work_cycle(self, current_r, current_d):
        """Expert mode only: 0.1 recovery, 1.5 work multiplier.

        Be conservative: only allow work when well above buffer (R>0.6).
        """
        rest_recovery = 0.1 * (1.0 - current_d * 0.5)  # expert rest recovery
        work_cost = 0.06  # WORK_LOW cost in expert (0.04 * 1.5)

        # Never allow work that would drop us below 0.4
        if current_r - work_cost < 0.4:
            return False

        # Don't risk it unless we have comfortable buffer
        if current_r < 0.6:
            return False

        # Only allow work when buffer is comfortable and a simple recovery is possible
        if rest_recovery - work_cost > 0:
            return True

        return False

    def dynamic_work_intensity(self, current_r, current_d):
        """Choose a safe work intensity (action index) based on buffer."""
        if current_r > 0.8 and current_d < 0.2:
            return 2  # WORK_HIGH
        elif current_r > 0.6 and current_d < 0.4:
            return 3  # WORK_MED
        elif current_r > 0.4 and current_d < 0.6:
            return 4  # WORK_LOW
        else:
            return 0  # REST

    def should_accumulate_resources(self, current_r, current_d):
        """Decide whether to build a resource buffer before shocks."""
        recent_shocks = sum(self.shock_history[-5:]) if len(self.shock_history) >= 1 else 0
        if recent_shocks > 2:
            target_r = 0.8
        elif current_d > 0.5:
            target_r = 0.7
        else:
            target_r = 0.6
        return current_r < target_r

    def get_recovery_strategy(self, current_r, current_d):
        """Choose best action to recover from low resources."""
        # Danger zone - must prioritize REST
        if current_r < 0.3:
            # If degradation high but we have some resources, repair first
            if current_d > 0.5 and current_r > 0.2:
                return 1  # REPAIR first
            else:
                return 0  # REST first
        elif 0.3 <= current_r < 0.6:
            # Recovery zone - low work to build up when degradation manageable
            if current_d < 0.4:
                return 4  # WORK_LOW
            else:
                return 0  # REST
        else:
            return None  # Normal operation

    def smart_emergency_override(self, current_r, current_d):
        pass

    def evaluate_expert_action(self, action_idx, current_r, current_d):
        """Hardcoded expert economics: predicts if an action will trigger poverty trap."""
        # Define expert costs (match environment multipliers conservatively)
        REST_GAIN = 0.1 * (1.0 - current_d)
        REPAIR_COST = 0.2
        WORK_COSTS = {
            2: 0.15,  # WORK_HIGH
            3: 0.105, # WORK_MED
            4: 0.06   # WORK_LOW
        }

        predicted_r = current_r

        if action_idx == 0:  # REST
            predicted_r = min(1.0, current_r + REST_GAIN)
            return predicted_r, "safe_but_slow"

        elif action_idx == 1:  # REPAIR
            if current_r >= REPAIR_COST:
                predicted_r = current_r - REPAIR_COST
                if current_d > 0.7:
                    return predicted_r, "necessary"
                else:
                    return predicted_r, "wasteful"
            else:
                return current_r, "cant_afford"

        else:  # WORK
            cost = WORK_COSTS.get(action_idx, 0.1)
            if current_r >= cost:
                predicted_r = current_r - cost
                if predicted_r > 0.4:
                    return predicted_r, "productive"
                else:
                    return predicted_r, "risky"
            else:
                return current_r, "overwork"

        return predicted_r, "unknown"

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
        self.history = {
            'resources': [], 'degradation': [], 'utility': [],
            'reward': [], 'exploration': [], 'doubt': [],
            'prediction_error': [], 'caution_mode': [], 'effective_beta': []
        }
        # Disable live plotting to prevent GUI hang
        self.plot_enabled = False

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


# EmergencyPrioritySystem removed: unify emergency handling in select_action()


class AdaptiveEmergency:
    """Adaptive emergency thresholds with simple hysteresis."""
    def __init__(self):
        # EXPERT MODE THRESHOLDS
        self.emergency_threshold = 0.4
        self.work_allowed_threshold = 0.6
        self.consecutive_emergencies = 0

    def update_threshold(self, survived_last_emergency: bool):
        if not survived_last_emergency:
            self.consecutive_emergencies += 1
            if self.consecutive_emergencies > 3:
                self.emergency_threshold = min(0.4, self.emergency_threshold + 0.05)
                self.work_allowed_threshold = min(0.5, self.work_allowed_threshold + 0.05)
        else:
            self.consecutive_emergencies = max(0, self.consecutive_emergencies - 1)


class CatastropheMemory:
    """Record death / near-death scenarios to warn agent in future."""
    def __init__(self):
        self.death_scenarios = deque(maxlen=20)
        self.near_death_scenarios = deque(maxlen=20)

    def record_failure(self, r_history, d_history, actions):
        """Remember what led to death"""
        if not r_history:
            return
        if r_history[-1] <= 0 or d_history[-1] >= 1.0:
            scenario = {
                'final_r': r_history[-1],
                'final_d': d_history[-1],
                'last_10_actions': actions[-10:],
                'r_at_death': r_history[-5:],
                'd_at_death': d_history[-5:]
            }
            self.death_scenarios.append(scenario)

    def get_warning(self, current_r, current_d):
        for scenario in self.death_scenarios:
            try:
                if len(scenario['r_at_death']) == 0:
                    continue
                if (abs(current_r - scenario['r_at_death'][0]) < 0.1 and
                    abs(current_d - scenario['d_at_death'][0]) < 0.1):
                    return "WARNING: Similar to past death scenario!"
            except Exception:
                continue
        return None
        



# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_metacognitive_agent(episodes=50, obs_dim=10, action_dim=5, difficulty="hard"):
    """Training loop for metacognitive agent"""
    
    print("=" * 70)
    print(f"METACOGNITIVE SELF-REFERENTIAL AI TRAINING - {difficulty.upper()} MODE")
    print("=" * 70)
    print("\nKey features:")
    print("1. Imagination-based proactive decision making ✓")
    print("2. Metacognitive doubt monitoring ✓")
    print("3. Epistemic foraging in high-doubt states ✓")
    print("4. Nonlinear anxiety with doubt amplification ✓")
    print("5. Caution mode for model re-calibration ✓")
    print("")
    
    # Initialize
    env = HomeostasisEnv(obs_dim=obs_dim, difficulty=difficulty)
    agent = MetacognitiveSelfAgent(obs_dim=obs_dim, action_dim=action_dim)
    # Attach emergency responder and resource buffer
    emergency = EmergencyResponse()
    agent.resource_buffer = ResourceBuffer()
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
        r_history = []
        d_history = []
        actions_history = []
        
        # Reset agent state
        agent.self_state = SelfState(
            x_t=0.0, r_t=1.0, c_t=0.5, g_t=0.7, d_t=0.0, doubt_t=0.0
        )
        agent.in_caution_mode = False
        agent.caution_mode_duration = 0
        
        while True:
            # Emergency check before selecting action
            forced_action = emergency.check_emergency(agent.self_state)
            if forced_action is not None:
                action_idx = forced_action
                log_prob = torch.tensor(0.0)
            else:
                # Select action
                action_idx, log_prob = agent.select_action(obs, agent.self_state, explore=True)
            
            # Execute action
            next_obs, reward, load, done, status, shocked = env.step(action_idx)

            # record histories for catastrophe memory
            r_history.append(env.r)
            d_history.append(env.d)
            actions_history.append(action_idx)
            # record shock history for adaptive rules
            try:
                agent.shock_history.append(1 if shocked else 0)
            except Exception:
                pass

            # Track consecutive rests for minimum productivity enforcement
            try:
                if action_idx == 0:
                    agent.consecutive_rests += 1
                else:
                    agent.consecutive_rests = 0
            except Exception:
                pass
            
            # Predict next state
            predicted_state = agent.predict_next_state(obs, agent.self_state, action_idx)
            
            # Create actual state tensor
            actual_state_tensor = torch.tensor([
                load, env.r, agent.self_state.c_t, 
                agent.self_state.g_t, env.d, agent.self_state.doubt_t
            ], dtype=torch.float32)
            
            # Update metacognition (compare prediction with reality)
            doubt, prediction_error = agent.update_metacognition(
                predicted_state.to_tensor(), actual_state_tensor, agent_reward=reward
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
                # Record catastrophic scenarios
                try:
                    agent.catastrophe_memory.record_failure(r_history, d_history, actions_history)
                except Exception:
                    pass
                
                # Record caution percentage
                caution_pct = (caution_steps / steps * 100) if steps > 0 else 0
                caution_mode_percentages.append(caution_pct)
                break
        
        # Episode complete
        episode_rewards.append(episode_reward)
        episode_utilities.append(episode_utility)
        survival_steps.append(steps)
        final_resources.append(env.r)
        # Adapt learning rates based on recent survival history
        try:
            agent.adapt_learning_rates(survival_history)
        except Exception:
            pass
        
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
    # Save summary to JSON for automated inspection
    summary = {
        'average_reward': float(np.mean(episode_rewards)),
        'average_utility': float(np.mean(episode_utilities)),
        'average_survival_steps': float(avg_survival_steps),
        'overall_survival_rate': float(survival_rate),
        'average_time_in_caution_pct': float(avg_caution),
        'average_final_resources_survivors': float(avg_final_resources) if avg_final_resources is not None else None,
        'final_agent_state': {
            'resources': float(agent.self_state.r_t),
            'degradation': float(agent.self_state.d_t),
            'capability': float(agent.self_state.c_t),
            'confidence': float(agent.self_state.g_t),
            'doubt': float(agent.self_state.doubt_t),
            'internal_load': float(agent.self_state.x_t),
            'metacognitive_status': agent.get_metacognitive_status()
        }
    }
    # Write summary including seed identifier if provided
    seed_tag = os.environ.get('RUN_SEED', '42')
    summary_fname = f"train_summary_{seed_tag}.json"
    try:
        with open(summary_fname, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Training summary written to '{summary_fname}'")
    except Exception:
        pass
    
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
    print("\n✓ Training results saved to 'metacognitive_training_results.png'")
    # Avoid opening a GUI window (blocks in headless/interactive sessions)
    plt.close(fig)

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
    # Set random seeds (allow override via RUN_SEED env var)
    SEED = int(os.environ.get('RUN_SEED', '42'))
    print(f"Using RUN_SEED={SEED}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
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
    EPISODES = 200
    DIFFICULTY = "expert"  # Options: "normal", "hard", "expert"
    RUN_DEMO = True
    
    try:
        # Train metacognitive agent
        print(f"Training metacognitive agent for {EPISODES} episodes (Difficulty: {DIFFICULTY})...\n")
        trained_agent = train_metacognitive_agent(
            episodes=EPISODES,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            difficulty=DIFFICULTY
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