Self-Referential AI with Proactive Imagination & Metacognition
A sophisticated implementation of a self-aware AI agent that uses self-referential reasoning, proactive imagination, and metacognitive processes to make optimal decisions in resource-constrained environments with homeostatic pressures.

Overview
This project implements an AI agent capable of:

Self-Referential Modeling: Understanding its own internal state (resources, degradation, capabilities)
Proactive Imagination: Simulating future scenarios before taking actions
Metacognitive Uncertainty: Tracking and managing self-doubt and confidence levels
Homeostatic Regulation: Maintaining internal balance by managing resource consumption, wear-and-tear, and work intensity
The agent learns to optimize survival through adaptive decision-making in dynamic environments with shocks, trade-offs, and critical failure thresholds.

Core Components
1. Self-State Representation (SelfState)
Tracks the agent's internal metrics:

x_t: Internal load (accumulated strain)
r_t: Resource availability (0-1 scale)
c_t: Capability estimate (self-assessed competence)
g_t: Confidence/prediction accuracy
d_t: Degradation level (wear and tear, 0-1)
doubt_t: Metacognitive uncertainty about self-model
2. Neural Network Architecture
SelfModel: Predicts next internal state given current state and actions

Input: Observation + Self-state + Action
Hidden layers: 128 → 64 neurons
Output: 6-dimensional next state (with sigmoid activation)
ActorCritic Network: Policy and value estimation

Shared encoding: 128 → 64 neurons
Actor head: Outputs probability distribution over 5 actions
Critic head: Outputs state value estimate
3. Homeostasis Environment (HomeostasisEnv)
A reinforcement learning environment featuring:

Actions:

REST: Recover resources
REPAIR: Reduce degradation at resource cost
WORK (3 intensities): Generate external reward with resource/degradation costs
Dynamics:

Random external shocks (5% probability per step)
Non-linear efficiency penalties
Death conditions: Resource depletion or complete degradation
Survival bonuses for lasting 100 steps
Reward Structure:

Task completion rewards (1.5x intensity)
Homeostatic maintenance rewards
Penalties for dangerous operating conditions
Severe penalties for terminal failure
4. Proactive Self-Referential Agent (ProactiveSelfReferentialAgent)
Key Features:

Nonlinear Anxiety (Effective β): Stress level increases near critical thresholds
Imagination-Based Planning: Simulates future trajectories to evaluate actions
Doubt-Based Exploration: Higher uncertainty → higher exploration rate
Composite Utility: Balances external rewards with internal homeostatic needs
Self-Model Adaptation: Learns to predict own behavior more accurately
Advanced Version (AdvancedProactiveAgent):

Multi-step imagination horizons
Trajectory importance weighting
Critic-based value estimation for imagination
Uncertainty-driven adaptive planning
Key Algorithms
Imagination-Based Action Selection
For each possible action:
  Simulate k-step future trajectories
  Compute average returns
  Weight by trajectory reliability (based on doubt)
Choose action with highest weighted expected value
Doubly Metacognitive Learning
Self-Model Learning: Learns to predict own state transitions
Policy Learning: Actor-Critic updates for action selection
Doubt Estimation: Uncertainty increases with prediction errors
Nonlinear Anxiety Dynamics
Effective anxiety (β) increases exponentially near:

Low resources (< 30%)
High degradation (> 60%)
This creates adaptive risk aversion when critical.

Training
The agent trains using:

Actor-Critic policy gradient with baseline
Self-model supervised learning with MSE loss
Experience replay from memory buffer
Exploration decay from 30% → 5% over episodes
Metrics tracked:

Episode rewards
Resource/degradation trajectories
Prediction errors (doubt level)
Action distribution
Visualization & Analysis
The implementation includes:

Real-time reward plotting during training
Self-state trajectory visualization
Prediction error analysis
Action frequency histograms
Requirements
numpy: Numerical computations
torch: Neural network training
matplotlib: Visualization
Usage
# Create environment and agent
env = HomeostasisEnv(obs_dim=10)
agent = AdvancedProactiveAgent(obs_dim=10, action_dim=5, beta=0.5)

# Train the agent
num_episodes = 1000
for episode in range(num_episodes):
    obs = env.reset()
    episode_reward = 0
    
    for step in range(env.max_steps):
        # Agent uses imagination-based decision making
        action = agent.choose_action(obs)
        
        next_obs, reward, load, done, status, shocked = env.step(action)
        
        # Update agent
        agent.update(obs, action, reward, next_obs, done, load)
        
        episode_reward += reward
        obs = next_obs
        
        if done:
            break
    
    agent.end_episode()
Philosophy
This implementation explores the intersection of:

Self-reference: Agents that model themselves
Imagination: Prospective simulation for decision-making
Metacognition: Awareness and uncertainty about one's own processes
Homeostasis: Internal regulation and adaptive stress response
The combination creates an agent that can handle resource constraints, adapt to uncertainty, and make decisions based on proactive reasoning about future states.

License
This project is provided as-is for research and educational purposes. Self-Referential AI with Proactive Imagination & Metacognition Complete self-contained implementation
