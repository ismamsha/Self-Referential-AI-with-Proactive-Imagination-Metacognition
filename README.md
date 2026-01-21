Overview
This project implements an AI agent capable of:

Self-Referential Modeling: Understanding its own internal state (resources, degradation, capabilities)
Proactive Imagination: Simulating future scenarios before taking actions
Metacognitive Uncertainty: Tracking and managing self-doubt and confidence levels
Homeostatic Regulation: Maintaining internal balance by managing resource consumption, wear-and-tear, and work intensity
The agent learns to optimize survival through adaptive decision-making in dynamic environments with shocks, trade-offs, and critical failure thresholds.

The Metacognitive Self-Referential Agent (MSRA) is a novel reinforcement learning architecture that integrates a predictive self-model with dynamic monitoring of endogenous uncertainty ("doubt"). Unlike standard RL agents that treat internal constraints as secondary, MSRA proactively imagines the physiological consequences of actions and modulates risk sensitivity based on its confidence in self-predictions.

Key Innovation
Endogenous Uncertainty Monitoring: The agent tracks prediction errors between imagined and actual internal states

Dynamic Risk Modulation: An "anxiety" parameter (β) adjusts based on doubt and resource scarcity

Caution Mode: Autonomous switching to conservative behavior when uncertainty exceeds thresholds

Perfect Homeostasis: Achieves 100% survival rates in stochastic resource-constrained environments

🎯 Features
Self-Modeling: Learns predictive models of internal state dynamics

Metacognitive Loop: Monitors and responds to prediction uncertainty

Risk-Sensitive Control: Dynamically adjusts behavior based on internal confidence

Homeostasis Maintenance: Optimally balances resource utilization and preservation

Benchmark Comparisons: Includes multiple baseline implementations for comparison

📊 Results
Agent Type	Survival Rate	Final Resources	Caution Mode
MSRA (Ours)	100%	0.98	4.4%
Standard MBRL	67%	0.62	N/A
Risk-Neutral PPO	73%	0.71	N/A
Fixed-β Controller	81%	0.79	0%
For detailed results, see our paper.
