# Project goal:
Invesitgate whether competitive multi-agent interaction can serve as an automatic data generation
and hard-state mining mechanism for robust humanoid locomotion.

Two humanoids will perform pursuit / evasion over increasingly difficult terrain using a pretrained
low-level locomotion controller and a learned high-level policy. Competitive rollouts will be used
to collect difficult trajectories and near-failure states. These data will then be used to train or
fine-tune a general locomotion controller and compared against equal-budget random-command,
aggressive-command, and terrain-curriculum baselines on held-out non-game locomotion tasks. 

The novelty is not the human tag itself. The research question is whether adaptive embodied
competition generates locomotion experience that improves downstream robustness beyond concentional
curricula.

# Scope
Current scope: MuJoCo/MJX, one humanoid morphololgy, hierarchical control, flat ground MVP first,
scripted agents before RL, then self-play, then terrain, then dataset analysis.


