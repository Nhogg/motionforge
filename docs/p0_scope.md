
  # P0 — Research scope

  ## Primary hypothesis

  Under an equal data-generation budget, locomotion training data produced by
  competitive pursuit/evasion contains more useful hard states than data produced
  by random commands, aggressively sampled commands, or a terrain curriculum.

  A low-level locomotion controller fine-tuned with TAG-generated data will
  therefore perform better on held-out, opponent-free robustness tasks.

  ## Minimum first-paper scope

  Use one humanoid morphology and a hierarchical controller in MuJoCo/MJX.

  1. Establish stable single-humanoid locomotion.
  2. Build a deterministic flat-ground two-humanoid tag environment.
  3. Validate it using scripted pursuit and evasion policies.
  4. Train flat-ground pursuit/evasion policies and introduce self-play.
  5. Generate equal-sized datasets from:
     - random commands
     - aggressive commands
     - terrain curriculum
     - TAG
  6. Fine-tune identical low-level controllers using equal data budgets.
  7. Evaluate on opponent-free locomotion robustness tasks.
  8. Compare state coverage, hard-state frequency, and downstream performance.

  ## Primary independent variable

  The source of additional locomotion training data:
  random, aggressive, terrain curriculum, or TAG.

  ## Primary outcome

  Aggregate performance across held-out, opponent-free locomotion robustness
  tasks under identical evaluation seeds and budgets.

  ## Secondary outcomes

  - fall rate
  - recovery rate
  - command-tracking error
  - state-space coverage
  - near-fall frequency
  - contact and slip statistics

  ## Explicit exclusions from the minimum scope

  - multiple humanoid morphologies
  - vision-based observations
  - learned terrain generation
  - adversarial environment generation
  - sim-to-real transfer
  - dexterous manipulation
  - multi-agent games beyond two-player pursuit/evasion
  - large-scale terrain experiments before flat-ground self-play is stable

  ## Reproducibility requirements

  Every experiment must record:

  - configuration
  - random seed
  - dependency versions
  - simulator/backend
  - source revision
  - machine-readable metrics and rollout metadata
