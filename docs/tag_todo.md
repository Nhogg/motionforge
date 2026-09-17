# P0 - determine scope / question

- [x] Determine primary hypothesis 
- [x] Define minimum first-paper / poster scope

# P1 - Establish MuJoCo Stack

- [x] Install and run standard MuJoCo
- [x] Run the same model under MJX (MuJoCo accelerator)
- [x] Test MJX-Warp on Mayo
- [x] Reproduce existing humanoid locomotion example
- [x] Run DeepMind's `dm_control` soccer environment and understand how multiple walkers are represented and stepped
- [x] Confirm that two humanoids can physically interact without simulation instability
- [x] Add a simple heightfield / uneven terrain and verify stable contacts

# P2 - Pick humanoid and locomotion prior

- [x] Choose humanoid morphology (unitree G1 / basic humanoid)
- [x] Find existing model / controller that can already do basic movements
- [x] Reproduce forward / backward walking, turning, stopping, and recovery
- [x] Define low-level controller interface
- [x] Initially expose high-level actions such as desired forward velocity, lateral velocity, and yaw rate
    (`high-level command -> locomotion controller -> joint actions -> MuJoCo`)

# P3 - Build MVP environment

- [x] Put two identical humanoids into one environment
- [x] Add pursuer and evader roles
- [x] Implement relative position and velocity observations
- [x] Implement tag / contact detection
- [x] Implement fall detection
- [x] Implement OOB detection
- [x] Implement deterministic reset logic
- [x] Add episode timeout
- [x] Add flat terrain FIRST
- [x] Integrate and smoke-test the complete P3 MJX-Warp environment

# P4 - Scripted tag

- [x] Write trivial scripted pursuer that moves toward the opponent
- [x] Write trivial scripted evader that moves away
- [x] Validate that low-level controller can survive rapid direction changes
- [x] Record complete rollouts
- [x] Fix observation-frame, contact, reset, and controller problems before training

# P5 - Logging / dataset infra

For each timestep record:

- [x] root pose and orientation
- [x] root linear / angular velocity
- [x] joint positions and velocities
- [x] controller commands
- [ ] foot contacts / contact forces
- [ ] opponent-relative state
- [ ] terrain info
- [ ] reward / game outcome
- [ ] tracking error
- [ ] fall / near-fall indicator

Also derive:

- [ ] acceleration
- [ ] yaw acceleration
- [ ] command derivatives
- [ ] roll / pitch excursions
- [ ] slip / contact events
- [ ] recovery events

# P6 - Non-TAG baselines

Generate equal-sized datasets from:

- [ ] ordinary random velocity commands
- [ ] deliberately aggressive command sampling
- [ ] terrain curriculum without an opponent

The aggressive baseline MUST contain hard turns, reversals, braking, lateral motion, and high yaw rates

# P7 - Train a TAG agent

- [ ] Freeze the evader as a scripted policy
- [ ] Train the pursuer
- [ ] Verify consistent pursuit success
- [ ] Freeze the pursuer and train an evader
- [ ] Compare their state distributions against scripted / random-command locomotion

# P8 - Introduce self-play

- [ ] Train pursuer and evader against one another
- [ ] Save historical policy snapshots
- [ ] Sample current and historical opponents
- [ ] Record win rates against the opponent population
- [ ] Look for cycling / catastrophic forgetting
- [ ] Add increasingly difficult terrain only after flat-ground self-play is stable

# P9 - Uneven / adversarial terrain

Start simple:

- [ ] slopes
- [ ] low-frequency heightfields
- [ ] bumps
- [ ] gaps / steps

Then, investigate whether agents exploit terrain strategically:

- [ ] evader chooses terrain that is difficult for pursuer
- [ ] pursuer takes risky shortcuts
- [ ] agents force rapid terrain transitions
- [ ] opponent causes near-failure states that terrain sampling alone does not

# P10 - First analysis

Compare:
`D_random`
`D_aggressive`
`D_terrain`
`D_TAG`
Measure distribution differences in:

- [ ] velocity
- [ ] acceleration
- [ ] angular velocity
- [ ] angular acceleration
- [ ] joint acceleration
- [ ] joint configurations
- [ ] contact patterns
- [ ] near-fall frequency
- [ ] recovery frequency
- [ ] state-space coverage

**Does TAG actually visit useful states that the engineered baselines fail to visit?**

# P11 - Hard state extraction

Define a hard state score using signals such as:

- [ ] large command-tracking error
- [ ] unusual roll / pitch
- [ ] unstable contact configuration
- [ ] slip
- [ ] large corrective action
- [ ] near-fall
- [ ] actual failure

Store the highest-value states or trajectory segments in a hard-state buffer

# P12 - key experiment

Train / fine-tune identical low-level controllers using equal data budgets:

- [ ] baseline locomotion data
- [ ] baseline + aggressive-command data
- [ ] baseline + terrain curriculum data
- [ ] baseline + TAG data
- [ ] optionally baseline + TAG hard-state replay

# P13 - Evaluate outside of TAG

Evaluate on tasks that contain **no opponent**:

- [ ] rapid 180 deg reversal
- [ ] abrupt velocity changes
- [ ] slalom
- [ ] rough terrain
- [ ] unseen terrain
- [ ] push recovery
- [ ] near-fall recovery
- [ ] terrain transitions
