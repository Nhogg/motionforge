# Experiment configuration

MotionForge command-line programs use Hydra 1.3 structured configs. Each entry
point defines a mutable `Config` dataclass, retains a plain `main(config)`
function for direct use in tests, and delegates command-line composition to
`motionforge.cli.run_hydra`.

The shared entry point registers the dataclass with Hydra's `ConfigStore` and
converts the resulting `DictConfig` back to a typed dataclass before calling the
experiment. This keeps configuration parsing separate from simulator,
controller, policy, logging, and evaluation logic.

## Overrides

Hydra uses `field=value` syntax. Field names match the Python dataclass and use
underscores:

```bash
uv run scripts/tests/test_tag_contact.py \
  seed=0 \
  separated_distance=2.0 \
  output=logs/p3/tag_contact.json
```

Use `null` for an optional value and lowercase `true` or `false` for booleans:

```bash
uv run scripts/evaluation/render_g1_checkpoint.py \
  checkpoint=logs/p2/training/example/checkpoints/000010000000 \
  stop_at=null \
  command_x=0.3 \
  command_y=0.0 \
  command_yaw=0.5
```

Run any entry point with `--help` to display its structured configuration.

## Output ownership

Experiments continue to expose explicit output paths and write their existing
machine-readable JSON, JSONL, checkpoint, and video artifacts. The shared Hydra
entry point disables Hydra's additional timestamped output directory, config
snapshot directory, and job log by default. This avoids two competing artifact
trees and preserves the repository's established experiment layout.

Those Hydra settings can still be overridden explicitly when its native run
directory behavior is useful.

## Type constraints

Hydra/OmegaConf cannot represent Python `Literal` annotations in structured
configs. CLI-facing finite-choice fields therefore use `str` annotations and
are checked by the entry point's existing configuration validation. Domain
dataclasses remain free to use stricter types and immutable instances.

Configuration dataclasses cannot be frozen because Hydra applies command-line
overrides by mutating its structured configuration container. This mutability is
limited to the CLI boundary; runtime state and model-layout dataclasses remain
unchanged.
