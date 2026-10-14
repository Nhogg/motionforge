"""CPU regression: render roles select distinct heads and correct agent policies."""

import json

import jax
import jax.numpy as jp
import numpy as np
from flax.core import freeze, unfreeze

from motionforge.envs.hierarchical_observations import role_encodings
from motionforge.evaluation.hierarchical_roles import audit_hierarchical_roles
from motionforge.models import HierarchicalPolicy
from motionforge.models.hierarchical import StrategyModule
from motionforge.rollout import HierarchicalRolloutRunner, HierarchicalRolloutSegment
from scripts.tests.test_hierarchical_policy import make_observations


def main():
    observations, locomotion = make_observations(2)
    policy = HierarchicalPolicy()
    parameters = policy.init(jax.random.PRNGKey(0), observations, locomotion)["params"]

    def marked_parameters(magnitude):
        tree = unfreeze(parameters)
        for name, sign in (("pursuer", 1), ("evader", -1)):
            command = tree["strategy"][f"{name}_head"]["command"]
            command["kernel"] = jp.zeros_like(command["kernel"])
            command["bias"] = jp.asarray([sign * magnitude, 0.0, 0.0])
        return freeze(tree)

    module = StrategyModule(
        policy.config.strategy_hidden_layer_sizes,
        policy.config.command_scale,
        policy.config.strategy_initial_log_std,
    )
    checks = {}
    for pursuer_index in (0, 1):
        roles = role_encodings(jp.asarray(pursuer_index))
        for learner_is_pursuer in (True, False):
            learner_index = pursuer_index if learner_is_pursuer else 1 - pursuer_index
            segment = HierarchicalRolloutSegment(
                learner_parameters=marked_parameters(0.7),
                opponent_parameters=marked_parameters(1.3),
                opponent_id="test",
                opponent_category="historical",
                learner_index=learner_index,
                learner_is_pursuer=learner_is_pursuer,
            )
            audit = audit_hierarchical_roles(
                pursuer_index=pursuer_index, segment=segment, roles=roles
            )
            stacked = HierarchicalRolloutRunner._agent_parameters(segment)
            output = jax.vmap(lambda p, o: module.apply({"params": p}, o))(
                stacked["strategy"], observations.replace(role=roles).as_array()
            )
            expected = [
                (1 if i == pursuer_index else -1) * (0.7 if i == learner_index else 1.3)
                for i in (0, 1)
            ]
            checks[f"pursuer_agent{pursuer_index}_learner_{segment.learner_role}"] = (
                bool(
                    np.allclose(np.asarray(output.location[:, 0]), expected)
                    and audit["agent_roles"].count("evader") == 1
                    and audit["agent_roles"].count("pursuer") == 1
                )
            )
            try:
                audit_hierarchical_roles(
                    pursuer_index=pursuer_index,
                    segment=segment,
                    roles=jp.asarray([[0.0, 1.0], [0.0, 1.0]]),
                )
            except ValueError:
                checks["duplicate_evader_roles_rejected"] = True
            else:
                checks["duplicate_evader_roles_rejected"] = False
    print(json.dumps({"checks": checks, "passed": all(checks.values())}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
