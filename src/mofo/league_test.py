"""CPU self-test for mofo.league: .venv/bin/python -m mofo.league_test [OUT_DIR]

Writes OUT_DIR/scripted_prob.png (default logs/mofo/league_debug): the scripted member's
probability over a simulated run where the learner learns to beat it.
"""

import os
import sys
import tempfile
from dataclasses import replace

import torch

from mofo.league import SCRIPTED_ID, League, LeagueCfg, Opponent

CFG = LeagueCfg()
zero = lambda obs: torch.zeros(len(obs), 2)


def league(n=0, wins=(), **kw) -> League:
  lg = League(replace(CFG, **kw), seed=0)
  lg.add(Opponent(SCRIPTED_ID, zero, "scripted"))
  for i in range(n):
    lg.snapshot(torch.nn.Linear(3, 2), i + 1)
  for o, w in zip(lg.members, wins):
    lg.win[o.opponent_id] = w
  return lg


def close(a, b, tol=1e-9):
  return all(abs(x - y) < tol for x, y in zip(a, b, strict=True))


def test_pfsp():
  p = list(league(2, (0.9, 0.5, 0.1), min_weight=0.0).probs(0).values())
  w = [(1 - x) ** 2 for x in (0.9, 0.5, 0.1)]
  assert close(p, [x / sum(w) for x in w]) and p[0] < p[1] < p[2], p
  print("1 pfsp", [round(x, 4) for x in p])


def test_floor():
  for lg in (league(2, (0.9999, 0.5, 0.1)), league(3, (1.0, 1.0, 1.0, 1.0)), league(), league(40, (0.999,) + (0.0,) * 40)):
    p = list(lg.probs(0).values())
    assert abs(sum(p) - 1) < 1e-9 and min(p) >= min(CFG.min_weight, 1 / len(p)) - 1e-12, p
  assert close(league(3, (1.0,) * 4).probs(0).values(), [0.25] * 4) and league().probs(0) == {SCRIPTED_ID: 1.0}
  p = list(league(2, (0.9999, 0.5, 0.1)).probs(0).values())
  print("2 floor", [round(x, 4) for x in p], "(32 members, scripted beaten: min", round(min(league(31, (1.0,) + (0.0,) * 31).probs(0).values()), 4), ")")


def test_uniform_recent():
  assert close(league(3, sampling="uniform").probs(0).values(), [0.25] * 4)
  p = list(league(3, sampling="recent", min_weight=0.0).probs(0).values())
  w = [0.9**3, 0.9**2, 0.9, 1.0]
  assert close(p, [x / sum(w) for x in w]), p
  print("3 uniform 0.25 x4; recent", [round(x, 4) for x in p])


def test_eviction():
  lg = league(5, max_size=4)
  ids = [o.opponent_id for o in lg.members]
  assert ids == [SCRIPTED_ID, "learner/3", "learner/4", "learner/5"] and set(lg.win) == set(ids), ids
  print("4 eviction", ids)


def test_schedule():
  lg = league(scripted_weight_start=1.0, scripted_weight_end=0.2, scripted_decay_iters=1000)
  v = [lg.scripted_weight(i) for i in (-5, 0, 500, 1000, 5000)]
  assert close(v, [1.0, 1.0, 0.6, 0.2, 0.2]), v
  p = league(1, (0.5, 0.5), min_weight=0.0, scripted_weight_start=1.0, scripted_weight_end=0.2, scripted_decay_iters=1000).probs(1000)
  assert close(p.values(), [0.2 / 1.2, 1 / 1.2]), p
  assert league().scripted_weight(10**6) == 1.0  # Default 1 -> 1: no effect.
  print("5 schedule", v)


def test_simulated_run(out_dir):
  """Learner win rate vs scripted climbs 0.2 -> 0.95 over 3000 iterations; vs each
  snapshot (every 250 iterations) it sits near 0.55."""
  lg, g = league(snapshot_every=250), torch.Generator().manual_seed(1)
  its, ps, size = range(3000), [], []
  for it in its:
    if it and it % lg.cfg.snapshot_every == 0:
      lg.snapshot(torch.nn.Linear(3, 2), it)
    for o in lg.members:
      p = 0.2 + 0.75 * it / 2999 if o.category == "scripted" else 0.55
      lg.record(o.opponent_id, torch.where(torch.rand(4, generator=g) < p, 1, -1))
    ps.append(lg.probs(it)[SCRIPTED_ID])
    size.append(len(lg.members))
  early, late = sum(ps[250:500]) / 250, sum(ps[-250:]) / 250
  assert late < early / 3 and late < 1 / size[-1] and min(ps) >= CFG.min_weight - 1e-12, (early, late)
  print(f"6 simulated: p_scripted mean {early:.3f} (its 250-500, pool 2) -> {late:.3f} (last 250, pool {size[-1]}); "
        f"win EMA vs scripted {lg.win[SCRIPTED_ID]:.2f}")
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  uniform = [1 / s for s in size]
  fig, ax = plt.subplots(figsize=(8, 4.5))
  ax.plot(its, ps, color="#2a78d6", linewidth=1.5, label="scripted/stage1 (PFSP)")
  ax.plot(its, uniform, color="#888", linewidth=1, linestyle="--", label="uniform share, 1 / pool size")
  ax.axhline(CFG.min_weight, color="#eb6834", linewidth=1, linestyle=":", label=f"floor min_weight = {CFG.min_weight}")
  ax.set(xlabel="iteration", ylabel="sampling probability", ylim=(0, 1.02),
         title="Scripted opponent probability as learner win rate rises 0.2 -> 0.95")
  ax.spines[["top", "right"]].set_visible(False)
  ax.legend(frameon=False)
  fig.tight_layout()
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, "scripted_prob.png")
  fig.savefig(path, dpi=120)
  print(f"  saved {path}")


def test_partition():
  for n, frac, k, sp, sizes in ((4096, 0.5, 8, 2048, [256] * 8), (4099, 0.5, 8, 2050, [256] * 1 + [257] + [256] * 6),
                                (10, 0.0, 3, 0, [4, 3, 3]), (8, 1.0, 4, 8, []), (5, 0.0, 8, 0, [1] * 5)):
    part = league(selfplay_frac=frac, groups=k).assign(n)
    got = [s.stop - s.start for s in part.groups]
    assert part.selfplay == slice(0, sp) and sorted(got, reverse=True) == sorted(sizes, reverse=True), (n, part)
    assert (part.groups[0].start if part.groups else n) == sp and (part.groups[-1].stop if part.groups else n) == n
    assert all(a.stop == b.start for a, b in zip(part.groups, part.groups[1:]))
  lg = league(3, groups=8, selfplay_frac=0.5)
  lg.assign(4096)
  assert len(lg.resample_groups(0)) == 8
  print("7 partition 4096 @ 0.5 x 8:", league(selfplay_frac=0.5, groups=8).assign(4096))


def test_determinism():
  def draws(seed):
    lg = League(replace(CFG, sampling="uniform"), seed=seed)
    for o in league(5).members:
      lg.add(o)
    return [lg.sample(i).opponent_id for i in range(50)]

  assert draws(0) == draws(0) and draws(0) != draws(1) and len(set(draws(0))) > 3
  print("8 determinism: same seed -> same 50 draws; seed 1 differs")


def test_snapshot_frozen():
  net, x = torch.nn.Linear(3, 2), torch.randn(5, 3)
  lg = league()
  snap = lg.snapshot(net, 7)
  before = snap.policy(x).clone()
  with torch.no_grad():
    net.weight.add_(1.0)
  assert torch.equal(snap.policy(x), before) and not torch.equal(net(x), before)
  assert not any(p.requires_grad for p in snap.policy.parameters())
  print("9 snapshot frozen")


def test_state_dict():
  def load(path):
    net = torch.nn.Linear(3, 2)
    net.load_state_dict(torch.load(path)["actor_state_dict"])
    return net.eval().requires_grad_(False)

  x = torch.randn(4, 3)
  with tempfile.TemporaryDirectory() as d:
    cfg = replace(CFG, max_size=4)
    a = League(cfg, seed=3, load=load, snapshot_dir=d)
    a.add(Opponent(SCRIPTED_ID, zero, "scripted"))
    for i in range(5):
      a.snapshot(torch.nn.Linear(3, 2), i)
    a.snapshot(torch.nn.Linear(3, 2).state_dict(), 9)  # The state-dict path reloads via load.
    a.record("learner/3", [1, 1, -1, 0])
    [a.sample(0) for _ in range(7)]
    b = League(cfg, seed=99, load=load, snapshot_dir=d)
    b.add(Opponent(SCRIPTED_ID, zero, "scripted"))  # Scripted policy is re-supplied, not saved.
    b.load_state_dict(a.state_dict())
    assert b.state_dict()["members"] == a.state_dict()["members"] and b.win == a.win and b.games == a.games
    assert all(torch.equal(p.policy(x), q.policy(x)) for p, q in zip(a.members[1:], b.members[1:]))
    assert [a.sample(i).opponent_id for i in range(30)] == [b.sample(i).opponent_id for i in range(30)]
  print("10 state_dict round-trip:", [m[0] for m in b.state_dict()["members"]])


def test_balanced_anchors():
  p = list(league(2, (0.0, 0.5, 0.3), min_weight=0.0, sampling="balanced").probs(0).values())
  w = [0.0, 0.25, 0.21]
  assert close(p, [x / sum(w) for x in w]), p
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "anchor.pt")
    torch.save({"actor_state_dict": torch.nn.Linear(3, 2).state_dict()}, path)
    load = lambda _: zero
    lg = League(replace(CFG, max_size=3, init_pool=[path], anchor_init_pool=True), load=load)
    lg.add(Opponent(SCRIPTED_ID, zero, "scripted"))
    for i in range(4):
      lg.snapshot(torch.nn.Linear(3, 2), i)
    ids = [o.opponent_id for o in lg.members]
    assert ids == [f"historical/{path}", SCRIPTED_ID, "learner/3"], ids
  for bad in ({"league_role": "both"}, {"sampling": "even"}):
    try:
      League(replace(CFG, **bad))
    except ValueError:
      continue
    raise AssertionError(bad)
  mixed = league(2, (0.0, 0.5, 0.3), min_weight=0.0, sampling="balanced", scripted_share=0.25).probs(0)
  assert close(mixed.values(), [0.25, 0.75 * 0.25 / 0.46, 0.75 * 0.21 / 0.46]), mixed
  print("11 balanced", [round(x, 4) for x in p], "; anchors kept:", ids)


if __name__ == "__main__":
  test_pfsp()
  test_floor()
  test_uniform_recent()
  test_eviction()
  test_schedule()
  test_simulated_run(sys.argv[1] if len(sys.argv) > 1 else "logs/mofo/league_debug")
  test_partition()
  test_determinism()
  test_snapshot_frozen()
  test_state_dict()
  test_balanced_anchors()
  print("league OK")
