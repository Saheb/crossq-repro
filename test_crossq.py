"""Sanity checks for crossq.py.  Run:  uv run test_crossq.py"""

from dataclasses import replace
from types import SimpleNamespace

import torch
from sb3_contrib.common.torch_layers import BatchRenorm1d  # reference implementation

from crossq import Agent, BatchRenorm, Config, PRESETS


def test_batchrenorm_matches_sb3_contrib():
    """Our F.batch_norm-based BatchRenorm must agree with sb3-contrib's explicit one
    in the warm-up phase, the renorm phase, and in eval mode (up to the biased/unbiased
    variance convention, hence the loose tolerance)."""
    torch.manual_seed(0)
    ours, ref = BatchRenorm(8, warmup_steps=3), BatchRenorm1d(8, warmup_steps=3)
    for step in range(12):  # first 4 calls are plain BN, then renorm with clipped r, d
        x = torch.randn(256, 8) * (1 + step) + 3 * step  # drifting input distribution
        torch.testing.assert_close(ours(x), ref(x), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(ours.running_mean, ref.ra_mean, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(ours.running_var, ref.ra_var, rtol=1e-2, atol=1e-2)
    ours.eval(), ref.eval()
    x = torch.randn(256, 8)
    torch.testing.assert_close(ours(x), ref(x), rtol=1e-2, atol=1e-2)


def test_batchrenorm_gradient_flows_through_batch_statistics():
    """BatchNorm-style gradients: shifting the whole batch must not change the output."""
    layer = BatchRenorm(4, warmup_steps=0)
    x = torch.randn(32, 4, requires_grad=True)
    layer(x).sum().backward()
    assert torch.allclose(x.grad.sum(0), torch.zeros(4), atol=1e-5)


def test_update_runs_for_both_presets():
    for algo in ("crossq", "sac"):
        cfg = replace(Config(algo=algo, **PRESETS[algo]), critic_hidden=(32, 32), actor_hidden=(32, 32), bn_warmup_steps=2)
        agent = Agent(obs_dim=5, act_dim=2, cfg=cfg, device=torch.device("cpu"))
        batch = SimpleNamespace(observations=torch.randn(16, 5), actions=torch.rand(16, 2) * 2 - 1, rewards=torch.randn(16, 1),
                                next_observations=torch.randn(16, 5), dones=torch.zeros(16, 1))
        for _ in range(6):  # enough for the delayed actor update and the end of BN warm-up
            info = agent.update(batch)
        assert all(torch.isfinite(v) for v in info.values()), info
        assert "actor_loss" in info
        assert (agent.critic_target is None) == (algo == "crossq")


def test_critic_running_stats_untouched_by_actor_update():
    """The actor loss evaluates the critic in eval mode, so its BN statistics must not move."""
    cfg = Config(critic_hidden=(32, 32), actor_hidden=(32, 32), policy_delay=1)
    agent = Agent(obs_dim=5, act_dim=2, cfg=cfg, device=torch.device("cpu"))
    bn = agent.critic.nets[0][0]
    obs = torch.randn(16, 5)
    agent.critic.eval()
    before = bn.running_mean.clone()
    agent.critic(obs, agent.actor.sample(obs)[0]).min(0).values.mean().backward()
    assert torch.equal(bn.running_mean, before)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
