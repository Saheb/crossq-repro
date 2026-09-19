"""
CrossQ in a single file (PyTorch).

Paper: Bhatt*, Palenicek*, Belousov, Argus, Amiranashvili, Brox, Peters.
       "CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater
       Sample Efficiency and Simplicity", ICLR 2024.  https://arxiv.org/abs/1902.05605

CrossQ is SAC with four changes (everything else is untouched):

  1. no target networks           -> bootstrap from the *live* critic
  2. BatchRenorm before every Linear layer of critic and actor
  3. one *joint* critic forward pass on the concatenated batch [(s,a); (s',a')]
     so the normalisation statistics come from the 50/50 mixture of both
     distributions (this is what makes 1+2 stable)
  4. wider critic (2048), Adam beta1 = 0.5, policy delay 3

The update-to-data ratio stays 1, i.e. one gradient step per environment step:
20x fewer gradient steps and 3-4x less wall-clock than REDQ / DroQ (UTD 20) at
the same sample efficiency.

Libraries do the boring parts: gymnasium (MuJoCo tasks + wrappers),
stable-baselines3 (replay buffer), torch (networks), tensorboard + csv (logs).
Everything CrossQ-specific is marked "CrossQ" in the comments.

Usage
    uv run crossq.py --algo crossq --env Hopper-v5 --seed 0 --steps 300_000
    uv run crossq.py --algo sac    --env Hopper-v5 --seed 0 --steps 300_000   # baseline
    uv run crossq.py --algo crossq --no-batch-norm --target-networks ...      # ablations
    uv run crossq.py --algo crossq --critic-hidden 512 512 --name crossq512   # narrower critic
Logs go to runs/<env>/<name or algo>/seed<seed>/{log.csv, tensorboard, config.json}.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --------------------------------------------------------------------------- config


@dataclass
class Config:
    algo: str = "crossq"  # "crossq" or "sac"; sets the preset below, flags override
    env: str = "Hopper-v5"
    seed: int = 0
    steps: int = 1_000_000  # environment steps
    learning_starts: int = 5_000  # random actions before training starts
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 1e-3
    adam_beta1: float = 0.5  # CrossQ: 0.5 (SAC: 0.9)
    critic_hidden: tuple[int, ...] = (2048, 2048)  # CrossQ: 2048 (SAC: 256)
    actor_hidden: tuple[int, ...] = (256, 256)
    n_critics: int = 2
    batch_norm: bool = True  # CrossQ: BatchRenorm in actor & critic
    bn_momentum: float = 0.01  # = paper's 0.99 running-average decay
    bn_warmup_steps: int = 100_000  # plain BatchNorm behaviour before this
    target_networks: bool = False  # CrossQ: none (SAC: Polyak-averaged critic copy)
    tau: float = 0.005  # Polyak rate, only used with target networks
    policy_delay: int = 3  # actor + temperature updated every k critic updates
    utd: int = 1  # gradient steps per environment step
    eval_every: int = 10_000
    eval_episodes: int = 5
    device: str = "auto"
    threads: int = 0  # torch CPU threads (0 = torch default); useful when running several seeds in parallel
    logdir: str = "runs"
    name: str = ""  # run name for the log directory (defaults to algo)


# Plain SAC (Haarnoja et al. 2018) as the baseline; everything not listed is shared.
PRESETS = {
    "crossq": {},
    "sac": dict(lr=3e-4, adam_beta1=0.9, critic_hidden=(256, 256), batch_norm=False,
                target_networks=True, policy_delay=1),
}


# --------------------------------------------------------------------------- layers


class BatchRenorm(nn.Module):
    """Batch Renormalization (Ioffe, 2017) as used by CrossQ.

    For the first `warmup_steps` training calls this is ordinary BatchNorm.
    Afterwards the batch statistics are still used, but corrected towards the
    running statistics through the clipped, gradient-free factors r and d, which
    makes the layer robust to the occasional outlier minibatch in a long RL run.
    In eval mode the running statistics are used (and not updated).
    """

    def __init__(self, num_features: int, momentum: float = 0.01, eps: float = 1e-3,
                 warmup_steps: int = 100_000, r_max: float = 3.0, d_max: float = 5.0):
        super().__init__()
        self.momentum, self.eps, self.warmup_steps = momentum, eps, warmup_steps
        self.r_max, self.d_max = r_max, d_max
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.num_batches_tracked = 0  # python int: avoids a device sync per call

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.batch_norm is the fused kernel: in training mode it normalises with batch
        # statistics (gradients flow through them) and updates the running statistics.
        if not self.training:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, eps=self.eps)
        warming_up = self.num_batches_tracked <= self.warmup_steps
        self.num_batches_tracked += 1
        if warming_up:  # plain BatchNorm
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, True, self.momentum, self.eps)
        # Renorm: x_hat = (x - mu_B) / sigma_B * r + d, with r and d clipped and treated as constants.
        with torch.no_grad():
            batch_var, batch_mean = torch.var_mean(x, dim=0, correction=0)
            running_std = (self.running_var + self.eps).sqrt()
            r = ((batch_var + self.eps).sqrt() / running_std).clamp(1 / self.r_max, self.r_max)
            d = ((batch_mean - self.running_mean) / running_std).clamp(-self.d_max, self.d_max)
        x_hat = F.batch_norm(x, self.running_mean, self.running_var, None, None, True, self.momentum, self.eps)
        return x_hat * (r * self.weight) + (d * self.weight + self.bias)  # == (x_hat * r + d) * weight + bias


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, cfg: Config) -> nn.Sequential:
    """ReLU MLP. With cfg.batch_norm a BatchRenorm precedes *every* Linear (incl. input & output)."""
    bn = lambda d: [BatchRenorm(d, cfg.bn_momentum, warmup_steps=cfg.bn_warmup_steps)] if cfg.batch_norm else []
    layers: list[nn.Module] = []
    for d_in, d_out in zip((in_dim, *hidden), hidden):
        layers += [*bn(d_in), nn.Linear(d_in, d_out), nn.ReLU()]
    layers += [*bn(hidden[-1]), nn.Linear(hidden[-1], out_dim)]
    return nn.Sequential(*layers)


LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class Actor(nn.Module):
    """Tanh-squashed diagonal Gaussian policy (standard SAC)."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: Config):
        super().__init__()
        self.net = mlp(obs_dim, cfg.actor_hidden, 2 * act_dim, cfg)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised action in [-1, 1] and its log-probability (with tanh correction)."""
        mean, log_std = self(obs)
        noise = torch.randn_like(mean)
        u = mean + log_std.exp() * noise
        action = torch.tanh(u)
        log_prob = (-0.5 * noise.pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)  # Gaussian density of u
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)  # change of variables for tanh
        return action, log_prob


class Critic(nn.Module):
    """Ensemble of n_critics Q-networks; returns Q-values of shape (n_critics, batch)."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: Config):
        super().__init__()
        self.nets = nn.ModuleList(mlp(obs_dim + act_dim, cfg.critic_hidden, 1, cfg) for _ in range(cfg.n_critics))

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return torch.stack([net(x).squeeze(-1) for net in self.nets])


# --------------------------------------------------------------------------- agent


class Agent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: Config, device: torch.device):
        self.cfg, self.device = cfg, device
        self.actor = Actor(obs_dim, act_dim, cfg).to(device)
        self.critic = Critic(obs_dim, act_dim, cfg).to(device)
        # SAC keeps a slowly-moving copy of the critic for bootstrap targets.
        # CrossQ: no target network at all -- the joint forward pass below replaces it.
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False) if cfg.target_networks else None
        # SAC automatic temperature tuning, target entropy = -|A|
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.target_entropy = -float(act_dim)

        adam = lambda params, betas: torch.optim.Adam(params, lr=cfg.lr, betas=betas, fused=True)
        self.actor_opt = adam(self.actor.parameters(), (cfg.adam_beta1, 0.999))  # CrossQ: beta1 = 0.5
        self.critic_opt = adam(self.critic.parameters(), (cfg.adam_beta1, 0.999))
        self.alpha_opt = adam([self.log_alpha], (0.9, 0.999))
        self.n_updates = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self.actor.eval()  # BatchRenorm uses running statistics when acting
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        action = torch.tanh(self.actor(obs_t)[0]) if deterministic else self.actor.sample(obs_t)[0]
        self.actor.train()
        return action.squeeze(0).cpu().numpy()

    def update(self, batch) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        s, a, r, s2, done = (batch.observations, batch.actions, batch.rewards.squeeze(-1),
                             batch.next_observations, batch.dones.squeeze(-1))
        alpha = self.log_alpha.exp().detach()

        # ---- critic --------------------------------------------------------------
        with torch.no_grad():  # a' ~ pi(.|s') with the actor's BN in eval mode (running stats)
            self.actor.eval()
            a2, logp2 = self.actor.sample(s2)
            self.actor.train()

        if self.critic_target is None:
            # CrossQ: a single forward pass on the concatenated batch. BatchRenorm then
            # normalises with statistics of the mixture of (s,a) and (s',a'), so the
            # live critic can be used for the target without diverging.
            q_joint = self.critic(torch.cat([s, s2]), torch.cat([a, a2]))  # (n_critics, 2B)
            q, q2 = q_joint.chunk(2, dim=1)
            q2 = q2.detach()
        else:
            # SAC: current Q from the live critic, next Q from the target critic.
            q = self.critic(s, a)
            with torch.no_grad():
                q2 = self.critic_target(s2, a2)

        target = r + cfg.gamma * (1.0 - done) * (q2.min(0).values - alpha * logp2)  # (B,)
        critic_loss = 0.5 * (q - target.detach()).pow(2).mean(1).sum()  # mean over batch, sum over critics
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.critic_target is not None:  # Polyak averaging (SAC only)
            with torch.no_grad():
                for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                    p_t.lerp_(p, cfg.tau)
        self.n_updates += 1
        # detached tensors, not .item(): converting would force a GPU sync every step
        info = {"critic_loss": critic_loss.detach(), "alpha": alpha, "q_mean": q.detach().mean()}

        # ---- actor + temperature (delayed) ---------------------------------------
        if self.n_updates % cfg.policy_delay == 0:
            a_pi, logp = self.actor.sample(s)  # actor BN in train mode: updates its running stats
            self.critic.eval()  # critic BN in eval mode: use, but do not pollute, its running stats
            q_pi = self.critic(s, a_pi).min(0).values
            self.critic.train()
            actor_loss = (alpha * logp - q_pi).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()

            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            info.update(actor_loss=actor_loss.detach(), entropy=-logp.detach().mean())
        return info


# --------------------------------------------------------------------------- env / eval


def make_env(name: str, seed: int) -> gym.Env:
    env = gym.make(name)
    env = gym.wrappers.RescaleAction(env, -1.0, 1.0)  # tanh policy lives in [-1, 1]
    env = gym.wrappers.DtypeObservation(env, np.float32)  # MuJoCo emits float64
    env = gym.wrappers.RecordEpisodeStatistics(env)  # info["episode"]["r"] at episode end
    env.action_space.seed(seed)
    return env


def evaluate(agent: Agent, env: gym.Env, episodes: int, seed: int) -> float:
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, deterministic=True))
            ret += float(reward)
            done = terminated or truncated
        returns.append(ret)
    return float(np.mean(returns))


# --------------------------------------------------------------------------- training loop


def train(cfg: Config) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    if cfg.device == "auto":
        cfg.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(cfg.device)

    run_dir = Path(cfg.logdir) / cfg.env / (cfg.name or cfg.algo) / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    writer = SummaryWriter(run_dir)
    csv_file = open(run_dir / "log.csv", "w", newline="")
    csv_log = csv.DictWriter(csv_file, fieldnames=["step", "eval_return", "train_return", "sps", "critic_loss", "q_mean", "alpha", "entropy"])
    csv_log.writeheader()

    env, eval_env = make_env(cfg.env, cfg.seed), make_env(cfg.env, cfg.seed + 10_000)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = Agent(obs_dim, act_dim, cfg, device)
    buffer = ReplayBuffer(cfg.buffer_size, env.observation_space, env.action_space, device=device,
                          handle_timeout_termination=True)  # bootstraps through time-limit truncations

    obs, _ = env.reset(seed=cfg.seed)
    last_train_return, info, t0 = float("nan"), {}, time.time()
    pbar = tqdm(range(1, cfg.steps + 1), desc=f"{cfg.env} {cfg.algo} s{cfg.seed}", dynamic_ncols=True, mininterval=5)
    for step in pbar:
        action = env.action_space.sample() if step <= cfg.learning_starts else agent.act(obs)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        buffer.add(obs, next_obs, action, reward, terminated or truncated, [{"TimeLimit.truncated": truncated and not terminated}])
        obs = next_obs
        if terminated or truncated:
            last_train_return = float(step_info["episode"]["r"])
            writer.add_scalar("train/return", last_train_return, step)
            obs, _ = env.reset()

        if step > cfg.learning_starts:
            for _ in range(cfg.utd):
                info = agent.update(buffer.sample(cfg.batch_size))

        if step % cfg.eval_every == 0:
            eval_return = evaluate(agent, eval_env, cfg.eval_episodes, seed=cfg.seed + step)
            sps = cfg.eval_every / (time.time() - t0)
            t0 = time.time()
            info = {k: float(v) for k, v in info.items()}
            row = {"step": step, "eval_return": eval_return, "train_return": last_train_return, "sps": round(sps, 1),
                   **{k: info.get(k, float("nan")) for k in ("critic_loss", "q_mean", "alpha", "entropy")}}
            csv_log.writerow(row)
            csv_file.flush()
            writer.add_scalar("eval/return", eval_return, step)
            for k, v in info.items():
                writer.add_scalar(f"train/{k}", v, step)
            pbar.set_postfix(eval=f"{eval_return:.0f}", train=f"{last_train_return:.0f}", sps=f"{sps:.0f}")

    torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()}, run_dir / "model.pt")
    csv_file.close()
    writer.close()
    return run_dir


# --------------------------------------------------------------------------- cli


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            p.add_argument(flag, default=None, action=argparse.BooleanOptionalAction)
        elif f.type.startswith("tuple"):
            p.add_argument(flag, default=None, type=int, nargs="+")
        else:
            p.add_argument(flag, default=None, type=type(f.default))
    args = {k: v for k, v in vars(p.parse_args()).items() if v is not None}
    algo = args.get("algo", Config.algo)
    cfg = Config(algo=algo, **PRESETS[algo])  # preset first ...
    for k, v in args.items():  # ... explicit flags override it
        setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    return cfg


if __name__ == "__main__":
    train(parse_args())
