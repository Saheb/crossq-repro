# Sample efficiency in deep RL: where it stands and where it is going

*Written September 2026, as companion to the CrossQ reproduction in this repo. Focus: model-free and model-based
RL for continuous control (MuJoCo / DMC / HumanoidBench / MyoSuite) with Atari-100k and real-robot work as
reference points. Links are to arXiv unless noted.*

---

## 0. TL;DR

1. **The field has largely stopped inventing new "algorithms" and started fixing optimisation.** Almost every
   headline gain since 2022 comes from *normalisation, network scale, regularisation and loss parameterisation*
   applied to plain SAC/TD3, not from new objectives. CrossQ (BatchNorm, no target net), SimBa/SimBaV2
   (LayerNorm / hyperspherical norm + residual MLPs), BRO (big regularised critics) and CrossQ+WN (weight
   normalisation) are all instances of this.
2. **The update-to-data (UTD) ratio is the exchange rate between samples and compute**, and its price is
   *plasticity loss* (a.k.a. primacy bias). 2021-23 bought sample efficiency with UTD 20 plus resets/dropout/
   ensembles (REDQ, DroQ, SR-SAC, BBF). 2024-26 buys it more cheaply with bigger, better-normalised networks at
   UTD 1-10, and treats plasticity with continuous fixes (weight norm, sample weight decay, parameter constraints)
   rather than resets.
3. **Scaling is now predictable.** "Value-based deep RL scales predictably" (ICML 2025) and "Compute-optimal
   scaling for value-based RL" (2025) give Pareto frontiers over data, compute, model size, batch size and UTD.
   Large critics tolerate large batches; small ones "TD-overfit".
4. **Classification-style critics won.** Categorical / HL-Gauss / quantile value heads are the default in new
   strong methods (SimBaV2, BRO, FastTD3, BBF, TD-MPC2).
5. **Model-based RL is back, but as a component.** TD-MPC2 style latent models, model-generated data to stabilise
   high UTD (MAD-TD), model-based *representations* without planning (MR.Q, DR.Q), and improved planning
   objectives (EfficientTDMPC, Dream-MPC, ICML 2026) currently hold the low-data SOTA on DMC-hard / HumanoidBench.
6. **Massively parallel simulation moved the goalposts.** When environment steps are nearly free (FastTD3, PQN),
   the frontier is wall-clock and compute, not samples. Sample-efficiency research is therefore migrating to
   where samples are truly scarce: real robots (RLPD → SERL → HIL-SERL → WorldSample), offline-to-online, and
   long-horizon / sparse-reward tasks (Q-chunking, horizon reduction).
7. **Priors from data are the biggest lever in the real world**: demonstrations in the replay buffer, pretrained
   VLA policies fine-tuned with RL, and video/world foundation models used as simulators.

---

## 1. What "sample efficient" means today

* **Benchmarks.** State-based: MuJoCo Gym (Hopper … Humanoid, 1-5M steps), DMC (incl. the "hard" dog/humanoid
  tasks, 1M steps), HumanoidBench, MyoSuite, Meta-World. Pixel-based: DMC from pixels, Atari-100k (2 h of play).
  Reported with rliable-style IQM and CIs since Agarwal et al. 2021.
* **Two currencies.** Environment steps (data) and gradient steps (compute). UTD = gradient steps per environment
  step. Rybkin et al. (ICML 2025) show that for a target return, achievable (data, compute) pairs lie on a
  Pareto frontier parameterised by UTD; you can buy samples with compute and vice versa
  ([2502.04327](https://arxiv.org/abs/2502.04327)).
* **What limits the exchange.** Plasticity loss: after many gradient steps on non-stationary targets, networks
  lose the ability to fit new targets (dormant neurons, feature-rank collapse, growing weight norm, shrinking
  effective learning rate). Survey: Klein et al., *Plasticity loss in deep RL: a survey*
  ([2411.04832](https://arxiv.org/abs/2411.04832)); benchmark suite: Plasticine
  ([2504.17490](https://arxiv.org/abs/2504.17490)).

---

## 2. The lineage that leads to CrossQ

| Year | Method | Key idea | Cost |
|---|---|---|---|
| 2018 | SAC / TD3 | max-entropy actor-critic; twin critics; target nets | UTD 1 |
| 2019 | MBPO ([1906.08253](https://arxiv.org/abs/1906.08253)) | ensemble dynamics model, short model rollouts | UTD 20-40, model |
| 2021 | REDQ ([2101.05982](https://arxiv.org/abs/2101.05982)) | 10 critics, min over random subset; showed UTD 20 works model-free | 10 critics × UTD 20 |
| 2022 | DroQ ([2110.02034](https://arxiv.org/abs/2110.02034)) | dropout + LayerNorm in 2 critics instead of 10 | UTD 20 |
| 2022 | Primacy bias / resets (Nikishin et al., [2205.07802](https://arxiv.org/abs/2205.07802)) | periodically re-initialise the networks, keep the buffer | UTD 32 (SR-SAC, D'Oro et al. ICLR 2023) |
| 2023 | BBF ([2305.19452](https://arxiv.org/abs/2305.19452)) | Atari-100k: bigger ResNet + shrink-and-perturb resets + RR 8 + annealed n-step/γ + SPR loss | RR 8 |
| 2024 | **CrossQ** ([1902.05605](https://arxiv.org/abs/1902.05605)) | remove target nets, BatchRenorm, joint (s,a)/(s',a') batch, wide critic, β₁=0.5 | **UTD 1** |

CrossQ's point was that most of REDQ/DroQ's sample-efficiency could be had for ~5 % of their gradient steps
by fixing the *optimisation* (normalisation, no stale target) rather than by hammering the data with updates.

---

## 3. The main threads (ideas, methods, techniques)

### 3.1 Normalisation and the target-network question

* **CrossQ** (Bhatt, Palenicek et al., ICLR 2024). BatchNorm in the critic was long "known" to fail; the paper
  shows it fails because (s,a) and (s',a') are differently distributed and the target net's stale statistics
  make the next-state batch look out-of-distribution. Passing both halves through the critic *in one batch*
  fixes it and makes target networks unnecessary. BatchRenorm instead of BatchNorm to survive long runs; wider
  critic (2048) exploits the stabler optimisation. Reference impls: authors' JAX
  ([github.com/adityab/CrossQ](https://github.com/adityab/CrossQ)), `sbx`, `sb3-contrib` (PyTorch).
* **PQN** (Gallici et al., *Simplifying deep temporal difference learning*, ICLR 2025,
  [2407.04811](https://arxiv.org/abs/2407.04811)). Proves that LayerNorm (+ ℓ2 regularisation) makes TD with
  function approximation provably convergent *without* target networks or replay buffers, even off-policy; a
  vectorised on-policy Q-learner that matches Rainbow on Atari at 50× the speed. Theory for what CrossQ found
  empirically.
* **Lyle et al., *Normalization and effective learning rates in RL*** ([2407.01800](https://arxiv.org/abs/2407.01800)):
  normalisation layers make the effective learning rate depend on the weight norm, which drifts in
  non-stationary training. Fix: keep the weight norm constant (Normalize-and-Project). This is the mechanism
  behind CrossQ+WN and SimBaV2.
* **Hussing et al., *Dissecting deep RL with high UTD*** ([2403.05996](https://arxiv.org/abs/2403.05996)):
  value divergence at high UTD is driven by feature norm growth; unit-ball normalisation of critic features fixes it.
* **SimBa** (Lee et al., ICLR 2025, [2410.09754](https://arxiv.org/abs/2410.09754)): running-statistics
  observation normalisation (RSNorm) + pre-LayerNorm residual MLP blocks + post-LayerNorm. Plug into SAC and it
  scales with parameters and UTD; matched TD-MPC2/DreamerV3 at a fraction of the compute.
* **SimBaV2** (Lee et al., ICML 2025 spotlight, [2502.15280](https://arxiv.org/abs/2502.15280)): replace
  LayerNorm by ℓ2 (hyperspherical) normalisation of features *and* weights, learnable-interpolation residuals,
  categorical critic with KL loss and reward scaling. SOTA on 57 tasks (MuJoCo, DMC, MyoSuite, HumanoidBench)
  with one hyper-parameter set, and it keeps improving with model size and UTD.
* **CrossQ + Weight Normalisation** (Palenicek et al., NeurIPS 2025,
  [2502.07523](https://arxiv.org/abs/2502.07523); short version [2506.03758](https://arxiv.org/abs/2506.03758)).
  Plain CrossQ degrades when UTD is raised; adding weight normalisation keeps the effective learning rate
  constant, prevents plasticity loss, and lets CrossQ scale reliably to high UTD, competitive on 25 DMC and
  MyoSuite tasks (incl. dog/humanoid) *without* resets. This is the direct successor of the paper reproduced here.

**Take-away:** BatchNorm (CrossQ) vs LayerNorm (DroQ, SimBa, PQN, RLPD) vs hyperspherical (SimBaV2) are three
routes to the same thing: bounded, well-conditioned features so that bootstrapping does not diverge. The
LayerNorm/ℓ2 family has become more popular because it is batch-independent (no joint-batch trick, no
train/eval mode, ensembles are easy); CrossQ's BatchNorm is arguably the most sample-efficient at UTD 1.

### 3.2 Replay ratio scaling and plasticity

* The **primacy bias** paper (ICML 2022) and **"Breaking the replay ratio barrier"** (D'Oro et al., ICLR 2023)
  showed resets let SAC/SPR use replay ratios of 16-32 productively.
* **Diagnostics and lighter interventions:** dormant neurons + ReDo (Sokar et al., ICML 2023,
  [2302.12902](https://arxiv.org/abs/2302.12902)); plasticity injection (Nikishin et al., NeurIPS 2023,
  [2305.15555](https://arxiv.org/abs/2305.15555)); PLASTIC ([2306.10711](https://arxiv.org/abs/2306.10711)).
* **Nauman et al., *Overestimation, overfitting and plasticity in actor-critic: the bitter lesson of RL***
  (ICML 2024, [2403.00514](https://arxiv.org/abs/2403.00514)): a 64-combination study concluding that network
  scaling + strong regularisation (LayerNorm, weight decay, resets) beats algorithmic tricks. Led directly to BRO.
* **MAD-TD** (Voelcker et al., ICLR 2025, [2410.08896](https://arxiv.org/abs/2410.08896)): high-UTD instability
  comes from the critic not generalising to unseen *on-policy* actions; a small amount of learned-model data fixes
  it. Model-based data as a plasticity/overfitting cure rather than for sample generation.
* **Forget-and-Grow** (Kang et al., ICML 2025, [2507.02712](https://arxiv.org/abs/2507.02712)): decay the
  sampling weight of old replay data ("forget") and add parameters during training ("grow"); beats BRO, SimBa and
  TD-MPC2 on 40+ tasks.
* **Sample Weight Decay** (Wu et al., ICLR 2026, [2604.01913](https://arxiv.org/abs/2604.01913)): attributes
  plasticity loss to NTK-Gram rank collapse and Θ(1/k) gradient decay; a lightweight per-sample weighting restores
  gradient magnitude across TD3/SAC(+SimBa)/DDQN.
* **Rethinking plasticity** (He, 2026, [2603.21173](https://arxiv.org/abs/2603.21173)): argues dormancy is a
  symptom of getting stuck in old local optima; recommends parameter constraints (norm bounds) over resets.
* **Churn reduction** (ICML 2025) and **adaptive linearity injection** ([2505.09486](https://arxiv.org/abs/2505.09486))
  are further continuous alternatives to resets.

**Direction:** from discrete surgery (resets, shrink-and-perturb) towards *always-on* constraints (weight norm,
hyperspherical weights, sample weighting) that keep the network trainable indefinitely. Resets are increasingly
seen as a symptom of missing normalisation.

### 3.3 Scale the network, not (only) the UTD

* **BRO** (Nauman et al., NeurIPS 2024, [2405.16158](https://arxiv.org/abs/2405.16158)): ~5M-parameter
  residual-LayerNorm critic ("BroNet"), weight decay, resets, quantile critic, optimistic exploration via
  pessimism-free action selection; UTD 10 (a UTD 2 variant is nearly as good). First model-free method to solve
  DMC dog/humanoid; reported 2.5× more sample-efficient than TD-MPC2.
* **Compute-optimal scaling** (Fu, Rybkin et al., 2025, [2508.14881](https://arxiv.org/abs/2508.14881)):
  larger critics can use larger batches; small critics suffer "TD-overfitting" when the batch grows. Gives rules
  for splitting a compute budget between model size, UTD and batch size.
* **Mixtures of experts** (Obando-Ceron et al., ICML 2024) and **Hadamax encodings**
  ([2505.15345](https://arxiv.org/abs/2505.15345)) show architecture alone moves Atari results; **1000-layer
  networks** for contrastive goal-reaching RL ([2503.14858](https://arxiv.org/abs/2503.14858)) show depth
  can matter for long-horizon reasoning.
* **Simplicial embeddings** (Obando-Ceron et al., ICLR 2026, [2510.13704](https://arxiv.org/abs/2510.13704)):
  a sparse, simplex-constrained representation layer improves sample efficiency of FastTD3, FastSAC and PPO at no
  runtime cost.
* **Scaling survey:** *Scaling DRL for decision making: data, network and training budget strategies*
  ([2508.03194](https://arxiv.org/abs/2508.03194)).

### 3.4 Losses and value parameterisation

* **Classification instead of regression:** HL-Gauss / two-hot cross-entropy (Farebrother et al., ICML 2024,
  [2403.03950](https://arxiv.org/abs/2403.03950)) scales value learning to large nets and transformers; now
  standard in SimBaV2, FastTD3, BBF, TD-MPC2, DreamerV3 (symlog two-hot). Learned categorical supports
  ([2607.01880](https://arxiv.org/abs/2607.01880)) and flow-matching critics (floq,
  [2509.06863](https://arxiv.org/abs/2509.06863)) are 2025-26 refinements.
* **Multi-step targets:** n-step / TD(λ) returns (PQN, BBF's annealed n-step), and **Q-chunking**
  (Li et al., NeurIPS 2025, [2507.07969](https://arxiv.org/abs/2507.07969)): run RL in a *chunked* action
  space so n-step backups are unbiased and exploration is temporally coherent; strong in offline-to-online.
  Decoupled Q-chunking ([2512.10926](https://arxiv.org/abs/2512.10926)) and adaptive chunking
  ([2605.10044](https://arxiv.org/abs/2605.10044)) follow.
* **Bias control:** clipped double-Q (TD3) remains default; REDQ's random-subset min, BRO's optimism, and
  pessimism-learning variants ([2110.03375](https://arxiv.org/abs/2110.03375)) adjust the pessimism level.

### 3.5 Model-based RL and learned representations

* **World-model agents:** DreamerV3 (Hafner et al., Nature 2025, [2301.04104](https://arxiv.org/abs/2301.04104)),
  **TD-MPC2** (Hansen et al., ICLR 2024, [2310.16828](https://arxiv.org/abs/2310.16828)): latent model +
  MPPI planning + TD value; robust across 100+ tasks with one config. EfficientZero
  ([2111.00210](https://arxiv.org/abs/2111.00210)) / **EfficientZero V2**
  ([2403.00564](https://arxiv.org/abs/2403.00564)) extend MuZero-style search to low-data continuous control.
  DIAMOND ([2405.12399](https://arxiv.org/abs/2405.12399)) trains in a diffusion world model (Atari-100k
  mean HNS 1.46).
* **2026 planning-objective work:** **EfficientTDMPC** ([2605.16692](https://arxiv.org/abs/2605.16692)) reports
  low-data SOTA on HumanoidBench-Hard and DMC-hard; **Dream-MPC** (Spieler & Behnke, ICML 2026,
  [2605.04568](https://arxiv.org/abs/2605.04568)) does gradient-based MPC in latent imagination; BOOM, TD-M(PC)²
  and BMPC fix planner/policy distillation in high-dimensional action spaces. Counterpoint: *The surprising
  difficulty of search in model-based RL* ([2601.21306](https://arxiv.org/abs/2601.21306)).
* **Model-based representations, model-free control:** TD7's SALE embeddings (Fujimoto et al., NeurIPS 2023),
  **MR.Q** (Fujimoto et al., ICLR 2025, [2501.16142](https://arxiv.org/abs/2501.16142)): learn a state-action
  embedding that is approximately linear in the value via next-latent/reward/termination prediction, then run TD3
  in latent space with one hyper-parameter set across Gym, DMC and Atari. **DR.Q** (Lyu et al., ICML 2026,
  [2605.11711](https://arxiv.org/abs/2605.11711)) debiases this with a mutual-information objective and faded
  prioritised replay; +15 % over SimBaV2 on DMC-hard, first >700 on dog-run within 1M steps.
* **Self-supervised auxiliary losses for pixels:** SPR (ICLR 2021), BBF; V-Simba (Aug 2026,
  [2608.07870](https://arxiv.org/abs/2608.07870)) shows SimBa-style architecture beats DrQ-v2 and DrM on DMC /
  Adroit / Meta-World from pixels with a *smaller* network.

### 3.6 Exploration and optimism

Sample-efficiency papers on standard benchmarks mostly rely on entropy bonuses (SAC), Gaussian noise (TD3) and
optimism in the *critic* (BRO's optimistic actor, OAC). Intrinsic-reward methods (RND, NovelD, etc.) matter for
hard-exploration tasks but are orthogonal to the optimisation story above; Q-chunking's temporally coherent
exploration is the notable 2025 crossover.

### 3.7 Data: demonstrations, offline-to-online, real robots

* **RLPD** (Ball et al., ICML 2023, [2302.02948](https://arxiv.org/abs/2302.02948)): symmetric sampling from
  demo and online buffers, LayerNorm critic ensembles, UTD 20; the backbone of **SERL** and **HIL-SERL**
  (Luo et al., 2024, [2410.21845](https://arxiv.org/abs/2410.21845)) which learn precise real-robot manipulation
  in 1-2.5 hours with human interventions. **WorldSample** (2026,
  [2607.02431](https://arxiv.org/abs/2607.02431)) adds a world model on top of HIL-SERL: success 56 → 82 %,
  training steps 56k → 23k.
* **Fine-tuning large pretrained policies:** VLA-RFT ([2510.00406](https://arxiv.org/abs/2510.00406)),
  world-action models (WAM-RL, Efficient-WAM, 2026) and video-model policies (Cosmos Policy, 2026) treat a
  video/world foundation model as simulator or prior; surveys: *World model for robot learning*
  ([2605.00080](https://arxiv.org/abs/2605.00080)).
* **Horizon is the scaling bottleneck in offline RL:** *Horizon reduction makes RL scalable* (Park et al.,
  2025, [2506.04168](https://arxiv.org/abs/2506.04168)); n-step/chunked/hierarchical methods keep scaling with
  data where 1-step TD saturates.

### 3.8 Massively parallel simulation changes the objective

* **FastTD3** (Seo et al., 2025, [2505.22642](https://arxiv.org/abs/2505.22642)): TD3 + thousands of parallel
  envs, batch 32k, distributional critic, clipped double-Q; solves HumanoidBench in < 3 h on one A100. FastSAC and
  simplicial embeddings build on it.
* **PQN** (above) and **Adaptive Batch Scaling** (2026, [2605.21557](https://arxiv.org/abs/2605.21557)):
  large-batch RL is possible if batch size tracks policy non-stationarity; "bigger network + bigger batch" scaling
  finally works for Q-learning on Atari.
* Consequence: on simulators, *compute efficiency* and *wall-clock* are the metrics that matter; the sample
  efficiency methods of §3.1-3.5 remain essential where each sample is expensive (hardware, slow sims, human
  data) or where exploration, not fitting, is the bottleneck.

---

## 4. Where CrossQ stands in 2026

* **Still the simplest strong baseline at UTD 1.** It is in `sb3-contrib`, `sbx`, and used as the base of
  CrossQ+WN (NeurIPS 2025). Its UTD-1 sample efficiency on MuJoCo Gym remains competitive with REDQ/DroQ.
* **Superseded on harder suites at higher budgets.** BRO, SimBa/SimBaV2, DR.Q, Forget-and-Grow and the 2026
  model-based methods report clearly better DMC-hard / HumanoidBench results, typically at UTD 2-10 and with
  5-25 M-parameter critics. CrossQ+WN closes much of that gap while keeping the CrossQ recipe.
* **Its lasting contributions:** (i) target networks are an optimisation crutch, not a necessity; (ii) the
  train/target distribution mismatch is the reason BN "did not work" in RL; (iii) wide critics + lower Adam β₁
  are cheap wins; (iv) compute matters (5 % of the gradient steps of REDQ for equal returns).
* **Known caveats:** BatchNorm makes ensembles, sequence models and eval-mode handling awkward; performance at
  UTD > 1 needs weight normalisation; BatchRenorm and its warm-up are required for long runs.

---

## 5. Direction of travel

1. **Optimisation hygiene as the recipe.** Normalised inputs, normalised features, constrained weight norm,
   residual MLPs, categorical critics, large critics, TD3/SAC base, UTD 2-8. SimBaV2 / BRO / CrossQ+WN are
   converging on this, and 2026 papers (DR.Q, SWD, Simplicial embeddings, V-Simba) are add-ons to it.
2. **Plasticity theory maturing** from phenomenology (dormant neurons, rank) to optimisation-based explanations
   with continuous remedies; resets are becoming a diagnostic rather than a method.
3. **Scaling laws for RL** make compute/data allocation a design decision, and put value-based RL on the same
   footing as supervised learning for planning experiments.
4. **Model-based components everywhere:** models as data augmenters (MAD-TD), representation teachers
   (MR.Q/DR.Q), or planners with better objectives (EfficientTDMPC, Dream-MPC). The pure model-free vs
   model-based split is dissolving.
5. **Temporal abstraction and horizon reduction** (action chunking, n-step, hierarchical) as the answer to
   long-horizon, sparse-reward sample efficiency.
6. **Priors from data and foundation models** (demos, offline data, VLAs, video world models) dominate real-world
   sample efficiency; online RL is becoming the *fine-tuning* stage.
7. **Benchmarks are shifting** from MuJoCo Gym to DMC-hard, HumanoidBench, MyoSuite and real hardware, and from
   "return at 1M steps" to Pareto curves over data and compute.

---

## 6. Things to try on top of `crossq.py`

Each is a small change and lines up with a paper above:

* `--utd 5` + weight normalisation on the Linear layers (CrossQ+WN).
* Swap BatchRenorm for LayerNorm and remove the joint pass but keep no-target-network (PQN / SimBa-style);
  compare stability.
* Replace the MSE critic loss by HL-Gauss cross-entropy over a fixed support (Stop Regressing).
* Critic width sweep 256 → 4096 and depth 2 → 4 with residual blocks (BRO / SimBa scaling).
* Periodic resets vs. sample weight decay at UTD 10, measuring dormant-neuron fraction.
* n-step (3-5) targets from the SB3 replay buffer (`n_steps` argument) and action chunking.

---

## 7. Compact reference list

Foundations: SAC ([1801.01290](https://arxiv.org/abs/1801.01290)), TD3 ([1802.09477](https://arxiv.org/abs/1802.09477)),
MBPO, REDQ, DroQ, Primacy bias, SR-SAC ([2305.19452](https://arxiv.org/abs/2305.19452) for BBF), DrQ
([2004.13649](https://arxiv.org/abs/2004.13649)), DrQ-v2 ([2107.09645](https://arxiv.org/abs/2107.09645)),
SPR ([2007.05929](https://arxiv.org/abs/2007.05929)).

Normalisation / architecture: CrossQ, PQN, Lyle et al. 2024, Hussing et al. 2024, SimBa, SimBaV2, CrossQ+WN,
BRO, Nauman et al. 2024 (bitter lesson), Simplicial embeddings, V-Simba, Hadamax, MoE-RL.

Plasticity: survey 2411.04832, Plasticine, ReDo, plasticity injection, MAD-TD, Forget-and-Grow, Sample Weight
Decay, Rethinking plasticity 2026.

Scaling: Rybkin et al. 2025, Fu et al. 2025, Horizon reduction, 1000-layer networks, Adaptive Batch Scaling,
scaling survey 2508.03194.

Losses: Stop Regressing (HL-Gauss), floq, Q-chunking, Decoupled Q-chunking.

Model-based / representations: DreamerV3, TD-MPC2, EfficientZero (V2), DIAMOND, EfficientTDMPC, Dream-MPC,
TD7, MR.Q, DR.Q, WIMLE ([2602.14351](https://arxiv.org/abs/2602.14351)), Discrete codebook world models
([2503.00653](https://arxiv.org/abs/2503.00653)).

Real world / data priors: RLPD, SERL, HIL-SERL, WorldSample, VLA-RFT, world-action models, FastTD3.
