---
title: "INFUSER: Influence-Guided Self-Evolution Improves Reasoning"
date: 2026-06-09
lastmod: 2026-06-09
author: "Siyu Chen, Miao Lu, Beining Wu, Heejune Sheen, Fengzhuo Zhang, Shuangning Li, Zhiyuan Li, Jose Blanchet, Tianhao Wang, and Zhuoran Yang"
description: "INFUSER turns unstructured documents into an adaptive reasoning curriculum by rewarding generated questions according to their optimizer-aware influence on a co-evolving solver."
cover: "/images/infuser/system_overview.png"
categories:
  - "Research Blog"
tags:
  - "LLM"
  - "Self-Evolution"
  - "RLVR"
  - "Influence Functions"
  - "Reasoning"
draft: false
math: true
toc: true
---

![INFUSER system overview](/images/infuser/system_overview.png)

**Paper:** INFUSER: Influence-Guided Self-Evolution Improves Reasoning

**Authors:** Siyu Chen, Miao Lu, Beining Wu, Heejune Sheen, Fengzhuo Zhang, Shuangning Li, Zhiyuan Li, Jose Blanchet, Tianhao Wang, and Zhuoran Yang.

<strong style="font-size: 0.85em; letter-spacing: 1px; color: #3B9DD9;">LINKS:</strong>
<span style="font-size: 0.9em;">
<strong style="color: #D98C3B;">arXiv:</strong> <a href="https://arxiv.org/abs/2606.09052" target="_blank"><strong>Paper</strong></a> &nbsp;|&nbsp;
<strong style="color: #4CAF7D;">GitHub:</strong> <a href="https://github.com/FFishy-git/INFUSER" target="_blank"><strong>Code</strong></a>
</span>

## TL;DR

**INFUSER turns self-evolution into influence-guided curriculum learning.** A co-evolving generator drafts questions and reference answers from unstructured documents, while a solver trains on them with correctness rewards. Instead of rewarding the generator for surface difficulty, INFUSER rewards each generated question by an optimizer-aware influence score: would training on this example move the solver toward the capabilities we want to improve?

This continuous and noisy influence reward is optimized with **DuGRPO**, a dual-normalized GRPO variant for generator training. Together, the generator, solver, and influence estimator turn a document pool into an adaptive curriculum that favors questions useful to the current solver, not merely harder ones.

On Qwen3-8B-Base, INFUSER outperforms strong self-evolution baselines, including over 20% relative improvement on Olympiad and SuperGPQA benchmarks. Its 8B co-evolving generator also outperforms a frozen 32B thinking generator on math and coding, while ablations confirm the importance of optimizer-aware influence and DuGRPO.

<figure>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start">
    <img src="/images/infuser/qw8bb-best-score-bar-gain.png" alt="Headline Qwen3-8B gains">
    <img src="/images/infuser/qw4bb-qw8bb-infuser-fixgen.png" alt="INFUSER training curves">
  </div>
  <figcaption><strong>INFUSER</strong> on Qwen3 base anchors. Left: relative accuracy gain over Qwen3-8B-Base on four headline benchmarks. Right: validation-set accuracy curves for INFUSER versus a fixed-generator baseline on Qwen3-4B-Base and Qwen3-8B-Base.</figcaption>
</figure>

## From Documents to Useful RL Signals

RLVR is powerful because it can train a solver from verifiable rewards, but it does not by itself solve the data problem. A textbook, paper, or web document may contain useful knowledge, yet an **unstructured** document does not directly provide the structured question, reference answer, and reward signal that RLVR consumes.

This is why self-evolution needs a generator: the generator converts raw documents into a curriculum of question-answer pairs, and the solver trains on that curriculum with RL. The remaining question is how to train the generator itself. If the generated data is heavily curated or teacher-generated, the system inherits the cost and bias of that teacher. If the generator is trained only by proxy signals, the most common proxy is difficulty against the current solver, as in recent self-evolution systems such as [R-Zero](https://arxiv.org/abs/2508.05004) and [SPICE](https://arxiv.org/abs/2510.24684).

Difficulty is useful, but incomplete. A generator can produce mislabeled or invalid questions that look difficult, but these examples may be irrelevant to the solver's next step or even harm training. Even when the generator poses valid questions, it is hard to know whether they are relevant to what we want the solver to improve on. INFUSER replaces this heuristic with a utility signal: **reward generated questions whose induced solver-gradient direction is useful for the dev-set objective.**

## INFUSER: Training for Usefulness

INFUSER has three coupled roles:

1. **Generator** $\pi_\phi$: reads documents from a corpus and drafts a curriculum $\mathcal{Q}_\phi$ of question-answer pairs $(q, a_\phi)$.
2. **Solver** $\pi_\theta$: answers generated questions and improves by RL training against the generated reference answers.
3. **Influence estimator**: scores each generated pair by how useful its solver update is for performance on the held-out development set.

![INFUSER training loop overview](/images/infuser/system_overview_paper.png)

At each iteration, INFUSER estimates a development reference gradient $\hat g_{\mathrm{dev}}$ as an RL gradient on a fixed dev-set. This gradient encodes the question:

<p style="text-align:center;font-size:1.05em;font-style:italic;font-weight:600">If the solver wants to improve on the dev-set, which direction should it move in parameter space?</p>

For each generated pair $(q, a_\phi)$, INFUSER asks the solver to sample rollouts on $q$ and uses the generated answer $a_\phi$ as the reference for RL rewards. From those rollouts, INFUSER estimates the **optimizer-aware** RL update direction $\hat\Gamma(q, a_\phi)$. This direction encodes the question:

<p style="text-align:center;font-size:1.05em;font-style:italic;font-weight:600">If the solver trains on this generated question, which direction would the optimizer move it?</p>

The influence reward is the cosine similarity

$$
\hat s(q, a_\phi) =
\mathrm{cossim}\bigl(\hat g_{\mathrm{dev}}, \hat\Gamma(q, a_\phi)\bigr).
$$

A large cosine similarity indicates that training on the generated question would move the solver in a direction aligned with dev-set improvement. A low or negative similarity indicates that the question is unhelpful or potentially harmful for the dev-set objective we want the solver to improve on.

The generator is updated using this influence reward. The solver is updated using correctness rewards on the same generated questions. Because the influence reward is continuous and noisy, INFUSER trains the generator with **DuGRPO**, a dual-normalized variant of GRPO designed to stabilize this setting. In DuGRPO, the generator advantage is computed as

$$
\hat A_{\mathrm{gen}}(q^k, a_{\phi}^k)
=
\frac{\hat s(q^k, a_{\phi}^k)-\mu_d}
{\sigma_d+\sigma_{\mathcal{B}}+\epsilon}.
$$

Here, \(\mu_d\) and \(\sigma_d\) are the mean and standard deviation of influence rewards for questions generated from the same document \(d\), while \(\sigma_{\mathcal{B}}\) is the average group standard deviation across the document batch. Unlike standard GRPO, the extra \(\sigma_{\mathcal{B}}\) term avoids amplifying noise in low-variance groups, while still preserving useful within-document ranking.

## INFUSER from the View of Bilevel Optimization

After the influence-score view, the deeper way to understand INFUSER is as a bilevel optimization problem. The generator does not merely emit data; it chooses a curriculum \(\mathcal{Q}_\phi\). The solver then learns on that curriculum, and the generator is judged by how well the resulting solver performs on the dev-set.

For this discussion, define the performance of a solver on any question-answer distribution \(\mathcal{D}\) as

$$
J_{\mathcal{D}}(\theta)
=
\mathbb{E}_{(q,a)\sim\mathcal{D},\;a'\sim\pi_\theta(\cdot\mid q)}
\bigl[r(a',a)\bigr].
$$

Let \(\mathcal{D}_{\mathrm{dev}}\) denote the held-out dev-set. Then the bilevel objective asks the generator to choose a curriculum \(\mathcal{Q}_\phi\) such that, after the solver trains on that curriculum, the resulting solver performs well on \(\mathcal{D}_{\mathrm{dev}}\):

$$
\max_\phi\; J_{\mathcal{D}_{\mathrm{dev}}}\!\bigl(\theta^*(\phi)\bigr),
\qquad
\theta^*(\phi)
=
\arg\max_\theta J_{\mathcal{Q}_\phi}(\theta).
$$

The lower level asks: if the generator fixed its curriculum \(\mathcal{Q}_\phi\), what solver would we get after training on it? The upper level asks: did that trained solver improve on the dev-set? This is why the generator is optimized for usefulness rather than difficulty.

<p style="text-align:center;font-size:1.05em;font-style:italic;font-weight:600">The solver learns from the curriculum; the generator learns which curriculum helps the solver.</p>

Exact bilevel optimization would require retraining the solver to convergence for every generator update, which is infeasible for LLMs. INFUSER therefore takes a local view: replace \(\theta^*(\phi)\) with the parameter after one optimizer step with respect to \(J_{\mathcal{Q}_\phi}\). To see the resulting tractable signal, start with the simpler SGD case. Let \(\hat g(q,a_\phi)\) denote the per-question RL gradient induced by training on the generated pair \((q,a_\phi)\). If the solver takes one SGD step on the generated curriculum \(\mathcal{Q}_\phi\), the resulting dev-set improvement decomposes over questions:

$$
J_{\mathcal{D}_{\mathrm{dev}}}\!\bigl(\theta_t+\Delta\theta_{\mathrm{SGD}}(\mathcal{Q}_\phi)\bigr)
-
J_{\mathcal{D}_{\mathrm{dev}}}(\theta_t)
\approx
\frac{\eta_s}{|\mathcal{Q}_\phi|}
\sum_{(q,a_\phi)\in\mathcal{Q}_\phi}
\langle
\hat g_{\mathrm{dev}},
\hat g(q,a_\phi)
\rangle.
$$

This gives a per-question contribution: questions whose RL gradient aligns with \(\hat g_{\mathrm{dev}}\) are useful for the dev-set objective. Therefore, optimizing \(J_{\mathcal{D}_{\mathrm{dev}}}\bigl(\theta_t+\Delta\theta_{\mathrm{SGD}}(\mathcal{Q}_\phi)\bigr)\) is equivalent to optimizing the average inner-product term, which means we can treat that inner product as the generator reward and apply RL optimization. INFUSER applies the same idea by replacing the raw SGD gradient \(\hat g(q,a_\phi)\) with the optimizer-aware RL update direction \(\hat\Gamma(q,a_\phi)\):

$$
\hat s(q,a_\phi)
=
\mathrm{cossim}\bigl(\hat g_{\mathrm{dev}}, \hat\Gamma(q,a_\phi)\bigr).
$$

Using cosine similarity keeps the reward focused on update direction and avoids spurious effects from sequence length or gradient norm. Since both the generator and solver objectives are now formulated as RL optimization problems, INFUSER optimizes them with alternating iterative updates.

## Main Results

Across both Qwen3-4B-Base and Qwen3-8B-Base anchors, INFUSER improves the base model on the four headline benchmarks. Relative gains over the corresponding base model are shown in parentheses.

| Benchmark | 4B Base | 4B INFUSER | 8B Base | 8B INFUSER |
| --- | ---: | ---: | ---: | ---: |
| MATH500 | 61.20 | **76.65** (+25.2%) | 76.05 | **82.77** (+8.8%) |
| OlympiadBench (Math) | 35.31 | **42.38** (+20.0%) | 40.36 | **50.24** (+24.5%) |
| MMLU-Pro | 52.98 | **60.20** (+13.6%) | 59.91 | **66.20** (+10.5%) |
| SuperGPQA | 25.88 | **33.48** (+29.4%) | 30.62 | **37.77** (+23.4%) |

Across the full benchmark suite, INFUSER's strongest gains appear on domains aligned with the document pool and dev-set, while still giving positive transfer on medical and coding benchmarks:

<table>
<thead>
<tr><th rowspan="2">Category Average</th><th colspan="4">Qwen3-4B-Base</th><th colspan="4">Qwen3-8B-Base</th></tr>
<tr><th>Base</th><th>R-Zero</th><th>AZR</th><th>INFUSER</th><th>Base</th><th>R-Zero</th><th>AZR</th><th>INFUSER</th></tr>
</thead>
<tbody>
<tr><td>General reasoning</td><td>29.46</td><td>32.18</td><td>32.93</td><td><strong>35.43</strong> (+20.3%)</td><td>34.43</td><td>37.14</td><td>37.61</td><td><strong>40.62</strong> (+18.0%)</td></tr>
<tr><td>Math &amp; physics</td><td>21.34</td><td>25.12</td><td><strong>26.51</strong></td><td>25.73 (+20.6%)</td><td>26.08</td><td>28.46</td><td>30.28</td><td><strong>31.49</strong> (+20.7%)</td></tr>
<tr><td>Medical (OOD)</td><td>34.24</td><td><strong>36.75</strong></td><td>36.14</td><td>36.32 (+6.1%)</td><td>39.34</td><td>40.17</td><td>39.89</td><td><strong>40.52</strong> (+3.0%)</td></tr>
<tr><td>Coding (OOD)</td><td>45.47</td><td>47.65</td><td>47.49</td><td><strong>48.63</strong> (+6.9%)</td><td>50.59</td><td>52.55</td><td>53.18</td><td><strong>53.29</strong> (+5.3%)</td></tr>
<tr><td><strong>Avg</strong></td><td>32.63</td><td>35.43</td><td>35.77</td><td><strong>36.53</strong> (+12.0%)</td><td>37.61</td><td>39.58</td><td>40.24</td><td><strong>41.48</strong> (+10.3%)</td></tr>
</tbody>
</table>

Three findings summarize the pattern:

1. **Aligned domains move the most.** General reasoning and math & physics are closest to the science document pool and dev-set, and both anchors see roughly 20% relative gains there.
2. **The gains transfer out of domain.** Even though medical and coding are OOD for the training signal, INFUSER still improves over the base model on both categories.
3. **The recipe scales to 8B.** At 8B, INFUSER leads on all four category averages, and its 4B-to-8B gain decay is smaller than competing self-evolution baselines.

## Results Analysis

<figure>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start">
    <img src="/images/infuser/qw8bb-gen-quality.png" alt="Generator quality over training" style="width:92%;margin:0 auto">
    <img src="/images/infuser/qw8bb-source-summary-bar.png" alt="Generator source and normalization ablations" style="width:82%;margin:0 auto">
  </div>
  <figcaption>Left: question quality rises over training, and the co-trained solver tracks the moving curriculum. Right: ablations compare generator source choices and generator-update variants.</figcaption>
</figure>

### The Generator Actually Improves

The generator-quality plot asks whether the generator is really improving, or merely drifting toward noisy hard-looking questions. We evaluate questions produced at different training iterations with three types of solvers:

1. **Fixed Qwen3-8B-Base solver.** If the trained generator produces harder questions than the initial generator, this fixed base solver should solve fewer of them.
2. **GPT-5.4 / GPT-5.4-mini reference solvers.** If question quality improves, with fewer invalid or ill-posed questions, strong solvers should solve more of them.
3. **Co-trained INFUSER solver.** This shows how our trained solver reacts to the rising curriculum produced by its co-evolving generator.

The key signal is the **strong-against-weak gap** (SWG): if GPT-5.4 improves relative to the fixed base solver, the questions are becoming harder for the base model while remaining answerable by a stronger solver.

The curve shows this rising-curriculum behavior. From iteration 0 to 30, accuracy drops, but the strong-against-weak gap more than doubles. From iteration 30 to 90, both solver accuracies rise, while GPT-5.4 rises faster and the gap widens further. Meanwhile, the INFUSER solver stays above the base model and tracks GPT-5.4-mini, indicating that the co-evolving solver is learning from the generator's increasingly useful curriculum. This suggests that the added difficulty reflects genuine reasoning challenge rather than invalid questions.

### Ablations

The ablations isolate the key design choices:

- A fixed generator trails INFUSER, showing that co-evolution matters.
- A larger frozen 32B generator helps on some knowledge-heavy domains, but INFUSER's 8B co-evolving generator remains stronger on math and coding.
- RLVR on the dev-set (Dev-only) improves the base model, but is narrower than influence-guided document generation.
- Optimizer-aware influence and DuGRPO normalization are important for stable generator learning.

The core pattern is consistent: documents provide knowledge, the dev-set provides direction, and influence-guided RL for the generator connects the two.

## Extensions

The main experiments start from pretrained Qwen3 base models and use only document-grounded self-evolution. The paper also tests two extensions that ask whether the same idea survives more realistic or mixed training settings.

<figure>
  <div style="display:grid;grid-template-columns:minmax(0,1.45fr) minmax(0,.85fr);gap:18px;align-items:start">
    <img src="/images/infuser/olmo-sft-radars.png" alt="OLMo instruction-finetuned anchor extension">
    <img src="/images/infuser/hybrid-length-accuracy-scatter.png" alt="Hybrid INFUSER math accuracy versus response length">
  </div>
  <figcaption>Left: INFUSER on OLMo-3-7B-Instruct-SFT, grouped by benchmark family. Right: math-and-physics accuracy tracks evaluation-time response length across science-only and math-RLVR-augmented INFUSER seeds.</figcaption>
</figure>

### Instruction-Finetuned Anchor

INFUSER still helps when the anchor has already been instruction-finetuned. On OLMo-3-7B-Instruct-SFT, INFUSER leads on 10 of 13 benchmarks compared with the instruction-finetuned base and a fixed-generator baseline, and it reaches the best overall average among the three variants. The gains are strongest on general reasoning, including +5.1 points on MMLU-Pro and +5.6 points on SuperGPQA, while the same aligned-domain pattern from the Qwen3 base experiments reappears.

### Math-RLVR Augmentation

INFUSER can also combine document-grounded science self-evolution with rule-verifiable math RLVR in a single loop. In the science-only setting, math performance is seed-sensitive because different seeds learn different response-length regimes. Adding verifiable math RLVR anchors long-CoT behavior: the math-and-physics cross-seed standard deviation drops from <span style="color:#1a6faf;font-weight:700">2.80</span> to <span style="color:#b30000;font-weight:700">0.48</span> percentage points, while the mean rises from <span style="color:#1a6faf;font-weight:700">31.49</span> to <span style="color:#b30000;font-weight:700">32.52</span>.

## More Than Best-of-k

![Pass@k curves](/images/infuser/qw8bb-passk-curves.png)

Pass@k results test whether INFUSER merely sharpens its most likely answer or broadens the support of correct reasoning paths. On several math benchmarks, INFUSER remains above the base model across many sampled attempts. This indicates that the model is not only getting better at selecting one lucky answer. Its sampled reasoning distribution has improved.

## Conclusion

INFUSER reframes self-evolution as influence-guided curriculum learning. The generator is rewarded for examples whose optimizer-aware gradient direction helps the solver on the dev-set. The solver trains on those examples. Because both roles evolve together, the system can turn unstructured documents into a moving curriculum matched to the current model.

<div class="infuser-tldr">
<p><strong>The broader lesson:</strong> self-generated data should be judged by training utility, not by surface difficulty alone.</p>
</div>

## Citation

<pre style="background:#f7f8fa;color:#1f2937;border:1px solid rgba(0,0,0,.12);border-radius:6px;padding:16px 18px;line-height:1.55;box-shadow:0 1px 2px rgba(0,0,0,.04);overflow:auto"><code>@article{chen2026infuser,
  title={INFUSER: Influence-Guided Self-Evolution Improves Reasoning},
  author={Chen, Siyu and Lu, Miao and Wu, Beining and Sheen, Heejune and Zhang, Fengzhuo and Li, Shuangning and Li, Zhiyuan and Blanchet, Jose and Wang, Tianhao and Yang, Zhuoran},
  journal={Manuscript},
  year={2026}
}</code></pre>
