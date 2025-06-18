---
title: "Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders"
date: 2024-03-25
lastmod: 2024-03-25
author: "Siyu Chen, Heejune Sheen, Xuyuan Xiong, Tianhao Wang, Zhuoran Yang"
cover: "/images/taming-sae-gba/bias_adap_illus.png"
categories:
  - "Research Blog"
tags:
  - "LLM"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "Feature Recovery"
draft: false
readingTime: 15
---

<script>
  window.location.href = "/taming-sae-gba/index.html";
</script>

<noscript>
  <meta http-equiv="refresh" content="0;url=/taming-sae-gba/index.html">
  <p>Please click <a href="/taming-sae-gba/index.html">here</a> to view the paper page.</p>
</noscript>

<!-- Paper title -->
<p><strong style="font-size: 0.8em; letter-spacing: 1px;">Paper:</strong> Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders</p>

<!-- Authors Section -->
<p><strong style="font-size: 0.8em; letter-spacing: 1px;">AUTHORS:</strong>
<small>
Siyu Chen*, Heejune Sheen*, Xuyuan Xiong†, Tianhao Wang§, and Zhuoran Yang*
</small></p>

<p><strong style="font-size: 0.8em; letter-spacing:1px;">AFFILIATIONS.</strong>
<small>
*Department of Statistics and Data Science, Yale University<br>
†Antai College of Economics and Management, Shanghai Jiao Tong University<br>
§Toyota Technological Institute at Chicago
</small></p>

<p><strong style="font-size: 0.8em; letter-spacing:1px;">LINKS.</strong>
<small>
<strong>ArXiv:</strong> <a href="https://arxiv.org/abs/<ARXIV PAPER ID>" target="_blank">arxiv.org/abs/<ARXIV PAPER ID></a>,
<strong>Github:</strong> <a href="https://github.com/FFishy-git/TamingSAE_GBA/tree/main/" target="_blank">https://github.com/FFishy-git/TamingSAE_GBA</a>
</small></p>

## Abstract

We address the challenge of theoretically grounded feature recovery using Sparse Autoencoders (SAEs) for interpreting Large Language Models. Current SAE training methods lack mathematical guarantees and face issues like hyperparameter sensitivity. We propose a statistical framework for feature recovery that models polysemantic features as sparse mixtures of monosemantic concepts. Based on this, we develop a "bias adaptation" SAE training algorithm that dynamically adjusts network biases for optimal sparsity. We **prove that this algorithm correctly recovers all monosemantic features** under our statistical model. Our improved variant, Group Bias Adaptation (GBA), **outperforms existing methods on LLMs up to 1.5 billion parameters** in terms of sparsity-loss trade-off and feature consistency. This work provides the first SAE algorithm with theoretical recovery guarantees, advancing interpretable and trustworthy AI through enhanced mechanistic understanding.

## Key Contributions

1. **A novel statistical framework** that rigorously formalizes feature recovery by modeling polysemantic features as sparse combinations of underlying monosemantic concepts, and establishes a precise notion of feature identifiability.

2. **An innovative SAE training algorithm**, **Group Bias Adaptation (GBA)**, which adaptively adjusts neural network bias parameters to enforce optimal activation sparsity, allowing distinct groups of neurons to target different activation frequencies.

3. **The first theoretical guarantee** proving that SAE training algorithm can provably recover all monosemantic features when the input data is sampled from our proposed statistical model.

## Results

### Algorithm Overview

![Algorithm Overview](/images/taming-sae-gba/algorithm.png)

### Performance Comparison

![Loss vs Sparsity](/images/taming-sae-gba/loss_sparsity.png)

### Feature Consistency

![Feature Consistency](/images/taming-sae-gba/consistency.png)

### Group Ablation Study

![Group Ablation](/images/taming-sae-gba/group_ablation.png)

### Bias Adaptation Illustration

![Bias Adaptation](/images/taming-sae-gba/bias_adap_illus.png)

### Selectivity Analysis

![Selectivity Analysis](/images/taming-sae-gba/selectivity_analysis.png)

### Z-Score Analysis

![Z-Score Analysis 1](/images/taming-sae-gba/zscore_1.png)
![Z-Score Analysis 2](/images/taming-sae-gba/zscore_2.png) 