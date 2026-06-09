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
math: false
toc: false
readingTime: 12
---

<script>
  window.location.href = "/infuser/";
</script>

<noscript>
  <meta http-equiv="refresh" content="0;url=/infuser/">
  <p>Please click <a href="/infuser/">here</a> to view the INFUSER blog post.</p>
</noscript>

<strong style="font-size: 0.85em; letter-spacing: 1px; color: #3B9DD9;">AUTHORS:</strong>
<small>Siyu Chen, Miao Lu, Beining Wu, Heejune Sheen, Fengzhuo Zhang, Shuangning Li, Zhiyuan Li, Jose Blanchet, Tianhao Wang, and Zhuoran Yang</small>

<strong style="font-size: 0.85em; letter-spacing: 1px; color: #3B9DD9;">LINKS:</strong>
<span style="font-size: 0.9em;">
<strong style="color: #D98C3B;">arXiv:</strong> <a href="https://arxiv.org/abs/2606.09052" target="_blank"><strong>Paper</strong></a> &nbsp;|&nbsp;
<strong style="color: #4CAF7D;">GitHub:</strong> <a href="https://github.com/FFishy-git/INFUSER" target="_blank"><strong>Code</strong></a>
</span>

INFUSER turns self-evolution into influence-guided curriculum learning. A co-evolving generator drafts questions and reference answers from unstructured documents, while a solver trains on them with correctness rewards.
