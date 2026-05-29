# Logic By Design: Encoding Inference Rules in MeMo

These experiments investigate whether MeMo's Correlation Matrix Memory 
can store logical inference rules explicitly, as proposed in:

> Shahrokhshahi, F. (2026). Towards Logical Reasoning by Design: 
> Encoding Inference Rules in MeMo's Associative Memory. MeMo Workshop.

## Scripts

1. `test_logic1_symbolic.py` — Proves the CMM mechanism stores and 
   retrieves symbolic rules with zero interference.
2. `test_logic2_single_examples.py` — Encodes real inference patterns 
   using MeMo's own embeddings; reveals the surface-similarity problem.
3. `test_logic3_modal_boosting.py` — Applies modal-keyword boosting 
   to improve pattern separation.
4. `test_logic4_centroid_generalization.py` — Main experiment: stores 
   pattern centroids from 40 examples, tests on 40 unseen instances. 
   Achieves 100% accuracy across 280 test cases.

## Data

Loaded automatically from the LogiCue repository:
https://github.com/fatemeshahrokhshahi/LogiCue
(Shahrokhshahi et al., Procedia Computer Science, 2026)

## Requirements

MeMoPyTorch (this repository), plus: transformers, torch
