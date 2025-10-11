# Any-SSR
This is the official code for Any-SSR [Analytic Subspace Routing: How Recursive Least Squares Works in Continual Learning of Large Language Model](https://arxiv.org/abs/2503.13575)

# Environment
We recommend using the [Anaconda](https://anaconda.org/) to install the development environment.

```bash
git clone --depth=1 git@https://github.com/ZHUANGHP/Any-SSR.git

cd Any-SSR
conda env create -n anyssr -f environment.yaml
conda activate anyssr
```
## Quick Start
All the data after processing can be downloaded from [Trace Benchmark](https://drive.google.com/file/d/1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV/view)

After finishing dataset downloading, use
```bash
python train_router_ana_continual.py
```
to train the router weight recursively, then use
```bash
python eval_router_ana.py
```
to generate routing accuracy.

## Lora Model Training
```bash
bash scripts/train_lora_20Minuten.sh
```
## Evaluate
```bash
bash scripts/inference.sh
```

# From new branch called Analytic Continual Learning
This is the first LLM member from the continual learning branch: [Analytic Continual Learning](https://github.com/ZHUANGHP/Analytic-continual-learning). We have published over 20 papers in this branch (check [My Scholar](https://scholar.google.com.sg/citations?user=vCXxuLkAAAAJ&hl=en))!
