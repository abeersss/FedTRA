# FedTRA — Reproducible Code

Repo: https://github.com/abeersss/FedTRA

Code for the paper *"FedTRA: Root-Free Broad-Spectrum Robust Aggregation for
Poisoning-Resilient Federated Intrusion Detection in IoT Networks"*
(Dr. Abeer Alshammari, CTU).

Pure NumPy/pandas — **no GPU, no deep-learning framework** required. Runs on a laptop.

## 1. Install
```
pip install -r requirements.txt
```

## 2. Get the datasets (public)
- **NSL-KDD**: `KDDTrain+.txt`, `KDDTest+.txt` (Kaggle `hassan06/nslkdd`).
- **N-BaIoT**: from Kaggle `mkashifn/nbaiot-dataset`, one device's CSVs, e.g.
  `1.benign.csv`, `1.mirai.udp.csv`, `2.gafgyt.tcp.csv`.

## 3. Reproduce results
```
# NSL-KDD  (Table I, Figs 1-2)  ~ a few minutes
python reproduce_nslkdd.py --train KDDTrain+.txt --test KDDTest+.txt

# N-BaIoT  (Table II, Fig 3)
python reproduce_nbaiot.py --benign 1.benign.csv --mirai 1.mirai.udp.csv --gafgyt 2.gafgyt.tcp.csv

# Figures from the CSVs
python make_figures.py
```
Outputs land in `results/metrics_nslkdd.csv`, `results/metrics_nbaiot.csv`, and `fig_*.png`.

## 4. Files
| File | Purpose |
|------|---------|
| `fl_experiment.py` | Core library: data loaders, NumPy MLP, FL simulation, the six aggregation rules (FedAvg, Median, Trimmed-Mean, Multi-Krum, FLTrust, **FedTRA**), and the three attacks (label-flip, sign-flip, backdoor). |
| `reproduce_nslkdd.py` | Full NSL-KDD grid (6 rules x 3 attacks x 5 malicious fractions x 3 seeds). |
| `reproduce_nbaiot.py` | Same grid on N-BaIoT. |
| `make_figures.py` | Regenerate the paper figures. |
| `requirements.txt` | numpy, pandas, matplotlib. |

## 5. FedTRA in one paragraph
FedTRA aggregates client updates by (1) **norm-bounding** each update to the cohort-median
norm (cancels scaling / boosted model-poisoning); (2) scoring each client with a
**multi-signal temporal reputation** (directional deviation, magnitude anomaly, kNN isolation,
smoothed by exponential momentum) to gate out blatant attackers; and (3) taking the
**coordinate-wise median over the trusted survivors** (suppresses stealthy backdoors).
It requires **no clean server-side root dataset**. See `FedTRA` and `run()` in `fl_experiment.py`.

## Config (defaults, editable in the reproduce scripts)
20 clients, non-IID Dirichlet(alpha=0.5), 1-hidden-layer MLP (32 units), 2 local epochs,
batch 64, lr 0.1, 25 rounds (NSL-KDD) / 20 (N-BaIoT), averaged over 3 seeds.
