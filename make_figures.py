#!/usr/bin/env python3
"""Regenerate the paper figures from results/*.csv"""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, os
rules=["fedavg","median","trimmed","krum","fltrust","fedtra"]
labels={"fedavg":"FedAvg","median":"Median","trimmed":"Trimmed-Mean","krum":"Multi-Krum","fltrust":"FLTrust","fedtra":"FedTRA (ours)"}
markers={"fedavg":"o","median":"s","trimmed":"^","krum":"D","fltrust":"v","fedtra":"*"}
def plot(csv,attack,metric,ylabel,fname,title):
    if not os.path.exists(csv): print("skip",csv); return
    df=pd.read_csv(csv); plt.figure(figsize=(4.6,3.4))
    for r in rules:
        sub=df[(df.attack==attack)&(df.rule==r)].sort_values("mal_frac")
        if len(sub)==0: continue
        lw=2.4 if r=="fedtra" else 1.4
        plt.plot(sub.mal_frac.values*100,sub[metric].values,marker=markers[r],label=labels[r],linewidth=lw,markersize=8 if r=="fedtra" else 6)
    plt.xlabel("Malicious clients (%)"); plt.ylabel(ylabel); plt.title(title,fontsize=10)
    plt.grid(True,alpha=0.3); plt.legend(fontsize=7,ncol=2); plt.tight_layout(); plt.savefig(fname,dpi=200); plt.close(); print("saved",fname)
plot("results/metrics_nslkdd.csv","backdoor","asr","Attack Success Rate","fig_backdoor_asr.png","(a) NSL-KDD Backdoor: ASR")
plot("results/metrics_nslkdd.csv","signflip","acc","Test Accuracy","fig_signflip_acc.png","(b) NSL-KDD Model-Poisoning: Accuracy")
plot("results/metrics_nbaiot.csv","signflip","acc","Test Accuracy","fig_nbaiot_signflip.png","N-BaIoT Model-Poisoning: Accuracy")
