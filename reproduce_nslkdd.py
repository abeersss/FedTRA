#!/usr/bin/env python3
"""Reproduce the NSL-KDD results (Table I + Figs 1-2).
Usage:
    python reproduce_nslkdd.py --train KDDTrain+.txt --test KDDTest+.txt
Outputs: results/metrics_nslkdd.csv
Runtime: a few minutes on a laptop (CPU only, numpy)."""
import argparse, os, importlib.util, numpy as np, pandas as pd
spec=importlib.util.spec_from_file_location("fx","fl_experiment.py"); fx=importlib.util.module_from_spec(spec); spec.loader.exec_module(fx)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",required=True); ap.add_argument("--test",required=True)
    ap.add_argument("--rounds",type=int,default=25); ap.add_argument("--seeds",type=int,default=3)
    ap.add_argument("--n_sub_train",type=int,default=20000); ap.add_argument("--n_sub_test",type=int,default=10000)
    a=ap.parse_args()
    Xtr,ytr,Xte,yte=fx.load_nsl(a.train,a.test)
    Xtr,ytr=fx.subsample(Xtr,ytr,a.n_sub_train); Xte,yte=fx.subsample(Xte,yte,a.n_sub_test,7)
    print(f"NSL-KDD: train {Xtr.shape} test {Xte.shape}")
    base=dict(data=(Xtr,ytr,Xte,yte),n_clients=20,d_hid=32,alpha=0.5,rounds=a.rounds,
              epochs=2,lr=0.1,server_lr=1.0,batch=64,beta=0.7)
    rules=["fedavg","median","trimmed","krum","fltrust","fedtra"]
    fracs=[0.0,0.1,0.2,0.3,0.4]; attacks=["label_flip","signflip","backdoor"]
    os.makedirs("results",exist_ok=True); rows=[]
    for attack in attacks:
        for mf in fracs:
            for rule in rules:
                accs=[];f1s=[];asrs=[]
                for sd in range(a.seeds):
                    r=fx.run(dict(base,rule=rule,mal_frac=mf,attack=attack,seed=sd))
                    accs.append(r["acc"]);f1s.append(r["f1"])
                    if r["asr"] is not None: asrs.append(r["asr"])
                rows.append(dict(attack=attack,mal_frac=mf,rule=rule,
                    acc=round(np.mean(accs),4),acc_std=round(np.std(accs),4),f1=round(np.mean(f1s),4),
                    asr=(round(np.mean(asrs),4) if asrs else None)))
                print(f"{attack:11s} mf={mf:.1f} {rule:8s} acc={np.mean(accs):.4f} asr={'' if not asrs else round(np.mean(asrs),4)}")
    pd.DataFrame(rows).to_csv("results/metrics_nslkdd.csv",index=False)
    print("Saved results/metrics_nslkdd.csv")

if __name__=="__main__": main()
