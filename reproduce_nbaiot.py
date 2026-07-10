#!/usr/bin/env python3
"""Reproduce the N-BaIoT results (Table II + Fig 3).
Usage (point to your uploaded CSVs):
    python reproduce_nbaiot.py --benign 1.benign.csv --mirai 1.mirai.udp.csv --gafgyt 2.gafgyt.tcp.csv
Outputs: results/metrics_nbaiot.csv
N-BaIoT CSVs: 115 numeric features, no header preprocessing needed."""
import argparse, os, importlib.util, numpy as np, pandas as pd
spec=importlib.util.spec_from_file_location("fx","fl_experiment.py"); fx=importlib.util.module_from_spec(spec); spec.loader.exec_module(fx)

def load_nbaiot(benign,mirai,gafgyt,seed=0):
    ben=pd.read_csv(benign); mir=pd.read_csv(mirai,nrows=30000); gaf=pd.read_csv(gafgyt)
    def samp(df,n,s): return df.sample(n=min(n,len(df)),random_state=s).values.astype(np.float32)
    Xben=samp(ben,10000,1); Xatt=np.vstack([samp(mir,7000,2),samp(gaf,3000,3)])
    X=np.vstack([Xben,Xatt]).astype(np.float32)
    y=np.concatenate([np.zeros(len(Xben),int),np.ones(len(Xatt),int)])
    p=np.random.default_rng(seed).permutation(len(X)); X,y=X[p],y[p]
    tr,te=[],[]
    for c in [0,1]:
        idx=np.where(y==c)[0]; np.random.default_rng(c+5).shuffle(idx); k=int(0.7*len(idx))
        tr+=idx[:k].tolist(); te+=idx[k:].tolist()
    tr,te=np.array(tr),np.array(te); Xtr,ytr,Xte,yte=X[tr],y[tr],X[te],y[te]
    mn=Xtr.min(0); mx=Xtr.max(0); rng=np.where((mx-mn)==0,1.0,mx-mn)
    Xtr=np.clip((Xtr-mn)/rng,0,1).astype(np.float32); Xte=np.clip((Xte-mn)/rng,0,1).astype(np.float32)
    return Xtr,ytr,Xte,yte

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benign",required=True); ap.add_argument("--mirai",required=True); ap.add_argument("--gafgyt",required=True)
    ap.add_argument("--rounds",type=int,default=20); ap.add_argument("--seeds",type=int,default=3)
    a=ap.parse_args()
    Xtr,ytr,Xte,yte=load_nbaiot(a.benign,a.mirai,a.gafgyt)
    print(f"N-BaIoT: train {Xtr.shape} test {Xte.shape} | prevalence {ytr.mean():.2f}")
    base=dict(data=(Xtr,ytr,Xte,yte),n_clients=20,d_hid=32,alpha=0.5,rounds=a.rounds,
              epochs=2,lr=0.1,server_lr=1.0,batch=64,beta=0.7)
    rules=["fedavg","median","trimmed","krum","fltrust","fedtra"]
    fracs=[0.0,0.1,0.2,0.3,0.4]; attacks=["label_flip","signflip","backdoor"]
    os.makedirs("results",exist_ok=True); rows=[]
    for attack in attacks:
        for mf in fracs:
            for rule in rules:
                accs=[];asrs=[]
                for sd in range(a.seeds):
                    r=fx.run(dict(base,rule=rule,mal_frac=mf,attack=attack,seed=sd))
                    accs.append(r["acc"])
                    if r["asr"] is not None: asrs.append(r["asr"])
                rows.append(dict(attack=attack,mal_frac=mf,rule=rule,acc=round(np.mean(accs),4),
                    asr=(round(np.mean(asrs),4) if asrs else None)))
                print(f"{attack:11s} mf={mf:.1f} {rule:8s} acc={np.mean(accs):.4f} asr={'' if not asrs else round(np.mean(asrs),4)}")
    pd.DataFrame(rows).to_csv("results/metrics_nbaiot.csv",index=False)
    print("Saved results/metrics_nbaiot.csv")

if __name__=="__main__": main()
