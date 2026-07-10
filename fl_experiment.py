#!/usr/bin/env python3
"""FedTRA: Temporal-Reputation Aggregation for poisoning-robust Federated
Learning in IoT intrusion detection. NumPy-only.
Baselines: FedAvg, Median, Trimmed-Mean, Multi-Krum, FLTrust. Proposed: FedTRA.
Attacks: label_flip, signflip (boosted model poisoning), backdoor."""
import os, json, argparse
import numpy as np, pandas as pd
RNG_SEED=42
NSL_COLUMNS=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
 "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
 "root_shell","su_attempted","num_root","num_file_creations","num_shells",
 "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
 "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
 "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
 "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
 "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
 "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
 "label","difficulty"]
CAT_COLS=["protocol_type","service","flag"]

def load_nsl(train_path,test_path):
    def rd(p):
        df=pd.read_csv(p,header=None)
        if df.shape[1]==43: df.columns=NSL_COLUMNS
        elif df.shape[1]==42: df.columns=NSL_COLUMNS[:-1]
        else: raise ValueError(f"cols {df.shape[1]}")
        return df
    tr=rd(train_path); te=rd(test_path)
    for df in (tr,te):
        if "difficulty" in df.columns: df.drop(columns=["difficulty"],inplace=True)
    ytr=(tr["label"]!="normal").astype(int).values; yte=(te["label"]!="normal").astype(int).values
    tr=tr.drop(columns=["label"]); te=te.drop(columns=["label"])
    both=pd.get_dummies(pd.concat([tr,te],ignore_index=True),columns=CAT_COLS)
    Xtr=both.iloc[:len(tr)].values.astype(np.float32); Xte=both.iloc[len(tr):].values.astype(np.float32)
    mn=Xtr.min(0); mx=Xtr.max(0); rng=np.where((mx-mn)==0,1.0,mx-mn)
    return (Xtr-mn)/rng,ytr,(Xte-mn)/rng,yte

def subsample(X,y,n,seed=RNG_SEED):
    if n is None or n>=len(X): return X,y
    idx=np.random.default_rng(seed).choice(len(X),n,replace=False); return X[idx],y[idx]

def dirichlet_partition(y,n_clients,alpha,seed=RNG_SEED):
    rng=np.random.default_rng(seed); nc=int(y.max()+1)
    idxc=[np.where(y==c)[0] for c in range(nc)]
    for a in idxc: rng.shuffle(a)
    ci=[[] for _ in range(n_clients)]
    for c in range(nc):
        p=rng.dirichlet(alpha*np.ones(n_clients))
        cuts=(np.cumsum(p)*len(idxc[c])).astype(int)[:-1]
        for i,s in enumerate(np.split(idxc[c],cuts)): ci[i].extend(s.tolist())
    return [np.array(sorted(c)) for c in ci]

def init_theta(d_in,dh,seed=RNG_SEED):
    r=np.random.default_rng(seed)
    W1=(r.standard_normal((d_in,dh))*np.sqrt(2/d_in)).astype(np.float32)
    W2=(r.standard_normal((dh,2))*np.sqrt(2/dh)).astype(np.float32)
    return np.concatenate([W1.ravel(),np.zeros(dh,np.float32),W2.ravel(),np.zeros(2,np.float32)])

def unpack(t,d_in,dh):
    i=0; W1=t[i:i+d_in*dh].reshape(d_in,dh); i+=d_in*dh
    b1=t[i:i+dh]; i+=dh; W2=t[i:i+dh*2].reshape(dh,2); i+=dh*2; b2=t[i:i+2]
    return W1,b1,W2,b2

def local_train(t0,X,y,dh,ep,lr,bs,seed):
    d_in=X.shape[1]; W1,b1,W2,b2=[a.copy() for a in unpack(t0,d_in,dh)]
    rng=np.random.default_rng(seed); n=len(X)
    for _ in range(ep):
        perm=rng.permutation(n)
        for s in range(0,n,bs):
            bi=perm[s:s+bs]
            if len(bi)==0: continue
            xb=X[bi]; yb=y[bi]; nb=len(bi)
            a1=np.maximum(xb@W1+b1,0); z2=a1@W2+b2
            z2=z2-z2.max(1,keepdims=True); e=np.exp(z2); p=e/e.sum(1,keepdims=True)
            yh=np.zeros_like(p); yh[np.arange(nb),yb]=1; dz2=(p-yh)/nb
            gW2=a1.T@dz2; gb2=dz2.sum(0); dz1=(dz2@W2.T)*(a1>0); gW1=xb.T@dz1; gb1=dz1.sum(0)
            W1-=lr*gW1; b1-=lr*gb1; W2-=lr*gW2; b2-=lr*gb2
    return np.concatenate([W1.ravel(),b1,W2.ravel(),b2])

def predict(t,d_in,dh,X):
    W1,b1,W2,b2=unpack(t,d_in,dh); a1=np.maximum(X@W1+b1,0); z2=a1@W2+b2; return z2.argmax(1)

BACKDOOR_FEATURES=[4,5]; BACKDOOR_TARGET=0
def apply_label_flip(y): return 1-y
def add_backdoor(X,y,frac,seed):
    rng=np.random.default_rng(seed); Xb=X.copy(); yb=y.copy(); n=int(frac*len(X))
    if n==0: return Xb,yb
    idx=rng.choice(len(X),n,replace=False); Xb[np.ix_(idx,BACKDOOR_FEATURES)]=0.99; yb[idx]=BACKDOOR_TARGET
    return Xb,yb

def agg_fedavg(u): return u.mean(0)
def agg_median(u): return np.median(u,0)
def agg_trimmed(u,beta=0.2):
    n=len(u); k=int(beta*n); s=np.sort(u,0); return s[k:n-k].mean(0) if n-2*k>0 else s.mean(0)
def _pd(u):
    n=len(u); D=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n): D[i,j]=D[j,i]=np.sum((u[i]-u[j])**2)
    return D
def agg_multikrum(u,f=None):
    n=len(u); f=max(1,int(0.2*n)) if f is None else f; D=_pd(u); m=max(n-f-2,1)
    sc=np.array([np.sort(D[i])[1:1+m].sum() for i in range(n)]); k=max(1,n-f)
    return u[np.argsort(sc)[:k]].mean(0)
def agg_fltrust(u,g0):
    ref=g0/(np.linalg.norm(g0)+1e-9); nr=np.linalg.norm(g0)
    ts=np.array([max(np.dot(x,ref)/(np.linalg.norm(x)+1e-9),0.0) for x in u])
    if ts.sum()==0: return np.zeros_like(u[0])
    sc=np.array([ts[i]*(nr/(np.linalg.norm(u[i])+1e-9))*u[i] for i in range(len(u))])
    return sc.sum(0)/ts.sum()
def robust_z(x):
    med=np.median(x); mad=np.median(np.abs(x-med))+1e-9; return (x-med)/(1.4826*mad)

class FedTRA:
    def __init__(self,n,beta=0.7,kappa=4.0,lam=1.0,knn=None,disable_rep=False):
        self.rep=np.ones(n)*0.5; self.beta=beta; self.kappa=kappa; self.lam=lam; self.knn=knn; self.t=0; self.disable_rep=disable_rep
    def aggregate(self,u,ids):
        n=len(u); knn=self.knn or max(1,n//2)
        # norm-bounding: clip each update to the cohort-median norm to neutralize
        # scaling / boosted model-poisoning; direction is preserved for the signals
        nr=np.linalg.norm(u,axis=1); med=np.median(nr)+1e-12
        u=np.array([u[i]*min(1.0,med/(nr[i]+1e-12)) for i in range(n)])
        if self.t==0: anchor=agg_trimmed(u,0.2)
        else:
            pr=np.array([self.rep[c] for c in ids]); wa=np.exp(self.kappa*pr); wa=wa/wa.sum(); anchor=(u*wa[:,None]).sum(0)
        a=anchor/(np.linalg.norm(anchor)+1e-9); norms=np.linalg.norm(u,axis=1)
        cos=np.array([np.dot(x,a)/(np.linalg.norm(x)+1e-9) for x in u]); s_dir=1-cos
        s_mag=np.abs(robust_z(norms)); D=np.sqrt(np.maximum(_pd(u),0))
        iso=np.array([np.sort(D[i])[1:1+knn].mean() for i in range(n)]); s_iso=np.clip(robust_z(iso),0,None)
        susp=1/(1+np.exp(-(robust_z(s_dir)+s_mag+s_iso))); inst=1-susp
        for k,c in enumerate(ids): self.rep[c]=self.beta*self.rep[c]+(1-self.beta)*inst[k]
        r=np.array([self.rep[c] for c in ids])
        # fuse temporal reputation with instantaneous trust: blatant single-round
        # anomalies (e.g. boosted sign-flip) are suppressed even before reputation converges
        score=np.sqrt(np.clip(r,1e-6,None)*np.clip(inst,1e-6,None))
        thr=np.median(score)-self.lam*(np.median(np.abs(score-np.median(score)))*1.4826)
        gate=(score>=thr)
        if self.disable_rep: gate=np.ones(n,bool)
        if gate.sum()==0: gate=np.ones(n,bool)
        self.t+=1
        # Hybrid aggregation: reputation gating removes magnitude/direction attackers;
        # coordinate-wise trimmed mean over survivors removes stealthy backdoor coordinates.
        surv=u[gate]
        agg=agg_median(surv) if len(surv)>=4 else surv.mean(0)
        return agg,r

def evaluate(t,d_in,dh,X,y):
    pred=predict(t,d_in,dh,X); acc=(pred==y).mean(); f1=[]
    for c in [0,1]:
        tp=((pred==c)&(y==c)).sum(); fp=((pred==c)&(y!=c)).sum(); fn=((pred!=c)&(y==c)).sum()
        pr=tp/(tp+fp+1e-9); rc=tp/(tp+fn+1e-9); f1.append(2*pr*rc/(pr+rc+1e-9))
    return float(acc),float(np.mean(f1))
def backdoor_asr(t,d_in,dh,X,y):
    # ASR among attacks the model correctly detects (isolates backdoor-induced flips)
    atk=np.where(y==1)[0]
    if len(atk)==0: return 0.0
    pc=predict(t,d_in,dh,X[atk])
    corr=atk[pc==1]
    if len(corr)==0: return 0.0
    Xt=X[corr].copy(); Xt[:,BACKDOOR_FEATURES]=0.99
    return float((predict(t,d_in,dh,Xt)==BACKDOOR_TARGET).mean())

def run(cfg):
    seed=cfg.get("seed",0); np.random.seed(RNG_SEED+seed)
    Xtr,ytr,Xte,yte=cfg["data"]; n=cfg["n_clients"]; dh=cfg["d_hid"]; d_in=Xtr.shape[1]
    parts=dirichlet_partition(ytr,n,cfg["alpha"],seed=RNG_SEED+seed)
    rng=np.random.default_rng(1+seed); root=rng.choice(len(Xtr),100,replace=False); Xr,yr=Xtr[root],ytr[root]
    mal=set(rng.choice(n,int(cfg["mal_frac"]*n),replace=False).tolist())
    theta=init_theta(d_in,dh); ft=FedTRA(n,beta=cfg.get("beta",0.7),disable_rep=cfg.get("disable_rep",False))
    for rd in range(cfg["rounds"]):
        U=[]; ids=[]
        for c in range(n):
            Xi,yi=Xtr[parts[c]],ytr[parts[c]]
            if len(Xi)==0: continue
            if c in mal:
                if cfg["attack"]=="label_flip": yi=apply_label_flip(yi)
                elif cfg["attack"]=="backdoor": Xi,yi=add_backdoor(Xi,yi,0.5,rd*97+c)
            upd=local_train(theta,Xi,yi,dh,cfg["epochs"],cfg["lr"],cfg["batch"],rd*1000+c)-theta
            if c in mal and cfg["attack"]=="signflip": upd=-cfg.get("boost",10.0)*upd
            U.append(upd); ids.append(c)
        U=np.array(U); rule=cfg["rule"]
        if rule=="fedavg": agg=agg_fedavg(U)
        elif rule=="median": agg=agg_median(U)
        elif rule=="trimmed": agg=agg_trimmed(U)
        elif rule=="krum": agg=agg_multikrum(U)
        elif rule=="fltrust":
            g0=local_train(theta,Xr,yr,dh,cfg["epochs"],cfg["lr"],cfg["batch"],rd)-theta; agg=agg_fltrust(U,g0)
        elif rule=="fedtra": agg,_=ft.aggregate(U,ids)
        theta=theta+cfg["server_lr"]*agg
    acc,f1=evaluate(theta,d_in,dh,Xte,yte)
    asr=backdoor_asr(theta,d_in,dh,Xte,yte) if cfg["attack"]=="backdoor" else None
    return {"acc":acc,"f1":f1,"asr":asr}
