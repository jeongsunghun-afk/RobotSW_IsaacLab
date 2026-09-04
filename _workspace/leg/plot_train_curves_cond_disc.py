# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""train_curves/*.npz 캐시로 판별기·손실·보상 곡선 4장을 그린다 (figures/train_*.png).

Run: python _workspace/leg/plot_train_curves_cond_disc.py
"""
import numpy as np, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SP="/home/lgb/IsaacLab-6.0/reports/leg_imitation/_comparisons/conditional_discriminator/metrics/train_curves"
OUT="/home/lgb/IsaacLab-6.0/reports/leg_imitation/_comparisons/conditional_discriminator/figures"
os.makedirs(OUT, exist_ok=True)
SEAM=16700
def load(f):
    z=np.load(os.path.join(SP,f)); d={}
    for k in z.files:
        if k.endswith("__val"): t=k[:-5]; d[t]=(z[t+"__step"], z[k])
    return d
base,A,B1,B2=load("base.npz"),load("A.npz"),load("B1.npz"),load("B2.npz")
B={}
for t in B1:
    s1,v1=B1[t]; m=s1<SEAM; s2,v2=B2[t]
    B[t]=(np.concatenate([s1[m],s2]), np.concatenate([v1[m],v2]))
for r in (base,A,B):
    se,ve=r["Loss/disc_expert_output"]; sp,vp=r["Loss/disc_policy_output"]
    n=min(len(se),len(sp)); r["margin"]=(se[:n], ve[:n]-vp[:n])
def ma(v,w=500):
    c=np.cumsum(np.insert(v.astype(float),0,0.0)); out=np.empty(len(v))
    for i in range(len(v)):
        a=max(0,i-w+1); out[i]=(c[i+1]-c[a])/(i+1-a)
    return out
def smooth(step,val,seam=None,w=500):
    if seam is None: return ma(val,w)
    m=step<seam; return np.concatenate([ma(val[m],w), ma(val[~m],w)])
RUNS=[("baseline (mlp, uncond)",base,"0.55",None),("A' cond-mlp",A,"tab:blue",None),("B' cond+drail",B,"tab:orange",SEAM)]
def panel(ax,tag,title,ylabel,logy=False,hline=None,skip=()):
    for name,r,c,seam in RUNS:
        if name in skip or tag not in r: continue
        s,v=r[tag]
        ax.plot(s,v,color=c,alpha=0.18,lw=0.5)
        ax.plot(s,smooth(s,v,seam),color=c,lw=1.8,label=name)
    if hline is not None: ax.axhline(hline,ls=":",color="k",lw=1)
    ax.axvline(SEAM,ls="--",color="tab:orange",lw=0.9,alpha=0.7)
    if logy: ax.set_yscale("log")
    ax.set_title(title,fontsize=10); ax.set_xlabel("iteration"); ax.set_ylabel(ylabel,fontsize=9)
    ax.grid(alpha=0.25)
def fig4(specs,fname,legend_ax=0):
    f,axs=plt.subplots(2,2,figsize=(13,8))
    for ax,sp in zip(axs.ravel(),specs): panel(ax,*sp[:4],hline=sp[5],skip=sp[6])
    axs.ravel()[legend_ax].legend(fontsize=8)
    f.suptitle("cond-AMP discriminator — leg_imitation_tracking_rma (resume seam @16,700 dashed)",fontsize=11)
    f.tight_layout(); f.savefig(os.path.join(OUT,fname),dpi=150); plt.close(f)
fig4([("Loss/disc_expert_output","D(expert) = sigmoid(logit)","prob",False,None,0.5,()),
      ("Loss/disc_policy_output","D(policy) = sigmoid(logit)","prob",False,None,0.5,()),
      ("margin","margin = D(expert) - D(policy)","prob diff",False,None,0.0,()),
      ("Episode_Reward/amp_reward","Episode_Reward/amp_reward","reward / s",False,None,None,())],
     "train_disc_outputs.png")
fig4([("Loss/disc_expert_loss","disc_expert_loss (BCE)","loss",False,None,np.log(2),()),
      ("Loss/disc_policy_loss","disc_policy_loss (BCE)","loss",False,None,np.log(2),()),
      ("Loss/disc_total_loss","disc_total_loss = 0.5(e+p)+GP+reg","loss",False,None,np.log(2),()),
      ("Loss/disc_grad_penalty","disc_grad_penalty (drail = 0 by code path)","penalty",False,None,None,())],
     "train_disc_losses.png")
fig4([("Policy/mean_noise_std","policy mean_noise_std (log y)","sigma",True,None,None,()),
      ("Episode_Reward/lin_vel_reward","Episode_Reward/lin_vel_reward","reward / s",False,None,None,()),
      ("Loss/entropy","Loss/entropy","nats",False,None,None,()),
      ("Loss/learning_rate","Loss/learning_rate (adaptive, log y)","lr",True,None,None,())],
     "train_policy_stats.png")
# reward balance
f,axs=plt.subplots(1,3,figsize=(16,4.5))
for name,r,c,seam in RUNS:
    sl,vl=r["Episode_Reward/lin_vel_reward"]; sa,va=r["Episode_Reward/amp_reward"]
    idx=np.clip(np.searchsorted(sa,sl),0,len(sa)-1); ok=sa[idx]==sl
    s=sl[ok]; ratio=vl[ok]/np.maximum(va[idx][ok],1e-9)
    axs[0].plot(s,ratio,color=c,alpha=0.18,lw=0.5); axs[0].plot(s,smooth(s,ratio,seam),color=c,lw=1.8,label=name)
    axs[1].plot(sl,smooth(sl,vl,seam),color=c,lw=1.8,label=name)
    axs[2].plot(sa,smooth(sa,va,seam),color=c,lw=1.8,label=name)
for ax,t,y in zip(axs,["lin_vel_reward / amp_reward","lin_vel_reward (500-iter MA)","amp_reward (500-iter MA)"],["ratio","reward / s","reward / s"]):
    ax.axvline(SEAM,ls="--",color="tab:orange",lw=0.9,alpha=0.7); ax.set_title(t,fontsize=10)
    ax.set_xlabel("iteration"); ax.set_ylabel(y,fontsize=9); ax.grid(alpha=0.25)
axs[0].legend(fontsize=8)
f.suptitle("task vs style reward balance (both Episode_Reward, divided by max_episode_length_s=20)",fontsize=11)
f.tight_layout(); f.savefig(os.path.join(OUT,"train_reward_balance.png"),dpi=150); plt.close(f)
print("plots ->",OUT, os.listdir(OUT))
