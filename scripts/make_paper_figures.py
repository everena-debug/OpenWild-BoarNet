"""Regenerates the data figures of the revised manuscript (Figs. 2, 3, 5-11) from the saved results.
Run from anywhere:  python scripts/make_paper_figures.py
(Fig. 1 is a Graphviz diagram; Fig. 4 shows dataset sample images.)"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from sklearn.metrics import roc_curve, roc_auc_score
import os
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R=os.path.join(BASE,'results')+'/'
os.makedirs(os.path.join(BASE,'paper_figures'),exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'font.family':'DejaVu Sans'})
W=3.3
def save(n):
    plt.tight_layout(); plt.savefig(os.path.join(BASE,'paper_figures',f'{n}.png'),dpi=300); plt.close()
ps=pd.read_csv(R+'per_sample_scores_and_predictions.csv.gz')
tr=ps[(ps.source=='boar_main')&(ps.split=='train')]; vk=ps[(ps.source=='boar_main')&(ps.split=='val')]; vu=ps[(ps.source=='wild_main')&(ps.split=='val')]
tk=ps[(ps.source=='boar_main')&(ps.split=='test')]; tu=ps[(ps.source=='wild_main')&(ps.split=='test')]
P='score__vit_b16__yoloroi__pipeline_yolo_evt'; E='score__vit_b16__yoloroi__evt_proto'
TAU=float(pd.read_csv(R+'tables/T14_frozen_thresholds_from_validation.csv').query("method=='vit_b16__yoloroi__pipeline_yolo_evt' and protocol=='P1_val_known_TPR95'").tau.iloc[0])
print('tau',TAU)
# Fig2/3 YOLO curves
c=pd.read_csv(R+'tables/T12_yolo_training_curves_all_seeds.csv')
for cols,lab,name,yl in [(['train/box_loss','train/cls_loss','train/dfl_loss'],['Box loss','Classification loss','DFL loss'],'fig2_yolo_loss','Training loss'),
                         (['metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)'],['Precision','Recall','mAP@50','mAP@50-95'],'fig3_yolo_val','Validation score')]:
    plt.figure(figsize=(W,3.0))
    for col,l in zip(cols,lab):
        s=c.groupby('epoch')[col].agg(['mean','std'])
        plt.plot(s.index,s['mean'],label=l,lw=1.4); plt.fill_between(s.index,s['mean']-s['std'],s['mean']+s['std'],alpha=.25)
    plt.xlabel('Epoch'); plt.ylabel(yl); plt.grid(alpha=.3); plt.legend(frameon=False,fontsize=10,ncol=2,loc='upper center',bbox_to_anchor=(0.45,-0.25))
    save(name)
# Fig6 Weibull fit (ROI view, tail 10%)
k,lam=6.6372,0.7090
s=tr[E].values; s=s[s<1]; d=lam*(-np.log(1-s))**(1/k)
n=int(round(0.10*len(tr))); tail=np.sort(d)[-n:]
xs=np.linspace(tail.min()*0.95,1.0,300)
fig,ax=plt.subplots(figsize=(W,3.0))
ax.hist(tail,bins=30,density=True,alpha=.45,color='tab:blue',label='Tail distances')
ax.plot(xs,weibull_min.pdf(xs,k,0,lam),color='tab:blue',lw=1.5,label='Weibull PDF')
ax.set_xlabel('Cosine distance to prototype'); ax.set_ylabel('Density')
ax2=ax.twinx(); ax2.plot(xs,weibull_min.cdf(xs,k,0,lam),'--',color='tab:red',lw=1.5,label='Weibull CDF'); ax2.set_ylabel('CDF')
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,frameon=False,fontsize=10,ncol=2,loc='upper center',bbox_to_anchor=(0.5,-0.25))
save('fig6_weibull')
print('tail n',n,len(s),'min tail',tail.min())
# Fig7 score distribution test
plt.figure(figsize=(W,3.2))
cap=lambda x: np.clip(x,0,1.0)
plt.hist(cap(tk[P]),bins=np.linspace(0,1,26),alpha=.6,density=True,label='Known wild boar')
plt.hist(cap(tu[P]),bins=np.linspace(0,1,26),alpha=.6,density=True,label='Unknown wildlife')
plt.axvline(TAU,color='k',ls='--',lw=1.2,label=f'Frozen τ = {TAU:.3f}')
plt.xlabel('Unknown score $S_U$'); plt.ylabel('Density'); plt.legend(frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(0.45,-0.25),ncol=1); save('fig7_scores')
print('mean scores known',cap(tk[P]).mean(),'unknown',cap(tu[P]).mean(),'median',np.median(cap(tk[P])),np.median(cap(tu[P])))
# Fig8 ROC
plt.figure(figsize=(W,3.9))
y=np.r_[np.zeros(len(tk)),np.ones(len(tu))]
for col,l in [(P,'Proposed'),('score__vit_b16__full__evt_proto','Full image EVT'),('score__vit_b16__yoloroi__mahalanobis','ROI Mahalanobis'),('score__vit_b16__yoloroi__mp_evt','ROI MP-EVT'),('score__yolov8n__image__yolo_conf','YOLO confidence')]:
    sc=np.r_[tk[col].values,tu[col].values]; f,t,_=roc_curve(y,sc); plt.plot(f,t,lw=1.3,label=f'{l} ({roc_auc_score(y,sc):.3f})')
plt.plot([0,1],[0,1],'k:',lw=.8); plt.xlabel('False-positive rate'); plt.ylabel('True-positive rate'); plt.legend(frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(0.42,-0.22),ncol=1); save('fig8_roc')
# Fig9 threshold sensitivity on validation (pipeline)
taus=np.round(np.arange(0.1,0.91,0.05),2); rows=[]
for t in taus:
    kr=(vk[P]<=t).mean(); ur=(vu[P]>t).mean(); acc=((vk[P]<=t).sum()+(vu[P]>t).sum())/(len(vk)+len(vu)); rows.append((t,kr,ur,(kr+ur)/2))
sv=pd.DataFrame(rows,columns=['tau','kr','ur','bal'])
plt.figure(figsize=(W,3.2))
plt.plot(sv.tau,sv.kr*100,marker='o',ms=3,label='Known recall'); plt.plot(sv.tau,sv.ur*100,marker='s',ms=3,label='Unknown recall'); plt.plot(sv.tau,sv.bal*100,marker='^',ms=3,label='Balanced accuracy')
plt.axvline(TAU,color='k',ls='--',lw=1); plt.xlabel('Threshold τ (validation)'); plt.ylabel('%'); plt.ylim(80,101); plt.grid(alpha=.3); plt.legend(frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(0.45,-0.25),ncol=1); save('fig9_val_threshold')
sv.to_csv(os.path.join(BASE,'results','tables','val_sensitivity_pipeline.csv'),index=False)
# Fig10 confusion matrix
cm=np.array([[ (tk[P]<=TAU).sum(), (tk[P]>TAU).sum()],[ (tu[P]<=TAU).sum(), (tu[P]>TAU).sum()]])
print('cm',cm)
plt.figure(figsize=(2.9,2.6)); plt.imshow(cm,cmap='Blues')
for i in range(2):
    for j in range(2): plt.text(j,i,f'{cm[i,j]}\n({100*cm[i,j]/cm[i].sum():.2f}%)',ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.xticks([0,1],['Known','Unknown']); plt.yticks([0,1],['Known','Unknown']); plt.xlabel('Predicted'); plt.ylabel('True'); save('fig10_cm')
# Fig11 external rejection (horizontal)
e=pd.read_csv(R+'tables/T06_external_near_ood_cross_source.csv')
q=e[(e.backbone=='vit_b16')&(e.view=='yoloroi')&(e.family=='pipeline_yolo_evt')&(e.protocol=='P1_val_known_TPR95')&(e.species!='ALL')].copy()
order={'known_ext':0,'near_ood':1,'far_ext':2}; q['o']=q.group.map(order); q=q.sort_values(['o','species'])
col={'known_ext':'tab:green','near_ood':'tab:orange','far_ext':'tab:red'}
plt.figure(figsize=(W,4.6)); yy=np.arange(len(q))[::-1]
plt.barh(yy,q.rejection_rate*100,color=[col[g] for g in q.group],xerr=[(q.rejection_rate-q.ci_lo)*100,(q.ci_hi-q.rejection_rate)*100],capsize=2,error_kw={'lw':.8})
plt.yticks(yy,q.species); [plt.text(v*100+1.5,yv,'0%',va='center',fontsize=10) for v,yv in zip(q.rejection_rate,yy) if v==0]; plt.xlabel('Rejected as unknown (%)'); plt.xlim(0,105)
from matplotlib.patches import Patch
plt.legend(handles=[Patch(color='tab:green',label='Boar, other source'),Patch(color='tab:orange',label='Near-OOD species'),Patch(color='tab:red',label='Far-OOD, other source')],frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(0.4,-0.13),ncol=1)
save('fig11_external')
# Fig5 YOLO false acceptance per unknown species (3 seeds)
e=pd.read_csv(R+'tables/T05_yolo_open_world_exposure_per_seed.csv')
q=e[(e.group=='unknown_test')&(e.species.isin(['buffalo','elephant','rhino','zebra']))]
sp=['buffalo','elephant','rhino','zebra']; lab=['Buffalo','Elephant','Rhino','Zebra']
m=[q[q.species==s].rate.mean()*100 for s in sp]; sd=[q[q.species==s].rate.std()*100 for s in sp]
n=[int(q[q.species==s].n.iloc[0]) for s in sp]
plt.figure(figsize=(W,2.6)); x=np.arange(4)
plt.bar(x,m,yerr=sd,capsize=3,color='tab:red',alpha=.8)
for s_i,s in enumerate(sp):
    v=q[q.species==s].rate.values*100; plt.scatter([s_i]*len(v),v,color='k',s=10,zorder=3)
plt.xticks(x,[f'{l}\n(n={k})' for l,k in zip(lab,n)]); plt.ylabel('Falsely accepted (%)'); plt.ylim(0,105); plt.grid(axis='y',alpha=.3)
save('fig5_yolo_species')
print('fig5',m,sd)
