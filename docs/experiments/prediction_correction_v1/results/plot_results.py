import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import file_hash
root=Path('docs/experiments/prediction_correction_v1/results')
s=json.loads((root/'summary.json').read_text());r=s['rows']
h=json.loads((root/'training_training_history.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
colors=['#546e7a','#1976d2','#ef6c00']
labels=['K0','K1','K2']
for ax,key,title in [(axes[0,0],'crps','CRPS (lower is better)'),(axes[0,1],'energy_score','Energy score (lower is better)')]:
 vals=[a[key] for a in r];ax.bar(labels,vals,color=colors)
 ax.set_title(title);ax.set_ylim(0,max(vals)*1.2)
 for i,val in enumerate(vals):ax.text(i,val,f'{val:.6f}',ha='center',va='bottom',fontsize=9)
ax=axes[1,0]
for row,label,color in zip(r,labels,colors):
 ax.scatter(row['sampling_seconds'],row['crps'],s=70,color=color)
 ax.annotate(label,(row['sampling_seconds'],row['crps']),xytext=(5,4),textcoords='offset points')
ax.margins(x=.15,y=.15);ax.set_title('Observed sampling cost vs CRPS');ax.set_xlabel('Seconds / 5,700 trajectories');ax.set_ylabel('CRPS')
ax=axes[1,1]
loss=np.array([x['loss'] for x in h]);smoothed=np.convolve(loss,np.ones(100)/100,mode='valid')
ax.plot(np.arange(100,len(loss)+1),smoothed,color='#1976d2')
ax.axvline(1000,color='#777777',ls='--',label='EMA copy')
ax.set_title('C epsilon MSE: 100-update moving mean');ax.set_xlabel('Successful C updates');ax.legend(frameon=False)
fig.suptitle('Frozen BASE-H0 + one trained corrector | seed 42 | 57 validation days',fontsize=12)
fig.savefig(root/'comparison.png',dpi=160)
manifest=json.loads((root/'archive_sha256.json').read_text());manifest['comparison.png']=file_hash(root/'comparison.png')
(root/'archive_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
