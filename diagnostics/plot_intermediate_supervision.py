"""Plot the frozen R04 primary four-cell comparison and seed-paired interactions."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(results, output):
    if output.exists():raise ValueError('Figure exists; choose a new destination')
    summary=json.loads((results/'summary.json').read_text())
    rows=[r for r in summary['rows'] if r['table']=='primary']
    models=['base','loop','loop_ds','untied','untied_ds']
    metrics=[('crps','Hourly CRPS'),('energy_score','ES'),('variogram_score','VS'),
             ('increment_1h_crps','1h increment CRPS'),('increment_3h_crps','3h increment CRPS'),('daily_mean_crps','Daily mean CRPS')]
    colors=['#547a9b','#258571','#d77828','#7b59a7','#bf5686']
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,(metric,title) in zip(axes.flat,metrics):
        for seed,marker in zip(summary['seeds'],['o','s','^']):
            values=[next(r[metric] for r in rows if r['model']==m and r['seed']==seed) for m in models]
            ax.plot(range(5),values,color='#bbbbbb',linewidth=.7,alpha=.6)
            for x,(value,color) in enumerate(zip(values,colors)):ax.scatter(x,value,color=color,marker=marker,s=32)
        ax.set_xticks(range(5),['BASE','LOOP','LOOP-DS','UNTIED','UNTIED-DS'],rotation=15)
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('R04: final readout, delayed EMA at update 3000 | validation only\n'
                 'Dots are seeds 42/43/44; lines pair the same training seed; all displayed scores lower is better')
    fig.tight_layout(rect=(0,0,1,.91));output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.results,a.output)
