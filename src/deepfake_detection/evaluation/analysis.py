"""Optional evaluation analyses; never infer demographics or tune test labels."""
import numpy as np
from deepfake_detection.evaluation.metrics import compute_metrics,_arrays
import math


def demographic_summaries(rows,labels,scores,*,threshold=.5):
    labels,scores=_arrays(labels,scores); rows=list(rows)
    if len(rows)!=len(labels): raise ValueError('Demographic rows and predictions differ in length')
    annotations=[row.get('demographics',{}) for row in rows]
    if any(not isinstance(item,dict) for item in annotations): raise ValueError('Demographic annotations must be dictionaries')
    fields=sorted({key for item in annotations for key in item})
    if not fields: raise ValueError('Demographic evaluation requires supplied annotations; attributes are never inferred')
    result={}
    for field in fields:
        groups=[item.get(field,'unannotated') for item in annotations]
        if any(not isinstance(group,str) or not group for group in groups): raise ValueError('Demographic groups must be explicit nonempty strings')
        result[field]={}
        for group in sorted(set(groups)):
            mask=np.asarray([value==group for value in groups])
            result[field][group]=compute_metrics(labels[mask],scores[mask],threshold=threshold)
    return result


def adversarial_robustness(model,loader,eps_list=(.01,.02,.05),*,mean=None,std=None):
    """Single-view uncalibrated FGSM AUC degradation; does not optimize weights."""
    import torch
    from deepfake_detection.training.adversarial import fgsm_attack
    from deepfake_detection.evaluation.metrics import real_probability
    epsilons=list(eps_list)
    if not epsilons or len(epsilons)>20 or any(not math.isfinite(eps) or not 0<=eps<=1 for eps in epsilons): raise ValueError('Attack epsilons must be bounded pixel budgets in[0,1]')
    parameter=next(model.parameters(),None); device=parameter.device if parameter is not None else torch.device('cpu')
    modes=[module.training for module in model.modules()]
    labels_all=[]; clean=[]; attacked=[[] for _ in epsilons]
    def score(inputs):
        with torch.no_grad(): output=model(inputs).float()
        if output.shape!=(len(inputs),1) or not bool(torch.isfinite(output).all()): raise ValueError('Robustness evaluation requires finite binary logits')
        return real_probability(output.cpu().reshape(-1).numpy()).tolist()
    try:
        model.eval()
        for batch in loader:
            images,labels=batch[:2]; images=images.to(device); labels=labels.to(device)
            if not bool(((labels==0)|(labels==1)).all()): raise ValueError('Robustness requires fake=0 real=1 labels')
            labels_all.extend(labels.cpu().reshape(-1).tolist()); clean.extend(score(images))
            for index,eps in enumerate(epsilons): attacked[index].extend(score(fgsm_attack(model,images,labels,eps,mean=mean,std=std)))
    finally:
        for module,mode in zip(model.modules(),modes): module.training=mode
    clean_metrics=compute_metrics(labels_all,clean)
    results=[]
    for eps,scores in zip(epsilons,attacked):
        metrics=compute_metrics(labels_all,scores); auc=metrics['roc_auc']; baseline=clean_metrics['roc_auc']
        results.append({'eps_pixels':eps,'auc':auc,'auc_degradation':baseline-auc if baseline is not None and auc is not None else None,'metrics':metrics})
    return {'clean_auc':clean_metrics['roc_auc'],'clean_metrics':clean_metrics,'attacks':results,'scoring':'single-view uncalibrated FGSM; distinct from calibrated/TTA report','samples':len(labels_all)}


def fairness_metrics(predictions,demographics):
    """FAKE-positive descriptive parity gaps from supplied categorical groups."""
    predictions=list(predictions); demographics=list(demographics)
    if not predictions or len(predictions)!=len(demographics): raise ValueError('Fairness requires matching predictions/demographics')
    labels=np.asarray([row['label'] for row in predictions]); decisions=np.asarray([row['prediction'] for row in predictions])
    if not np.isin(labels,[0,1]).all() or not np.isin(decisions,[0,1]).all(): raise ValueError('Fairness requires fake=0 real=1 truth and predictions')
    if all(isinstance(value,str) for value in demographics): demographics=[{'group':value} for value in demographics]
    if any(not isinstance(value,dict) for value in demographics): raise ValueError('Fairness demographic annotations must be strings or dictionaries')
    fields=sorted({key for value in demographics for key in value})
    if not fields: raise ValueError('Fairness requires supplied demographic annotations')
    result={}
    for field in fields:
        values=[value.get(field,'unannotated') for value in demographics]
        if any(not isinstance(value,str) or not value for value in values): raise ValueError('Fairness requires explicit nonempty categorical groups')
        groups={}
        for value in sorted(set(values)):
            mask=np.asarray([item==value for item in values]); truth=labels[mask]; predicted=decisions[mask]
            def rate(condition,denominator): return float(condition[denominator].mean()) if denominator.any() else None
            groups[value]={'samples':int(mask.sum()),'selection_rate_fake':float((predicted==0).mean()),'fake_tpr':rate(predicted==0,truth==0),'real_fpr':rate(predicted==0,truth==1),'fake_precision':rate(truth==0,predicted==0)}
        eligible=[group for value,group in groups.items() if value!='unannotated']
        def gap(key):
            numbers=[group[key] for group in eligible]
            return max(numbers)-min(numbers) if len(numbers)>=2 and all(number is not None for number in numbers) else None
        tpr=gap('fake_tpr'); fpr=gap('real_fpr')
        result[field]={'groups':groups,'demographic_parity_gap':gap('selection_rate_fake'),'equalized_odds_gap':max(tpr,fpr) if tpr is not None and fpr is not None else None,'predictive_parity_gap':gap('fake_precision'),'limitations':['Descriptive pooled-domain gaps; unannotated groups excluded from comparisons. Undefined class denominators or fewer than two annotated groups yield null. Small strata require uncertainty analysis; rates are not causal fairness conclusions.']}
    return result


def save_failure_cases(rows,scores,labels,output_dir,top_k=50,*,threshold=.5,workspace=None):
    """Copy top-confidence wrong crops with immutable source/hash provenance."""
    from deepfake_detection.common.paths import Workspace,atomic_bytes,digest_file,fingerprint,write_jsonl
    from pathlib import Path
    labels,scores=_arrays(labels,scores); rows=list(rows)
    if len(rows)!=len(labels) or any(row.get('label')!=label for row,label in zip(rows,labels)): raise ValueError('Failure rows/labels differ')
    if not isinstance(top_k,int) or not 0<=top_k<=1000 or not math.isfinite(threshold) or not 0<=threshold<=1: raise ValueError('Invalid failure-export budget/threshold')
    ws=workspace or Workspace(); output=ws.output(output_dir)
    predictions=np.where(1-scores>=threshold,0,1)
    confidence=np.where(predictions==0,1-scores,scores)
    wrong=[index for index in range(len(rows)) if predictions[index]!=labels[index]]
    wrong.sort(key=lambda index:(-confidence[index],str(rows[index]['sample_id'])))
    records=[]
    for rank,index in enumerate(wrong[:top_k],1):
        row=rows[index]; source=ws.output(row['crop_path']); sha=digest_file(source)
        if row.get('crop_sha256') and row['crop_sha256']!=sha: raise ValueError(f'Failure source hash differs: {source}')
        target=ws.output(output/f'{rank:03d}-{fingerprint(row["sample_id"])[:16]}.png')
        atomic_bytes(target,Path(source).read_bytes(),ws)
        records.append({'rank':rank,'sample_id':row['sample_id'],'true_label':int(labels[index]),'prediction':int(predictions[index]),'confidence':float(confidence[index]),'score_real':float(scores[index]),'score_fake':float(1-scores[index]),'source_path':str(source),'saved_path':str(target),'crop_sha256':sha,'dataset':row.get('dataset'),'method':row.get('method')})
    write_jsonl(output/'failures.jsonl',records,ws)
    return {'wrong_total':len(wrong),'saved':len(records),'manifest_path':str(output/'failures.jsonl'),'threshold':threshold}


def paired_bootstrap_test(labels,scores_a,scores_b,n=10000,*,seed=42,clusters=None):
    from sklearn.metrics import roc_auc_score
    labels,a=_arrays(labels,scores_a); _,b=_arrays(labels,scores_b)
    if len(np.unique(labels))!=2 or not isinstance(n,int) or not 20<=n<=100000: raise ValueError('Paired bootstrap requires both classes and20..100000 resamples')
    observed=float(roc_auc_score(labels,a)-roc_auc_score(labels,b)); rng=np.random.default_rng(seed)
    if clusters is not None:
        if len(clusters)!=len(labels): raise ValueError('Cluster/prediction counts differ')
        groups={}
        for index,cluster in enumerate(clusters): groups.setdefault(str(cluster),[]).append(index)
        units=list(groups.values())
        if len(units)<2: return {'auc_difference':observed,'p_value':None,'interval':None,'valid_resamples':0,'sampling_unit':'dependency_family','limitation':'fewer_than_two_clusters'}
    else: units=None
    differences=[]
    for _ in range(n):
        if units is not None: indices=np.concatenate([units[index] for index in rng.integers(0,len(units),len(units))])
        else:
            indices=np.concatenate([rng.choice(np.flatnonzero(labels==label),size=int((labels==label).sum()),replace=True) for label in (0,1)])
        if len(np.unique(labels[indices]))<2: continue
        differences.append(float(roc_auc_score(labels[indices],a[indices])-roc_auc_score(labels[indices],b[indices])))
    if not differences: return {'auc_difference':observed,'p_value':None,'interval':None,'valid_resamples':0,'limitation':'all_resamples_single_class'}
    values=np.asarray(differences)
    p=float((1+np.count_nonzero(np.abs(values-observed)>=abs(observed)))/(len(values)+1))
    return {'auc_difference':observed,'p_value':p,'interval':np.quantile(values,[.025,.975]).tolist(),'valid_resamples':len(values),'seed':seed,'sampling_unit':'dependency_family' if units is not None else 'stratified_iid_faces','limitations':['Exploratory centered-bootstrap two-sided test; IID faces ignore crop dependencies unless clusters are supplied.']}


def delong_test(labels,scores_a,scores_b):
    labels,a=_arrays(labels,scores_a); _,b=_arrays(labels,scores_b)
    if min(int((labels==0).sum()),int((labels==1).sum()))<2: raise ValueError('DeLong requires at least two examples of each class')
    def influences(scores):
        positive=1-scores[labels==0]; negative=1-scores[labels==1]
        vp=np.zeros(len(positive)); vn=np.zeros(len(negative))
        for start in range(0,len(positive),512):
            for end in range(0,len(negative),512):
                p=positive[start:start+512,None]; q=negative[None,end:end+512]
                wins=(p>q).astype(float)+.5*(p==q)
                vp[start:start+512]+=wins.sum(1); vn[end:end+512]+=wins.sum(0)
        return vp/len(negative),vn/len(positive)
    ap,an=influences(a); bp,bn=influences(b)
    covariance=np.cov(np.stack([ap,bp]),ddof=1)/len(ap)+np.cov(np.stack([an,bn]),ddof=1)/len(an)
    variance=max(0,float(covariance[0,0]+covariance[1,1]-2*covariance[0,1])); difference=float(ap.mean()-bp.mean())
    z=difference/math.sqrt(variance) if variance>0 else None
    probability=math.erfc(abs(z)/math.sqrt(2)) if z is not None else (1. if difference==0 else 0.)
    return {'auc_difference':difference,'variance':variance,'z_score':z,'p_value':probability,'sampling_unit':'iid_faces','limitations':['Standard paired DeLong assumes independent examples; crop/frame dependencies violate this assumption. Prefer paired family bootstrap. Degenerate variance yields an explicit limit.']}


def save_learning_curves(history_json,output_dir,*,workspace=None):
    from deepfake_detection.common.paths import Workspace,read_json,atomic_bytes
    from io import BytesIO
    import os
    ws=workspace or Workspace(); source=ws.output(history_json); output=ws.output(output_dir)
    history=read_json(source)
    if not isinstance(history,list) or not history: raise ValueError('Learning curves require nonempty saved epoch history')
    fields=('train_loss','validation_loss','validation_macro_auc')
    if any(any(not isinstance(row.get(field),(int,float)) or not math.isfinite(row[field]) for field in fields) for row in history): raise ValueError('Learning curves require finite loss/AUC fields')
    cache=ws.output(ws.runs/'.cache'/'matplotlib'); cache.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(10,4)); epochs=list(range(1,len(history)+1))
    try:
        axes[0].plot(epochs,[row['train_loss'] for row in history],label='Training loss')
        axes[0].plot(epochs,[row['validation_loss'] for row in history],label='Validation loss')
        axes[0].set(xlabel='Completed epoch',ylabel='Loss'); axes[0].legend()
        axes[1].plot(epochs,[row['validation_macro_auc'] for row in history],label='Validation macro AUC')
        axes[1].set(xlabel='Completed epoch',ylabel='Macro AUC',ylim=(0,1)); axes[1].legend()
        for index in range(1,len(history)):
            if history[index].get('phase')!=history[index-1].get('phase'):
                for axis in axes: axis.axvline(index+.5,color='gray',linestyle='--')
        figure.tight_layout(); buffer=BytesIO(); figure.savefig(buffer,format='png',dpi=150)
        path=ws.output(output/'learning_curves.png'); atomic_bytes(path,buffer.getvalue(),ws)
    finally: plt.close(figure)
    return {'path':str(path),'epochs':len(history),'source':str(source)}
