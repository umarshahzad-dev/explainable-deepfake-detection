"""Explicit local-run ensembles; no implicit model downloads or test fitting."""
from copy import deepcopy
import numpy as np


def _weights(weights,count):
    values=np.ones(count,dtype=float) if weights is None else np.asarray(weights,dtype=float)
    if count<1 or values.shape!=(count,) or not np.isfinite(values).all() or (values<0).any() or values.sum()<=0:
        raise ValueError('Ensemble weights must be finite nonnegative, match members, and have positive sum')
    return values/values.sum()


def bayesian_ensemble(predictions,weights):
    """Finite weighted predictive mixture; supplied weights are not inferred posteriors.

    Predictions are member-major, with either one scalar or N faces per member.
    Member disagreement describes this ensemble and is not an OOD guarantee.
    """
    scores=np.asarray(predictions,dtype=float)
    if scores.ndim not in (1,2) or not scores.size or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('Predictions must be finite member-major probabilities')
    mixture=_weights(weights,len(scores)); shape=(-1,)+(1,)*(scores.ndim-1)
    mean=(scores*mixture.reshape(shape)).sum(0)
    epistemic=((scores-mean)**2*mixture.reshape(shape)).sum(0)
    aleatoric=(scores*(1-scores)*mixture.reshape(shape)).sum(0)
    return {'mean':mean.tolist(),'epistemic_variance':epistemic.tolist(),'aleatoric_variance':aleatoric.tolist(),'total_variance':(epistemic+aleatoric).tolist(),'weights':mixture.tolist(),'uncertainty_note':'Weighted member disagreement; no learned posterior or OOD coverage guarantee'}


class EnsembleScorer:
    """Sequential member scoring with strict face correspondence and abstention.

    Member probability calibration is retained; the ensemble decision itself
    requires independent calibration. Its explicit default threshold is .5.
    """
    def __init__(self,engines,weights=None,*,threshold=.5,fusion='weighted_average'):
        self.engines=list(engines); self.weights=_weights(weights,len(self.engines))
        hashes={engine.metadata.get('preprocessing_hash') for engine in self.engines}
        if len(hashes)!=1 or None in hashes: raise ValueError('Ensemble members require the same verified preprocessing contract')
        if not np.isfinite(threshold) or not 0<=threshold<=1: raise ValueError('Invalid ensemble threshold')
        self.threshold=float(threshold)
        if fusion not in {'weighted_average','mean','max','min','vote'}: raise ValueError('Unknown ensemble fusion mode')
        self.fusion=fusion
        from deepfake_detection.common.paths import fingerprint
        self.cfg=deepcopy(getattr(self.engines[0],'cfg',{'preprocessing':{'max_pixels':8000000}}))
        member_hashes=[engine.metadata['weights_sha256'] for engine in self.engines]
        self.metadata={'run_id':'ensemble:'+','.join(engine.metadata['run_id'] for engine in self.engines),'weights_sha256':fingerprint({'members':member_hashes,'weights':self.weights.tolist(),'fusion':fusion,'threshold':threshold}),'preprocessing_hash':next(iter(hashes)),'pilot':any(engine.metadata.get('pilot',False) for engine in self.engines)}
        self.calibration={'temperature':1.,'threshold':self.threshold,'applied':False,'limitation':'ensemble_threshold_requires_independent_calibration'}

    @classmethod
    def from_runs(cls,run_names,weights=None,*,workspace=None,device='cpu',threshold=.5,fusion='weighted_average'):
        from deepfake_detection.inference.engine import InferenceEngine
        engines=[InferenceEngine.from_run(name,workspace,device=device,allow_ensemble=False) for name in run_names]
        if isinstance(weights,dict):
            selected=[]
            for engine in engines:
                name=engine.cfg['model']['name']
                key=engine.metadata['run_id'] if engine.metadata['run_id'] in weights else ('xception' if 'xception' in name else 'b0' if 'efficientnet_b0' in name else name)
                if key not in weights: raise ValueError('Ensemble weight mapping must cover every run or model; provide an explicit list')
                selected.append(weights[key])
            weights=selected
        return cls(engines,weights,threshold=threshold,fusion=fusion)

    def analyze(self,image,*,explain=False):
        packets=[engine.analyze(image,explain=explain) for engine in self.engines]
        result=deepcopy(packets[0]); faces=result['faces']
        for packet in packets[1:]:
            if len(packet['faces'])!=len(faces): raise ValueError('Ensemble face correspondence differs between members')
            for face,other in zip(faces,packet['faces']):
                if face['face_index']!=other['face_index'] or not np.allclose(face['box'],other['box'],atol=1e-6,rtol=0):
                    raise ValueError('Ensemble face correspondence differs between members')
        for index,face in enumerate(faces):
            members=[packet['faces'][index] for packet in packets]
            face['heatmap_data_url']=None; face['overlay_data_url']=None
            if any(member.get('status')!='accepted' or member.get('fake_score') is None for member in members):
                face.update(status='ensemble_abstained',label=None,fake_score=None,real_score=None,confidence=None)
                continue
            scores=np.asarray([member['fake_score'] for member in members],dtype=float)
            if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any(): raise ValueError('Invalid ensemble member probability')
            if self.fusion=='weighted_average': fake=float(scores@self.weights)
            elif self.fusion=='mean': fake=float(scores.mean())
            elif self.fusion=='max': fake=float(scores.max())
            elif self.fusion=='min': fake=float(scores.min())
            else: fake=float((scores>=.5).mean())
            label='FAKE' if fake>=self.threshold else 'REAL'
            face.update(fake_score=fake,real_score=1-fake,label=label,confidence=fake if label=='FAKE' else 1-fake,uncertain=abs(fake-self.threshold)<.05,member_fake_scores=scores.tolist())
        result.update(ensemble_members=[engine.metadata['run_id'] for engine in self.engines],ensemble_weights=self.weights.tolist(),ensemble_fusion=self.fusion,ensemble_weights_sha256=[engine.metadata['weights_sha256'] for engine in self.engines],threshold=self.threshold,calibration_applied=False,training_pilot=any(packet.get('training_pilot',False) for packet in packets),explanation_note='Member attributions are not an attribution of the fused ensemble decision.')
        result.update(run_id=self.metadata['run_id'],weights_sha256=self.metadata['weights_sha256'],temperature=1.)
        return result


def _local_engines(run_names,workspace,device):
    from deepfake_detection.inference.engine import InferenceEngine
    names=list(run_names)
    if len(names)<2 or len(set(names))!=len(names): raise ValueError('Run ensemble requires at least two distinct run names')
    engines=[InferenceEngine.from_run(name,workspace,device=device) for name in names]
    if len({engine.cfg['model']['name'] for engine in engines})!=1:
        raise ValueError('Homogeneous run ensembles require the same model architecture')
    if len({engine.metadata['weights_sha256'] for engine in engines})!=len(engines):
        raise ValueError('Run ensemble members must have distinct weights')
    return engines


class MultiSeedEnsemble(EnsembleScorer):
    def __init__(self,run_names,*,workspace=None,device='cpu',weights=None,threshold=.5):
        engines=_local_engines(run_names,workspace,device)
        if len({engine.cfg['seed'] for engine in engines})!=len(engines): raise ValueError('Multi-seed ensemble requires distinct seeds')
        super().__init__(engines,weights,threshold=threshold)


AUGMENTATION_KEYS={
    'augmentation','mixup_alpha','cutmix_alpha','fouriermix_prob','fouriermix_alpha',
    'fft_amplitude_mix_prob','fft_amplitude_mix_alpha','dct_mask_prob','dct_num_masks','dct_mask_size',
    'gaussian_noise_std','salt_pepper_prob','sharpen_prob','sharpen_factor_range',
    'hue_jitter','saturation_jitter','color_jitter_prob','gamma_prob','gamma_range',
    'jpeg_prob','jpeg_quality_range','rescale_prob','rescale_range','rotation_range',
    'blur_prob','blur_sigma_range','motion_blur_prob','motion_blur_kernel',
    'coarse_dropout_prob','coarse_dropout_holes','coarse_dropout_ratio',
    'grid_shuffle_prob','grid_shuffle_size','crop_pad_prob','crop_pad_ratio',
    'perspective_prob','perspective_scale','elastic_prob','elastic_alpha','elastic_sigma',
    'random_erase_prob','grayscale_prob','randaugment','trivialaugment','augmix',
    'artifact_injection','artifact_label_policy','artifact_prob','style_transfer','sbi_prob',
    'randaugment_num_ops','randaugment_magnitude','trivialaugment_bins',
    'augmix_width','augmix_depth','augmix_alpha','style_images_path','style_weights_path',
}


class MultiAugmentationEnsemble(EnsembleScorer):
    def __init__(self,run_names,*,workspace=None,device='cpu',weights=None,threshold=.5):
        from deepfake_detection.common.paths import fingerprint
        engines=_local_engines(run_names,workspace,device)
        recipes=[fingerprint({key:value for key,value in engine.cfg['training'].items() if key in AUGMENTATION_KEYS}) for engine in engines]
        if len(set(recipes))!=len(engines): raise ValueError('Multi-augmentation ensemble requires distinct augmentation recipes')
        super().__init__(engines,weights,threshold=threshold)


def kfold_manifests(rows,n_splits=3,seed=42):
    """Connected-family folds over training pool; leaves fixed evaluation alone."""
    import random
    rows=list(rows)
    if not rows or any(row.get('split')!='train' or row.get('dataset')=='dfd' for row in rows):
        raise ValueError('K-fold pool must contain training-only records, never DFD/held-out data')
    if not isinstance(n_splits,int) or not 2<=n_splits<=10: raise ValueError('K-fold count must be 2..10')
    parents=list(range(len(rows)))
    def root(index):
        while parents[index]!=index:
            parents[index]=parents[parents[index]]; index=parents[index]
        return index
    seen={}
    for index,row in enumerate(rows):
        if row.get('label') not in (0,1) or not row.get('parents'): raise ValueError('K-fold requires binary labels and resolved parent provenance')
        tokens=[('parent',token) for token in row['parents']]+[('person',token) for token in row.get('known_person_ids',[])]+[('sample',row['sample_id'])]
        for field in ('sha256','crop_sha256'):
            if row.get(field): tokens.append((field,row[field]))
        for token in tokens:
            if token in seen: parents[root(index)]=root(seen[token])
            else: seen[token]=index
    groups={}
    for index in range(len(rows)): groups.setdefault(root(index),[]).append(index)
    if len(groups)<n_splits: raise ValueError('Not enough independent parent/person families for requested folds')
    # Stable input order makes seeded fold membership independent of manifest order.
    families=sorted(groups.values(),key=lambda group:sorted(str(rows[i]['sample_id']) for i in group))
    rng=random.Random(seed); rng.shuffle(families); families.sort(key=len,reverse=True)
    assignments=[[] for _ in range(n_splits)]; counts=np.zeros((n_splits,2),dtype=int)
    for group in families:
        label_counts=np.bincount([rows[i]['label'] for i in group],minlength=2)
        chosen=min(range(n_splits),key=lambda fold:(int(counts[fold].sum()),int(counts[fold]@label_counts),fold))
        assignments[chosen].extend(group); counts[chosen]+=label_counts
    result=[]
    for fold,held in enumerate(assignments):
        held=set(held); training=[]; validation=[]
        for index,row in enumerate(rows):
            copy=deepcopy(row); copy.update(original_split='train',cv_fold=fold,split='val' if index in held else 'train')
            (validation if index in held else training).append(copy)
        result.append({'fold':fold,'train_rows':training,'validation_rows':validation,'identity_claim':'source/known-person disjoint; unknown persons remain unverified'})
    return result


def _provenance_overlap(fit,held):
    for field in ('parents','known_person_ids'):
        if {token for row in fit for token in row.get(field,[])} & {token for row in held for token in row.get(field,[])}: return field
    for field in ('video_id','sample_id','sha256','crop_sha256'):
        if {row[field] for row in fit if row.get(field)} & {row[field] for row in held if row.get(field)}: return field
    return None


def _crop_score(engine,row):
    if hasattr(engine,'score_crop'): return float(engine.score_crop(row))
    from PIL import Image
    import torch
    from deepfake_detection.data.preprocessing import model_tensor
    from deepfake_detection.evaluation.metrics import real_probability
    from deepfake_detection.evaluation.calibration import apply_calibration
    if row.get('preprocessing_hash')!=engine.metadata['preprocessing_hash']:
        raise ValueError('Stacking crop preprocessing contract differs from base model')
    with Image.open(row['crop_path']) as image:
        image.load()
        if image.mode!='RGB' or image.size!=(engine.cfg['preprocessing']['master_size'],)*2: raise ValueError('Invalid stacking master crop')
        image=image.copy()
        tensor=model_tensor(image,engine.cfg['model']).unsqueeze(0).to(engine.device)
    from deepfake_detection.evaluation.test_time_augmentation import inference_augmented,inference_score
    with engine.lock,torch.no_grad():
        if inference_augmented(engine.cfg): logits=np.asarray([inference_score(engine.model,image,engine.cfg,device=engine.device)['logit']])
        else: logits=engine.model(tensor).float().cpu().reshape(-1).numpy()
    if len(logits)!=1: raise ValueError('Stacking requires binary base logits')
    return 1-float(apply_calibration(real_probability(logits),engine.calibration)[0])


class StackingEnsemble(EnsembleScorer):
    def __init__(self,base_runs,*,engines=None,workspace=None,device='cpu',threshold=.5):
        from deepfake_detection.common.paths import Workspace,read_json
        names=list(base_runs)
        if len(names)<2 or len(set(names))!=len(names): raise ValueError('Stacking requires distinct base runs')
        if engines is None:
            from deepfake_detection.inference.engine import InferenceEngine
            ws=workspace or Workspace()
            engines=[InferenceEngine.from_run(name,ws,device=device) for name in names]
            from deepfake_detection.models.registry import resolve_model_path
            for name,engine in zip(names,engines): engine.fit_rows=read_json(resolve_model_path(name,ws)/'input_manifest.json')['rows']
        if len(engines)!=len(names): raise ValueError('Stacking run/member counts differ')
        super().__init__(engines,threshold=threshold)
        self.meta_classifier=None; self.fit_report=None

    def fit(self,validation_rows):
        from sklearn.linear_model import LogisticRegression
        from deepfake_detection.common.paths import fingerprint
        rows=list(validation_rows)
        if not rows or any(row.get('split') not in {'val','dev'} or row.get('dataset')=='dfd' or not row.get('parents') for row in rows):
            raise ValueError('Stacking accepts resolved-parent validation/development rows only')
        if {row.get('label') for row in rows}!={0,1}: raise ValueError('Stacking requires both binary classes')
        for engine in self.engines:
            provenance=getattr(engine,'fit_rows',None)
            trained=[row for row in provenance or [] if row.get('split')=='train']
            if not trained: raise ValueError('Base model training provenance is missing')
            overlap=_provenance_overlap(trained,rows)
            if overlap: raise ValueError(f'Stacking meta-fit overlaps base training {overlap}')
        scores=np.asarray([[_crop_score(engine,row) for engine in self.engines] for row in rows])
        if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any(): raise ValueError('Invalid stacking probabilities')
        classifier=LogisticRegression(C=1,random_state=42,max_iter=1000)
        classifier.fit(scores,[int(row['label']==0) for row in rows])
        self.meta_classifier=classifier
        self.fit_report={'samples':len(rows),'fit_manifest_sha256':fingerprint(rows),'base_weights_sha256':[engine.metadata['weights_sha256'] for engine in self.engines],'limitations':['Validation fitted stack needs independent calibration/test evaluation; unknown person identities remain unverified.']}
        return deepcopy(self.fit_report)

    def analyze(self,image,*,explain=False):
        if self.meta_classifier is None: raise ValueError('Stacking requires an explicit fit before prediction')
        if self.fit_report['base_weights_sha256']!=[engine.metadata['weights_sha256'] for engine in self.engines]: raise ValueError('Base weights changed after stacking fit')
        packet=super().analyze(image,explain=explain)
        for face in packet['faces']:
            if face.get('fake_score') is None: continue
            fake=float(self.meta_classifier.predict_proba([face['member_fake_scores']])[0,1])
            label='FAKE' if fake>=self.threshold else 'REAL'
            face.update(fake_score=fake,real_score=1-fake,label=label,confidence=fake if label=='FAKE' else 1-fake,uncertain=abs(fake-self.threshold)<.05)
        packet['stacking_fit']=deepcopy(self.fit_report)
        return packet

    def predict(self,image): return self.analyze(image)


def publish_snapshot(model,run,cfg,*,epoch,workspace=None):
    """Commit immutable epoch weights before atomically adding registry metadata."""
    import hashlib
    from safetensors.torch import save
    from deepfake_detection.common.paths import Workspace,atomic_bytes,read_json,write_json,fingerprint
    from deepfake_detection.data.preprocessing import preprocessing_hash
    ws=workspace or Workspace(); run=ws.output(run)
    if not isinstance(epoch,int) or epoch<1: raise ValueError('Snapshot epoch must be positive')
    payload=save({key:value.detach().cpu().contiguous() for key,value in model.state_dict().items()})
    sha=hashlib.sha256(payload).hexdigest(); filename=f'snapshot-{epoch}-{sha}.safetensors'
    registry=read_json(run/'snapshots.json') if (run/'snapshots.json').exists() else {'snapshots':[]}
    entry={'epoch':epoch,'run_id':f'{run.name}-snapshot-{epoch}','weights_file':filename,'weights_sha256':sha,'preprocessing_hash':preprocessing_hash(cfg),'model_transform_hash':fingerprint(cfg['model']),'pilot':bool(cfg.get('pilot'))}
    for previous in registry['snapshots']:
        if previous['epoch']==epoch:
            if previous!=entry: raise ValueError('Snapshot epoch already committed with different weights/metadata')
            return previous
    atomic_bytes(run/filename,payload,ws)
    registry['snapshots'].append(entry); registry['snapshots'].sort(key=lambda row:row['epoch'])
    write_json(run/'snapshots.json',registry,ws)
    return entry


class SnapshotEnsemble(EnsembleScorer):
    def __init__(self,model_dir,*,workspace=None,device='cpu',weights=None,threshold=.5):
        from pathlib import Path
        from deepfake_detection.common.paths import Workspace,read_json,fingerprint
        from deepfake_detection.models.model_loader import architecture,_local_state,verify_weights
        from deepfake_detection.data.preprocessing import preprocessing_hash
        from deepfake_detection.inference.engine import InferenceEngine
        ws=workspace or Workspace(); run=ws.output(model_dir); cfg=read_json(run/'config.json')
        if cfg['model']['class_map']!={'fake':0,'real':1}: raise ValueError('Snapshot class semantics differ')
        registry=read_json(run/'snapshots.json'); entries=registry.get('snapshots',[])
        if len(entries)<2: raise ValueError('Snapshot ensemble requires at least two explicitly committed epochs')
        if len({entry['epoch'] for entry in entries})!=len(entries): raise ValueError('Duplicate snapshot epochs')
        admitted=[]
        for entry in entries:
            filename=entry.get('weights_file','')
            if not filename or Path(filename).name!=filename: raise ValueError('Snapshot weights must be a filename inside the run')
            if entry.get('preprocessing_hash')!=preprocessing_hash(cfg) or entry.get('model_transform_hash')!=fingerprint(cfg['model']): raise ValueError('Snapshot contract differs from saved config')
            path=ws.output(run/filename)
            try: verify_weights(path,expected_size=None,expected_sha=entry['weights_sha256'])
            except (OSError,ValueError) as exc: raise ValueError(f'Snapshot hash/file admission failed: {path}; {exc}') from exc
            admitted.append((entry,path))
        engines=[]; detector=None
        for entry,path in admitted:
            import torch
            state=_local_state(path)
            if not all(bool(torch.isfinite(value).all()) for value in state.values()): raise ValueError('Snapshot contains nonfinite tensors')
            model=architecture(cfg['model']); model.load_state_dict(state,strict=True)
            engine=InferenceEngine(model,cfg,entry,detector=detector,device=device)
            detector=engine.detector; engines.append(engine)
        super().__init__(engines,weights,threshold=threshold)
