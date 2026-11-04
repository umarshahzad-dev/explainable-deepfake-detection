"""JSON-serializable probability calibrators fitted on independent groups."""
import math
import numpy as np
from deepfake_detection.evaluation.metrics import _arrays,real_probability


def _logits(scores):
    bounded=np.clip(scores,1e-12,1-1e-12)
    return np.log(bounded)-np.log1p(-bounded)


def fit_platt(labels,real_scores):
    from sklearn.linear_model import LogisticRegression
    labels,scores=_arrays(labels,real_scores)
    if len(np.unique(labels))!=2: raise ValueError('Platt fit requires both classes')
    model=LogisticRegression(C=1,random_state=42,max_iter=1000)
    model.fit(_logits(scores).reshape(-1,1),labels)
    return {'method':'platt','coef':float(model.coef_[0,0]),'intercept':float(model.intercept_[0]),'temperature':1.}


def fit_isotonic(labels,real_scores):
    from sklearn.isotonic import IsotonicRegression
    labels,scores=_arrays(labels,real_scores)
    if len(np.unique(labels))!=2: raise ValueError('Isotonic fit requires both classes')
    fitted=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(scores,labels)
    return {'method':'isotonic','x':fitted.X_thresholds_.tolist(),'y':fitted.y_thresholds_.tolist(),'temperature':1.}


def apply_calibration(real_scores,calibration):
    scores=np.asarray(real_scores,dtype=float)
    if scores.ndim!=1 or not scores.size or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('Calibration requires finite one-dimensional probabilities')
    method=calibration.get('method','temperature')
    if method=='temperature': return real_probability(_logits(scores),calibration.get('temperature',1.))
    if method=='platt':
        coefficient=float(calibration['coef']); intercept=float(calibration['intercept'])
        if not math.isfinite(coefficient) or not math.isfinite(intercept): raise ValueError('Nonfinite Platt coefficients')
        return real_probability(_logits(scores)*coefficient+intercept)
    if method=='isotonic':
        x=np.asarray(calibration['x'],dtype=float); y=np.asarray(calibration['y'],dtype=float)
        if x.ndim!=1 or not x.size or y.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or (np.diff(x)<=0).any() or (np.diff(y)<0).any() or ((x<0)|(x>1)|(y<0)|(y>1)).any():
            raise ValueError('Invalid isotonic interpolation knots')
        return np.interp(scores,x,y)
    if method=='median':
        members=calibration.get('calibrators',[])
        if len(members)!=3 or {member.get('method') for member in members}!={'temperature','platt','isotonic'}:
            raise ValueError('Median calibration requires exactly temperature, Platt, and isotonic fits')
        return np.median([apply_calibration(scores,member) for member in members],axis=0)
    raise ValueError(f'Unknown calibration method: {method}')


class CalibratorEnsemble:
    def __init__(self): self.record=None

    def fit(self,labels,real_scores):
        from deepfake_detection.evaluation.metrics import fit_temperature
        labels,scores=_arrays(labels,real_scores)
        fitted={'method':'median','temperature':1.,'calibrators':[
            {'method':'temperature','temperature':fit_temperature(labels,_logits(scores))},
            fit_platt(labels,scores),fit_isotonic(labels,scores)]}
        self.record=fitted
        from copy import deepcopy
        return deepcopy(fitted)

    def transform(self,scores):
        if self.record is None: raise ValueError('Calibrator ensemble requires explicit fit')
        return apply_calibration(scores,self.record)


def conformal_threshold(labels,scores,alpha=.05):
    """Split-conformal quantile for nonconformity1-P(true class).

    Marginal1-alpha coverage requires a fixed predictor and exchangeable
    independently held-out calibration/test examples. Correlated crops and
    domain shift do not satisfy this guarantee automatically.
    """
    labels,real=_arrays(labels,scores)
    if not math.isfinite(alpha) or not 0<alpha<1: raise ValueError('Conformal alpha must be strictly between0and1')
    nonconformity=np.where(labels==1,1-real,real)
    rank=math.ceil((len(labels)+1)*(1-alpha))
    return 1. if rank>len(labels) else float(np.sort(nonconformity)[rank-1])


def conformal_predict(real_scores,threshold):
    scores=np.asarray(real_scores,dtype=float)
    _arrays(np.zeros(len(scores)),scores)
    if not math.isfinite(threshold) or not 0<=threshold<=1: raise ValueError('Invalid conformal threshold')
    return [[label for label,nonconformity in ((0,real),(1,1-real)) if nonconformity<=threshold] for real in scores]
