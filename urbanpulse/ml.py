"""UrbanPulse supervised ML: Random Forest + XGBoost + spatial validation."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from .config import CLASSES, CLASS_IDX

LABEL_COLUMN = "label"
META_COLUMNS = {
    "cell_id", "cell_uid", "year", "row", "col", "label", "geometry",
    "predicted_class", "class_idx", "predicted_class_idx", "class_idx_raw",
    "explanation", "built_gate", "confidence", "informal_probability",
}

def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols=[]
    for c in df.columns:
        if c in META_COLUMNS or c.endswith("_class"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]): cols.append(c)
    return sorted(cols)

def load_labels(path_or_file) -> pd.DataFrame:
    name=getattr(path_or_file,"name",str(path_or_file)).lower()
    if name.endswith('.csv'):
        labels=pd.read_csv(path_or_file)
    else:
        import geopandas as gpd
        labels=gpd.read_file(path_or_file)
    labels=labels.rename(columns={"class":"label"})
    if "label" not in labels.columns: raise ValueError("Labels need a 'label' or 'class' column.")
    if "year" not in labels.columns: labels["year"]=2025
    labels["label"]=labels["label"].fillna("").astype(str).str.strip()
    labels=labels[labels["label"]!=""].copy()
    bad=sorted(set(labels["label"])-set(CLASSES))
    if bad: raise ValueError(f"Unknown labels: {bad}. Allowed: {CLASSES}")
    return labels

def prepare_labels(labels: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    if "cell_uid" in labels.columns and "cell_uid" in cells.columns:
        return labels[["cell_uid","label"] + (["year"] if "year" in labels.columns else [])].drop_duplicates("cell_uid")
    if {"cell_id","year"}.issubset(labels.columns):
        return labels[["cell_id","year","label"]].drop_duplicates(["cell_id","year"])
    if "geometry" not in labels.columns: raise ValueError("Polygon labels need geometry, or provide cell_uid/cell_id.")
    import geopandas as gpd
    polys=gpd.GeoDataFrame(labels,geometry="geometry",crs=labels.crs)
    if polys.crs is None: raise ValueError("Label polygons need a CRS.")
    polys=polys.to_crs(cells.crs)
    if "year" not in polys.columns: polys["year"]=2025
    centroids=cells[["cell_id","geometry"]].copy(); centroids["geometry"]=centroids.geometry.centroid
    j=gpd.sjoin(centroids,polys[["label","year","geometry"]],predicate="within",how="left")
    return j.dropna(subset=["label"])[["cell_id","year","label"]].drop_duplicates(["cell_id","year"])

def build_training_table(features, labels):
    keys=["cell_uid"] if "cell_uid" in features.columns and "cell_uid" in labels.columns else ["cell_id","year"]
    d=features.merge(labels,on=keys,how="inner")
    if d.empty: raise ValueError("No labelled cells matched the feature table.")
    d=d[d.label.isin(CLASSES)].copy()
    cols=numeric_feature_columns(d)
    d[cols]=d[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    return d,cols

def spatial_groups(d,block_size_cells=5):
    return (d["row"].astype(int)//block_size_cells)*100000+(d["col"].astype(int)//block_size_cells)

def _metrics(y,pred):
    report=classification_report(y,pred,labels=sorted(np.unique(y)),target_names=[CLASSES[i] for i in sorted(np.unique(y))],output_dict=True,zero_division=0)
    cm=confusion_matrix(y,pred,labels=sorted(np.unique(y)))
    return report,cm

def _spatial_eval(model,X,y,groups):
    n=int(pd.Series(groups).nunique())
    if n>=2:
        cv=GroupKFold(n_splits=min(5,n)); pred=cross_val_predict(model,X,y,cv=cv,groups=groups,n_jobs=1)
        cv_name=f"{min(5,n)}-fold spatial block CV"
    else:
        model.fit(X,y); pred=model.predict(X); cv_name="training-set evaluation (insufficient spatial blocks)"
    return _metrics(y,pred),cv_name

def _spatial_holdout(model,X,y,groups,seed=42):
    if pd.Series(groups).nunique()<4:
        return None,None
    for rs in [seed,seed+1,seed+2,seed+3,seed+4]:
        splitter=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=rs)
        tr,te=next(splitter.split(X,y,groups))
        if len(np.unique(y[te]))>=2 and len(np.unique(y[tr]))>=2:
            m=model.__class__(**model.get_params()); m.fit(X.iloc[tr],y[tr]); pred=m.predict(X.iloc[te])
            rep,cm=_metrics(y[te],pred)
            return (rep,cm, len(tr),len(te),rs),m
    return None,None

def make_models(seed=42,n_estimators=500):
    rf=RandomForestClassifier(n_estimators=n_estimators,min_samples_leaf=2,max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=seed)
    try:
        from xgboost import XGBClassifier
        xgb=XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.85,colsample_bytree=0.85,min_child_weight=2,objective="multi:softprob",eval_metric="mlogloss",tree_method="hist",n_jobs=-1,random_state=seed)
    except ImportError:
        xgb=None
    return rf,xgb

def train_models(features,labels,seed=42,n_estimators=500,block_size_cells=5):
    d,cols=build_training_table(features,labels); X=d[cols]; y=d.label.map(CLASS_IDX).astype(int).to_numpy(); groups=spatial_groups(d,block_size_cells)
    rf,xgb=make_models(seed,n_estimators); models={}; evaluations={}
    for name,model in [("random_forest",rf),("xgboost",xgb)]:
        if model is None: continue
        (cv_report,cv_cm),cv_name=_spatial_eval(model,X,y,groups)
        hold,hold_model=_spatial_holdout(model,X,y,groups,seed)
        model.fit(X,y); models[name]=model
        evaluations[name]={"cv":cv_name,"cv_report":cv_report,"cv_confusion_matrix":cv_cm.tolist(),"holdout":None}
        if hold:
            rep,cm,ntr,nte,rs=hold; evaluations[name]["holdout"]={"report":rep,"confusion_matrix":cm.tolist(),"n_train":ntr,"n_test":nte,"seed":rs}
    # Choose by spatial CV macro-F1, not training score.
    selected=max(evaluations,key=lambda k:evaluations[k]["cv_report"]["macro avg"]["f1-score"])
    metadata={"classes":CLASSES,"features":cols,"n_rows":len(d),"n_spatial_blocks":int(groups.nunique()),"label_counts":d.label.value_counts().to_dict(),"selected_model":selected,"models":{}}
    for name,e in evaluations.items():
        metadata["models"][name]={"cv":e["cv"],"cv_macro_f1":e["cv_report"]["macro avg"]["f1-score"],"cv_weighted_f1":e["cv_report"]["weighted avg"]["f1-score"],"holdout_macro_f1":(e["holdout"]["report"]["macro avg"]["f1-score"] if e["holdout"] else None)}
    return models,cols,evaluations,metadata,d

def predict(model,cols,features):
    out=features.copy(); X=out[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); proba=model.predict_proba(X); classes=np.asarray(getattr(model,"classes_",np.arange(proba.shape[1])),dtype=int); best=proba.argmax(1)
    out["predicted_class"]=[CLASSES[int(classes[i])] for i in best]; out["confidence"]=proba.max(1)
    ii=np.where(classes==CLASS_IDX["informal"])[0]; out["informal_probability"]=proba[:,ii[0]] if len(ii) else 0.0
    return out

def save_bundle(path,model,cols,metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); joblib.dump({"model":model,"features":cols,"metadata":metadata},path); path.with_suffix('.json').write_text(json.dumps(metadata,indent=2,default=float))

def load_model(path): return joblib.load(path)

def explain_row(model,cols,row,top_n=8):
    X=pd.DataFrame([row[cols].to_dict()]).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    try:
        import shap
        ex=shap.TreeExplainer(model); values=ex.shap_values(X); pred_idx=int(np.argmax(model.predict_proba(X)[0])); arr=np.asarray(values)
        if isinstance(values,list): vals=np.asarray(values[pred_idx])[0]
        elif arr.ndim==3: vals=arr[0,:,pred_idx]
        else: vals=arr[0]
        order=np.argsort(np.abs(vals))[::-1][:top_n]
        return {"method":"TreeSHAP","items":[(cols[i],float(vals[i])) for i in order]}
    except Exception as e:
        imp=pd.Series(getattr(model,"feature_importances_",np.zeros(len(cols))),index=cols).sort_values(ascending=False).head(top_n)
        return {"method":"Tree feature importance (SHAP unavailable)","items":[(k,float(v)) for k,v in imp.items()],"note":str(e)}
