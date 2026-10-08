import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

# Metadata / leakage columns excluded from model features.  Numeric morphology,
# spectral, surface and network features are discovered automatically so the
# same model code scales when new feature columns are added to the pipeline.
FEATURE_EXCLUDE = {
    'cell_id', 'id', 'row', 'col', 'cell_area_m2', 'year', 'cell_uid',
    'geometry', 'label', 'predicted_class', 'group_id', 'imi',
    'informal_probability', 'confidence'
}

class NumericFeatureColumns(list):
    def __init__(self, default_cols=None):
        super().__init__(default_cols or [])

    def __call__(self, df=None):
        if df is None:
            return list(self)
        return [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in FEATURE_EXCLUDE]

numeric_feature_columns = NumericFeatureColumns()

def load_labels(file_path_or_buffer):
    if isinstance(file_path_or_buffer, str):
        if file_path_or_buffer.lower().endswith('.csv'):
            return pd.read_csv(file_path_or_buffer)
        import geopandas as gpd
        return gpd.read_file(file_path_or_buffer)
    try:
        return pd.read_csv(file_path_or_buffer)
    except Exception:
        import geopandas as gpd
        return gpd.read_file(file_path_or_buffer)

def prepare_labels(gdf, labels_df):
    df = gdf.copy()
    if isinstance(labels_df, pd.DataFrame) and 'label' in labels_df.columns:
        if 'cell_id' in labels_df.columns and 'cell_id' in df.columns:
            lab = labels_df[['cell_id', 'label']].copy()
            df = df.drop(columns=['label'], errors='ignore').merge(lab, on='cell_id', how='left')
        elif len(labels_df) == len(df):
            df['label'] = labels_df['label'].values
    target_col = 'label' if 'label' in df.columns else None
    if target_col is None:
        for candidate in ['predicted_class', 'class', 'auto_label']:
            if candidate in df.columns:
                target_col = candidate
                break
    if target_col:
        out = df.dropna(subset=[target_col]).copy()
        out = out[out[target_col].astype(str).str.len() > 0].copy()
        return out, out[target_col].astype(str).values
    return df, None

def _spatial_groups(gdf, block_size_cells=4):
    if {'row', 'col'}.issubset(gdf.columns):
        return ((gdf['row'].astype(int) // block_size_cells) * 100000 +
                (gdf['col'].astype(int) // block_size_cells)).to_numpy()
    if 'group_id' in gdf.columns:
        return gdf['group_id'].to_numpy()
    # Last-resort fallback; app data normally contains row/col.
    return np.arange(len(gdf))

def _spatial_eval(model, X, y, groups, seed=42):
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError('Need at least 2 spatial groups for validation.')
    n_splits = min(5, n_groups)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=1)
    return (classification_report(y, pred, output_dict=True, zero_division=0),
            confusion_matrix(y, pred), n_splits)

def train_models(gdf, labels):
    feature_cols = numeric_feature_columns(gdf)
    if not feature_cols:
        raise ValueError('No numeric morphology features found.')

    work = gdf.copy()
    if labels is not None:
        work['label'] = np.asarray(labels).ravel()
    work = work.dropna(subset=['label']).copy()
    work['label'] = work['label'].astype(str)
    X = work[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    le = LabelEncoder()
    y = le.fit_transform(work['label'].to_numpy())
    if len(le.classes_) < 2:
        raise ValueError('Training requires at least 2 classes.')
    groups = _spatial_groups(work, block_size_cells=4)

    xgb_model = xgb.XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
        objective='multi:softprob', eval_metric='mlogloss',
        tree_method='hist', n_jobs=-1, random_state=42
    )
    rf_model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, max_features='sqrt',
        class_weight='balanced_subsample', n_jobs=-1, random_state=42
    )

    xgb_report, xgb_cm, n_splits = _spatial_eval(xgb_model, X, y, groups)
    rf_report, rf_cm, _ = _spatial_eval(rf_model, X, y, groups)
    xgb_model.fit(X, y)
    rf_model.fit(X, y)

    evaluations = {
        'xgboost': {'model_name':'xgboost','spatial_cv_name':f'StratifiedGroupKFold ({n_splits} spatial folds)',
                    'cv_report':xgb_report,'cv_cm':xgb_cm},
        'random_forest': {'model_name':'random_forest','spatial_cv_name':f'StratifiedGroupKFold ({n_splits} spatial folds)',
                          'cv_report':rf_report,'cv_cm':rf_cm}
    }
    scores = {k: v['cv_report']['macro avg']['f1-score'] for k,v in evaluations.items()}
    selected = max(scores, key=scores.get)
    models = {'xgboost': xgb_model, 'random_forest': rf_model}
    counts = work['label'].value_counts().to_dict()
    metadata = {
        'selected_model': selected, 'n_rows': len(work), 'n_samples': len(work),
        'label_counts': {str(k): int(v) for k,v in counts.items()},
        'label_encoder': le, 'classes': le.classes_.tolist(),
        'spatial_block_cells': 4, 'spatial_groups': int(len(np.unique(groups))),
        'feature_count': len(feature_cols),
        'label_note': 'Promoted weak/reference labels; not an independently verified ground-truth benchmark.'
    }
    training_table = {'feature_count':len(feature_cols),'sample_count':len(work),
                      'classes_mapped':dict(zip(range(len(le.classes_)), le.classes_))}
    return models, feature_cols, evaluations, metadata, training_table

def predict_grid(gdf, model, feature_cols, label_encoder):
    df = gdf.copy()
    X = df[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    df['predicted_class'] = label_encoder.inverse_transform(preds.astype(int))
    classes = list(label_encoder.classes_)
    if 'informal' in classes:
        df['informal_probability'] = probs[:, classes.index('informal')]
    else:
        df['informal_probability'] = probs.max(axis=1)
    df['confidence'] = probs.max(axis=1)
    df['imi'] = df['informal_probability']
    return df

def explain_cell_prediction(model, cell_features, feature_cols):
    X_single = cell_features[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0).values.reshape(1,-1)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_single)
        arr = np.asarray(raw)
        pred = int(model.predict(X_single)[0])
        if arr.ndim == 3:
            vals = arr[0, :, pred]
        elif arr.ndim == 2:
            vals = arr[0]
        else:
            vals = arr.reshape(-1)[:len(feature_cols)]
        method = 'TreeSHAP'
    except Exception:
        imp = getattr(model, 'feature_importances_', np.ones(len(feature_cols))/len(feature_cols))
        vals = imp * X_single[0]
        method = 'feature-importance fallback'
    df = pd.DataFrame({'feature':feature_cols,'shap_value':np.asarray(vals).flatten()[:len(feature_cols)],
                       'feature_value':X_single.flatten()[:len(feature_cols)]})
    return df.sort_values('shap_value', key=lambda s: s.abs(), ascending=False), method

def save_bundle(output_path, models, feature_cols, metadata):
    bundle = {'model': models[metadata['selected_model']], 'models': models,
              'features': feature_cols, 'feature_cols': feature_cols,
              'metadata': metadata, 'label_encoder': metadata['label_encoder']}
    out = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(bundle, f)
    return out
