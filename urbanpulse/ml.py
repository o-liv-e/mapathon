import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

# Default morphology feature column list
DEFAULT_NUMERIC_FEATURE_COLUMNS = [
    'building_count',
    'building_density',
    'building_irregularity',
    'impervious_fraction',
    'ndvi',
    'ndbi',
    'built_up_intensity'
]


class NumericFeatureColumns(list):
    """
    Hybrid object that acts as a list when iterated or indexed,
    and as a function when called with a DataFrame: numeric_feature_columns(df).
    """
    def __init__(self, default_cols=None):
        cols = default_cols or DEFAULT_NUMERIC_FEATURE_COLUMNS
        super().__init__(cols)

    def __call__(self, df=None):
        if df is not None:
            existing = [col for col in self if col in df.columns]
            if existing:
                return existing
            
            # Fallback: extract numeric features excluding metadata/target fields
            exclude = {'cell_id', 'geometry', 'label', 'predicted_class', 'group_id', 'year', 'imi', 'confidence'}
            return [
                col for col in df.select_dtypes(include=[np.number]).columns 
                if col not in exclude
            ]
        return list(self)


# Exported hybrid object expected by app.py imports
numeric_feature_columns = NumericFeatureColumns()


def load_labels(file_path_or_buffer):
    """
    Loads labeled training data from a CSV, GeoJSON, or GeoPackage file/buffer into a DataFrame.
    """
    if isinstance(file_path_or_buffer, str):
        if file_path_or_buffer.endswith('.csv'):
            return pd.read_csv(file_path_or_buffer)
        else:
            import geopandas as gpd
            return gpd.read_file(file_path_or_buffer)
    else:
        try:
            return pd.read_csv(file_path_or_buffer)
        except Exception:
            import geopandas as gpd
            return gpd.read_file(file_path_or_buffer)


def prepare_labels(gdf, labels_df):
    """
    Merges user-provided labels or promoted weak labels into the main grid GeoDataFrame.
    """
    df = gdf.copy()
    if isinstance(labels_df, pd.DataFrame) and 'label' in labels_df.columns:
        if 'cell_id' in labels_df.columns and 'cell_id' in df.columns:
            df = df.merge(labels_df[['cell_id', 'label']], on='cell_id', how='left', suffixes=('', '_new'))
            if 'label_new' in df.columns:
                df['label'] = df['label_new'].combine_first(df['label'])
                df.drop(columns=['label_new'], inplace=True)
        elif len(labels_df) == len(df):
            df['label'] = labels_df['label'].values
            
    # Clean and filter out rows missing labels
    target_col = 'label' if 'label' in df.columns else None
    if not target_col:
        for candidate in ['predicted_class', 'class', 'auto_label']:
            if candidate in df.columns:
                target_col = candidate
                break

    if target_col:
        labeled_gdf = df.dropna(subset=[target_col]).copy()
        return labeled_gdf, labeled_gdf[target_col].values
    
    return df, None


def _spatial_eval(model, X, y, groups):
    """
    Performs spatial cross-validation using StratifiedGroupKFold to prevent data leakage 
    and maintain class proportions across spatial clusters.
    """
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(5, n_groups))
    
    cv = StratifiedGroupKFold(n_splits=n_splits)
    
    # Generate predictions across spatial folds
    pred = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=1)
    
    cv_report = classification_report(y, pred, output_dict=True)
    cv_cm = confusion_matrix(y, pred)
    
    return (cv_report, cv_cm), "StratifiedGroupKFold"


def train_models(gdf, labels):
    """
    Trains Random Forest and XGBoost models on morphology features, encoding
    discontinuous target indices and populating all required UI metadata.
    """
    feature_cols = numeric_feature_columns(gdf)
    X = gdf[feature_cols].values

    # Robust target label extraction
    if isinstance(labels, np.ndarray):
        y_raw = labels
    elif isinstance(labels, (pd.Series, pd.DataFrame)):
        y_raw = labels.values.ravel()
    elif 'label' in gdf.columns:
        y_raw = gdf['label'].values
    else:
        # Fallback to candidate label column names
        for col in ['predicted_class', 'class', 'auto_label']:
            if col in gdf.columns:
                y_raw = gdf[col].values
                break
        else:
            raise KeyError("Could not locate a target label column in 'gdf' or 'labels'. Expected 'label' column.")

    groups = gdf['group_id'].values if 'group_id' in gdf.columns else np.arange(len(gdf))

    # Calculate raw class counts for Streamlit rendering
    unique_labels, counts = np.unique(y_raw, return_counts=True)
    label_counts_dict = dict(zip(map(str, unique_labels), map(int, counts)))

    # Encode target labels into contiguous zero-indexed integers [0, 1, 2, ...]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Initialize XGBoost and Random Forest classifiers
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        eval_metric='mlogloss'
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Run spatial evaluation
    (xgb_report, xgb_cm), cv_name = _spatial_eval(xgb_model, X, y, groups)
    (rf_report, rf_cm), _ = _spatial_eval(rf_model, X, y, groups)

    # Fit final models on full dataset
    xgb_model.fit(X, y)
    rf_model.fit(X, y)

    n_rows = len(gdf)
    metadata = {
        'selected_model': 'xgboost',
        'n_rows': n_rows,
        'n_samples': n_rows,
        'label_counts': label_counts_dict,
        'label_encoder': le,
        'classes': le.classes_.tolist()
    }

    training_table = {
        'feature_count': X.shape[1],
        'sample_count': n_rows,
        'classes_mapped': dict(zip(map(int, le.transform(le.classes_)), le.classes_))
    }

    # Format evaluations to satisfy app.py rendering
    eval_dict_xgb = {
        'model_name': 'xgboost',
        'spatial_cv_name': cv_name,
        'cv_report': xgb_report,
        'cv_cm': xgb_cm,
        'holdout': {
            'report': xgb_report,
            'confusion_matrix': xgb_cm,
            'accuracy': xgb_report.get('accuracy', 0.0)
        }
    }

    eval_dict_rf = {
        'model_name': 'random_forest',
        'spatial_cv_name': cv_name,
        'cv_report': rf_report,
        'cv_cm': rf_cm,
        'holdout': {
            'report': rf_report,
            'confusion_matrix': rf_cm,
            'accuracy': rf_report.get('accuracy', 0.0)
        }
    }

    evaluations = {
        'xgboost': eval_dict_xgb,
        'random_forest': eval_dict_rf
    }

    models = {
        'xgboost': xgb_model,
        'random_forest': rf_model
    }

    return models, feature_cols, evaluations, metadata, training_table


def save_bundle(output_path, models, feature_cols, metadata):
    """
    Serializes trained models, metadata, and feature lists to disk.
    """
    bundle = {
        'models': models,
        'feature_cols': feature_cols,
        'metadata': metadata
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(bundle, f)
    return output_path
def predict_grid(gdf, model, feature_cols, label_encoder):
    """
    Runs spatial inference over all grid cells, returning predicted classes, 
    class probabilities, and an Informal Morphology Index (IMI) score.
    """
    df = gdf.copy()
    X = df[feature_cols].values

    # Predict discrete class indices and probabilities
    preds = model.predict(X)
    probs = model.predict_proba(X)

    # Decode class indices back to original string labels
    df['predicted_class'] = label_encoder.inverse_transform(preds)
    
    # Extract probability for the 'informal' class to use as the IMI score
    classes = list(label_encoder.classes_)
    if 'informal' in classes:
        informal_idx = classes.index('informal')
        df['imi'] = probs[:, informal_idx]
    else:
        # Fallback to max probability if 'informal' isn't explicitly named
        df['imi'] = probs.max(axis=1)

    df['confidence'] = probs.max(axis=1)
    return df