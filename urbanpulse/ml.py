import numpy as np
import xgboost as xgb
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
    Hybrid object that acts as a list of feature column names when iterated or indexed,
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


# Exported object expected by app.py imports
numeric_feature_columns = NumericFeatureColumns()


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
    Trains ML models (XGBoost) on morphology features, encoding 
    discontinuous class target integers into continuous [0, 1, 2, ...] ranges.
    """
    # Extract feature columns using the hybrid resolver
    feature_cols = numeric_feature_columns(gdf)
    
    X = gdf[feature_cols].values
    y_raw = labels if isinstance(labels, np.ndarray) else gdf['label'].values
    groups = gdf['group_id'].values if 'group_id' in gdf.columns else np.arange(len(gdf))

    # Fix: Encode target labels into contiguous zero-indexed integers [0, 1, 2, ...]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Initialize XGBoost model
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        eval_metric='mlogloss'
    )

    # Run spatial evaluation with encoded continuous targets
    (cv_report, cv_cm), cv_name = _spatial_eval(model, X, y, groups)

    # Fit final model on complete dataset
    model.fit(X, y)

    # Store mapping metadata for decoding predictions later
    metadata = {
        'label_encoder': le,
        'classes': le.classes_.tolist()
    }

    training_table = {
        'feature_count': X.shape[1],
        'sample_count': X.shape[0],
        'classes_mapped': dict(zip(map(int, le.transform(le.classes_)), le.classes_))
    }

    evaluations = {
        'spatial_cv_name': cv_name,
        'report': cv_report,
        'confusion_matrix': cv_cm
    }

    models = {'xgboost': model}

    return models, feature_cols, evaluations, metadata, training_table