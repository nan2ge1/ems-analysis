import xgboost as xgb
import pandas as pd
from typing import Tuple

def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    n_estimators: int = 1000,
    learning_rate: float = 0.01,
    early_stopping_rounds: int = 50
) -> xgb.XGBRegressor:
    """
    Train an XGBoost regressor.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        n_estimators: Number of trees.
        learning_rate: Learning rate.
        early_stopping_rounds: Early stopping rounds.
        
    Returns:
        Trained XGBRegressor model.
    """
    reg = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        objective='reg:squarederror'
    )
    
    reg.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )
    
    return reg

def predict_and_clip(model: xgb.XGBRegressor, X: pd.DataFrame) -> pd.Series:
    """
    Make predictions and clip negative values to 0.
    
    Args:
        model: Trained XGBRegressor.
        X: Features to predict on.
        
    Returns:
        Series of predictions.
    """
    predictions = model.predict(X)
    # Return as series to match index, but model.predict returns numpy array.
    pred_series = pd.Series(predictions, index=X.index)
    return pred_series.clip(lower=0)
