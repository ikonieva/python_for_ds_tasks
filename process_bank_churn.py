from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# =============================================================================
# Constants
# =============================================================================

TARGET_COLUMN = "Exited"

ID_COLUMN = "CustomerId"
SURNAME_COLUMN = "Surname"

CATEGORICAL_COLUMNS = ["Geography", "Gender"]

DROP_COLUMNS = [ID_COLUMN, SURNAME_COLUMN]


# =============================================================================
# Data model
# =============================================================================

@dataclass(slots=True)
class PreprocessedData:
    """Container for processed datasets and fitted transformers."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series

    feature_names: list[str]

    scaler: Optional[MinMaxScaler]
    categorical_encoder: OneHotEncoder


# =============================================================================
# Dataset splitting
# =============================================================================

def split_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and validation sets.

    Args:
        df: Input dataframe.
        target_column: Name of the target column.
        test_size: Fraction of samples for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Training features, validation features,
        training targets, and validation targets.
    """
    input_columns = list(df.columns)[1:-1]

    return train_test_split(
        df[input_columns],
        df[target_column],
        test_size=test_size,
        random_state=random_state,
    )


# =============================================================================
# Data cleaning
# =============================================================================

def remove_unused_columns(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove columns that are not used during training.

    Args:
        train_df: Training feature dataframe.
        val_df: Validation feature dataframe.

    Returns:
        Cleaned training and validation dataframes.
    """
    return (
        train_df.drop(columns=DROP_COLUMNS),
        val_df.drop(columns=DROP_COLUMNS),
    )


# =============================================================================
# Feature encoding
# =============================================================================

def encode_categorical_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    One-hot encode all categorical features.

    Args:
        train_df: Training feature dataframe.
        val_df: Validation feature dataframe.

    Returns:
        Encoded training dataframe, encoded validation dataframe,
        and the fitted OneHotEncoder.
    """
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    train_encoded = encoder.fit_transform(train_df[CATEGORICAL_COLUMNS])
    val_encoded = encoder.transform(val_df[CATEGORICAL_COLUMNS])

    encoded_columns = encoder.get_feature_names_out(CATEGORICAL_COLUMNS)

    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoded_columns,
        index=train_df.index,
    )

    val_encoded_df = pd.DataFrame(
        val_encoded,
        columns=encoded_columns,
        index=val_df.index,
    )

    train_df = pd.concat(
        [
            train_df.drop(columns=CATEGORICAL_COLUMNS),
            train_encoded_df,
        ],
        axis=1,
    )

    val_df = pd.concat(
        [
            val_df.drop(columns=CATEGORICAL_COLUMNS),
            val_encoded_df,
        ],
        axis=1,
    )

    return train_df, val_df, encoder


# =============================================================================
# Feature scaling
# =============================================================================

def scale_numeric_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scale numeric features using MinMaxScaler.

    Args:
        train_df: Training feature dataframe.
        val_df: Validation feature dataframe.

    Returns:
        Scaled training dataframe, scaled validation dataframe,
        and the fitted scaler.
    """
    numeric_columns = train_df.select_dtypes(include="number").columns

    scaler = MinMaxScaler()

    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df[numeric_columns] = scaler.fit_transform(train_df[numeric_columns])
    val_df[numeric_columns] = scaler.transform(val_df[numeric_columns])

    return train_df, val_df, scaler


# =============================================================================
# Public API
# =============================================================================

def preprocess_data(
    raw_df: pd.DataFrame,
    scale_numeric: bool = True,
) -> PreprocessedData:
    """
    Prepare raw data for model training.

    This is the main entry point that should be called by the rest
    of the application.

    Args:
        raw_df: Raw input dataframe.
        scale_numeric: Whether numeric features should be scaled.

    Returns:
        A PreprocessedData object containing processed datasets
        and fitted preprocessing objects.
    """
    X_train, X_val, y_train, y_val = split_dataset(raw_df)

    X_train, X_val = remove_unused_columns(
        X_train,
        X_val,
    )

    X_train, X_val, categorical_encoder = encode_categorical_features(
        X_train,
        X_val,
    )

    scaler: Optional[MinMaxScaler] = None

    if scale_numeric:
        X_train, X_val, scaler = scale_numeric_features(
            X_train,
            X_val,
        )

    return PreprocessedData(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_names=X_train.columns.tolist(),
        scaler=scaler,
        categorical_encoder=categorical_encoder,
    )

def preprocess_new_data(
    df: pd.DataFrame,
    categorical_encoder: OneHotEncoder,
    scaler: MinMaxScaler | None = None,
) -> pd.DataFrame:
    """
    Preprocess new data using previously fitted transformers.

    This function applies the same preprocessing steps as were used
    during training without fitting any new transformers.

    Args:
        df: Raw input dataframe.
        categorical_encoder: Previously fitted OneHotEncoder.
        scaler: Previously fitted MinMaxScaler. If None, numeric
            features are not scaled.

    Returns:
        Preprocessed dataframe ready for prediction.
    """
    # Remove columns not used by the model
    df = df.drop(columns=DROP_COLUMNS).copy()

    # One-hot encode categorical features
    encoded = categorical_encoder.transform(df[CATEGORICAL_COLUMNS])

    encoded_columns = categorical_encoder.get_feature_names_out(
        CATEGORICAL_COLUMNS
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_columns,
        index=df.index,
    )

    df = pd.concat(
        [
            df.drop(columns=CATEGORICAL_COLUMNS),
            encoded_df,
        ],
        axis=1,
    )

    # Scale numeric features if a scaler is provided
    if scaler is not None:
        numeric_columns = df.select_dtypes(include="number").columns
        df[numeric_columns] = scaler.transform(df[numeric_columns])

    return df