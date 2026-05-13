"""
preprocessor.py
---------------
Responsável pela limpeza, transformação e preparação dos dados
para o treinamento dos modelos.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# Colunas onde "ausente" significa que o imóvel não tem aquela feature
# (ex: sem garagem, sem porão) — preenchemos com "None" ou 0
NONE_FILL_CATEGORICAL = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

ZERO_FILL_NUMERIC = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
    "MasVnrArea",
]


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata valores ausentes de forma contextualizada:
    - Categóricas "ausentes por design" → 'None'
    - Numéricas "ausentes por design"   → 0
    - Demais categóricas                → moda
    - Demais numéricas                  → mediana

    Args:
        df: DataFrame bruto.

    Returns:
        DataFrame com nulos tratados.
    """
    df = df.copy()

    # Ausência com significado semântico
    for col in NONE_FILL_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in ZERO_FILL_NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Demais colunas
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"✅ Valores nulos restantes: {df.isnull().sum().sum()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features que podem melhorar a performance do modelo.

    Novas colunas criadas:
        - TotalSF        : área total do imóvel (porão + andares)
        - HouseAge       : idade da casa no ano de venda
        - RemodAge       : anos desde a última reforma
        - HasGarage      : flag binária (1 = tem garagem)
        - HasPool        : flag binária (1 = tem piscina)
        - HasFireplace   : flag binária (1 = tem lareira)
        - TotalBathrooms : total de banheiros

    Args:
        df: DataFrame após tratamento de nulos.

    Returns:
        DataFrame com novas features.
    """
    df = df.copy()

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["TotalBathrooms"] = (
        df["FullBath"] + 0.5 * df["HalfBath"] +
        df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte variáveis categóricas em numéricas via Label Encoding.
    Para produção, considere One-Hot Encoding para modelos lineares.

    Args:
        df: DataFrame com features já criadas.

    Returns:
        DataFrame totalmente numérico.
    """
    df = df.copy()
    le = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Normaliza as features com StandardScaler (média 0, desvio 1).
    O scaler é ajustado APENAS no treino para evitar data leakage.

    Args:
        X_train: Features de treino.
        X_test:  Features de teste.

    Returns:
        Tupla (X_train_scaled, X_test_scaled) como arrays numpy.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def prepare_data(df: pd.DataFrame, target_col: str = "SalePrice") -> tuple[pd.DataFrame, pd.Series]:
    """
    Pipeline completo de pré-processamento:
    fill_missing → engineer_features → encode_categoricals → separa X e y.

    Args:
        df:         DataFrame bruto de treino.
        target_col: Nome da coluna alvo.

    Returns:
        Tupla (X, y) prontos para uso com sklearn.
    """
    df = fill_missing_values(df)
    df = engineer_features(df)
    df = encode_categoricals(df)

    # Remove colunas com variância zero (não ajudam o modelo)
    df = df.drop(columns=["Id"], errors="ignore")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Log-transform no alvo: SalePrice tem distribuição assimétrica
    y_log = np.log1p(y)

    print(f"✅ Features finais: {X.shape[1]} | Amostras: {X.shape[0]}")
    return X, y_log
