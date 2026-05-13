"""
data_loader.py
--------------
Responsável por carregar e inspecionar os dados brutos do dataset
House Prices (Ames Housing).

Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
"""

import pandas as pd
import numpy as np
import os


def load_data(train_path: str, test_path: str = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Carrega os arquivos CSV do dataset.

    Args:
        train_path: Caminho para o CSV de treino.
        test_path: Caminho para o CSV de teste (opcional).

    Returns:
        Tupla (df_train, df_test). df_test é None se não for fornecido.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {train_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path) if test_path and os.path.exists(test_path) else None

    print(f"✅ Dados de treino carregados: {df_train.shape[0]} linhas, {df_train.shape[1]} colunas")
    if df_test is not None:
        print(f"✅ Dados de teste carregados:  {df_test.shape[0]} linhas, {df_test.shape[1]} colunas")

    return df_train, df_test


def summarize(df: pd.DataFrame) -> None:
    """
    Exibe um resumo rápido do DataFrame: tipos, nulos e estatísticas básicas.

    Args:
        df: DataFrame a ser inspecionado.
    """
    print("\n--- Informações Gerais ---")
    print(df.info())

    print("\n--- Valores Nulos (top 20) ---")
    nulls = df.isnull().sum().sort_values(ascending=False)
    print(nulls[nulls > 0].head(20))

    print("\n--- Estatísticas Descritivas (numéricas) ---")
    print(df.describe().T)


def get_feature_types(df: pd.DataFrame) -> tuple[list, list]:
    """
    Separa as colunas em numéricas e categóricas.

    Args:
        df: DataFrame de entrada.

    Returns:
        Tupla (colunas_numericas, colunas_categoricas).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove o alvo se estiver presente
    if "SalePrice" in numeric_cols:
        numeric_cols.remove("SalePrice")

    return numeric_cols, categorical_cols
