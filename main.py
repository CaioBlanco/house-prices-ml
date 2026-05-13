"""
main.py
-------
Script principal do pipeline de Machine Learning.

Executa em sequência:
    1. Carregamento dos dados
    2. Pré-processamento
    3. Comparação de modelos via Cross-Validation
    4. Treinamento do melhor modelo
    5. Geração de gráficos e métricas

Uso:
    python main.py

Requisitos:
    pip install -r requirements.txt
    Coloque train.csv (e opcionalmente test.csv) na pasta data/
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_data, summarize, get_feature_types
from src.preprocessor import prepare_data
from src.model import compare_models, train_best_model, plot_feature_importance, plot_predictions_vs_actual


def main():
    print("=" * 55)
    print("  🏠  House Prices — Pipeline de ML")
    print("=" * 55)

    # ── 1. Carregar dados ──────────────────────────────────
    print("\n[1/5] Carregando dados...")
    df_train, _ = load_data(
        train_path="data/train.csv",
        test_path="data/test.csv",
    )

    # ── 2. Resumo exploratório ─────────────────────────────
    print("\n[2/5] Resumo dos dados...")
    summarize(df_train)

    # ── 3. Pré-processamento ───────────────────────────────
    print("\n[3/5] Pré-processando dados...")
    X, y = prepare_data(df_train)

    # Split treino/teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

    # ── 4. Comparar modelos ────────────────────────────────
    print("\n[4/5] Comparando modelos (Cross-Validation 5-fold)...")
    df_results = compare_models(X_train.values, y_train.values)

    print("\n📋 Ranking de Modelos:")
    print(df_results.to_string(index=False))

    # Salva resultados
    os.makedirs("outputs", exist_ok=True)
    df_results.to_csv("outputs/model_comparison.csv", index=False)

    # ── 5. Treinar melhor modelo e gerar gráficos ──────────
    print("\n[5/5] Treinando o melhor modelo no conjunto completo...")
    result = train_best_model(
        X_train.values, y_train.values,
        X_test.values, y_test.values,
    )

    # Gráficos
    plot_feature_importance(
        result["model"],
        feature_names=list(X.columns),
        save_path="outputs/feature_importance.png"
    )

    preds_log = result["model"].predict(X_test.values)
    plot_predictions_vs_actual(
        y_true=np.expm1(y_test.values),
        y_pred=np.expm1(preds_log),
        save_path="outputs/predictions_vs_actual.png"
    )

    print("\n✅ Pipeline concluído! Resultados salvos em outputs/")
    print("=" * 55)


if __name__ == "__main__":
    main()
