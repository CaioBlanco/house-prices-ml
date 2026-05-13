"""
model.py
--------
Define, treina e avalia os modelos de Machine Learning.

Modelos implementados:
    1. Regressão Linear   → baseline simples
    2. Random Forest      → ensemble baseado em árvores
    3. XGBoost            → gradient boosting (geralmente o melhor)

Métricas utilizadas:
    - RMSE  (Root Mean Squared Error)   → penaliza erros grandes
    - MAE   (Mean Absolute Error)       → erro médio absoluto
    - R²    (Coeficiente de determinação) → % da variância explicada
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # backend sem janela gráfica

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost não instalado. Rode: pip install xgboost")


# ─────────────────────────────────────────────
# Definição dos modelos
# ─────────────────────────────────────────────

def get_models() -> dict:
    """
    Retorna um dicionário com os modelos a serem comparados.

    Returns:
        Dict {nome: instância_do_modelo}
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

    return models


# ─────────────────────────────────────────────
# Avaliação com Cross-Validation
# ─────────────────────────────────────────────

def evaluate_with_cv(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """
    Avalia um modelo usando K-Fold Cross-Validation.
    Usamos o log do preço como alvo (já transformado em prepare_data).

    Args:
        model:    Modelo sklearn.
        X:        Features de treino.
        y:        Alvo (log-transformado).
        n_splits: Número de folds.

    Returns:
        Dict com métricas médias e desvio padrão.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # RMSE via cross_val_score (usa neg_mean_squared_error)
    rmse_scores = np.sqrt(
        -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
    )
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

    return {
        "RMSE_mean": rmse_scores.mean(),
        "RMSE_std": rmse_scores.std(),
        "R2_mean": r2_scores.mean(),
        "R2_std": r2_scores.std(),
    }


def compare_models(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Compara todos os modelos via Cross-Validation e retorna um ranking.

    Args:
        X: Features.
        y: Alvo (log-transformado).

    Returns:
        DataFrame com métricas de todos os modelos, ordenado por RMSE.
    """
    models = get_models()
    results = []

    for name, model in models.items():
        print(f"  → Avaliando: {name}...")
        metrics = evaluate_with_cv(model, X, y)
        results.append({"Modelo": name, **metrics})

    df_results = pd.DataFrame(results).sort_values("RMSE_mean")
    return df_results


# ─────────────────────────────────────────────
# Treinamento final e importância de features
# ─────────────────────────────────────────────

def train_best_model(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Treina o melhor modelo (XGBoost ou Random Forest) no conjunto completo
    e calcula métricas no conjunto de teste.

    Args:
        X_train, y_train: Dados de treino.
        X_test,  y_test:  Dados de teste (hold-out).

    Returns:
        Dict com modelo treinado e métricas no teste.
    """
    model = (
        get_models()["XGBoost"] if XGBOOST_AVAILABLE
        else get_models()["Random Forest"]
    )
    model_name = "XGBoost" if XGBOOST_AVAILABLE else "Random Forest"

    model.fit(X_train, y_train)
    preds_log = model.predict(X_test)

    # Reverte log-transform para calcular métricas no preço real
    preds = np.expm1(preds_log)
    y_real = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_real, preds))
    mae = mean_absolute_error(y_real, preds)
    r2 = r2_score(y_real, preds)

    print(f"\n📊 Resultados do {model_name} no conjunto de teste:")
    print(f"   RMSE : $ {rmse:,.0f}")
    print(f"   MAE  : $ {mae:,.0f}")
    print(f"   R²   : {r2:.4f}")

    return {"model": model, "name": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}


def plot_feature_importance(model, feature_names: list, top_n: int = 20,
                             save_path: str = "outputs/feature_importance.png") -> None:
    """
    Plota as N features mais importantes do modelo.

    Args:
        model:         Modelo treinado (Random Forest ou XGBoost).
        feature_names: Lista com nomes das colunas.
        top_n:         Quantas features exibir.
        save_path:     Onde salvar o gráfico.
    """
    if not hasattr(model, "feature_importances_"):
        print("⚠️  Modelo não suporta feature_importances_")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 10))
    top.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(f"Top {top_n} Features Mais Importantes", fontsize=14)
    ax.set_xlabel("Importância")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Gráfico salvo em: {save_path}")


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                                save_path: str = "outputs/predictions_vs_actual.png") -> None:
    """
    Scatter plot: valores reais vs predições do modelo.
    Uma linha diagonal perfeita indica predições sem erro.

    Args:
        y_true:    Valores reais (escala original).
        y_pred:    Predições do modelo (escala original).
        save_path: Onde salvar o gráfico.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, color="#4C72B0", s=15)

    # Linha de predição perfeita
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Predição Perfeita")

    ax.set_xlabel("Preço Real (USD)")
    ax.set_ylabel("Preço Predito (USD)")
    ax.set_title("Valores Reais vs. Predições do Modelo")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Gráfico salvo em: {save_path}")
