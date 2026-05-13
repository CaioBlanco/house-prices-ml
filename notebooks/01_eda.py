# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 01 — Análise Exploratória de Dados (EDA)
#
# Neste notebook vamos entender a estrutura do dataset antes de qualquer modelagem:
# - Distribuição do alvo (SalePrice)
# - Correlação das features numéricas com o preço
# - Análise de valores nulos
# - Distribuição de variáveis categóricas importantes

# ## 1. Imports e Configurações

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

import sys
sys.path.append("..")
from src.data_loader import load_data, get_feature_types

# ## 2. Carregar Dados

df, _ = load_data("../data/train.csv")
df.head()

# ## 3. Distribuição do Preço (Variável Alvo)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribuição original
sns.histplot(df["SalePrice"], kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("SalePrice — Distribuição Original")
axes[0].set_xlabel("Preço (USD)")

# Distribuição após log-transform
sns.histplot(np.log1p(df["SalePrice"]), kde=True, ax=axes[1], color="#55A868")
axes[1].set_title("log(SalePrice + 1) — Mais Normal")
axes[1].set_xlabel("log(Preço)")

plt.tight_layout()
plt.savefig("../outputs/eda_target_distribution.png", dpi=150)
plt.show()

print(f"Assimetria original:         {df['SalePrice'].skew():.2f}")
print(f"Assimetria após log-transform: {np.log1p(df['SalePrice']).skew():.2f}")

# ## 4. Mapa de Calor — Correlações com SalePrice

numeric_cols, _ = get_feature_types(df)
corr = df[numeric_cols + ["SalePrice"]].corr()

# Top 15 correlações com o alvo
top_corr = corr["SalePrice"].abs().sort_values(ascending=False).head(15).index

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    df[top_corr].corr(),
    annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, ax=ax
)
ax.set_title("Correlação — Top 15 Features com SalePrice")
plt.tight_layout()
plt.savefig("../outputs/eda_correlation_heatmap.png", dpi=150)
plt.show()

# ## 5. Análise de Valores Nulos

null_counts = df.isnull().sum().sort_values(ascending=False)
null_pct = (null_counts / len(df) * 100).round(1)
null_df = pd.DataFrame({"Nulos": null_counts, "%": null_pct})
null_df = null_df[null_df["Nulos"] > 0]

print(f"Colunas com valores nulos: {len(null_df)}")

fig, ax = plt.subplots(figsize=(10, 6))
null_df["%"].sort_values().plot(kind="barh", ax=ax, color="#C44E52")
ax.set_title("Porcentagem de Valores Nulos por Coluna")
ax.set_xlabel("% de Nulos")
plt.tight_layout()
plt.savefig("../outputs/eda_missing_values.png", dpi=150)
plt.show()

# ## 6. Top Features Numéricas vs SalePrice

top_features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath"]

fig, axes = plt.subplots(1, len(top_features), figsize=(18, 5))

for i, feat in enumerate(top_features):
    if df[feat].nunique() < 12:
        # Boxplot para variáveis discretas
        sns.boxplot(data=df, x=feat, y="SalePrice", ax=axes[i], palette="Blues")
    else:
        # Scatter para variáveis contínuas
        axes[i].scatter(df[feat], df["SalePrice"], alpha=0.3, s=10, color="#4C72B0")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("SalePrice")

    axes[i].set_title(f"{feat}")

plt.suptitle("Features Mais Correlacionadas com SalePrice", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("../outputs/eda_top_features.png", dpi=150)
plt.show()

# ## 7. Distribuição por Bairro

fig, ax = plt.subplots(figsize=(14, 6))
neighborhood_median = df.groupby("Neighborhood")["SalePrice"].median().sort_values()
neighborhood_median.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("Preço Mediano por Bairro (Neighborhood)")
ax.set_xlabel("Preço Mediano (USD)")
plt.tight_layout()
plt.savefig("../outputs/eda_neighborhood.png", dpi=150)
plt.show()

print("✅ EDA concluída! Gráficos salvos em outputs/")
