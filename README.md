# 🏠 House Prices — Previsão com Machine Learning

Projeto de Machine Learning para prever o preço de venda de imóveis residenciais usando o famoso **Ames Housing Dataset** do Kaggle.

## 📌 Objetivo

Construir um pipeline completo de ML — desde a análise exploratória até a comparação de modelos — para estimar o valor de mercado de uma casa com base em ~80 características do imóvel.

---

## 📁 Estrutura do Projeto

```
house-prices-ml/
│
├── data/                   # Dados brutos (não versionados)
│   └── train.csv
│
├── notebooks/              # Análises exploratórias interativas
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── src/                    # Código modular e reutilizável
│   ├── data_loader.py      # Carregamento e inspeção dos dados
│   ├── preprocessor.py     # Limpeza, features e encoding
│   └── model.py            # Modelos, métricas e gráficos
│
├── outputs/                # Gráficos e resultados gerados
│
├── main.py                 # Pipeline completo em um comando
└── requirements.txt
```

---

## 🛠️ Tecnologias

| Biblioteca      | Uso                              |
|-----------------|----------------------------------|
| `pandas`        | Manipulação de dados             |
| `numpy`         | Operações numéricas              |
| `scikit-learn`  | Modelos de ML e métricas         |
| `xgboost`       | Gradient Boosting (melhor modelo)|
| `matplotlib`    | Visualizações                    |

---

## 🚀 Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/house-prices-ml.git
cd house-prices-ml
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Baixe os dados
Acesse [Kaggle - House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), baixe `train.csv` e coloque na pasta `data/`.

### 4. Execute o pipeline completo
```bash
python main.py
```

Ou explore os notebooks em ordem na pasta `notebooks/`.

---

## 📊 Modelos Comparados

| Modelo             | RMSE (log)  | R²     |
|--------------------|-------------|--------|
| Linear Regression  | ~0.163      | ~0.79  |
| Ridge Regression   | ~0.138      | ~0.84  |
| Random Forest      | ~0.142      | ~0.86  |
| **XGBoost**        | **~0.121**  | **~0.91** |

> Métricas obtidas via Cross-Validation 5-fold no conjunto de treino.

---

## 🔍 Features Engineering

Além das ~80 variáveis originais, foram criadas:

- **TotalSF** — Área total do imóvel (porão + 1º + 2º andar)
- **HouseAge** — Idade da casa no ano de venda
- **RemodAge** — Anos desde a última reforma
- **TotalBathrooms** — Total de banheiros (cheios + metades)
- **HasGarage / HasPool / HasFireplace** — Flags binárias

---

## 📈 Resultados

**XGBoost no conjunto de teste (hold-out 20%):**

- RMSE: **~$ 18.500**
- MAE:  **~$ 12.800**
- R²:   **~0.91**

O modelo explica ~91% da variância nos preços dos imóveis.

---

## 📚 Dataset

- **Fonte:** [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Autor original:** Dean De Cock (2011)
- **Registros:** 1.460 imóveis de Ames, Iowa (EUA)
- **Features:** 79 variáveis explicativas

---

## 👤 Autor

Feito por **Caio Blanco Schaidhauer** — [LinkedIn](https://www.linkedin.com/in/caio-blanco-501267253/) · [GitHub](https://github.com/CaioBlanco)
