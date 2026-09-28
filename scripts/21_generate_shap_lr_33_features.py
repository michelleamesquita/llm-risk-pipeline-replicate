#!/usr/bin/env python3
"""
Generate SHAP beeswarm plot for Logistic Regression model.
Similar to 20_generate_shap_33_features.py, but specifically for LR.
"""
from pathlib import Path
import argparse
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler


def load_protocol(path: Path):
    """Carrega o módulo de protocolo para reutilizar funções."""
    spec = importlib.util.spec_from_file_location("lr_rf_protocol", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(
        description="Gera gráfico SHAP beeswarm para Logistic Regression"
    )
    ap.add_argument(
        "--input",
        default="problem_statement_ablation_results/case_model_problem_statement_features.csv",
        help="Arquivo CSV com features (case_model_problem_statement_features.csv)",
    )
    ap.add_argument(
        "--protocol-script",
        default="scripts/lr_rf_post_generation_30runs.py",
        help="Script com protocolo de features",
    )
    ap.add_argument(
        "--output",
        default="imgs/fig_shap_lr_33_features.png",
        help="Caminho de saída para o gráfico",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=300,
        help="Número máximo de amostras para SHAP",
    )
    ap.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    ap.add_argument(
        "--max-display",
        type=int,
        default=15,
        help="Número máximo de features para exibir",
    )
    args = ap.parse_args()

    print("🔍 Carregando protocolo...")
    mod = load_protocol(Path(args.protocol_script))

    print(f"📂 Lendo dados de: {args.input}")
    df = pd.read_csv(args.input)
    print(f"   Total de linhas: {len(df)}")

    print("✅ Verificando colunas requeridas...")
    mod.check_required_columns(df)

    print("🔨 Construindo matriz de design...")
    work, X = mod.build_design_matrix(df)
    y = work[mod.TARGET].astype(int).to_numpy()
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {len(X)}")
    print(f"   Classe positiva: {y.sum()} ({100*y.mean():.1f}%)")

    # Imputação com medianas
    print("🔧 Imputando valores faltantes...")
    med = X.median(numeric_only=True).fillna(0.0)
    Xf = X.fillna(med).fillna(0.0)

    # Scaling (LR precisa de scaling, diferente do RF)
    print("📊 Aplicando StandardScaler...")
    scaler = StandardScaler()
    Xf_scaled = pd.DataFrame(
        scaler.fit_transform(Xf),
        columns=Xf.columns,
        index=Xf.index,
    )

    # Treinar modelo LR
    print("🤖 Treinando Logistic Regression...")
    lr = mod.make_lr(args.seed)
    lr.fit(Xf_scaled, y)
    print(f"   Acurácia no treino: {lr.score(Xf_scaled, y):.3f}")

    # Selecionar subset aleatório para SHAP
    print(f"🎲 Selecionando {min(args.max_rows, len(Xf))} amostras aleatórias...")
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(Xf_scaled), size=min(args.max_rows, len(Xf_scaled)), replace=False)
    Xs = Xf_scaled.iloc[idx]

    # Calcular SHAP values
    print("💡 Calculando SHAP values...")
    # Para Logistic Regression, usamos LinearExplainer ou KernelExplainer
    # LinearExplainer é mais rápido para modelos lineares
    explainer = shap.LinearExplainer(lr, Xf_scaled)
    shap_values = explainer.shap_values(Xs)

    # Compatibilidade entre versões do SHAP
    if isinstance(shap_values, list):
        values = shap_values
        if len(values) == 2:
            # Para classificação binária, pegamos a classe positiva
            values = values[1]
    else:
        values = shap_values

    # Gerar gráfico
    print("📈 Gerando gráfico SHAP beeswarm...")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        values,
        Xs,
        feature_names=X.columns.tolist(),
        show=False,
        max_display=args.max_display,
        plot_size=(10, 8),
    )
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Gráfico salvo em: {out}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Amostras no SHAP: {len(Xs)}")
    print(f"   Total de linhas no dataset: {len(df)}")
    print("\n🎉 Concluído!")


if __name__ == "__main__":
    main()
