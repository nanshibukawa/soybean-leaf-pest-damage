#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark - Gráficos e Avaliação de Predição
Gera tabelas de desempenho comparativo, análise multi-objetivo de Pareto (Acurácia vs. Latência),
análise estratificada por resolução (Impacto do tamanho das imagens no INSECT12C)
e Heatmaps de F1-Score por classe para pragas de soja.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuração de caminhos do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "artifacts/model_evaluation/benchmark_data"
OUTPUT_DIR = ROOT_DIR / "graphics/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV = INPUT_DIR / "output_dataframe.csv"
INFERENCE_CSV = INPUT_DIR / "dataset.csv"
CLASSES_JSON = INPUT_DIR / "classes.json"

# Mapeamento legível dos modelos
MODEL_NAME_MAPPING = {
    'efficientnetv2b1_trained': 'EfficientNetV2-B1 (ImageNet)',
    'efficientnetv2b1_ip102_finetuned': 'EfficientNetV2-B1 (IP102 + Fine-Tuning)',
    'efficientnetv2b0_best': 'EfficientNetV2-B0 (IP102 + Fine-Tuning)',
    'mobilenetv3large_trained': 'MobileNetV3-Large (ImageNet)',
    'mobilenetv3large_best': 'MobileNetV3-Large (IP102 + Fine-Tuning)',
    'mobilenetv3small_trained': 'MobileNetV3-Small (ImageNet)',
    'mobilenetv3small_best': 'MobileNetV3-Small (IP102 + Fine-Tuning)',
    'mobilevit_custom': 'MobileViT (From Scratch)'
}

# Mapeamento legível dos datasets
DATASET_NAME_MAPPING = {
    'datasetpests': 'DatasetPests (Validação)',
    'insect12c': 'INSECT12C (Teste Robustez)'
}

# Mapeamento das 10 classes de pragas de soja
DEFAULT_CLASS_LABELS = {
    0: "A. gemmatalis",
    1: "Coccinellidae",
    2: "D. speciosa",
    3: "E. meditabunda",
    4: "E. heros",
    5: "Gastropoda",
    6: "L. villosa",
    7: "N. viridula",
    8: "R. schistocercoides",
    9: "S. albula"
}

def load_class_labels():
    """Carrega os nomes das classes se o arquivo classes.json existir."""
    if CLASSES_JSON.exists():
        with open(CLASSES_JSON, "r", encoding="utf-8") as f:
            classes = json.load(f)
            return {i: DEFAULT_CLASS_LABELS.get(i, name) for i, name in enumerate(classes)}
    return DEFAULT_CLASS_LABELS

def find_pareto_optimal(df, objectives_to_maximize, objectives_to_minimize):
    """Identifica os pontos Pareto-ótimos em um DataFrame."""
    is_pareto = pd.Series(True, index=df.index)
    for i, row_i in df.iterrows():
        if not is_pareto.loc[i]:
            continue
        for j, row_j in df.iterrows():
            if i == j:
                continue

            j_is_better_or_equal_to_i_in_all = True
            j_is_strictly_better_than_i_in_at_least_one = False

            for obj in objectives_to_maximize:
                if row_j[obj] < row_i[obj]:
                    j_is_better_or_equal_to_i_in_all = False
                    break
                if row_j[obj] > row_i[obj]:
                    j_is_strictly_better_than_i_in_at_least_one = True

            if not j_is_better_or_equal_to_i_in_all:
                continue

            for obj in objectives_to_minimize:
                if row_j[obj] > row_i[obj]:
                    j_is_better_or_equal_to_i_in_all = False
                    break
                if row_j[obj] < row_i[obj]:
                    j_is_strictly_better_than_i_in_at_least_one = True

            if not j_is_better_or_equal_to_i_in_all:
                continue

            if j_is_better_or_equal_to_i_in_all and j_is_strictly_better_than_i_in_at_least_one:
                is_pareto.loc[i] = False
                break
    return df[is_pareto]

def compute_performance_metrics(df_pred, df_inf):
    """Calcula as métricas consolidadas de classificação e mescla com os tempos de inferência."""
    print("📊 Calculando métricas globais de desempenho...")
    
    # 1. Acurácia
    def calc_acc(group):
        return accuracy_score(group['y_true_idx'], group['y_pred_idx'])
    acc_df = df_pred.groupby(['modelo', 'dataset']).apply(calc_acc, include_groups=False).reset_index(name='accuracy')

    # 2. Precisão Macro e Weighted
    def calc_prec_macro(group):
        return precision_score(group['y_true_idx'], group['y_pred_idx'], average='macro', zero_division=0)
    def calc_prec_weighted(group):
        return precision_score(group['y_true_idx'], group['y_pred_idx'], average='weighted', zero_division=0)
    
    prec_macro_df = df_pred.groupby(['modelo', 'dataset']).apply(calc_prec_macro, include_groups=False).reset_index(name='Precision')
    prec_weighted_df = df_pred.groupby(['modelo', 'dataset']).apply(calc_prec_weighted, include_groups=False).reset_index(name='wP')

    # 3. Recall Macro e Weighted
    def calc_rec_macro(group):
        return recall_score(group['y_true_idx'], group['y_pred_idx'], average='macro', zero_division=0)
    def calc_rec_weighted(group):
        return recall_score(group['y_true_idx'], group['y_pred_idx'], average='weighted', zero_division=0)

    rec_macro_df = df_pred.groupby(['modelo', 'dataset']).apply(calc_rec_macro, include_groups=False).reset_index(name='Recall')
    rec_weighted_df = df_pred.groupby(['modelo', 'dataset']).apply(calc_rec_weighted, include_groups=False).reset_index(name='wR')

    # 4. F1-Score Macro e Weighted
    def calc_f1_macro(group):
        return f1_score(group['y_true_idx'], group['y_pred_idx'], average='macro', zero_division=0)
    def calc_f1_weighted(group):
        return f1_score(group['y_true_idx'], group['y_pred_idx'], average='weighted', zero_division=0)

    f1_macro_df = df_pred.groupby(['modelo', 'dataset']).apply(calc_f1_macro, include_groups=False).reset_index(name='F1-score')
    f1_weighted_df = df_pred.groupby(['modelo', 'dataset']).apply(calc_f1_weighted, include_groups=False).reset_index(name='wF1')

    # Mesclar todas as métricas
    perf_df = acc_df.merge(prec_macro_df, on=['modelo', 'dataset'])
    perf_df = perf_df.merge(rec_macro_df, on=['modelo', 'dataset'])
    perf_df = perf_df.merge(f1_macro_df, on=['modelo', 'dataset'])
    perf_df = perf_df.merge(prec_weighted_df, on=['modelo', 'dataset'])
    perf_df = perf_df.merge(rec_weighted_df, on=['modelo', 'dataset'])
    perf_df = perf_df.merge(f1_weighted_df, on=['modelo', 'dataset'])

    # Desvio padrão por predição
    df_pred_copy = df_pred.copy()
    df_pred_copy['is_accurate'] = (df_pred_copy['y_true_idx'] == df_pred_copy['y_pred_idx']).astype(int)
    std_df = df_pred_copy.groupby(['modelo', 'dataset'])['is_accurate'].std().reset_index(name='std_deviation')
    perf_df = perf_df.merge(std_df, on=['modelo', 'dataset'])

    # Guardar acurácia numérica antes da formatação
    perf_df['numeric_accuracy'] = perf_df['accuracy']

    # Formatar acurácia com desvio padrão
    perf_df['accuracy_formatted'] = perf_df['accuracy'].apply(lambda x: f"{x:.4f}") + ' ' + perf_df['std_deviation'].apply(lambda x: f"(±{x:.2f})")

    # 5. Mesclar tempos de inferência
    if df_inf is not None and not df_inf.empty:
        mean_time = df_inf.groupby(['modelo', 'dataset', 'device'])['inference_time'].mean().reset_index()
        time_pivot = mean_time.pivot_table(
            index=['modelo', 'dataset'],
            columns='device',
            values='inference_time'
        ).reset_index()
        time_pivot.columns.name = None

        rename_map = {
            'Fast-end': 'time_high_end',
            'Mid-end': 'time_mid_end',
            'Slow-end': 'time_low_end',
            'GPU': 'time_high_end',
            'CPU': 'time_mid_end'
        }
        time_pivot = time_pivot.rename(columns=rename_map)
        perf_df = perf_df.merge(time_pivot, on=['modelo', 'dataset'], how='left')

    # Garantir colunas de tempo caso ausentes
    for t_col in ['time_high_end', 'time_mid_end', 'time_low_end']:
        if t_col not in perf_df.columns:
            perf_df[t_col] = 0.0
        perf_df[t_col] = perf_df[t_col].round(2)

    # 6. Cálculo da Fronteira de Pareto
    perf_df['is_pareto_high_end'] = False
    perf_df['is_pareto_mid_end'] = False
    perf_df['is_pareto_low_end'] = False

    for d_name in perf_df['dataset'].unique():
        sub = perf_df[perf_df['dataset'] == d_name].copy()
        for tier, flag in [('time_high_end', 'is_pareto_high_end'),
                           ('time_mid_end', 'is_pareto_mid_end'),
                           ('time_low_end', 'is_pareto_low_end')]:
            pareto_pts = find_pareto_optimal(sub, objectives_to_maximize=['numeric_accuracy'], objectives_to_minimize=[tier])
            for m in pareto_pts['modelo'].unique():
                perf_df.loc[(perf_df['dataset'] == d_name) & (perf_df['modelo'] == m), flag] = True

    return perf_df

def plot_pareto_frontier(perf_df):
    """Gera gráficos de dispersão da Acurácia vs. Tempo de Inferência com a Fronteira de Pareto."""
    print("📈 Gerando gráficos de Fronteira de Pareto (Acurácia vs. Tempo de Inferência)...")
    
    device_plot_info = {
        'time_high_end': {'title': 'Fast-end (GPU)', 'pareto_flag': 'is_pareto_high_end'},
        'time_mid_end': {'title': 'Mid-end (CPU/Edge)', 'pareto_flag': 'is_pareto_mid_end'},
        'time_low_end': {'title': 'Slow-end (Mobile/IoT)', 'pareto_flag': 'is_pareto_low_end'}
    }

    unique_models = perf_df['modelo'].unique()
    model_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', '*', 'X']
    marker_map = {m: model_markers[i % len(model_markers)] for i, m in enumerate(unique_models)}
    
    palette = sns.color_palette('tab10', n_colors=len(unique_models))
    color_map = {m: palette[i] for i, m in enumerate(unique_models)}

    datasets = perf_df['dataset'].unique()
    pareto_colors = ['#1f77b4', '#2ca02c', '#d62728']

    # 1. Gráfico consolidado com todos os datasets
    fig, axes = plt.subplots(len(datasets), 3, figsize=(20, 6 * len(datasets)), squeeze=False)

    legend_elements = {}
    for m in unique_models:
        label = MODEL_NAME_MAPPING.get(m, m)
        legend_elements[label] = plt.Line2D([0], [0], marker=marker_map[m], color=color_map[m], markersize=12, linestyle='None')

    for row_idx, d_name in enumerate(datasets):
        df_sub = perf_df[perf_df['dataset'] == d_name].copy()
        d_display_name = DATASET_NAME_MAPPING.get(d_name, d_name.capitalize())

        f_label = f'Fronteira de Pareto ({d_display_name})'
        legend_elements[f_label] = plt.Line2D([0], [0], color=pareto_colors[row_idx % len(pareto_colors)], linestyle='--', linewidth=2.5)

        min_acc = df_sub['numeric_accuracy'].min()
        max_acc = df_sub['numeric_accuracy'].max()
        margin = max(0.04, (max_acc - min_acc) * 0.2)
        y_lim = (max(0.0, min_acc - margin), min(1.02, max_acc + margin))

        for col_idx, (col_time, info) in enumerate(device_plot_info.items()):
            ax = axes[row_idx, col_idx]

            sns.scatterplot(
                data=df_sub,
                x=col_time,
                y='numeric_accuracy',
                hue='modelo',
                palette=color_map,
                style='modelo',
                markers=marker_map,
                s=350,
                ax=ax,
                legend=False,
                edgecolor='black',
                linewidth=1.2
            )

            pareto_df = df_sub[df_sub[info['pareto_flag']] == True].sort_values(by=col_time)
            if not pareto_df.empty:
                sns.lineplot(
                    data=pareto_df,
                    x=col_time,
                    y='numeric_accuracy',
                    color=pareto_colors[row_idx % len(pareto_colors)],
                    linestyle='--',
                    linewidth=2.5,
                    ax=ax,
                    legend=False
                )

            ax.set_title(f'{d_display_name}\nAcurácia vs. {info["title"]}', fontsize=15, fontweight='bold')
            ax.set_xlabel(f'{info["title"]} (ms)', fontsize=14)
            if col_idx == 0:
                ax.set_ylabel('Acurácia Global', fontsize=14)
            else:
                ax.set_ylabel('')

            ax.tick_params(axis='both', labelsize=13)
            ax.set_ylim(y_lim)
            ax.set_xscale('log')
            ax.grid(True, which="both", ls="--", alpha=0.6)

    handles = list(legend_elements.values())
    labels = list(legend_elements.keys())
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(labels), 4), fontsize=14, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    pdf_path = OUTPUT_DIR / "accuracy_vs_inference_time_by_dataset.pdf"
    png_path = OUTPUT_DIR / "accuracy_vs_inference_time_by_dataset.png"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvo: {pdf_path}")

    # 2. Gráficos individuais por dataset
    for d_name in datasets:
        df_sub = perf_df[perf_df['dataset'] == d_name].copy()
        d_display_name = DATASET_NAME_MAPPING.get(d_name, d_name.capitalize())

        fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=False)
        fig.suptitle(f'Acurácia vs. Tempo de Inferência - {d_display_name}', fontsize=18, fontweight='bold', y=1.03)

        min_acc = df_sub['numeric_accuracy'].min()
        max_acc = df_sub['numeric_accuracy'].max()
        margin = max(0.04, (max_acc - min_acc) * 0.2)
        y_lim = (max(0.0, min_acc - margin), min(1.02, max_acc + margin))

        for col_idx, (col_time, info) in enumerate(device_plot_info.items()):
            ax = axes[col_idx]
            sns.scatterplot(
                data=df_sub,
                x=col_time,
                y='numeric_accuracy',
                hue='modelo',
                palette=color_map,
                style='modelo',
                markers=marker_map,
                s=350,
                ax=ax,
                legend=False,
                edgecolor='black',
                linewidth=1.2
            )

            pareto_df = df_sub[df_sub[info['pareto_flag']] == True].sort_values(by=col_time)
            if not pareto_df.empty:
                sns.lineplot(
                    data=pareto_df,
                    x=col_time,
                    y='numeric_accuracy',
                    color='#1f77b4',
                    linestyle='--',
                    linewidth=2.5,
                    ax=ax,
                    legend=False
                )

            ax.set_title(f'{info["title"]}', fontsize=15)
            ax.set_xlabel(f'{info["title"]} (ms)', fontsize=14)
            if col_idx == 0:
                ax.set_ylabel('Acurácia Global', fontsize=14)
            else:
                ax.set_ylabel('')

            ax.tick_params(axis='both', labelsize=13)
            ax.set_ylim(y_lim)
            ax.set_xscale('log')
            ax.grid(True, which="both", ls="--", alpha=0.6)

        fig.legend(handles[:len(unique_models)] + [handles[-1]], labels[:len(unique_models)] + [labels[-1]],
                   loc='lower center', bbox_to_anchor=(0.5, -0.05),
                   ncol=min(len(labels), 4), fontsize=14, frameon=True, fancybox=True, shadow=True)

        plt.tight_layout(rect=[0, 0.1, 1, 0.96])
        d_pdf = OUTPUT_DIR / f"accuracy_vs_inference_time_{d_name}.pdf"
        d_png = OUTPUT_DIR / f"accuracy_vs_inference_time_{d_name}.png"
        plt.savefig(d_pdf, format='pdf', bbox_inches='tight')
        plt.savefig(d_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Salvo: {d_pdf}")

def plot_resolution_impact_analysis(df_pred):
    """
    Gera a análise e gráfico de barras comparativo de impacto de resolução no INSECT12C:
    1. Todas as Imagens (Sem filtro)
    2. Imagens Médias (> 60x60 px)
    3. Imagens Grandes (> 100x100 px)
    """
    print("\n📐 Gerando Análise de Impacto da Resolução no INSECT12C...")
    df_insect = df_pred[df_pred['dataset'] == 'insect12c'].copy()

    if df_insect.empty or 'orig_width' not in df_insect.columns:
        print("⚠️ Colunas de dimensão 'orig_width' não encontradas em df_pred. Pulando análise estratificada.")
        return

    scenarios = {
        'Todas (Sem filtro)': df_insect,
        'Médias (> 60×60 px)': df_insect[(df_insect['orig_width'] > 60) & (df_insect['orig_height'] > 60)],
        'Grandes (> 100×100 px)': df_insect[(df_insect['orig_width'] > 100) & (df_insect['orig_height'] > 100)]
    }

    records = []
    class_labels_map = load_class_labels()
    unique_classes = sorted(df_insect['y_true_idx'].unique())
    class_display_names = [class_labels_map.get(c, f"Classe {c}") for c in unique_classes]

    for sc_name, sc_df in scenarios.items():
        n_samples = len(sc_df['modelo'].unique()) and int(len(sc_df) / len(sc_df['modelo'].unique()))
        for m in sorted(sc_df['modelo'].unique()):
            m_sub = sc_df[sc_df['modelo'] == m]
            acc = accuracy_score(m_sub['y_true_idx'], m_sub['y_pred_idx'])
            f1_macro = f1_score(m_sub['y_true_idx'], m_sub['y_pred_idx'], average='macro', zero_division=0)
            f1_weighted = f1_score(m_sub['y_true_idx'], m_sub['y_pred_idx'], average='weighted', zero_division=0)
            
            records.append({
                'Cenário de Resolução': sc_name,
                'Amostras': n_samples,
                'Modelo': MODEL_NAME_MAPPING.get(m, m),
                'modelo_id': m,
                'Acurácia': acc,
                'Macro F1': f1_macro,
                'Weighted F1': f1_weighted
            })

    res_df = pd.DataFrame(records)
    res_csv = OUTPUT_DIR / "resolution_summary_insect12c.csv"
    res_df.to_csv(res_csv, index=False)
    print(f"✅ Salvo resumo de resolução em: {res_csv}")

    # 1. Gráfico de Barras Agrupadas: Acurácia vs. Faixa de Resolução
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    
    palette = sns.color_palette("Set2", n_colors=len(res_df['Modelo'].unique()))
    ax = sns.barplot(
        data=res_df,
        x='Cenário de Resolução',
        y='Acurácia',
        hue='Modelo',
        palette=palette,
        edgecolor='black',
        linewidth=1.2
    )

    # Anotações de porcentagem no topo das barras
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height*100:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold',
                        xytext=(0, 4), textcoords='offset points')

    plt.title("Impacto da Resolução na Robustez Zero-Shot (INSECT12C)", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Faixas de Resolução dos Recortes", fontsize=15, fontweight='bold', labelpad=10)
    plt.ylabel("Acurácia Global", fontsize=15, fontweight='bold', labelpad=10)
    plt.ylim(0.40, 1.0)
    plt.legend(title="Modelos", fontsize=13, title_fontsize=14, loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()

    bar_pdf = OUTPUT_DIR / "resolution_impact_insect12c.pdf"
    bar_png = OUTPUT_DIR / "resolution_impact_insect12c.png"
    plt.savefig(bar_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(bar_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvo gráfico de barras: {bar_pdf}")

    # 2. Gerar Heatmaps para cada cenário de resolução
    scenario_files = {
        'Todas (Sem filtro)': 'all',
        'Médias (> 60×60 px)': 'gt60',
        'Grandes (> 100×100 px)': 'gt100'
    }

    for sc_name, file_suffix in scenario_files.items():
        sc_df = scenarios[sc_name]
        unique_models = sorted(sc_df['modelo'].unique())
        f1_matrix = {}
        for m in unique_models:
            m_data = sc_df[sc_df['modelo'] == m]
            y_t = m_data['y_true_idx']
            y_p = m_data['y_pred_idx']
            f1_matrix[m] = f1_score(y_t, y_p, average=None, labels=unique_classes, zero_division=0)

        f1_df = pd.DataFrame(f1_matrix).T
        f1_df.columns = class_display_names
        f1_df.index = [MODEL_NAME_MAPPING.get(m, m) for m in f1_df.index]

        # Salvar CSV
        f1_df.to_csv(OUTPUT_DIR / f"f1-score-insect12c-{file_suffix}.csv")

        # Plotar
        fig, ax = plt.subplots(figsize=(14, 7))
        heatmap_data = f1_df.to_numpy()
        cmap = plt.cm.viridis
        vmin, vmax = 0.2, 1.0

        im = ax.imshow(heatmap_data, cmap=cmap, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(class_display_names)))
        ax.set_yticks(np.arange(len(f1_df.index)))
        ax.set_xticklabels(class_display_names, rotation=35, ha='right', fontsize=14, fontweight='bold')
        ax.set_yticklabels(f1_df.index, fontsize=14, fontweight='bold')

        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                val = heatmap_data[i, j]
                text_color = "black" if val > 0.65 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=12, fontweight='bold')

        ax.set_xlabel('Classes de Pragas de Soja', fontsize=16, fontweight='bold', labelpad=12)
        ax.set_ylabel('Modelos Avaliados', fontsize=16, fontweight='bold', labelpad=12)
        ax.set_title(f'F1-Score por Classe - INSECT12C [{sc_name}]', fontsize=18, fontweight='bold', pad=15)

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('F1-Score', fontsize=15, fontweight='bold')
        cbar.ax.tick_params(labelsize=13)

        plt.tight_layout()
        pdf_p = OUTPUT_DIR / f"f1-score-insect12c-{file_suffix}.pdf"
        png_p = OUTPUT_DIR / f"f1-score-insect12c-{file_suffix}.png"
        plt.savefig(pdf_p, format='pdf', bbox_inches='tight')
        plt.savefig(png_p, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Salvo Heatmap [{sc_name}]: {pdf_p}")

def plot_f1_heatmaps(df_pred):
    """Gera Matrizes de Calor (Heatmaps) de F1-Score por classe para cada dataset."""
    print("\n🔥 Gerando Heatmaps de F1-Score por classe...")
    class_labels_map = load_class_labels()
    unique_classes = sorted(df_pred['y_true_idx'].unique())
    class_display_names = [class_labels_map.get(c, f"Classe {c}") for c in unique_classes]

    datasets = df_pred['dataset'].unique()

    for d_name in datasets:
        df_sub = df_pred[df_pred['dataset'] == d_name].copy()
        d_display_name = DATASET_NAME_MAPPING.get(d_name, d_name.capitalize())
        unique_models = sorted(df_sub['modelo'].unique())

        f1_matrix = {}
        for m in unique_models:
            m_data = df_sub[df_sub['modelo'] == m]
            y_t = m_data['y_true_idx']
            y_p = m_data['y_pred_idx']
            f1_per_cls = f1_score(y_t, y_p, average=None, labels=unique_classes, zero_division=0)
            f1_matrix[m] = f1_per_cls

        f1_df = pd.DataFrame(f1_matrix).T
        f1_df.columns = class_display_names
        f1_df.index = [MODEL_NAME_MAPPING.get(m, m) for m in f1_df.index]

        csv_path = OUTPUT_DIR / f"f1-score-{d_name}.csv"
        f1_df.to_csv(csv_path)
        print(f"✅ Salvo: {csv_path}")

        fig, ax = plt.subplots(figsize=(14, 7))
        heatmap_data = f1_df.to_numpy()

        cmap = plt.cm.viridis
        vmin = max(0.0, np.nanmin(heatmap_data) - 0.05)
        vmax = min(1.0, np.nanmax(heatmap_data) + 0.05)
        
        im = ax.imshow(heatmap_data, cmap=cmap, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(len(class_display_names)))
        ax.set_yticks(np.arange(len(f1_df.index)))
        ax.set_xticklabels(class_display_names, rotation=35, ha='right', fontsize=14, fontweight='bold')
        ax.set_yticklabels(f1_df.index, fontsize=14, fontweight='bold')

        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                val = heatmap_data[i, j]
                text_color = "black" if val > (vmin + vmax) / 2 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=12, fontweight='bold')

        ax.set_xlabel('Classes de Pragas de Soja', fontsize=16, fontweight='bold', labelpad=12)
        ax.set_ylabel('Modelos Avaliados', fontsize=16, fontweight='bold', labelpad=12)
        ax.set_title(f'F1-Score por Classe - {d_display_name}', fontsize=18, fontweight='bold', pad=15)

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('F1-Score', fontsize=15, fontweight='bold')
        cbar.ax.tick_params(labelsize=13)

        plt.tight_layout()
        pdf_path = OUTPUT_DIR / f"f1-score-{d_name}.pdf"
        png_path = OUTPUT_DIR / f"f1-score-{d_name}.png"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Salvo: {pdf_path}")

def main():
    print("\n" + "=" * 80)
    print(" 🚀 INICIANDO PROCESSAMENTO DE BENCHMARK E GERAÇÃO DE GRÁFICOS")
    print("=" * 80)

    if not PREDICTIONS_CSV.exists():
        print(f"❌ Arquivo de predições não encontrado em: {PREDICTIONS_CSV}")
        print("💡 Execute primeiro o script 'scripts/generate_benchmark_data.py' para gerar os dados.")
        sys.exit(1)

    print(f"📂 Lendo predições de: {PREDICTIONS_CSV.name}")
    df_pred = pd.read_csv(PREDICTIONS_CSV)

    df_inf = None
    if INFERENCE_CSV.exists():
        print(f"📂 Lendo tempos de inferência de: {INFERENCE_CSV.name}")
        df_inf = pd.read_csv(INFERENCE_CSV)

    # 1. Calcular métricas consolidadas
    performance_df = compute_performance_metrics(df_pred, df_inf)

    # 2. Salvar tabela consolidada
    cols_order = ['modelo', 'dataset', 'accuracy_formatted', 'numeric_accuracy', 'Precision', 'Recall', 'F1-score', 'wP', 'wR', 'wF1',
                  'time_high_end', 'time_mid_end', 'time_low_end', 'is_pareto_high_end', 'is_pareto_mid_end', 'is_pareto_low_end']
    existing_cols = [c for c in cols_order if c in performance_df.columns]
    performance_df = performance_df[existing_cols]

    perf_csv_path = OUTPUT_DIR / "performance_summary.csv"
    performance_df.to_csv(perf_csv_path, index=False)
    print(f"✅ Salvo resumo de desempenho em: {perf_csv_path}")

    # 3. Gerar gráficos de Pareto
    plot_pareto_frontier(performance_df)

    # 4. Gerar Heatmaps de F1 por classe globais
    plot_f1_heatmaps(df_pred)

    # 5. Gerar Análise Estratificada de Resolução
    plot_resolution_impact_analysis(df_pred)

    # 6. Exibir Resumo no Terminal
    print("\n" + "=" * 80)
    print(" 📋 TABELA CONSOLIDADA DE RESULTADOS (BENCHMARK)")
    print("=" * 80)
    display_df = performance_df.copy()
    display_df['modelo'] = display_df['modelo'].map(lambda m: MODEL_NAME_MAPPING.get(m, m))
    display_df['dataset'] = display_df['dataset'].map(lambda d: DATASET_NAME_MAPPING.get(d, d))
    
    summary_cols = ['modelo', 'dataset', 'accuracy_formatted', 'Precision', 'Recall', 'F1-score', 'time_high_end']
    available_summary_cols = [c for c in summary_cols if c in display_df.columns]
    print(display_df[available_summary_cols].to_string(index=False))

    print("=" * 80)
    print(f" 🎉 TODOS OS GRÁFICOS E TABELAS FORAM GERADOS COM SUCESSO EM:\n    {OUTPUT_DIR}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()