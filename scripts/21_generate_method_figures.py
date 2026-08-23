#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

CORE_FEATURES = [
    'patch_lines','patch_added','patch_removed','patch_files_touched','patch_hunks','patch_churn','patch_net',
    'prompt_chars','prompt_lines','prompt_tokens',
    'patch_density','add_remove_ratio','net_per_line','hunks_per_file','prompt_density','prompt_token_density',
    'patch_complexity','change_intensity',
]


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def add_derived(df):
    df = df.copy()
    for c in ['patch_lines','patch_added','patch_removed','patch_files_touched','patch_hunks','patch_churn','patch_net',
              'prompt_chars','prompt_lines','prompt_tokens']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['patch_density'] = safe_div(df['patch_churn'], np.maximum(df['patch_lines'],1))
    df['add_remove_ratio'] = safe_div(df['patch_added'], np.maximum(df['patch_removed'],1))
    df['net_per_line'] = safe_div(df['patch_net'], np.maximum(df['patch_lines'],1))
    df['hunks_per_file'] = safe_div(df['patch_hunks'], np.maximum(df['patch_files_touched'],1))
    df['prompt_density'] = safe_div(df['prompt_chars'], np.maximum(df['prompt_lines'],1))
    df['prompt_token_density'] = safe_div(df['prompt_chars'], np.maximum(df['prompt_tokens'],1))
    df['patch_complexity'] = df['patch_hunks'] * df['patch_files_touched']
    df['change_intensity'] = safe_div(df['patch_churn'], np.maximum(df['patch_files_touched'],1))
    return df


def corr_figure(df, out_path):
    df = add_derived(df)
    cols = [c for c in CORE_FEATURES if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12.2, 10.5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    ax.set_title('Matriz de correlação entre atributos numéricos pré-SAST', fontsize=13)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i,j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5.5,
                    color='black' if abs(val) < 0.65 else 'white')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlação de Pearson')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def add_box(ax, x, y, w, h, text, fc, ec='#444444', fontsize=10):
    rect = Rectangle((x,y), w,h, facecolor=fc, edgecolor=ec, linewidth=1.4)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)
    return rect


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle='-|>', mutation_scale=14,
                                 linewidth=1.2, color='black'))


def pipeline_figure(out_path):
    fig, ax = plt.subplots(figsize=(15, 3.9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4)
    ax.axis('off')

    y, h = 1.35, 1.15
    boxes = [
        (0.25, 1.45, 'Base caso × modelo\n170 cases × 4 LLMs', '#ffffff'),
        (2.45, 1.60, 'Pré-processamento\nimputação + escala + one-hot', '#fff0c9'),
        (5.10, 1.80, 'Engenharia de atributos\n33 features pré-SAST', '#dce9ff'),
        (7.75, 1.75, 'Classificadores\nRL (baseline) + RF', '#ffe1c4'),
        (10.30, 1.85, 'Score pré-SAST\nprobabilidade de finding', '#f7d3d3'),
        (13.00, 1.65, 'Avaliação agrupada\nGroupShuffleSplit 80/20\n30 repetições', '#e7daf2'),
    ]
    rects=[]
    for x,w,text,fc in boxes:
        rects.append(add_box(ax,x,y,w,h,text,fc,fontsize=9.5))
    for i in range(len(rects)-1):
        r1, r2 = rects[i], rects[i+1]
        arrow(ax, r1.get_x()+r1.get_width(), y+h/2, r2.get_x(), y+h/2)

    # branch from score to operational use
    x_score = rects[4].get_x()+rects[4].get_width()/2
    ax.plot([x_score,x_score],[y,0.85], color='black', linewidth=1.1)
    arrow(ax, x_score,0.85, x_score,0.52)
    add_box(ax, 9.95,0.05,2.55,0.55,'Priorização da fila antes do SAST','#e3f3dc', fontsize=9)

    ax.text(7.5,3.55,'Pipeline de dados e modelagem pré-SAST',ha='center',va='center',fontsize=15,fontweight='bold')
    ax.text(7.5,3.08,'Nenhum atributo produzido pelo Bandit entra no classificador',ha='center',va='center',fontsize=10,style='italic')
    fig.tight_layout()
    fig.savefig(out_path,dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-dir', required=True)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.input)
    corr_figure(df, out/'corr_pre_sast_numeric.png')
    pipeline_figure(out/'pipeline_dados_pre_sast.png')
    print(out/'corr_pre_sast_numeric.png')
    print(out/'pipeline_dados_pre_sast.png')

if __name__=='__main__':
    main()
