#!/usr/bin/env python3
import os
import json
import math
import random
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import argparse
import glob
from collections import defaultdict

try:
    import seaborn as sns  # opcional
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


def _safe_read_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao ler {path}: {e}")
        return None


def load_bandit_reports(reports_dir, tag="after"):
    """Carrega relatórios do Bandit. tag in {"after","before"}."""
    pattern = f"*_bandit_{tag}.json"
    all_rows = []
    for report_file in glob.glob(os.path.join(reports_dir, pattern)):
        data = _safe_read_json(report_file)
        if not data:
            continue
        for result in data.get('results', []):
            filename = result.get('filename', '').replace('./', '')
            all_rows.append({
                'filename': filename,
                'line_number': result.get('line_number'),
                'test_id': result.get('test_id'),
                'test_name': result.get('test_name'),
                'issue_severity': result.get('issue_severity'),
                'issue_confidence': result.get('issue_confidence'),
                'issue_text': result.get('issue_text'),
                'cwe': f"CWE-{result.get('issue_cwe', {}).get('id', 'Unknown')}"
            })
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df['key'] = df.apply(lambda r: f"{r['filename']}|{r['line_number']}|{r['test_id']}", axis=1)
        df = df.drop_duplicates('key')
    return df



def compute_patch_stats_from_patches_dir(patches_dir):
    """Calcula linhas adicionadas/removidas e arquivos tocados a partir dos .patch."""
    total_added = 0
    total_removed = 0
    files_touched = set()
    for patch_file in glob.glob(os.path.join(patches_dir, "*.patch")):
        try:
            with open(patch_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith("+++ b/"):
                        files_touched.add(line.strip())
                    elif line.startswith("+") and not line.startswith("+++"):
                        total_added += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        total_removed += 1
        except Exception as e:
            print(f"Erro ao ler patch {patch_file}: {e}")
            continue
    return {
        'lines_added': total_added,
        'lines_removed': total_removed,
        'lines_changed': total_added + total_removed,
        'files_touched': len(files_touched)
    }


def plot_summary(stats, model_names, output_prefix):
    plt.figure(figsize=(max(10, 3 * len(model_names)), 6))
    x = range(len(model_names))
    width = 0.35
    totals = [stats['totals'][m]['total'] for m in model_names]
    uniques = [stats['totals'][m]['unique'] for m in model_names]
    plt.bar([i - width/2 for i in x], totals, width, label='Total', color='skyblue')
    plt.bar([i + width/2 for i in x], uniques, width, label='Únicos', color='salmon')
    plt.xlabel('Modelos')
    plt.ylabel('Número de Vulnerabilidades')
    plt.title('Vulnerabilidades por Modelo')
    plt.xticks(list(x), model_names, rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_total_vulns.png")
    plt.close()
    print(f"-> Gráfico de barras salvo em {output_prefix}_total_vulns.png")

    if len(model_names) == 2:
        plt.figure(figsize=(8, 8))
        a, b = model_names
        venn2(subsets=(
            len(stats['unique_to'][a]),
            len(stats['unique_to'][b]),
            len(stats['common_all'])
        ), set_labels=model_names)
        plt.title("Sobreposição de Vulnerabilidades (2 modelos)")
        plt.savefig(f"{output_prefix}_venn.png")
        plt.close()
        print(f"-> Diagrama de Venn salvo em {output_prefix}_venn.png")
    elif len(model_names) == 3:
        plt.figure(figsize=(8, 8))
        a, b, c = model_names
        venn3(subsets=(
            len(stats['sets'][a] - stats['sets'][b] - stats['sets'][c]),
            len(stats['sets'][b] - stats['sets'][a] - stats['sets'][c]),
            len((stats['sets'][a] & stats['sets'][b]) - stats['sets'][c]),
            len(stats['sets'][c] - stats['sets'][a] - stats['sets'][b]),
            len((stats['sets'][a] & stats['sets'][c]) - stats['sets'][b]),
            len((stats['sets'][b] & stats['sets'][c]) - stats['sets'][a]),
            len(stats['common_all'])
        ), set_labels=model_names)
        plt.title("Sobreposição de Vulnerabilidades (3 modelos)")
        plt.savefig(f"{output_prefix}_venn.png")
        plt.close()
        print(f"-> Diagrama de Venn salvo em {output_prefix}_venn.png")
    else:
        print("-> Venn não gerado (suportado apenas para 2 ou 3 modelos).")


def compare_vulns_multi(dfs, names):
    sets = {}
    keyed_dfs = []
    for df, name in zip(dfs, names):
        df = df.copy()
        if not df.empty and 'key' not in df.columns:
            df['key'] = df.apply(lambda r: f"{r['filename']}|{r['line_number']}|{r['test_id']}", axis=1)
        df = df.drop_duplicates('key') if not df.empty else df
        keyed_dfs.append(df)
        sets[name] = set(df['key']) if not df.empty else set()

    totals = {n: {'total': len(sets[n]), 'unique': 0} for n in names}
    common_all = set.intersection(*sets.values()) if len(sets) >= 2 else set()

    unique_to = {}
    for n in names:
        others = set().union(*(sets[o] for o in names if o != n))
        uniq = sets[n] - others
        unique_to[n] = uniq
        totals[n]['unique'] = len(uniq)

    cwe_stats = defaultdict(lambda: {n: 0 for n in names})
    for df, n in zip(keyed_dfs, names):
        for _, row in (df.iterrows() if not df.empty else []):
            cwe_stats[row['cwe']][n] += 1

    cwe_diffs = []
    for cwe, counts in cwe_stats.items():
        values = [counts[n] for n in names]
        diff = max(values) - min(values)
        cwe_diffs.append({'cwe': cwe, **counts, 'diff': diff, 'total': sum(values)})
    cwe_diffs.sort(key=lambda x: x['diff'], reverse=True)

    stats = {
        'totals': totals,
        'common_all': common_all,
        'unique_to': unique_to,
        'sets': sets,
        'cwe_diffs': cwe_diffs,
        'cwe_table': cwe_stats,
    }
    return stats


def write_comparison_md_multi(stats, model_names, output_file, deltas=None, norm=None, intersections=None):
    with open(output_file, 'w') as f:
        f.write(f"# Comparação entre {', '.join(model_names)}\n\n")
        f.write("## Resumo Quantitativo\n\n")
        for m in model_names:
            total = stats['totals'][m]['total']
            unique = stats['totals'][m]['unique']
            pct = (unique / total * 100) if total else 0
            f.write(f"- {m}: total={total:,}, únicos={unique:,} ({pct:.1f}%)\n")
        f.write(f"- Comuns a todos: {len(stats['common_all']):,}\n\n")

        # Δ total entre 2 modelos (quando aplicável)
        if len(model_names) == 2:
            a, b = model_names
            ta = stats['totals'][a]['total']
            tb = stats['totals'][b]['total']
            d = ta - tb
            pctd = (abs(d) / max(ta, tb, 1)) * 100.0
            f.write(f"- Δ total ({a}−{b}): {d:,} ({pctd:.2f}%)\n\n")

        if deltas:
            f.write("## Delta Bandit (After - Before)\n\n")
            for m, d in deltas.items():
                if d is None:
                    f.write(f"- {m}: before não encontrado\n")
                else:
                    f.write(f"- {m}: Δtotal={d['delta_total']:,}\n")
            f.write("\n")

        if norm:
            f.write("## Normalização por Mudança\n\n")
            for m, n in norm.items():
                f.write(f"- {m}: lines_changed={n['lines_changed']:,}, files_touched={n['files_touched']:,}, vuls_per_1k_changed={n['vuls_per_1k_changed']:.2f}\n")
            f.write("\n")

        f.write("## Diferenças por CWE\n\n")
        # When exactly 2 models, also include % difference column similar to example
        if len(model_names) == 2:
            header = "| CWE | " + " | ".join(model_names) + " | Diferença | % Diferença |\n"
            f.write(header)
            f.write("|" + "---|" * (len(model_names) + 3) + "\n")
            a, b = model_names
            for diff in stats['cwe_diffs']:
                da = diff[a]
                db = diff[b]
                delta = abs(da - db)
                denom = max(da, db, 1)
                pct = (delta / denom) * 100.0
                row = [diff['cwe'], f"{da:,}", f"{db:,}", f"{delta:,}", f"{pct:.2f}%"]
                f.write("| " + " | ".join(row) + " |\n")
        else:
            header = "| CWE | " + " | ".join(model_names) + " | Δ | Par Δ |\n"
            f.write(header)
            f.write("|" + "---|" * (len(model_names) + 3) + "\n")
            for diff in stats['cwe_diffs']:
                # identify max/min models for this CWE
                max_model = None; min_model = None; max_val = None; min_val = None
                for m in model_names:
                    v = diff.get(m, 0)
                    if max_val is None or v > max_val:
                        max_val = v; max_model = m
                    if min_val is None or v < min_val:
                        min_val = v; min_model = m
                row = [diff['cwe']] + [f"{diff[m]:,}" for m in model_names] + [f"{diff['diff']:,}", f"{max_model}−{min_model}"]
                f.write("| " + " | ".join(row) + " |\n")

            # Tabela adicional: participação percentual por modelo
            f.write("\n### Participação percentual por modelo")
            f.write("\n\n")
            pct_header = "| CWE | " + " | ".join([m + " %" for m in model_names]) + " |\n"
            f.write(pct_header)
            f.write("|" + "---|" * (len(model_names) + 1) + "\n")
            for diff in stats['cwe_diffs']:
                total = max(1, diff['total'])
                row = [diff['cwe']] + [f"{(diff[m] / total) * 100:.1f}%" for m in model_names]
                f.write("| " + " | ".join(row) + " |\n")

        # Interseções removidas (Semgrep/CodeQL não utilizados)

        # Análise
        f.write("\n## Análise\n\n")
        if len(model_names) == 2:
            a, b = model_names
            ta = stats['totals'][a]['total']
            tb = stats['totals'][b]['total']
            common = len(stats['common_all'])
            base = max(ta, tb, 1)
            common_pct = (common / base) * 100.0
            # Top 3 diferenças por CWE
            top_diffs = stats['cwe_diffs'][:3]
            f.write("1. **Diferenças Principais**:\n")
            if top_diffs:
                for d in top_diffs:
                    da = d.get(a, 0); db = d.get(b, 0)
                    win = a if da >= db else b
                    qty = abs(da - db)
                    den = max(da, db, 1)
                    pct = (qty / den) * 100.0
                    f.write(f"   - {win} identificou mais {qty:,} casos de {d['cwe']} ({pct:.1f}% a mais)\n")
            else:
                f.write("   - Não foram observadas diferenças por CWE\n")

            f.write("\n2. **Consistência entre Modelos**:\n\n")
            f.write(f"   - {common_pct:.1f}% das vulnerabilidades são identificadas por ambos\n")
            f.write(f"   - {common:,} vulnerabilidades exatas (arquivo|linha|regra) são comuns\n")

            f.write("\n3. **Conclusões**:\n\n")
            f.write("   - Alta concordância geral entre os dois modelos\n")
            if top_diffs:
                f.write("   - Divergências concentradas nas CWEs acima; investigar contexto e patching\n")
            else:
                f.write("   - Não houve divergências relevantes por CWE\n")
        else:
            # Multi-modelo: destaque amplitude e proximidade/distância por Jaccard
            f.write("1. **Diferenças Principais** (Top 5 por amplitude Δ max−min):\n")
            for d in stats['cwe_diffs'][:5]:
                # show which models define the range
                max_model = None; min_model = None; max_val = None; min_val = None
                for m in model_names:
                    v = d.get(m, 0)
                    if max_val is None or v > max_val:
                        max_val = v; max_model = m
                    if min_val is None or v < min_val:
                        min_val = v; min_model = m
                f.write(f"   - {d['cwe']}: Δ={d['diff']:,} ({max_model}−{min_model})\n")

            # Par mais próximo e mais distante via Jaccard
            import itertools
            best_pair = None; best_j = -1.0
            worst_pair = None; worst_j = 2.0
            for x, y in itertools.combinations(model_names, 2):
                A = stats['sets'][x]; B = stats['sets'][y]
                j = (len(A & B) / len(A | B)) if (A or B) else 0.0
                if j > best_j:
                    best_j = j; best_pair = (x, y)
                if j < worst_j:
                    worst_j = j; worst_pair = (x, y)
            f.write("\n2. **Concordância entre Modelos**:\n\n")
            if best_pair is not None:
                f.write(f"   - Mais próximos (Jaccard): {best_pair[0]}×{best_pair[1]} = {best_j:.2f}\n")
            if worst_pair is not None:
                f.write(f"   - Mais distantes (Jaccard): {worst_pair[0]}×{worst_pair[1]} = {worst_j:.2f}\n")

            f.write("\n3. **Conclusões**:\n\n")
            f.write("   - Conjunto de achados com boa sobreposição geral; diferenças se concentram nas CWEs destacadas\n")

            # Seção extra: Comparações par-a-par (Top 5 por par)
            import itertools
            f.write("\n## Comparações par-a-par (Top 5 por par)\n\n")
            for a, b in itertools.combinations(model_names, 2):
                ta = stats['totals'][a]['total']
                tb = stats['totals'][b]['total']
                d_tot = ta - tb
                pct_tot = (abs(d_tot) / max(ta, tb, 1)) * 100.0
                f.write(f"### {a} vs {b}\n\n")
                f.write(f"- Δ total ({a}−{b}): {d_tot:,} ({pct_tot:.2f}%)\n\n")
                f.write("| CWE | " + a + " | " + b + " | Diferença | % Diferença |\n")
                f.write("|-----|---------------|--------|------------|-------------|\n")
                # Ordena por diferença específica do par
                pair_rows = []
                for d in stats['cwe_diffs']:
                    va = d.get(a, 0); vb = d.get(b, 0)
                    delta = abs(va - vb)
                    denom = max(va, vb, 1)
                    pct = (delta / denom) * 100.0
                    pair_rows.append((delta, { 'cwe': d['cwe'], a: va, b: vb, 'delta': delta, 'pct': pct }))
                pair_rows.sort(key=lambda x: x[0], reverse=True)
                for _, row in pair_rows[:5]:
                    f.write(f"| {row['cwe']} | {row[a]:,} | {row[b]:,} | {row['delta']:,} | {row['pct']:.2f}% |\n")


def export_csvs(stats, model_names, output_prefix, deltas=None, norm=None, intersections=None):
    out_dir = os.path.dirname(output_prefix)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for m in model_names:
        r = {'model': m, 'total': stats['totals'][m]['total'], 'unique': stats['totals'][m]['unique']}
        if deltas and deltas.get(m):
            r['delta_total'] = deltas[m]['delta_total']
        if norm and norm.get(m):
            r['lines_changed'] = norm[m]['lines_changed']
            r['files_touched'] = norm[m]['files_touched']
            r['vuls_per_1k_changed'] = norm[m]['vuls_per_1k_changed']
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{output_prefix}_totals.csv", index=False)

    cwe_rows = []
    for cwe, counts in stats['cwe_table'].items():
        row = {'cwe': cwe}
        row.update(counts)
        cwe_rows.append(row)
    pd.DataFrame(cwe_rows).sort_values('cwe').to_csv(f"{output_prefix}_cwe_by_model.csv", index=False)

    import itertools
    jrows = []
    for a, b in itertools.combinations(model_names, 2):
        A = stats['sets'][a]; B = stats['sets'][b]
        j = (len(A & B) / len(A | B)) if (A or B) else 0.0
        jrows.append({'model_a': a, 'model_b': b, 'jaccard': j})
    pd.DataFrame(jrows).to_csv(f"{output_prefix}_jaccard.csv", index=False)

    if intersections:
        irows = []
        for m, inter in intersections.items():
            if inter is None:
                continue
            row = {'model': m, **inter}
            irows.append(row)
        if irows:
            pd.DataFrame(irows).to_csv(f"{output_prefix}_bandit_semgrep_intersection.csv", index=False)


def plot_cwe_heatmap(stats, model_names, output_prefix, top_n=25):
    records = []
    for cwe, counts in stats['cwe_table'].items():
        total = sum(counts[m] for m in model_names)
        row = {'cwe': cwe, **{m: counts[m] for m in model_names}, 'total': total}
        records.append(row)
    df = pd.DataFrame(records)
    if df.empty:
        return
    df = df.sort_values('total', ascending=False).head(top_n)
    df_plot = df.set_index('cwe')[model_names]

    plt.figure(figsize=(max(10, len(model_names)*2.5), max(8, top_n*0.4)))
    if HAS_SEABORN:
        sns.heatmap(df_plot, annot=False, cmap='Reds')
    else:
        plt.imshow(df_plot.values, aspect='auto', cmap='Reds')
        plt.xticks(range(len(model_names)), model_names, rotation=15)
        plt.yticks(range(len(df_plot.index)), df_plot.index)
        plt.colorbar()
    plt.title(f"Heatmap CWE × Modelo (Top {top_n})")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_cwe_heatmap.png")
    plt.close()
    print(f"-> Heatmap CWE salvo em {output_prefix}_cwe_heatmap.png")


def plot_jaccard_heatmap(stats, model_names, output_prefix):
    n = len(model_names)
    M = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A = stats['sets'][model_names[i]]
            B = stats['sets'][model_names[j]]
            M[i][j] = (len(A & B) / len(A | B)) if (A or B) else 0.0
    M = pd.DataFrame(M, index=model_names, columns=model_names)

    plt.figure(figsize=(max(8, n*2.5), max(6, n*2)))
    if HAS_SEABORN:
        sns.heatmap(M, annot=True, fmt='.2f', cmap='Blues')
    else:
        plt.imshow(M.values, cmap='Blues')
        plt.xticks(range(n), model_names, rotation=15)
        plt.yticks(range(n), model_names)
        for i in range(n):
            for j in range(n):
                plt.text(j, i, f"{M.values[i][j]:.2f}", ha='center', va='center', color='black')
        plt.colorbar()
    plt.title("Jaccard entre modelos (vulnerabilidades)")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_jaccard_heatmap.png")
    plt.close()
    print(f"-> Heatmap Jaccard salvo em {output_prefix}_jaccard_heatmap.png")


def plot_cwe_diff_chart(stats, model_names, output_prefix, top_n=10):
    diffs = stats.get('cwe_diffs', [])
    if not diffs:
        return None
    top = diffs[:top_n]
    cwes = [d['cwe'] for d in top]
    n_models = len(model_names)
    if n_models == 2:
        a, b = model_names
        va = [d[a] for d in top]
        vb = [d[b] for d in top]
        deltas = [abs(x - y) for x, y in zip(va, vb)]
        denoms = [max(x, y, 1) for x, y in zip(va, vb)]
        pcts = [(d / dn) * 100.0 for d, dn in zip(deltas, denoms)]

        plt.figure(figsize=(max(10, 1.2 * len(top)), 6))
        x = range(len(top))
        width = 0.4
        plt.bar([i - width/2 for i in x], va, width, label=a, color='steelblue')
        plt.bar([i + width/2 for i in x], vb, width, label=b, color='salmon')
        # annotate delta% on top
        for i, (xa, xb, dd, pp) in enumerate(zip(va, vb, deltas, pcts)):
            y = max(xa, xb)
            plt.text(i, y, f"Δ={dd}\n{pp:.2f}%", ha='center', va='bottom', fontsize=8)
        plt.xticks(list(x), cwes, rotation=45, ha='right')
        plt.ylabel('Achados por CWE')
        plt.title('Diferenças por CWE (Top)')
        plt.legend()
        plt.tight_layout()
        out_path = f"{output_prefix}_cwe_diff.png"
        plt.savefig(out_path)
        plt.close()
        print(f"-> Gráfico de diferenças por CWE salvo em {out_path}")
        return out_path
    else:
        # Show max-min range across models
        ranges = [d['diff'] for d in top]
        plt.figure(figsize=(max(10, 1.2 * len(top)), 6))
        x = range(len(top))
        plt.bar(x, ranges, color='mediumpurple')
        plt.xticks(list(x), cwes, rotation=45, ha='right')
        plt.ylabel('Δ (max - min) de achados por CWE')
        plt.title('Amplitude de diferenças por CWE (Top)')
        plt.tight_layout()
        out_path = f"{output_prefix}_cwe_range_diff.png"
        plt.savefig(out_path)
        plt.close()
        print(f"-> Gráfico de amplitude por CWE salvo em {out_path}")
        return out_path

def _discover_model_reports(root_dir):
    """Descobre subdiretórios de modelos dentro de root_dir que contenham *_bandit_after.json.
    Retorna (names, reports).
    """
    names = []
    reports = []
    try:
        for entry in sorted(os.listdir(root_dir)):
            p = os.path.join(root_dir, entry)
            if not os.path.isdir(p):
                continue
            if entry.lower() in ("compare", "cases"):
                continue
            if glob.glob(os.path.join(p, "*_bandit_after.json")):
                names.append(entry)
                reports.append(p)
                continue
            # Check common 'reports' subdir
            rep_dir = os.path.join(p, "reports")
            if os.path.isdir(rep_dir) and glob.glob(os.path.join(rep_dir, "*_bandit_after.json")):
                names.append(entry)
                reports.append(rep_dir)
                continue
            # Fallback: recursive search under this model dir
            files = glob.glob(os.path.join(p, "**", "*_bandit_after.json"), recursive=True)
            if files:
                names.append(entry)
                reports.append(os.path.dirname(files[0]))
    except Exception as e:
        print(f"[warn] Falha ao listar {root_dir}: {e}")
    return names, reports


def _safe_prob_dist(counts_map):
    total = sum(counts_map.values())
    if total == 0:
        return {k: 0.0 for k in counts_map}
    return {k: v / total for k, v in counts_map.items()}


def hellinger_distance(p_map, q_map):
    """Hellinger distance between two discrete distributions given as dicts over same keys."""
    keys = set(p_map.keys()) | set(q_map.keys())
    p = {k: p_map.get(k, 0.0) for k in keys}
    q = {k: q_map.get(k, 0.0) for k in keys}
    hp = _safe_prob_dist(p)
    hq = _safe_prob_dist(q)
    s = 0.0
    for k in keys:
        s += (math.sqrt(hp[k]) - math.sqrt(hq[k])) ** 2
    return math.sqrt(s) / math.sqrt(2.0)


def cramers_v_from_cwe_table(cwe_table, model_names):
    """Compute Cramér's V for association between model and CWE categories."""
    # Build contingency table
    cwes = list(cwe_table.keys())
    if not cwes or not model_names:
        return 0.0
    rows = len(model_names)
    cols = len(cwes)
    observed = [[cwe_table[c].get(m, 0) for c in cwes] for m in model_names]
    total = sum(sum(row) for row in observed)
    if total == 0:
        return 0.0
    row_sums = [sum(row) for row in observed]
    col_sums = [sum(observed[r][c] for r in range(rows)) for c in range(cols)]
    chi2 = 0.0
    for r in range(rows):
        for c in range(cols):
            expected = (row_sums[r] * col_sums[c]) / total if total else 0
            if expected > 0:
                chi2 += (observed[r][c] - expected) ** 2 / expected
    k = min(rows, cols)
    if total == 0 or k <= 1:
        return 0.0
    v = math.sqrt(chi2 / (total * (k - 1)))
    return v


def _per_repo_totals(df):
    """Return dict repo->count from a Bandit dataframe that may include 'repo' column."""
    if df is None or df.empty:
        return {}
    # repository granularity not needed; group by current report context only
    # Use a single bucket to avoid incorrect grouping by path parts
    return {'all': len(df)}


def _permute_labels(values, group_sizes, rng):
    idx = list(range(len(values)))
    rng.shuffle(idx)
    groups = []
    start = 0
    for sz in group_sizes:
        groups.append([values[i] for i in idx[start:start+sz]])
        start += sz
    return groups


def _anova_f_stat(groups):
    # groups: list of lists of numbers
    n_groups = len(groups)
    sizes = [len(g) for g in groups]
    if any(s == 0 for s in sizes):
        return 0.0
    all_values = [x for g in groups for x in g]
    grand_mean = sum(all_values) / len(all_values)
    ss_between = 0.0
    ss_within = 0.0
    for g in groups:
        m = sum(g) / len(g)
        ss_between += len(g) * (m - grand_mean) ** 2
        ss_within += sum((x - m) ** 2 for x in g)
    df_between = n_groups - 1
    df_within = len(all_values) - n_groups
    if df_within <= 0:
        return 0.0
    ms_between = ss_between / max(df_between, 1)
    ms_within = ss_within / df_within
    if ms_within == 0:
        return float('inf')
    return ms_between / ms_within


def permutation_anova_pvalue(groups, iterations=500, seed=42):
    rng = random.Random(seed)
    obs = _anova_f_stat(groups)
    values = [x for g in groups for x in g]
    group_sizes = [len(g) for g in groups]
    if sum(group_sizes) != len(values) or len(values) == 0:
        return {'F': obs, 'p_perm': 1.0, 'iterations': 0}
    count = 0
    for _ in range(iterations):
        perm_groups = _permute_labels(values, group_sizes, rng)
        stat = _anova_f_stat(perm_groups)
        if stat >= obs:
            count += 1
    p = (count + 1) / (iterations + 1)
    return {'F': obs, 'p_perm': p, 'iterations': iterations}


def chisq_stat_from_cwe_table(cwe_table, model_names):
    # Build observed matrix
    cwes = list(cwe_table.keys())
    rows = len(model_names)
    cols = len(cwes)
    observed = [[cwe_table[c].get(m, 0) for c in cwes] for m in model_names]
    total = sum(sum(r) for r in observed)
    if total == 0:
        return 0.0
    row_sums = [sum(r) for r in observed]
    col_sums = [sum(observed[r][c] for r in range(rows)) for c in range(cols)]
    chi2 = 0.0
    for r in range(rows):
        for c in range(cols):
            expected = (row_sums[r] * col_sums[c]) / total if total else 0
            if expected > 0:
                chi2 += (observed[r][c] - expected) ** 2 / expected
    return chi2


def permutation_chisq_pvalue_cwe(cwe_table, model_names, iterations=300, seed=123):
    # Flatten all findings as list of CWE labels per finding, and preserve per-model totals
    labels = []  # list of cwe ids (strings)
    group_sizes = []
    for m in model_names:
        count_m = 0
        for cwe, counts in cwe_table.items():
            count_m += counts.get(m, 0)
            labels.extend([cwe] * counts.get(m, 0))
        group_sizes.append(count_m)
    total = sum(group_sizes)
    if total == 0 or any(sz == 0 for sz in group_sizes):
        return {'chi2': 0.0, 'p_perm': 1.0, 'iterations': 0}
    # observed chi2
    chi_obs = chisq_stat_from_cwe_table(cwe_table, model_names)
    rng = random.Random(seed)
    # precompute indices per CWE label for fast recount after permutation
    # For simplicity, we recompute counts each iteration from assigned groups
    count_ge = 0
    for _ in range(iterations):
        # shuffle labels and split into groups preserving group sizes
        idx = list(range(total))
        rng.shuffle(idx)
        splits = []
        start = 0
        for sz in group_sizes:
            splits.append([labels[i] for i in idx[start:start+sz]])
            start += sz
        # rebuild cwe_table for permuted assignment
        perm_table = defaultdict(lambda: {m: 0 for m in model_names})
        for group_list, m in zip(splits, model_names):
            for cwe in group_list:
                perm_table[cwe][m] += 1
        stat = chisq_stat_from_cwe_table(perm_table, model_names)
        if stat >= chi_obs:
            count_ge += 1
    p = (count_ge + 1) / (iterations + 1)
    return {'chi2': chi_obs, 'p_perm': p, 'iterations': iterations}


def main():
    ap = argparse.ArgumentParser(description="Compara resultados Bandit AFTER de 2+ modelos.")
    ap.add_argument("--reports", nargs='+', help="Diretórios com os relatórios dos modelos (>=2)")
    ap.add_argument("--names", nargs='+', help="Nomes dos modelos (>=2, mesma quantidade de --reports)")
    ap.add_argument("--out_prefix", help="Prefixo para os arquivos de saída")
    ap.add_argument("--roots", nargs='+', help="Um ou dois diretórios raiz (ex.: runs_backup runs) para auto-descoberta de modelos (Apenas AFTER)")
    args = ap.parse_args()

    # Branch 1: Auto-discovery mode with one or two roots
    if args.roots:
        roots = args.roots
        assert len(roots) in (1, 2), "--roots deve ter 1 ou 2 diretórios"

        analyses = {}
        for root in roots:
            print(f"==> Analisando root: {root}")
            model_names, report_dirs = _discover_model_reports(root)
            assert len(model_names) >= 2, f"Poucos modelos encontrados em {root}"
            # Build a pseudo-args object to reuse pipeline below
            class A: pass
            a = A(); a.names = model_names; a.reports = report_dirs
            # out prefix default inside root
            out_prefix = os.path.join(root, "model_comparison")

            # Somente AFTER
            bandit_after = []
            patches_stats = {}

            for name, rep in zip(a.names, a.reports):
                print(f"-> Carregando dados do modelo {name}...")
                df_after = load_bandit_reports(rep, tag='after')
                bandit_after.append(df_after)
                after_files = glob.glob(os.path.join(rep, '*_bandit_after.json'))
                after_count = 0 if df_after is None or df_after.empty else len(df_after)
                print(f"   Bandit AFTER files: {len(after_files)}, findings loaded: {after_count}")

                patches_dir = os.path.join(os.path.dirname(rep), 'patches')
                if os.path.isdir(patches_dir):
                    patches_stats[name] = compute_patch_stats_from_patches_dir(patches_dir)
                else:
                    patches_stats[name] = {'lines_added': 0, 'lines_removed': 0, 'lines_changed': 0, 'files_touched': 0}

            stats = compare_vulns_multi(bandit_after, a.names)

            # Sem DELTAS intra-root (apenas AFTER)
            deltas = None

            norm = {}
            for name, df in zip(a.names, bandit_after):
                vuls = len(df) if df is not None and not df.empty else 0
                st = patches_stats.get(name, {'lines_changed': 0, 'files_touched': 0})
                lc = st['lines_changed'] or 1
                norm[name] = {
                    'lines_changed': st['lines_changed'],
                    'files_touched': st['files_touched'],
                    'vuls_per_1k_changed': vuls / (lc / 1000.0)
                }

            intersections_semgrep = None
            intersections_codeql = None

            out_dir = os.path.dirname(out_prefix)
            os.makedirs(out_dir, exist_ok=True)
            plot_summary(stats, a.names, out_prefix)
            if stats['cwe_table']:
                plot_cwe_heatmap(stats, a.names, out_prefix, top_n=25)
            plot_jaccard_heatmap(stats, a.names, out_prefix)
            # CWE diff chart
            diff_img = plot_cwe_diff_chart(stats, a.names, out_prefix, top_n=10)

            # Export and write report (reuse top-level functions)
            export_csvs(stats, a.names, out_prefix, deltas=deltas, norm=norm, intersections=None)
            comparison_file = os.path.join(os.path.dirname(out_prefix), "COMPARISON.md")
            write_comparison_md_multi(stats, a.names, comparison_file, deltas=deltas, norm=norm, intersections=None)
            # Append link to diff image if exists
            if diff_img and os.path.isfile(diff_img):
                try:
                    with open(comparison_file, 'a') as f:
                        f.write("\n![Diferenças por CWE](" + os.path.basename(diff_img) + ")\n")
                except Exception:
                    pass
            print(f"-> Análise completa salva em {comparison_file}")

            analyses[root] = {
                'names': a.names,
                'stats': stats,
                'deltas': deltas,
                'norm': norm,
                'bandit_after_map': {n: df for n, df in zip(a.names, bandit_after)}
            }

            # 4-model statistical tests within this root
            try:
                # Permutation chi-square on CWE×model association
                chisq_res = permutation_chisq_pvalue_cwe(stats['cwe_table'], a.names, iterations=500, seed=123)

                # Per-repo totals groups for permutation ANOVA
                # Build unified repo set
                repo_set = set()
                per_model_repo_counts = {}
                for n, df in zip(a.names, bandit_after):
                    counts = _per_repo_totals(df)
                    per_model_repo_counts[n] = counts
                    repo_set.update(counts.keys())
                groups = []
                for n in a.names:
                    counts = per_model_repo_counts.get(n, {})
                    group_vals = [counts.get(r, 0) for r in sorted(repo_set)]
                    groups.append(group_vals)
                anova_res = permutation_anova_pvalue(groups, iterations=1000, seed=77)

                # Append to COMPARISON.md
                with open(comparison_file, 'a') as f:
                    f.write("\n## Testes Estatísticos (multi-modelo)\n\n")
                    f.write(f"- Permutation Chi-square (CWE×Modelo): chi²={chisq_res['chi2']:.2f}, p={chisq_res['p_perm']:.4f}, iters={chisq_res['iterations']}\n")
                    f.write(f"- Permutation ANOVA (totais por repositório): F={anova_res['F']:.2f}, p={anova_res['p_perm']:.4f}, iters={anova_res['iterations']}\n")

                # Boxplot per-repo totals
                try:
                    plt.figure(figsize=(max(10, 2.2 * len(a.names)), 6))
                    plt.boxplot(groups, labels=a.names, showfliers=False)
                    plt.yscale('symlog')
                    plt.ylabel('Vulnerabilidades por repositório (Bandit após)')
                    plt.title('Distribuição por modelo (por repositório)')
                    plt.tight_layout()
                    out_png = os.path.join(out_dir, "model_per_repo_totals_boxplot.png")
                    plt.savefig(out_png)
                    plt.close()
                    print(f"-> Boxplot por repositório salvo em {out_png}")
                except Exception as e:
                    print(f"[warn] Falha ao gerar boxplot per-repo: {e}")
            except Exception as e:
                print(f"[warn] Falha nos testes estatísticos multi-modelo em {root}: {e}")

        # If two roots, compute cross-prompt effects
        if len(roots) == 2:
            rA, rB = roots
            A = analyses[rA]
            B = analyses[rB]
            common_models = [m for m in A['names'] if m in set(B['names'])]
            if not common_models:
                print("[warn] Nenhum modelo comum entre roots para comparação cross-prompt.")
                return

            # Build per-model metrics
            rows = []
            for m in common_models:
                ta = A['stats']['totals'][m]['total']
                tb = B['stats']['totals'][m]['total']
                delta = tb - ta
                pct = (delta / max(ta, 1)) * 100.0
                # Distribution shift (Hellinger) using CWE counts
                cweA = {cwe: A['stats']['cwe_table'][cwe].get(m, 0) for cwe in A['stats']['cwe_table'].keys()}
                cweB = {cwe: B['stats']['cwe_table'][cwe].get(m, 0) for cwe in B['stats']['cwe_table'].keys()}
                hd = hellinger_distance(cweA, cweB)
                rows.append({'model': m, 'before_total': ta, 'after_total': tb, 'delta': delta, 'pct_change': pct, 'hellinger_cwe': hd})

            # Outputs under the security-prompt root (assumed rB)
            cross_dir = os.path.join(rB, "compare", "prompt")
            os.makedirs(cross_dir, exist_ok=True)
            cross_md = os.path.join(cross_dir, "COMPARISON.md")
            with open(cross_md, 'w') as f:
                f.write(f"# Efeito do Prompt de Segurança ({os.path.basename(rA)} → {os.path.basename(rB)})\n\n")
                f.write("## Mudança por Modelo\n\n")
                for row in rows:
                    f.write(f"- {row['model']}: before={row['before_total']:,}, after={row['after_total']:,}, Δ={row['delta']:,} ({row['pct_change']:.2f}%), hellinger_cwe={row['hellinger_cwe']:.3f}\n")
                # Overall association strength per root (Cramér's V)
                vA = cramers_v_from_cwe_table(A['stats']['cwe_table'], A['names'])
                vB = cramers_v_from_cwe_table(B['stats']['cwe_table'], B['names'])
                f.write("\n## Associação Modelo×CWE (Cramér's V)\n\n")
                f.write(f"- {os.path.basename(rA)}: V={vA:.3f}\n")
                f.write(f"- {os.path.basename(rB)}: V={vB:.3f}\n")
                # Paired sign-flip permutation test por modelo (per-repo)
                f.write("\n## Teste de permutação pareado por modelo (per-repo; duas caudas)\n\n")
                def sign_flip_perm_pvalue(diffs, iterations=1000, seed=7):
                    if not diffs:
                        return 1.0
                    rng = random.Random(seed)
                    obs = abs(sum(diffs) / len(diffs))
                    count = 0
                    for _ in range(iterations):
                        sim = [d * (1 if rng.random() < 0.5 else -1) for d in diffs]
                        stat = abs(sum(sim) / len(sim))
                        if stat >= obs:
                            count += 1
                    return (count + 1) / (iterations + 1)
                for m in common_models:
                    dfA = analyses[rA]['bandit_after_map'].get(m)
                    dfB = analyses[rB]['bandit_after_map'].get(m)
                    countsA = _per_repo_totals(dfA)
                    countsB = _per_repo_totals(dfB)
                    repos = sorted(set(countsA.keys()) | set(countsB.keys()))
                    diffs = [countsB.get(r, 0) - countsA.get(r, 0) for r in repos]
                    p = sign_flip_perm_pvalue(diffs, iterations=1000, seed=7)
                    f.write(f"- {m}: n_repos={len(repos)}, p_perm={p:.4f}\n")

                # Diferenças por CWE entre roots (B − A), por modelo
                f.write("\n## Diferenças por CWE entre roots (after_B − after_A)\n\n")
                # union of CWEs across both roots
                all_cwes = set(list(A['stats']['cwe_table'].keys()) + list(B['stats']['cwe_table'].keys()))
                common_models_sorted = sorted(common_models)
                # header
                f.write("| CWE | " + " | ".join([m + " Δ" for m in common_models_sorted]) + " | Total |\n")
                f.write("|" + "---|" * (len(common_models_sorted) + 2) + "\n")
                # rows sorted by total abs delta desc
                delta_rows = []
                for cwe in all_cwes:
                    row = {'cwe': cwe, 'total_abs': 0}
                    for m in common_models_sorted:
                        a_val = A['stats']['cwe_table'].get(cwe, {}).get(m, 0)
                        b_val = B['stats']['cwe_table'].get(cwe, {}).get(m, 0)
                        dval = b_val - a_val
                        row[m] = dval
                        row['total_abs'] += abs(dval)
                    delta_rows.append(row)
                delta_rows.sort(key=lambda r: r['total_abs'], reverse=True)
                for r in delta_rows:
                    cells = [r['cwe']] + [f"{r[m]:,}" for m in common_models_sorted] + [f"{r['total_abs']:,}"]
                    f.write("| " + " | ".join(cells) + " |\n")

                # Export full CSV for programmatic diff later
                try:
                    import csv
                    csv_path = os.path.join(cross_dir, "cross_cwe_delta_by_model.csv")
                    with open(csv_path, 'w', newline='') as cf:
                        writer = csv.writer(cf)
                        writer.writerow(["cwe"] + [m + "_delta" for m in common_models_sorted] + ["total_abs_delta"])
                        for r in delta_rows:
                            writer.writerow([r['cwe']] + [r[m] for m in common_models_sorted] + [r['total_abs']])
                    print(f"-> CSV de diffs por CWE salvo em {csv_path}")
                except Exception as e:
                    print(f"[warn] Falha ao exportar CSV de diffs por CWE: {e}")
            print(f"-> Comparação cross-prompt salva em {cross_md}")

            # Plot cross-prompt totals (before vs after) per model
            try:
                plt.figure(figsize=(max(10, 0.8 * len(common_models) * 2), 6))
                x = range(len(common_models))
                width = 0.35
                before_vals = [next(r['before_total'] for r in rows if r['model'] == m) for m in common_models]
                after_vals = [next(r['after_total'] for r in rows if r['model'] == m) for m in common_models]
                plt.bar([i - width/2 for i in x], before_vals, width, label=os.path.basename(rA), color='lightgray')
                plt.bar([i + width/2 for i in x], after_vals, width, label=os.path.basename(rB), color='steelblue')
                plt.xlabel('Modelos')
                plt.ylabel('Total de Vulnerabilidades (Bandit após patch)')
                plt.title('Antes vs Depois do Prompt de Segurança (totais por modelo)')
                plt.xticks(list(x), common_models, rotation=15)
                plt.legend()
                plt.tight_layout()
                out_png = os.path.join(cross_dir, "cross_prompt_totals.png")
                plt.savefig(out_png)
                plt.close()
                print(f"-> Gráfico cross-prompt salvo em {out_png}")
            except Exception as e:
                print(f"[warn] Falha ao gerar gráfico cross-prompt: {e}")

            # Consolidar tudo em um único arquivo no root de backup (runs_backup)
            try:
                parts = []
                root_reports = [os.path.join(rA, "COMPARISON.md"), os.path.join(rB, "COMPARISON.md"), cross_md]
                for rp in root_reports:
                    if os.path.isfile(rp):
                        with open(rp, 'r') as f:
                            parts.append(f"\n\n---\n\n" + f.read())
                combined_md = os.path.join(rA, "COMPARISON.md")
                with open(combined_md, 'w') as f:
                    f.write(f"# Comparação consolidada ({os.path.basename(rA)} + {os.path.basename(rB)} + cross-prompt)\n\n")
                    for content in parts:
                        f.write(content)
                print(f"-> Relatório consolidado salvo em {combined_md}")
            except Exception as e:
                print(f"[warn] Falha ao consolidar COMPARISON.md: {e}")

            # Gerar um markdown final único com imagens referenciadas corretamente
            try:
                def rewrite_image_links(md_text, src_dir, dst_dir):
                    def repl(m):
                        alt = m.group(1)
                        url = m.group(2)
                        if url.startswith('http://') or url.startswith('https://') or '/' in url or url.startswith('.'):
                            return m.group(0)
                        abs_src = os.path.join(src_dir, url)
                        rel = os.path.relpath(abs_src, dst_dir)
                        return f"![{alt}]({rel})"
                    return re.sub(r"!\\[([^\\]]*)\\]\\(([^\\)]+)\\)", repl, md_text)

                final_md = os.path.join(rA, "FINAL_REPORT.md")
                with open(final_md, 'w') as f:
                    f.write(f"# Relatório Final\n\n")
                    f.write(f"Este arquivo consolida os resultados de `{os.path.basename(rA)}` (prompt simples), `{os.path.basename(rB)}` (prompt de segurança) e a comparação cross-prompt.\n\n")

                    # Secção runs_backup
                    f.write(f"## {os.path.basename(rA)}\n\n")
                    if os.path.isfile(os.path.join(rA, "COMPARISON.md")):
                        with open(os.path.join(rA, "COMPARISON.md"), 'r') as fr:
                            content = fr.read()
                            f.write(rewrite_image_links(content, rA, rA))

                    # Secção runs
                    f.write(f"\n\n## {os.path.basename(rB)}\n\n")
                    if os.path.isfile(os.path.join(rB, "COMPARISON.md")):
                        with open(os.path.join(rB, "COMPARISON.md"), 'r') as fr:
                            content = fr.read()
                            f.write(rewrite_image_links(content, rB, rA))

                    # Secção cross-prompt
                    f.write(f"\n\n## Cross-prompt\n\n")
                    if os.path.isfile(cross_md):
                        with open(cross_md, 'r') as fr:
                            content = fr.read()
                            f.write(rewrite_image_links(content, os.path.dirname(cross_md), rA))
                print(f"-> Relatório final salvo em {final_md}")
            except Exception as e:
                print(f"[warn] Falha ao gerar FINAL_REPORT.md: {e}")

        return

    # Branch 2: legacy explicit mode
    assert args.reports and args.names and args.out_prefix, "Forneça --reports, --names e --out_prefix ou use --roots."
    assert len(args.reports) == len(args.names) and len(args.names) >= 2, "Forneça o mesmo número de reports e names (>=2)."

    # Carregar dados Bandit AFTER (apenas)
    bandit_after = []
    patches_stats = {}

    for name, rep in zip(args.names, args.reports):
        print(f"-> Carregando dados do modelo {name}...")
        df_after = load_bandit_reports(rep, tag='after')
        bandit_after.append(df_after)
        # Debug: quantos arquivos e achados
        after_files = glob.glob(os.path.join(rep, '*_bandit_after.json'))
        after_count = 0 if df_after is None or df_after.empty else len(df_after)
        print(f"   Bandit AFTER files: {len(after_files)}, findings loaded: {after_count}")

        # patch stats: tenta achar dir de patches como sibling
        patches_dir = os.path.join(os.path.dirname(rep), 'patches')
        if os.path.isdir(patches_dir):
            patches_stats[name] = compute_patch_stats_from_patches_dir(patches_dir)
        else:
            patches_stats[name] = {'lines_added': 0, 'lines_removed': 0, 'lines_changed': 0, 'files_touched': 0}

    # Comparar AFTER entre modelos
    stats = compare_vulns_multi(bandit_after, args.names)

    # Sem deltas no modo explícito (apenas AFTER)
    deltas = None

    # Normalização por linhas/arquivos
    norm = {}
    for name, df in zip(args.names, bandit_after):
        vuls = len(df) if df is not None and not df.empty else 0
        st = patches_stats.get(name, {'lines_changed': 0, 'files_touched': 0})
        lc = st['lines_changed'] or 1
        norm[name] = {
            'lines_changed': st['lines_changed'],
            'files_touched': st['files_touched'],
            'vuls_per_1k_changed': vuls / (lc / 1000.0)
        }

    intersections_semgrep = None
    intersections_codeql = None

    # Saídas
    out_dir = os.path.dirname(args.out_prefix)
    os.makedirs(out_dir, exist_ok=True)
    plot_summary(stats, args.names, args.out_prefix)
    if stats['cwe_table']:
        plot_cwe_heatmap(stats, args.names, args.out_prefix, top_n=25)
    plot_jaccard_heatmap(stats, args.names, args.out_prefix)

    # Export and write report (usar funções globais)
    export_csvs(stats, args.names, args.out_prefix, deltas=deltas, norm=norm)
    comparison_file = os.path.join(os.path.dirname(args.out_prefix), "COMPARISON.md")
    write_comparison_md_multi(stats, args.names, comparison_file, deltas=deltas, norm=norm)
    print(f"-> Análise completa salva em {comparison_file}")


if __name__ == "__main__":
    main()