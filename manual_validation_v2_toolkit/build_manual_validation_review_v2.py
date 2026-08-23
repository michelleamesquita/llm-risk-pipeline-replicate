#!/usr/bin/env python3
"""
Prepara a versão FINAL da validação manual do Bandit.

Entrada:
  --blinded  manual_validation_sample_blinded.csv
  --key      manual_validation_key.csv

Saída padrão (~80 itens):
  - mantém TODOS os findings novos (is_new_finding=1);
  - deduplica os demais por finding equivalente no mesmo case/arquivo/regra
    usando a linha real do código AFTER, não apenas line_number;
  - seleciona controles preexistentes de forma estratificada/diversa até
    completar --target-size;
  - recalcula BEFORE/AFTER/DIFF quando o checkout Git patchado ainda existe;
  - mapeia corretamente a linha AFTER para a linha correspondente BEFORE
    usando os hunks do unified diff;
  - REMOVE do arquivo cego: model, severity, Bandit confidence,
    is_new_finding, existed_before, scores etc.;
  - gera CSV cego + chave + HTML amigável para o avaliador.

Uso:
  python build_manual_validation_review_v2.py \
      --blinded manual_validation/manual_validation_sample_blinded.csv \
      --key manual_validation/manual_validation_key.csv \
      --outdir manual_validation_v2 \
      --target-size 80 \
      --seed 42

IMPORTANTE:
  Rode enquanto os checkouts reconstruídos ainda existem, se possível.
  O script usa `filename` da chave para localizar o arquivo AFTER e
  `git show HEAD:<arquivo>` para reconstruir BEFORE.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


VALIDITY = [
    "VALID_SECURITY_FINDING",
    "LIKELY_FALSE_POSITIVE",
    "UNCERTAIN",
]
ATTRIBUTION = [
    "INTRODUCED_OR_AFFECTED_BY_PATCH",
    "PRE_EXISTING",
    "UNCERTAIN",
]


def s(v) -> str:
    if pd.isna(v):
        return ""
    return str(v)


def normalize_code_line(line: str) -> str:
    line = re.sub(r"^\s*>>\s*\d+\s*:\s*", "", s(line))
    line = re.sub(r"^\s*\d+\s*:\s*", "", line)
    return re.sub(r"\s+", " ", line.strip())


def pointed_line(context: str) -> str:
    for line in s(context).splitlines():
        if line.lstrip().startswith(">>"):
            return normalize_code_line(line)
    return ""


def context_window(text: str, line_number: int | None, radius: int = 8) -> str:
    if not text or not line_number:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    idx = max(0, min(len(lines) - 1, int(line_number) - 1))
    lo = max(0, idx - radius)
    hi = min(len(lines), idx + radius + 1)
    out = []
    for i in range(lo, hi):
        marker = ">>" if i == idx else "  "
        out.append(f"{marker} {i+1:5d}: {lines[i]}")
    return "\n".join(out)


HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


def parse_hunks(diff_text: str):
    """Retorna hunks com linhas e coordenadas do unified diff."""
    lines = s(diff_text).splitlines()
    hunks = []
    current = None
    for raw in lines:
        m = HUNK_RE.match(raw)
        if m:
            if current is not None:
                hunks.append(current)
            current = {
                "old_start": int(m.group("old_start")),
                "old_count": int(m.group("old_count") or 1),
                "new_start": int(m.group("new_start")),
                "new_count": int(m.group("new_count") or 1),
                "lines": [],
            }
        elif current is not None:
            # Para no próximo cabeçalho de arquivo, se houver múltiplos diffs.
            if raw.startswith("diff --git "):
                hunks.append(current)
                current = None
            else:
                current["lines"].append(raw)
    if current is not None:
        hunks.append(current)
    return hunks


def map_after_to_before(after_line: int, diff_text: str):
    """
    Mapeia linha do arquivo AFTER para BEFORE.

    Retorna:
      mapped_before_line: int | None
      relation:
        UNCHANGED_OUTSIDE_DIFF
        CONTEXT_LINE_IN_HUNK
        ADDED_IN_PATCH
        MAPPING_FAILED
    """
    hunks = parse_hunks(diff_text)
    if not hunks:
        return after_line, "UNCHANGED_NO_DIFF"

    # offset = old_line - new_line no trecho entre hunks.
    offset = 0

    for h in hunks:
        ns = h["new_start"]
        os_ = h["old_start"]

        # Linha antes deste hunk: aplica offset acumulado dos hunks anteriores.
        if after_line < ns:
            return after_line + offset, "UNCHANGED_OUTSIDE_DIFF"

        old_cursor = os_
        new_cursor = ns

        for raw in h["lines"]:
            # Ignora metadados especiais.
            if raw.startswith("\\ No newline"):
                continue

            prefix = raw[:1] if raw else " "
            if prefix == " ":
                if new_cursor == after_line:
                    return old_cursor, "CONTEXT_LINE_IN_HUNK"
                old_cursor += 1
                new_cursor += 1
            elif prefix == "+" and not raw.startswith("+++"):
                if new_cursor == after_line:
                    # Linha adicionada não possui linha BEFORE direta.
                    # Usa o ponto de inserção como âncora visual.
                    anchor = max(1, old_cursor)
                    return anchor, "ADDED_IN_PATCH"
                new_cursor += 1
            elif prefix == "-" and not raw.startswith("---"):
                old_cursor += 1
            else:
                # Linha inesperada dentro de hunk.
                pass

        # Depois do hunk, atualiza offset old-new.
        offset = (h["old_start"] + h["old_count"]) - (
            h["new_start"] + h["new_count"]
        )

    return after_line + offset, "UNCHANGED_OUTSIDE_DIFF"


def infer_repo_root(after_file: Path, relative_filename: str) -> Path | None:
    try:
        root = after_file
        for _ in Path(relative_filename).parts:
            root = root.parent
        return root
    except Exception:
        return None


def git_show(repo_root: Path, relative_filename: str) -> str:
    try:
        p = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"HEAD:{relative_filename}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return p.stdout if p.returncode == 0 else ""
    except Exception:
        return ""


def git_diff(repo_root: Path, relative_filename: str) -> str:
    try:
        p = subprocess.run(
            [
                "git", "-C", str(repo_root),
                "diff", "--unified=12", "--", relative_filename
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return p.stdout if p.returncode == 0 else ""
    except Exception:
        return ""


def reconstruct_context(row: pd.Series, radius: int = 8):
    """
    Recalcula o contexto se `filename` ainda existir.
    Caso contrário, usa o conteúdo já presente no CSV.
    """
    after_line = int(float(row["line_number"]))
    relative_filename = s(row.get("relative_filename"))
    after_path = Path(s(row.get("filename")))

    existing_before = s(row.get("code_context_before"))
    existing_after = s(row.get("code_context_after"))
    existing_diff = s(row.get("patch_diff_context"))

    out = {
        "after_line_number": after_line,
        "before_line_number_mapped": np.nan,
        "line_mapping_relation": "FALLBACK_EXISTING_CONTEXT",
        "code_context_before_mapped": existing_before,
        "patch_diff_context_v2": existing_diff,
        "code_context_after_v2": existing_after,
        "context_status_v2": s(row.get("context_status")) or "UNKNOWN",
    }

    if not after_path.exists() or not after_path.is_file():
        # Ainda tenta mapear a linha usando o diff já salvo.
        if existing_diff:
            mapped, relation = map_after_to_before(after_line, existing_diff)
            out["before_line_number_mapped"] = mapped
            out["line_mapping_relation"] = relation + "_NO_FULL_BEFORE_FILE"
        return out

    try:
        after_text = after_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out

    repo_root = infer_repo_root(after_path, relative_filename)
    if repo_root is None or not repo_root.exists():
        out["code_context_after_v2"] = context_window(after_text, after_line, radius)
        out["context_status_v2"] = "AFTER_ONLY"
        return out

    before_text = git_show(repo_root, relative_filename)
    diff_text = git_diff(repo_root, relative_filename) or existing_diff

    mapped, relation = map_after_to_before(after_line, diff_text)
    out["before_line_number_mapped"] = mapped
    out["line_mapping_relation"] = relation
    out["code_context_after_v2"] = context_window(after_text, after_line, radius)
    out["patch_diff_context_v2"] = diff_text

    if before_text and mapped:
        out["code_context_before_mapped"] = context_window(
            before_text, int(mapped), radius
        )

    if before_text and diff_text:
        out["context_status_v2"] = "BEFORE_AFTER_DIFF_MAPPED"
    elif before_text:
        out["context_status_v2"] = "BEFORE_AFTER_MAPPED"
    elif diff_text:
        out["context_status_v2"] = "AFTER_DIFF"
    else:
        out["context_status_v2"] = "AFTER_ONLY"

    return out


def stable_hash(*parts) -> str:
    text = "||".join(s(x).strip().lower() for x in parts)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def make_equivalence_key(row: pd.Series) -> str:
    """
    Dedup conservador para CONTROLES.

    Mantém cases diferentes separados, mas collapse repetições do mesmo finding
    em múltiplos modelos quando o código AFTER apontado é equivalente.
    """
    code = pointed_line(row.get("code_context_after_v2", ""))
    if not code:
        code = f"LINE:{row.get('line_number', '')}"
    return stable_hash(
        row.get("case"),
        row.get("relative_filename"),
        row.get("test_id"),
        row.get("cwe"),
        row.get("details"),
        code,
    )


def choose_controls(controls: pd.DataFrame, n_needed: int, seed: int):
    if n_needed <= 0:
        return controls.iloc[0:0].copy()

    rng = np.random.default_rng(seed)

    # Dedup equivalente, priorizando melhor contexto.
    work = controls.copy()
    quality = {
        "BEFORE_AFTER_DIFF_MAPPED": 5,
        "BEFORE_AFTER_DIFF": 4,
        "BEFORE_AFTER_MAPPED": 3,
        "BEFORE_AFTER": 2,
        "AFTER_DIFF": 1,
    }
    work["_context_quality"] = (
        work["context_status_v2"].map(quality).fillna(0)
    )
    work["_rand"] = rng.random(len(work))
    work = work.sort_values(
        ["_equivalence_key", "_context_quality", "_rand"],
        ascending=[True, False, True],
    )
    dedup = work.drop_duplicates("_equivalence_key", keep="first").copy()

    # Stratum por rule + severity escondida, para diversidade.
    dedup["_stratum"] = (
        dedup["test_id"].astype(str)
        + " | "
        + dedup["severity"].astype(str).str.upper()
    )

    strata = sorted(dedup["_stratum"].unique())
    pools = {}
    for st in strata:
        idx = dedup.index[dedup["_stratum"] == st].to_numpy()
        rng.shuffle(idx)
        pools[st] = list(idx)

    chosen = []
    while len(chosen) < min(n_needed, len(dedup)):
        progressed = False
        for st in strata:
            if pools[st] and len(chosen) < n_needed:
                chosen.append(pools[st].pop())
                progressed = True
        if not progressed:
            break

    return dedup.loc[chosen].drop(
        columns=["_context_quality", "_rand", "_stratum"],
        errors="ignore",
    )


def reviewer_html(df: pd.DataFrame, out_path: Path):
    """
    Gera HTML estático com cartões e botão de exportação CSV.
    Nenhuma informação cega é embutida.
    """
    cards = []
    for _, r in df.iterrows():
        rid = html.escape(s(r["review_id_v2"]))
        claim = html.escape(s(r["details"]))
        before = html.escape(s(r["code_context_before"]))
        diff = html.escape(s(r["patch_diff_context"]))
        after = html.escape(s(r["code_context_after"]))
        file = html.escape(s(r["relative_filename"]))
        test = html.escape(f"{s(r['test_id'])} — {s(r['test_name'])}")
        cwe = html.escape(s(r["cwe"]))
        relation = html.escape(s(r["line_mapping_relation"]))

        validity_opts = "".join(
            f'<option value="{x}">{x}</option>' for x in [""] + VALIDITY
        )
        attr_opts = "".join(
            f'<option value="{x}">{x}</option>' for x in [""] + ATTRIBUTION
        )

        cards.append(f"""
<section class="card" data-review-id="{rid}">
  <h2>{rid}</h2>
  <div class="meta"><b>Arquivo:</b> {file}</div>
  <div class="meta"><b>Regra:</b> {test}</div>
  <div class="meta"><b>CWE:</b> {cwe}</div>
  <div class="meta"><b>Alegação do Bandit:</b> {claim}</div>
  <details>
    <summary>BEFORE</summary>
    <pre>{before}</pre>
  </details>
  <details open>
    <summary>PATCH / DIFF</summary>
    <pre>{diff}</pre>
  </details>
  <details open>
    <summary>AFTER</summary>
    <pre>{after}</pre>
  </details>
  <div class="mapping"><b>Mapeamento técnico:</b> {relation}</div>

  <label>1. Validade do finding</label>
  <select class="validity">{validity_opts}</select>

  <label>2. Atribuição ao patch</label>
  <select class="attribution">{attr_opts}</select>

  <label>3. Confiança do avaliador (1–5)</label>
  <select class="confidence">
    <option value=""></option>
    <option>1</option><option>2</option><option>3</option>
    <option>4</option><option>5</option>
  </select>

  <label>4. Observação</label>
  <textarea class="notes" rows="4"></textarea>
</section>
""")

    document = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Validação manual — Bandit</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px auto; max-width: 1100px; line-height: 1.4; }}
h1 {{ margin-bottom: 4px; }}
.notice {{ background:#f3f3f3; padding:14px; border-radius:8px; margin-bottom:20px; }}
.card {{ border:1px solid #bbb; border-radius:10px; padding:18px; margin:20px 0; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#f7f7f7; padding:12px; border-radius:6px; }}
label {{ display:block; font-weight:bold; margin-top:14px; }}
select, textarea {{ width:100%; padding:8px; margin-top:4px; }}
.meta {{ margin:5px 0; }}
.mapping {{ font-size:0.9em; margin-top:8px; }}
.actions {{ position:sticky; top:0; background:white; padding:10px 0; border-bottom:1px solid #ddd; z-index:10; }}
button {{ padding:10px 16px; font-size:16px; }}
</style>
</head>
<body>
<h1>Validação manual dos findings do Bandit</h1>
<div class="notice">
<b>Não tente inferir qual LLM gerou o patch.</b><br>
Avalie validade e atribuição separadamente. Quando o contexto não for suficiente,
use <code>UNCERTAIN</code>.
</div>
<div class="actions">
<label style="display:inline-block;margin-right:8px">Avaliador:</label>
<input id="reviewerId" placeholder="ex.: R1">
<button onclick="exportCSV()">Exportar respostas CSV</button>
<span id="progress"></span>
</div>

{''.join(cards)}

<script>
function escCSV(x) {{
  x = (x ?? '').toString();
  return '"' + x.replaceAll('"','""') + '"';
}}
function updateProgress() {{
  const cards = [...document.querySelectorAll('.card')];
  const done = cards.filter(c => c.querySelector('.validity').value !== '').length;
  document.getElementById('progress').textContent = `Preenchidos: ${{done}}/${{cards.length}}`;
}}
document.querySelectorAll('select').forEach(x => x.addEventListener('change', updateProgress));
updateProgress();

function exportCSV() {{
  const reviewer = document.getElementById('reviewerId').value.trim();
  if (!reviewer) {{
    alert('Informe o identificador do avaliador.');
    return;
  }}
  const rows = [[
    'review_id_v2','reviewer_id','validity','patch_attribution',
    'reviewer_confidence_1_to_5','reviewer_notes'
  ]];
  document.querySelectorAll('.card').forEach(c => {{
    rows.push([
      c.dataset.reviewId,
      reviewer,
      c.querySelector('.validity').value,
      c.querySelector('.attribution').value,
      c.querySelector('.confidence').value,
      c.querySelector('.notes').value
    ]);
  }});
  const text = rows.map(r => r.map(escCSV).join(',')).join('\\n');
  const blob = new Blob([text], {{type:'text/csv;charset=utf-8'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `manual_review_${{reviewer}}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}}
</script>
</body>
</html>
"""
    out_path.write_text(document, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blinded", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--findings", default="", help="CSV original all_findings_flat_robust_common.csv; recomendado para recuperar filename via _row_id_original.")
    ap.add_argument("--outdir", default="manual_validation_v2")
    ap.add_argument("--target-size", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context-radius", type=int, default=8)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    blinded = pd.read_csv(args.blinded)
    key = pd.read_csv(args.key)

    if "review_id" not in blinded or "review_id" not in key:
        raise ValueError("Os dois arquivos precisam conter review_id.")

    # Evita trazer colunas duplicadas da chave.
    key_cols = [
        c for c in key.columns
        if c == "review_id" or c not in blinded.columns
    ]
    merged = blinded.merge(
        key[key_cols],
        on="review_id",
        how="left",
        validate="one_to_one",
    )

    # A primeira versão da chave não incluía `filename`. Recupera do CSV
    # original usando _row_id_original, quando fornecido.
    if "filename" not in merged.columns and args.findings:
        findings = pd.read_csv(args.findings)
        if "_row_id_original" in merged.columns:
            lookup = findings.copy()
            lookup["_row_id_original"] = np.arange(len(lookup))
            extra_cols = [
                c for c in [
                    "_row_id_original", "filename", "backup_dir",
                    "patch_apply_success", "report_file"
                ] if c in lookup.columns
            ]
            merged = merged.merge(
                lookup[extra_cols],
                on="_row_id_original",
                how="left",
                validate="many_to_one",
            )

    # Sem filename ainda é possível produzir o pacote usando os contextos já
    # reconstruídos; apenas o remapeamento BEFORE será fallback.
    if "filename" not in merged.columns:
        merged["filename"] = ""

    required = {
        "review_id", "case", "relative_filename", "line_number",
        "test_id", "test_name", "cwe", "details",
        "is_new_finding", "existed_before", "severity", "model",
    }
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(
            "A chave/CSV não contém colunas necessárias: "
            + ", ".join(sorted(missing))
        )

    # Recalcula contextos e mapeamento.
    recon = []
    for _, row in merged.iterrows():
        recon.append(reconstruct_context(row, args.context_radius))
    recon_df = pd.DataFrame(recon)

    # Remove contextos antigos para evitar confusão.
    merged = merged.drop(
        columns=[
            "code_context_before", "code_context_after",
            "patch_diff_context", "context_status",
        ],
        errors="ignore",
    )
    merged = pd.concat(
        [merged.reset_index(drop=True), recon_df.reset_index(drop=True)],
        axis=1,
    )

    merged["_equivalence_key"] = merged.apply(
        make_equivalence_key, axis=1
    )

    # TODOS os novos.
    is_new = (
        pd.to_numeric(merged["is_new_finding"], errors="coerce")
        .fillna(0).astype(int).eq(1)
    )
    new_items = merged[is_new].copy()
    new_items["sample_group_v2"] = "ALL_NEW_FINDINGS"

    # Controles = não novos; dedup por equivalência e seleção diversa.
    controls = merged[~is_new].copy()
    n_controls = max(0, args.target_size - len(new_items))
    chosen_controls = choose_controls(
        controls, n_controls, args.seed
    ).copy()
    chosen_controls["sample_group_v2"] = "STRATIFIED_DEDUP_CONTROL"

    selected = pd.concat(
        [new_items, chosen_controls],
        ignore_index=True,
    )

    # Randomiza a ordem novamente.
    selected = selected.sample(
        frac=1.0, random_state=args.seed
    ).reset_index(drop=True)
    selected["review_id_v2"] = [
        f"RV{i:03d}" for i in range(1, len(selected) + 1)
    ]

    # ID de grupo de validade equivalente, mantido só na chave.
    selected["validity_equivalence_group"] = selected["_equivalence_key"]

    # Contextos finais com nomes simples.
    selected["code_context_before"] = selected["code_context_before_mapped"]
    selected["patch_diff_context"] = selected["patch_diff_context_v2"]
    selected["code_context_after"] = selected["code_context_after_v2"]

    # ----------------
    # Arquivo CEGO
    # ----------------
    blinded_cols = [
        "review_id_v2",
        "case",
        "relative_filename",
        "after_line_number",
        "before_line_number_mapped",
        "line_mapping_relation",
        "test_id",
        "test_name",
        "cwe",
        "details",
        "code_context_before",
        "patch_diff_context",
        "code_context_after",
        "context_status_v2",
    ]
    review = selected[blinded_cols].copy()

    # Não incluir confidence/severity/model/new/existed no arquivo cego.
    for prefix in ["reviewer_1", "reviewer_2"]:
        review[f"{prefix}_validity"] = ""
        review[f"{prefix}_patch_attribution"] = ""
        review[f"{prefix}_confidence_1_to_5"] = ""
        review[f"{prefix}_notes"] = ""
    review["adjudicated_validity"] = ""
    review["adjudicated_patch_attribution"] = ""
    review["adjudication_notes"] = ""

    review.to_csv(
        outdir / "manual_validation_review_blinded_v2.csv",
        index=False,
    )

    # ----------------
    # CHAVE NÃO CEGA
    # ----------------
    key_out_cols = [
        "review_id_v2", "review_id",
        "sample_group_v2",
        "validity_equivalence_group",
        "model", "case", "repo",
        "filename", "relative_filename",
        "line_number", "after_line_number",
        "before_line_number_mapped",
        "line_mapping_relation",
        "test_id", "test_name", "cwe",
        "severity", "confidence", "details",
        "finding_fingerprint",
        "existed_before", "is_new_finding",
        "file_touched_by_patch",
        "strict_patch_apply", "fuzzy_patch_apply",
        "context_status_v2",
    ]
    key_out_cols = [c for c in key_out_cols if c in selected.columns]
    selected[key_out_cols].to_csv(
        outdir / "manual_validation_key_v2.csv",
        index=False,
    )

    # HTML amigável.
    reviewer_html(
        review,
        outdir / "manual_validation_review_packet_v2.html",
    )

    # Composição.
    composition = (
        selected.groupby(
            ["sample_group_v2", "test_id", "severity"],
            dropna=False,
        )
        .size()
        .reset_index(name="n")
    )
    composition.to_csv(
        outdir / "manual_validation_composition_v2.csv",
        index=False,
    )

    # Auditoria de deduplicação.
    control_audit = {
        "original_rows_in_blinded": int(len(merged)),
        "new_findings_kept_all": int(len(new_items)),
        "control_candidates_before_dedup": int(len(controls)),
        "control_unique_equivalence_groups": int(
            controls["_equivalence_key"].nunique()
        ),
        "controls_selected": int(len(chosen_controls)),
        "final_review_items": int(len(selected)),
        "target_size_requested": int(args.target_size),
        "seed": int(args.seed),
        "context_status_v2": (
            selected["context_status_v2"].value_counts().to_dict()
        ),
        "line_mapping_relation": (
            selected["line_mapping_relation"].value_counts().to_dict()
        ),
        "blinded_fields_removed": [
            "model",
            "severity",
            "Bandit confidence",
            "is_new_finding",
            "existed_before",
            "sample_group",
            "LR/RF scores",
        ],
    }
    (
        outdir / "manual_validation_v2_summary.json"
    ).write_text(
        json.dumps(control_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rubric = """# Rubrica — validação manual v2

Avalie cada item independentemente.

## 1. Validade do finding
Escolha exatamente um:
- VALID_SECURITY_FINDING
- LIKELY_FALSE_POSITIVE
- UNCERTAIN

VALID_SECURITY_FINDING:
o código apresentado dá suporte plausível ao problema de segurança descrito.

LIKELY_FALSE_POSITIVE:
o alerta não representa um problema de segurança plausível no contexto mostrado
(ex.: regra genérica disparada em contexto de teste/controlado sem exposição
de segurança aparente).

UNCERTAIN:
o contexto fornecido não permite concluir.

## 2. Atribuição ao patch
Escolha exatamente um:
- INTRODUCED_OR_AFFECTED_BY_PATCH
- PRE_EXISTING
- UNCERTAIN

INTRODUCED_OR_AFFECTED_BY_PATCH:
o diff criou ou alterou materialmente a condição que originou o finding.

PRE_EXISTING:
a condição já estava presente no BEFORE e o patch não a criou nem a alterou
materialmente.

UNCERTAIN:
não há evidência suficiente para atribuir.

## Regras para os avaliadores
1. Não assumir comportamento que não aparece no código/contexto.
2. Validade e atribuição são perguntas diferentes.
3. Não tentar identificar qual LLM gerou o patch.
4. O fato de uma regra do Bandit disparar não prova vulnerabilidade.
5. Se o finding está em arquivo de teste, avalie o contexto real; não marque
   automaticamente como válido ou falso.
6. Se o AFTER mostra uma linha adicionada e o BEFORE mostra o ponto de inserção,
   use o diff para decidir a atribuição.
7. Confidence 1–5 representa a confiança DO AVALIADOR no próprio julgamento.
"""
    (outdir / "MANUAL_REVIEW_RUBRIC_V2.md").write_text(
        rubric, encoding="utf-8"
    )

    print("=" * 88)
    print("VALIDAÇÃO MANUAL V2 CRIADA")
    print("=" * 88)
    print(json.dumps(control_audit, indent=2, ensure_ascii=False))
    print("\nArquivos principais:")
    print(" -", outdir / "manual_validation_review_packet_v2.html")
    print(" -", outdir / "manual_validation_review_blinded_v2.csv")
    print(" -", outdir / "manual_validation_key_v2.csv")
    print(" -", outdir / "MANUAL_REVIEW_RUBRIC_V2.md")


if __name__ == "__main__":
    main()
