# Validação manual do Bandit — protocolo v2

Este fluxo corrige quatro problemas da primeira amostra:

1. reduz redundância entre findings equivalentes;
2. mantém **todos os findings novos**;
3. mapeia corretamente `AFTER line -> BEFORE line` pelo unified diff;
4. remove do avaliador informações que podem causar viés, inclusive
   `model`, `severity`, **Bandit confidence**, `is_new_finding`,
   `existed_before` e scores LR/RF.

O tamanho padrão é **80 itens**:
- 27 findings novos, todos mantidos;
- 53 controles deduplicados/estratificados.

## 1. Gerar o pacote final

Rode enquanto os checkouts reconstruídos/patchados ainda existirem.

```bash
python build_manual_validation_review_v2.py \
  --blinded manual_validation/manual_validation_sample_blinded.csv \
  --key manual_validation/manual_validation_key.csv \
  --findings all_findings_flat_robust_common.csv \
  --outdir manual_validation_v2 \
  --target-size 80 \
  --seed 42
```

Arquivos principais:

- `manual_validation_review_packet_v2.html`
  - abrir no navegador;
  - um cartão por finding;
  - avaliador preenche validade, atribuição, confiança e nota;
  - botão **Exportar respostas CSV**.

- `manual_validation_review_blinded_v2.csv`
  - alternativa ao HTML.

- `manual_validation_key_v2.csv`
  - NÃO entregar ao avaliador.

- `MANUAL_REVIEW_RUBRIC_V2.md`
  - rubrica.

- `manual_validation_v2_summary.json`
  - auditoria da amostragem/mapeamento.

### Antes de mandar ao avaliador

Abra `manual_validation_v2_summary.json`.

O ideal é que a maior parte/total esteja como:

```text
context_status_v2 = BEFORE_AFTER_DIFF_MAPPED
```

Se aparecer `FALLBACK_EXISTING_CONTEXT` ou `...NO_FULL_BEFORE_FILE`,
o checkout original/patchado não estava disponível para remapear o BEFORE.
Nesse caso, regenere enquanto os checkouts estiverem acessíveis.

## 2. Revisão humana

Envie a cada avaliador apenas:

- `manual_validation_review_packet_v2.html`
- `MANUAL_REVIEW_RUBRIC_V2.md`

Não envie a chave.

Cada avaliador abre uma cópia do HTML, informa um ID (`R1`, `R2`) e ao fim
exporta:

```text
manual_review_R1.csv
manual_review_R2.csv
```

Rótulos de validade:

```text
VALID_SECURITY_FINDING
LIKELY_FALSE_POSITIVE
UNCERTAIN
```

Rótulos de atribuição:

```text
INTRODUCED_OR_AFFECTED_BY_PATCH
PRE_EXISTING
UNCERTAIN
```

## 3. Primeiro resultado: concordância e divergências

```bash
python analyze_manual_validation_v2.py \
  --reviewer1 manual_review_R1.csv \
  --reviewer2 manual_review_R2.csv \
  --key manual_validation_v2/manual_validation_key_v2.csv \
  --outdir manual_validation_v2_analysis
```

Isso gera:

- `interrater_v2.csv`
- `manual_validation_v2_summary.csv`
- `disagreements_for_adjudication.csv`
- `all_new_findings_human_review.csv`

O arquivo `disagreements_for_adjudication.csv` deve ser preenchido por consenso
ou terceiro avaliador.

Depois rode novamente:

```bash
python analyze_manual_validation_v2.py \
  --reviewer1 manual_review_R1.csv \
  --reviewer2 manual_review_R2.csv \
  --key manual_validation_v2/manual_validation_key_v2.csv \
  --adjudication manual_validation_v2_analysis/disagreements_for_adjudication.csv \
  --outdir manual_validation_v2_analysis_final
```

## 4. LangSmith / LLM-as-a-judge

O LLM judge é complementar; os humanos são a referência principal.

```bash
pip install -U langsmith openai pydantic

export LANGSMITH_API_KEY="..."
export OPENAI_API_KEY="..."

python langsmith_bandit_judge_v2.py \
  --input manual_validation_v2/manual_validation_review_blinded_v2.csv \
  --judge-model "<modelo>" \
  --outdir langsmith_judge_v2
```

O judge não recebe os campos cegos.

Depois compare com o julgamento humano final:

```bash
python analyze_manual_validation_v2.py \
  --reviewer1 manual_review_R1.csv \
  --reviewer2 manual_review_R2.csv \
  --key manual_validation_v2/manual_validation_key_v2.csv \
  --adjudication manual_validation_v2_analysis/disagreements_for_adjudication.csv \
  --judge langsmith_judge_v2/langsmith_bandit_judge_v2_results.csv \
  --outdir manual_validation_v2_analysis_final
```

## Como relatar no artigo

O resultado primário é a revisão humana.

Exemplo de formulação, substituindo pelos números obtidos:

> We manually inspected all newly detected Bandit findings and a stratified,
> deduplicated sample of pre-existing findings. Two reviewers independently
> assessed finding validity and patch attribution using blinded before/diff/after
> code context. Inter-rater agreement was quantified using Cohen's kappa, and
> disagreements were adjudicated by consensus.

Para o LLM judge:

> An LLM-as-a-judge analysis was conducted as a complementary robustness check
> and was compared against the human consensus; it was not treated as ground truth.
