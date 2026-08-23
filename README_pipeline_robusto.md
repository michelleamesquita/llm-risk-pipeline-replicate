# Pipeline robusto para o artigo

## Objetivo

Este pipeline mantém o desenho do seu experimento, mas corrige os dois problemas
mais importantes:

1. o Bandit passa a analisar **somente os arquivos efetivamente modificados pelo patch**;
2. para cada finding AFTER, o CSV registra se ele **já existia no BEFORE** ou se é
   **novo após o patch**.

Assim você poderá trabalhar com duas definições sem misturá-las:

- `is_risky_after`: existe HIGH no arquivo modificado após a geração;
- `is_risky_new`: existe HIGH novo no arquivo modificado após a geração.

`is_risky` é mantido no CSV por compatibilidade e equivale a `is_risky_after`.

## Arquivos

Copie para o projeto:

- `model_profiles_article.yaml` -> `configs/model_profiles.yaml`
- `04_apply_patch_robust.py` -> `scripts/04_apply_patch_robust.py`
- `05_sast_touched_before_after.py` -> `scripts/05_sast_touched_before_after.py`
- `08_security_patches_robust.py` -> `scripts/08_security_patches_robust.py`
- `10_build_flat_findings_robust.py` -> `scripts/10_build_flat_findings_robust.py`

Seu `02_prompt_builder.py`, `03_model_adapter.py` e `prompt_template.md` podem continuar.

## Modelos do artigo

- Claude Sonnet 4
- GPT-4o
- DeepSeek V3
- CodeLlama-34B

O protocolo atual padroniza somente o prompt e o limite máximo de saída.
`temperature`, `top_p` e `top_k` não são usados como fatores experimentais,
pois os quatro endpoints não expõem os mesmos controles.

## Primeiro teste

Não comece pelo conjunto completo. Prepare e tente 5 casos por modelo:

```bash
make prepare-cases MAX_CASES=5
make fetch-repos MAX_CASES=5
python scripts/08_security_patches_robust_replicate.py --profile gpt-4o --max-cases 5
python scripts/08_security_patches_robust_replicate.py --profile claude --max-cases 5
python scripts/08_security_patches_robust_replicate.py --profile deepseek --max-cases 5
python scripts/08_security_patches_robust_replicate.py --profile codellama-tuned --max-cases 5
```

O target `prepare-cases` limpa casos antigos por padrão para que `MAX_CASES`
seja aplicado ao mesmo conjunto em todas as etapas. Reexecuções reutilizam
gerações completas e não chamam o Replicate novamente, salvo com
`FORCE_GENERATE=1`/`--force-generate`.

O prompt recebe o código real no `base_commit`: arquivos pequenos completos e,
para arquivos grandes, trechos recuperados lexicalmente pelos termos do issue.
O prompt completo é limitado a 5.000 caracteres, com os mesmos limites de issue
e código-fonte para os quatro modelos, para caber na janela do CodeLlama. O
gold patch não é incluído; somente a lista de arquivos relevantes já usada pelo
protocolo.

Depois gere o CSV:

```bash
python scripts/10_build_flat_findings_robust.py   --runs-root runs_backup   --profiles configs/model_profiles.yaml   --out-csv all_findings_flat_robust.csv   --summary-csv case_model_before_after_summary.csv
```

## Campos novos do CSV final

Além das colunas antigas:

- `relative_filename`: caminho normalizado dentro do repositório;
- `file_touched_by_patch`: sempre 1 na base principal;
- `patch_apply_success`: 1 somente para patches reais aplicados;
- `finding_fingerprint`: identidade estável do finding sem depender da linha;
- `existed_before`: finding equivalente existia antes do patch;
- `is_new_finding`: finding apareceu somente após o patch;
- `is_risky_after`: o arquivo possui pelo menos um HIGH no AFTER;
- `is_risky_new`: o arquivo possui pelo menos um HIGH novo;
- `severity_score`: LOW=1, MEDIUM=2, HIGH=3.

Também é produzido `case_model_before_after_summary.csv`, uma linha por
`case x model`, com `findings_before`, `findings_after`, `findings_new`,
`high_before`, `high_after`, `high_new`, `delta_high` etc.

## Importante sobre o artigo

Para a pergunta principal sobre "risco presente no código produzido", use
`is_risky_after` / métricas AFTER.

Para afirmar "vulnerabilidade introduzida pelo LLM", use `is_new_finding`
ou `is_risky_new`.

Isso permite apresentar as duas análises sem exagerar a conclusão.

`experiment_valid`/`pipeline_valid` significam somente: patch aplicado e
Bandit BEFORE/AFTER concluído. Os testes funcionais SWE-bench não são executados
aqui. Casos aplicados via `patch --fuzz` são marcados em `fuzzy_patch_apply` e
devem ser reportados separadamente dos casos `strict_patch_apply`.
