# Replicate-only: quatro modelos do artigo

## Disponibilidade verificada

- GPT-4o -> `openai/gpt-4o`
- Claude Sonnet 4 -> `anthropic/claude-4-sonnet`
- DeepSeek V3 -> `deepseek-ai/deepseek-v3`
- CodeLlama-34B-Instruct -> `meta/codellama-34b-instruct:eeb928567781f4e90d2aba57a51baef235de53f907c214a4ab42adabf5bb9736`

## Atenção metodológica: temperature

O Replicate não oferece o mesmo conjunto de parâmetros nos quatro endpoints.
Por isso o protocolo atual padroniza somente o prompt e o limite máximo de
saída. O perfil `codellama-tuned` usa explicitamente `temperature=0.2` para
reduzir alucinações; os demais mantêm os defaults dos respectivos endpoints.

O metadata registra os controles efetivos. Essa diferença deve ser declarada;
não escreva no artigo que uma mesma temperature foi aplicada aos quatro.

## Instalação

```bash
pip install replicate pyyaml
export REPLICATE_API_TOKEN="SEU_TOKEN"
```

## Copiar

- `model_profiles_replicate.yaml` -> `configs/model_profiles.yaml`
- `03_model_adapter_replicate.py` -> `scripts/03_model_adapter_replicate.py`
- `08_security_patches_robust_replicate.py` -> `scripts/08_security_patches_robust_replicate.py`

Mantenha também os arquivos robustos já gerados:
- 04_apply_patch_robust.py
- 05_sast_touched_before_after.py
- 10_build_flat_findings_robust.py

## Smoke test: 2 casos

```bash
python scripts/08_security_patches_robust_replicate.py --profile gpt-4o --max-cases 2
python scripts/08_security_patches_robust_replicate.py --profile claude --max-cases 2
python scripts/08_security_patches_robust_replicate.py --profile deepseek --max-cases 2
python scripts/08_security_patches_robust_replicate.py --profile codellama-tuned --max-cases 2
```

Cada caso usa no máximo três gerações por padrão, inclusive quando o diff é
inválido ou inaplicável. Use `--max-generation-attempts 1` para limitar cada
caso a uma chamada. Falhas permanecem registradas e não recebem placeholders.

Depois gere o flat robusto normalmente.
