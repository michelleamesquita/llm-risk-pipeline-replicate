ALTERAÇÕES PARA META DE 300 CASOS COMUNS VÁLIDOS

Arquivos para substituir em scripts/:
1) 08_security_patches_robust_replicate.py
   - MAX_CASES agora conta apenas casos incompletos que realmente precisam de trabalho.
   - casos experiment_valid já completos são retomados/ignorados e NÃO consomem a cota do batch.
   - geração válida existente continua sendo reutilizada; falhas são tentadas seletivamente.

2) 10_build_flat_findings_robust.py
   - cria all_findings_before_flat_robust.csv.
   - adiciona no resumo case x model:
       risk_before
       risk_after
       delta_risk
       risk_introduced
   - score: LOW=1, MEDIUM=2, HIGH=3.
   - mantém matching BEFORE/AFTER por fingerprint e findings novos.

3) run_until_300_common.sh
   - coloque em scripts/.
   - mede a interseção dos quatro modelos antes de gastar API.
   - seleciona casos ainda não comuns e chama somente os modelos ausentes.
   - prioriza casos que precisam de menos modelos para entrar na interseção.
   - executa os perfis sequencialmente para evitar rate limit compartilhado.
   - não repete o mesmo par caso/modelo em rodadas posteriores.
   - padrão: TARGET_COMMON=300, BATCH_CASES=100, GENERATION_ATTEMPTS=3.

Uso sugerido:
  export REPLICATE_API_TOKEN='...'
  TARGET_COMMON=300 BATCH_CASES=100 GENERATION_ATTEMPTS=3 bash scripts/run_until_300_common.sh

Para uma primeira rodada mais conservadora:
  TARGET_COMMON=300 BATCH_CASES=50 GENERATION_ATTEMPTS=3 bash scripts/run_until_300_common.sh

IMPORTANTE:
- NÃO use --force-generate para esta continuação, pois isso desperdiçaria chamadas já válidas.
- Não apague runs_backup: ele é usado para detectar os casos já completos.
- O estado fica em runs_backup/target_common_state.json.
- Use RESET_TARGET_STATE=1 somente se quiser permitir uma nova tentativa dos
  pares selecionados anteriormente.
- O script pressupõe os run_dir já usados pelo projeto:
  gpt-4o_backup, claud-sonnet_backup, deepseek_backup, codellama_tuned_backup.
