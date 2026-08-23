SHELL := /bin/bash

# ============================================================
# AMBIENTE
# ============================================================

PY := python3
VENV := .venv
PYBIN := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
HF := $(VENV)/bin/huggingface-cli

ifeq ($(wildcard $(PYBIN)),)
	PYBIN := $(PY)
endif

# ============================================================
# DIRETÓRIOS
# ============================================================

DATA_DIR := data/swe-bench
CASES_DIR := runs/cases
RUNS_ROOT := runs_backup

PROFILES := configs/model_profiles.yaml
PROMPT_TEMPLATE := configs/prompt_template.md

FINAL_CSV := all_findings_flat_robust.csv
SUMMARY_CSV := case_model_before_after_summary.csv
COMMON_FINAL_CSV := all_findings_flat_robust_common.csv
COMMON_SUMMARY_CSV := case_model_before_after_summary_common.csv
STATUS_CSV := case_model_execution_status.csv

CLASSIFIER_DIR := classifier_article_results
SEMANTIC_DIR := semantic_prompt_results
SEMANTIC_GROUP_DIR := semantic_group_ablation_results
PROBLEM_STMT_DIR := problem_statement_ablation_results
PROMPT_FEATURE_DIR := prompt_feature_risk_results
MULTIVARIATE_DIR := multivariate_prompt_risk_results
FINAL_PROMPT_DIR := final_prompt_risk_classifier_results
FINAL_MODEL_DIR := final_model_vulnerability_results

SEMANTIC_FEATURES_CSV := $(SEMANTIC_DIR)/case_model_with_prompt_semantic_features.csv
PROBLEM_STMT_FEATURES_CSV := $(PROBLEM_STMT_DIR)/case_model_problem_statement_features.csv

# ============================================================
# CONFIGURAÇÃO DO EXPERIMENTO
# ============================================================

# gpt-4o | claude | deepseek | codellama-tuned
PROFILE ?= gpt-4o

# Exemplo:
# make run PROFILE=gpt-4o MAX_CASES=2
#
# Vazio = todos os casos disponíveis em runs/cases
MAX_CASES ?=
# Arquivo opcional com um case_id por linha
CASE_LIST ?=
# 1 = remove JSONs antigos antes de preparar novo conjunto
CLEAN_CASES ?= 1
# 1 = continua/retorna sucesso quando parte dos patches falha
ALLOW_PARTIAL ?= 0
# 1 = ignora patches existentes e chama o Replicate novamente
FORCE_GENERATE ?= 0
# Máximo uniforme de gerações por caso (inclui patch inaplicável)
GENERATION_ATTEMPTS ?= 3
# 1 = autoriza execução paga sem limite de casos
CONFIRM_ALL ?= 0
# Análises 11–18 (classificador / ablação / risco)
ANALYSIS_RUNS ?= 30
ANALYSIS_SEED ?= 42
ANALYSIS_BOOTSTRAP ?= 10000

# ============================================================
# PHONY
# ============================================================

.PHONY: \
	help setup check-token \
	download-swe prepare-cases fetch-repos \
	run run-gpt run-claude run-deepseek run-codellama-tuned \
	smoke smoke-all run-all run-all-parallel \
	build-flat validate clean-results

# ============================================================
# HELP
# ============================================================

help:
	@echo ""
	@echo "LLM Security Risk Pipeline - Robust Version"
	@echo "==========================================="
	@echo ""
	@echo "Preparação:"
	@echo "  make setup"
	@echo "  make download-swe"
	@echo "  make prepare-cases MAX_CASES=10"
	@echo ""
	@echo "Teste rápido:"
	@echo "  make smoke PROFILE=gpt-4o"
	@echo "  make smoke-all"
	@echo ""
	@echo "Rodar modelo:"
	@echo "  make run PROFILE=gpt-4o MAX_CASES=10"
	@echo "  make run PROFILE=claude MAX_CASES=10"
	@echo "  make run PROFILE=deepseek MAX_CASES=10"
	@echo "  make run PROFILE=codellama-tuned MAX_CASES=10"
	@echo ""
	@echo "Rodar os quatro:"
	@echo "  make run-all MAX_CASES=10"
	@echo "  make run-all-parallel MAX_CASES=10"
	@echo ""
	@echo "CSV final e análises 11–18:"
	@echo "  make build-flat"
	@echo ""
	@echo "Variáveis:"
	@echo "  PROFILE=$(PROFILE)"
	@echo "  MAX_CASES=$(MAX_CASES)"
	@echo "  CLEAN_CASES=$(CLEAN_CASES)"
	@echo "  ALLOW_PARTIAL=$(ALLOW_PARTIAL)"
	@echo "  FORCE_GENERATE=$(FORCE_GENERATE)"
	@echo "  GENERATION_ATTEMPTS=$(GENERATION_ATTEMPTS)"
	@echo "  CONFIRM_ALL=$(CONFIRM_ALL)"
	@echo "  ANALYSIS_RUNS=$(ANALYSIS_RUNS)"
	@echo "  ANALYSIS_SEED=$(ANALYSIS_SEED)"
	@echo "  ANALYSIS_BOOTSTRAP=$(ANALYSIS_BOOTSTRAP)"
	@echo ""

# ============================================================
# SETUP
# ============================================================

setup:
	$(PY) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install \
		replicate \
		pyyaml \
		bandit==1.7.9 \
		pandas==2.2.2 \
		numpy \
		scipy \
		scikit-learn \
		tiktoken \
		datasets \
		huggingface_hub \
		tqdm \
		matplotlib
	@echo ""
	@echo "Setup concluído."
	@echo "Agora defina:"
	@echo '  export REPLICATE_API_TOKEN="seu_token"'

check-token:
	@if [ -z "$$REPLICATE_API_TOKEN" ]; then \
		echo "ERRO: REPLICATE_API_TOKEN não definido."; \
		echo 'Execute: export REPLICATE_API_TOKEN="seu_token"'; \
		exit 1; \
	fi
	@echo "REPLICATE_API_TOKEN configurado."

# ============================================================
# DATASET
# ============================================================

download-swe:
	mkdir -p $(DATA_DIR)
	$(HF) download princeton-nlp/SWE-bench \
		--repo-type dataset \
		--local-dir $(DATA_DIR)

prepare-cases:
	mkdir -p $(CASES_DIR)
ifneq ($(MAX_CASES),)
	$(PYBIN) scripts/01_prepare_cases.py \
		--max_cases $(MAX_CASES) \
		$(if $(filter 1,$(CLEAN_CASES)),--clean,)
else
	$(PYBIN) scripts/01_prepare_cases.py \
		--max_cases 0 \
		$(if $(filter 1,$(CLEAN_CASES)),--clean,)
endif

# ============================================================
# REPOSITÓRIOS
# ============================================================

fetch-repos:
	@echo "Preparando repositórios dos casos..."
	$(PYBIN) scripts/00_fetch_repo.py \
		--cases_dir "$(CASES_DIR)" \
		--out_root "$(DATA_DIR)/repos" \
		$(if $(MAX_CASES),--max_cases $(MAX_CASES),)

# ============================================================
# PIPELINE ROBUSTO
# ============================================================

run: check-token
	@if [ -z "$(MAX_CASES)" ] && [ "$(CONFIRM_ALL)" != "1" ]; then \
		echo "ERRO: execução sem limite recusada para evitar gasto acidental."; \
		echo "Defina MAX_CASES=N ou CONFIRM_ALL=1."; \
		exit 1; \
	fi
	@echo ""
	@echo "=========================================="
	@echo "Profile: $(PROFILE)"
	@echo "=========================================="
	@echo ""
	$(PYBIN) scripts/08_security_patches_robust_replicate.py \
		--profile $(PROFILE) \
		--profiles $(PROFILES) \
		--cases-dir $(CASES_DIR) \
		$(if $(CASE_LIST),--case-list "$(CASE_LIST)",) \
		--template-md $(PROMPT_TEMPLATE) \
		--max-generation-attempts $(GENERATION_ATTEMPTS) \
		$(if $(MAX_CASES),--max-cases $(MAX_CASES),) \
		$(if $(filter 1,$(ALLOW_PARTIAL)),--allow-partial,) \
		$(if $(filter 1,$(FORCE_GENERATE)),--force-generate,)

run-gpt:
	$(MAKE) run PROFILE=gpt-4o MAX_CASES="$(MAX_CASES)"

run-claude:
	$(MAKE) run PROFILE=claude MAX_CASES="$(MAX_CASES)"

run-deepseek:
	$(MAKE) run PROFILE=deepseek MAX_CASES="$(MAX_CASES)"

run-codellama-tuned:
	$(MAKE) run PROFILE=codellama-tuned MAX_CASES="$(MAX_CASES)"

# ============================================================
# SMOKE TESTS
# ============================================================

smoke:
	$(MAKE) run PROFILE=$(PROFILE) MAX_CASES=2

smoke-all:
	$(MAKE) run PROFILE=gpt-4o MAX_CASES=2
	$(MAKE) run PROFILE=claude MAX_CASES=2
	$(MAKE) run PROFILE=deepseek MAX_CASES=2
	$(MAKE) run PROFILE=codellama-tuned MAX_CASES=2
	$(MAKE) build-flat

# ============================================================
# TODOS OS MODELOS
# ============================================================

run-all:
	$(MAKE) run PROFILE=gpt-4o MAX_CASES="$(MAX_CASES)" ALLOW_PARTIAL=1
	$(MAKE) run PROFILE=claude MAX_CASES="$(MAX_CASES)" ALLOW_PARTIAL=1
	$(MAKE) run PROFILE=deepseek MAX_CASES="$(MAX_CASES)" ALLOW_PARTIAL=1
	$(MAKE) run PROFILE=codellama-tuned MAX_CASES="$(MAX_CASES)" ALLOW_PARTIAL=1
	$(MAKE) build-flat

run-all-parallel:
	MAX_CASES="$(MAX_CASES)" \
	GENERATION_ATTEMPTS="$(GENERATION_ATTEMPTS)" \
	CONFIRM_ALL="$(CONFIRM_ALL)" \
	./scripts/run_all_parallel.sh

# ============================================================
# CSV FINAL + ANÁLISES 11–18
# ============================================================

build-flat:
	$(PYBIN) scripts/10_build_flat_findings_robust.py \
		--runs-root $(RUNS_ROOT) \
		--profiles $(PROFILES) \
		--out-csv $(FINAL_CSV) \
		--summary-csv $(SUMMARY_CSV) \
		--common-out-csv $(COMMON_FINAL_CSV) \
		--common-summary-csv $(COMMON_SUMMARY_CSV) \
		--status-csv $(STATUS_CSV)

	$(PYBIN) scripts/11_classifier_article_protocol.py \
		--input $(COMMON_SUMMARY_CSV) \
		--runs $(ANALYSIS_RUNS) \
		--seed $(ANALYSIS_SEED) \
		--out-dir $(CLASSIFIER_DIR)

	$(PYBIN) scripts/12_prompt_semantic_features_and_ablation.py \
		--input $(COMMON_SUMMARY_CSV) \
		--runs-root $(RUNS_ROOT) \
		--cases-dir $(CASES_DIR) \
		--runs $(ANALYSIS_RUNS) \
		--seed $(ANALYSIS_SEED) \
		--out-dir $(SEMANTIC_DIR)

	$(PYBIN) scripts/13_prompt_semantic_group_ablation_revised.py \
		--input $(SEMANTIC_FEATURES_CSV) \
		--runs-root $(RUNS_ROOT) \
		--cases-dir $(CASES_DIR) \
		--runs $(ANALYSIS_RUNS) \
		--seed $(ANALYSIS_SEED) \
		--bootstrap-resamples $(ANALYSIS_BOOTSTRAP) \
		--out-dir $(SEMANTIC_GROUP_DIR)

	$(PYBIN) scripts/14_problem_statement_ablation.py \
		--input $(COMMON_SUMMARY_CSV) \
		--cases-dir $(CASES_DIR) \
		--runs $(ANALYSIS_RUNS) \
		--seed $(ANALYSIS_SEED) \
		--bootstrap-resamples $(ANALYSIS_BOOTSTRAP) \
		--out-dir $(PROBLEM_STMT_DIR)

	$(PYBIN) scripts/15_prompt_feature_risk_analysis.py \
		--input $(PROBLEM_STMT_FEATURES_CSV) \
		--out-dir $(PROMPT_FEATURE_DIR)

	$(PYBIN) scripts/16_multivariate_prompt_risk.py \
		--input $(PROBLEM_STMT_FEATURES_CSV) \
		--runs $(ANALYSIS_RUNS) \
		--seed $(ANALYSIS_SEED) \
		--bootstrap-resamples $(ANALYSIS_BOOTSTRAP) \
		--out-dir $(MULTIVARIATE_DIR)

	$(PYBIN) scripts/17_final_prompt_risk_classifier.py \
		--input $(PROBLEM_STMT_FEATURES_CSV) \
		--runs $(ANALYSIS_RUNS) \
		--seed $(ANALYSIS_SEED) \
		--bootstrap-resamples $(ANALYSIS_BOOTSTRAP) \
		--out-dir $(FINAL_PROMPT_DIR)

	$(PYBIN) scripts/18_final_model_vulnerability_analysis.py \
		--input $(COMMON_SUMMARY_CSV) \
		--features-csv $(PROBLEM_STMT_FEATURES_CSV) \
		--runs $(ANALYSIS_RUNS) \
		--seed $(ANALYSIS_SEED) \
		--bootstrap-resamples $(ANALYSIS_BOOTSTRAP) \
		--out-dir $(FINAL_MODEL_DIR)

	@echo ""
	@echo "Gerados:"
	@echo "  $(FINAL_CSV)"
	@echo "  $(SUMMARY_CSV)"
	@echo "  $(COMMON_FINAL_CSV)"
	@echo "  $(COMMON_SUMMARY_CSV)"
	@echo "  $(STATUS_CSV)"
	@echo "  $(CLASSIFIER_DIR)/"
	@echo "  $(SEMANTIC_DIR)/"
	@echo "  $(SEMANTIC_GROUP_DIR)/"
	@echo "  $(PROBLEM_STMT_DIR)/"
	@echo "  $(PROMPT_FEATURE_DIR)/"
	@echo "  $(MULTIVARIATE_DIR)/"
	@echo "  $(FINAL_PROMPT_DIR)/"
	@echo "  $(FINAL_MODEL_DIR)/"

# ============================================================
# VALIDAÇÃO RÁPIDA
# ============================================================

validate:
	@echo ""
	@echo "Arquivos obrigatórios:"
	@test -f scripts/00_fetch_repo.py && echo "OK 00_fetch_repo.py" || echo "FALTA 00_fetch_repo.py"
	@test -f scripts/repo_paths.py && echo "OK repo_paths.py" || echo "FALTA repo_paths.py"
	@test -f scripts/01_prepare_cases.py && echo "OK 01_prepare_cases.py" || echo "FALTA 01_prepare_cases.py"
	@test -f scripts/02_prompt_builder.py && echo "OK 02_prompt_builder.py" || echo "FALTA 02_prompt_builder.py"
	@test -f scripts/03_model_adapter_replicate.py && echo "OK 03_model_adapter_replicate.py" || echo "FALTA 03_model_adapter_replicate.py"
	@test -f scripts/04_apply_patch_robust.py && echo "OK 04_apply_patch_robust.py" || echo "FALTA 04_apply_patch_robust.py"
	@test -f scripts/05_sast_touched_before_after.py && echo "OK 05_sast_touched_before_after.py" || echo "FALTA 05_sast_touched_before_after.py"
	@test -f scripts/08_security_patches_robust_replicate.py && echo "OK 08_security_patches_robust_replicate.py" || echo "FALTA 08_security_patches_robust_replicate.py"
	@test -f scripts/10_build_flat_findings_robust.py && echo "OK 10_build_flat_findings_robust.py" || echo "FALTA 10_build_flat_findings_robust.py"
	@test -f scripts/11_classifier_article_protocol.py && echo "OK 11_classifier_article_protocol.py" || echo "FALTA 11_classifier_article_protocol.py"
	@test -f scripts/12_prompt_semantic_features_and_ablation.py && echo "OK 12_prompt_semantic_features_and_ablation.py" || echo "FALTA 12_prompt_semantic_features_and_ablation.py"
	@test -f scripts/13_prompt_semantic_group_ablation_revised.py && echo "OK 13_prompt_semantic_group_ablation_revised.py" || echo "FALTA 13_prompt_semantic_group_ablation_revised.py"
	@test -f scripts/14_problem_statement_ablation.py && echo "OK 14_problem_statement_ablation.py" || echo "FALTA 14_problem_statement_ablation.py"
	@test -f scripts/15_prompt_feature_risk_analysis.py && echo "OK 15_prompt_feature_risk_analysis.py" || echo "FALTA 15_prompt_feature_risk_analysis.py"
	@test -f scripts/16_multivariate_prompt_risk.py && echo "OK 16_multivariate_prompt_risk.py" || echo "FALTA 16_multivariate_prompt_risk.py"
	@test -f scripts/17_final_prompt_risk_classifier.py && echo "OK 17_final_prompt_risk_classifier.py" || echo "FALTA 17_final_prompt_risk_classifier.py"
	@test -f scripts/18_final_model_vulnerability_analysis.py && echo "OK 18_final_model_vulnerability_analysis.py" || echo "FALTA 18_final_model_vulnerability_analysis.py"
	@test -f configs/model_profiles.yaml && echo "OK model_profiles.yaml" || echo "FALTA model_profiles.yaml"
	@test -f configs/prompt_template.md && echo "OK prompt_template.md" || echo "FALTA prompt_template.md"

# ============================================================
# LIMPEZA DE RESULTADOS
# ============================================================

clean-results:
	rm -rf runs_backup/claud-sonnet_backup
	rm -rf runs_backup/gpt-4o_backup
	rm -rf runs_backup/deepseek_backup
	rm -rf runs_backup/codellama_backup
	rm -rf runs_backup/codellama_tuned_backup
	rm -f $(FINAL_CSV)
	rm -f $(SUMMARY_CSV)
	rm -f $(COMMON_FINAL_CSV)
	rm -f $(COMMON_SUMMARY_CSV)
	rm -f $(STATUS_CSV)
	rm -rf $(CLASSIFIER_DIR)
	rm -rf $(SEMANTIC_DIR)
	rm -rf $(SEMANTIC_GROUP_DIR)
	rm -rf $(PROBLEM_STMT_DIR)
	rm -rf $(PROMPT_FEATURE_DIR)
	rm -rf $(MULTIVARIATE_DIR)
	rm -rf $(FINAL_PROMPT_DIR)
	rm -rf $(FINAL_MODEL_DIR)