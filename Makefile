.PHONY: clean activate_env notebook tidy_data mf_data werk_data lint requirements train_model_old train_model load_model_and_predict load_model_and_show_inputs prepare_mysql read_parquet load_model_and_predict_json

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
export THIS_PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = manufuture_part_price_prediction
PYTHON_INTERPRETER = python3
RAW_WERK_DATA = data/raw/werk_data
EXTERNAL_WERK_DATA = data/external/werk_data
EXTERNAL_MF_DATA = data/external/mf_data
RAW_MF_DATA = data/raw/mf_data
RAW_MF_PRICES = data/raw/mf_prices
INTERIM_WERK_DATA = data/interim/werk_data
INTERIM_MF_DATA = data/interim/mf_data
PROCESSED_DATA = data/processed


ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Fetch Manufuture Data from MySQL
mf_data: ## requirements
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option prices_in_csvs_to_parquets --io $(EXTERNAL_MF_DATA) $(RAW_MF_PRICES)
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option manufuture_db_to_parquets --io none $(RAW_MF_DATA)


## Activate python environment
activate_env: ## requirements
	@echo ">>> Activation command: source activate $(PROJECT_NAME)"


## Make Werk Dataset
process_werk_data: ## requirements
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option process_werk_data --io $(EXTERNAL_WERK_DATA) $(INTERIM_WERK_DATA)

## Prepare manufuture_rnd database
prepare_mysql: ## requirements
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option prepare_mysql


## Prepare tidy data (handle Manufuture and Werk data if needed)
tidy_data: werk_data process_werk_data mf_data
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option prepare_tidy_data --mf_data_filepath $(RAW_MF_DATA) --mf_prices_filepath $(RAW_MF_PRICES) --werk_input_filepath $(INTERIM_WERK_DATA) --output_filepath $(PROCESSED_DATA)


## Run jupyter notebook server
notebook: ## requirements
	jupyter notebook

### Train pricing model on processed data and save it to models
#train_model_old: tidy_data
#	$(PYTHON_INTERPRETER) src/models/train_model_and_save.py $(PROCESSED_DATA) models

## Train a model and save it to models
train_model:
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option train_model_and_save --model_output_filepath models

## Evaluate model: make evaluate_model MODEL_NAME=MA_128_64_32_MIH_0x576d7_TH_0x432_E_10_BS_32_LR_0.01
evaluate_model:
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option evaluate_model --model_name $(MODEL_NAME)

## Evaluate model: make show_model_details MODEL_NAME=MA_128_64_32_MIH_0x576d7_TH_0x432_E_10_BS_32_LR_0.01
show_model_details:
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option show_model_details --model_name $(MODEL_NAME)

## Load model and predict on PART_FEATURES_PRED_INPUT: make load_model_and_predict MODEL_NAME=model__[100, 50, 20, 10]__T__part_price_training_table_646_training
load_model_and_predict:
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option load_model_and_predict --model_name $(MODEL_NAME)

## Load model and show inputs: make load_model_and_show_inputs MODEL_NAME=model__[100, 50, 20, 10]__T__part_price_training_table_646_training
load_model_and_show_inputs:
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option load_model_and_show_inputs --model_name $(MODEL_NAME)

## Load model and predict on raw werk: make load_model_and_predict_on_raw MODEL_NAME=MA_128_64_32_MIH_0x576d7_TH_0x432_E_10_BS_32_LR_0.01
load_model_and_predict_on_raw:
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option load_model_and_predict_on_raw --model_name $(MODEL_NAME)

## Load model and predict on json: make load_model_and_predict_json MODEL_NAME=MA_128_64_32_MIH_0x6d79_TH_0x432_E_150_BS_30_LR_0.001
load_model_and_predict_json:
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option load_model_and_predict_json --model_name $(MODEL_NAME)

drop_all_models:
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option drop_all_models

## read_parquet: make read_parquet PARQUET_PATH=data/raw/mf_data/wp_type_part.parquet
make read_parquet:
	$(PYTHON_INTERPRETER) src/data/entry_point.py --option read_parquet --parquet_path $(PARQUET_PATH)


## Delete all compiled Python files, and all parquet files
clean: drop_all_models
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find data/processed -type f -name "*.parquet" -delete
	find data/interim -type f -name "*.parquet" -delete
	find data/raw -type f -name "*.parquet" -delete
	find data/raw/mf_data -type f -name "*.parquet" -delete
	find data/raw/mf_prices -type f -name "*.parquet" -delete



## Lint using flake8
lint:
	flake8 src

# ## Upload Data to S3
# sync_data_to_s3:
# ifeq (default,$(PROFILE))
# 	aws s3 sync data/ s3://$(BUCKET)/data/
# else
# 	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
# endif
#
# ## Download Data from S3
# sync_data_from_s3:
# ifeq (default,$(PROFILE))
# 	aws s3 sync s3://$(BUCKET)/data/ data/
# else
# 	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
# endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3.9
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
		@echo
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
