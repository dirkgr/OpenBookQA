#!/usr/bin/env bash

set -e
set -x
	
config_file=$1
shift

if [ ! -n "${config_file}" ] ; then
  echo "${config_file} is empty"
  echo "Use:"
  echo "$0 training_config/qa/multi_choice/openbookqa/your_allennlp_config.json"
  exit
fi

DATASET_NAME_SHORT=obqa
EXPERIMENTS_OUTPUT_DIR_BASE=_experiments

# question and choices
experiment_prefix_base=${DATASET_NAME_SHORT}_$(basename $config_file)_$(date +%y-%m-%d-%H-%M-%S)-r${RANDOM}
experiment_out_dir=${EXPERIMENTS_OUTPUT_DIR_BASE}/${experiment_prefix_base}

bash scripts/experiments/qa/exec_run_with_5_runs_partial_know.sh ${config_file} ${experiment_out_dir} $*

