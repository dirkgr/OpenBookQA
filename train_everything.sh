#!/usr/bin/env bash

set -Eeuxo pipefail

FINAL_EXPERIMENT_DIR=$1
shift
test -e $FINAL_EXPERIMENT_DIR && (echo "$FINAL_EXPERIMENT_DIR exists, bailing" && exit 1)

NUMBER_OF_RUNS=5

function run_experiment() {
  set -Eeuxo pipefail

  CONFIG=$1
  shift
  SPLIT=$1
  shift
  OUTPUT_DIR=$1
  shift

  mkdir -p "$OUTPUT_DIR/$(basename $CONFIG .json)"
  LOG_OUTPUT="$OUTPUT_DIR/$(basename $CONFIG .json)/log$SPLIT.txt"
  OUTPUT_DIR="$OUTPUT_DIR/$(basename $CONFIG .json)/run$SPLIT"

  GPU_COUNT=$(nvidia-smi -L | wc -l)
  for GPU_NUMBER in $(seq 0 $(($GPU_COUNT-1))); do
    flock -w 1 /tmp/gpu$GPU_NUMBER.lck echo "Using GPU $GPU_NUMBER" && break
  done
  
  (
    flock -s 200
    export CUDA_VISIBLE_DEVICES=$GPU_NUMBER
    export RANDOM_SEED=${RANDOM}
    echo "*** Training $CONFIG/$SPLIT ***" | tee "$LOG_OUTPUT"
    python -u obqa/run.py train "${CONFIG}" -s "${OUTPUT_DIR}" -o "{trainer:{cuda_device:0}}" "$@" 2>&1 | tee -a "$LOG_OUTPUT"
  ) 200> /tmp/gpu$GPU_NUMBER.lck

  # evaluate without attentions
  echo "*** Evaluating $CONFIG/$SPLIT ***" | tee -a "$LOG_OUTPUT"
  python obqa/run.py evaluate_predictions_qa_mc --archive_file "${OUTPUT_DIR}/model.tar.gz" --output_file "${OUTPUT_DIR}/predictions" 2>&1 | tee -a "$LOG_OUTPUT"

  # convert evaluation to aristo-eval json
  echo "*** Making aristo predictions for $CONFIG/$SPLIT ***" | tee -a "$LOG_OUTPUT"
  python tools/predictions_to_aristo_eval_json.py "${OUTPUT_DIR}/predictions_dev.txt" > "${OUTPUT_DIR}/aristo_evaluator_predictions_dev.txt" 2>&1 | tee -a "$LOG_OUTPUT"
  python tools/predictions_to_aristo_eval_json.py "${OUTPUT_DIR}/predictions_test.txt" > "${OUTPUT_DIR}/aristo_evaluator_predictions_test.txt" 2>&1 | tee -a "$LOG_OUTPUT"

  # try to export also attentions. This will fail for no-knowledge models
  knowledge_model_name="qa_multi_choice_know_reader_v1"
  if grep -q ${knowledge_model_name} "${CONFIG}"; then
      echo "*** Exporting attention values for dev and test for $CONFIG/$SPLIT ***" | tee -a "$LOG_OUTPUT"
      python obqa/run.py evaluate_predictions_qa_mc_know_visualize --archive_file "${OUTPUT_DIR}/model.tar.gz" --output_file "${OUTPUT_DIR}/predictions_visual" 2>&1 | tee -a "$LOG_OUTPUT"
  fi
  echo "*** Done with $CONFIG/$SPLIT! ***" | tee -a "$LOG_OUTPUT"
}
export -f run_experiment  # needed so parallel can call the function

CONFIGS=( \
  training_config/qa/multi_choice/openbookqa/reader_mc_qa_question_to_choice.json \
  training_config/qa/multi_choice/openbookqa/reader_mc_qa_question_to_choice_elmo.json \
  training_config/qa/multi_choice/openbookqa/reader_mc_qa_esim.json \
  training_config/qa/multi_choice/openbookqa/reader_mc_qa_esim_elmo.json \
  training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_oracle_openbook_plus_cn5omcs.json \
  training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_oracle_openbook_plus_cn5wordnet.json \
  training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_cn5omcs.json \
  training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_cn5wordnet.json \
  training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_openbook_plus_cn5omcs.json \
  training_config/qa/multi_choice/openbookqa/knowreader_v1_mc_qa_multi_source_openbook_plus_cn5wordnet.json \
)

parallel --halt 2 --line-buffer --jobs $(nvidia-smi -L | wc -l) -q run_experiment {2} {1} "$FINAL_EXPERIMENT_DIR" "$@" ::: $(seq $NUMBER_OF_RUNS) ::: ${CONFIGS[*]}

for CONFIG in ${CONFIGS[*]}; do
  CONFIG=$(basename $CONFIG .json)
  python tools/merge_metrics_files.py \
    $FINAL_EXPERIMENT_DIR/$CONFIG/run?/metrics.json \
    $FINAL_EXPERIMENT_DIR/$CONFIG.json
done

echo "Success! Results are in $FINAL_EXPERIMENT_DIR"
