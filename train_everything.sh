#!/usr/bin/env bash

set -x
set -e

FINAL_EXPERIMENT_DIR=$1
shift
test -e $FINAL_EXPERIMENT_DIR && (echo "$FINAL_EXPERIMENT_DIR exists, bailing" && exit 1)

NUMBER_OF_RUNS=5

function run_experiment() {
  set -x
  set -e

  CONFIG=$1
  shift
  SPLIT=$1
  shift
  OUTPUT_DIR=$1
  shift
  OUTPUT_DIR="$OUTPUT_DIR/$(basename $CONFIG .json)/run$SPLIT"

  export RANDOM_SEED=${RANDOM}
  python -u obqa/run.py train "${CONFIG}" -s "${OUTPUT_DIR}" "$@"
  # evaluate without attentions
  python obqa/run.py evaluate_predictions_qa_mc --archive_file "${OUTPUT_DIR}/model.tar.gz" --output_file "${OUTPUT_DIR}/predictions"

  # convert evaluation to aristo-eval json
  python tools/predictions_to_aristo_eval_json.py "${OUTPUT_DIR}/predictions_dev.txt" > "${OUTPUT_DIR}/aristo_evaluator_predictions_dev.txt"
  python tools/predictions_to_aristo_eval_json.py "${OUTPUT_DIR}/predictions_test.txt" > "${OUTPUT_DIR}/aristo_evaluator_predictions_test.txt"

  # try to export also attentions. This will fail for no-knowledge models
  knowledge_model_name="qa_multi_choice_know_reader_v1"
  if grep -q ${knowledge_model_name} "${CONFIG}"; then
      echo "${knowledge_model_name} is used in the config $(basename $CONFIG). Exporting attentions values for dev and test."
      python obqa/run.py evaluate_predictions_qa_mc_know_visualize --archive_file "${OUTPUT_DIR}/model.tar.gz" --output_file "${OUTPUT_DIR}/predictions_visual"
  fi
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

parallel --shuf --halt now,fail=1 --line-buffer -j3 -q run_experiment {1} {2} "$FINAL_EXPERIMENT_DIR" "$@" ::: ${CONFIGS[*]} ::: $(seq $NUMBER_OF_RUNS)

for CONFIG in ${CONFIGS[*]}; do
  CONFIG=$(basename $CONFIG .json)
  python tools/merge_metrics_files.py \
    $FINAL_EXPERIMENT_DIR/$CONFIG/run?/metrics.json \
    $FINAL_EXPERIMENT_DIR/$CONFIG.json
done

echo "Success! Results are in $FINAL_EXPERIMENT_DIR"
