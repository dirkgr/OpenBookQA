#!/usr/bin/env python3

from allennlp.common.file_utils import cached_path
from obqa.data.dataset_readers.quark import utilities
import sys
from typing import *
import itertools

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare model predictions across runs')
    parser.add_argument(
        '--questions', '-q',
        help='Filename of the question set to try',
        required=True)
    parser.add_argument(
        'prediction_files',
        type=str,
        nargs='+',
        help="files with predictions, one file per run")

    args = parser.parse_args()

    correct_answers = {
        q['id']: q['answerKey']
        for q in utilities.json_from_file(cached_path(args.questions))
    }

    prediction_files = args.prediction_files
    prediction_files.sort()
    predictions_per_file = {}
    for prediction_file in prediction_files:
        predictions_in_this_file = {}
        for p in utilities.json_from_file(prediction_file):
            qid = p['id']
            predictions = ((c['label'], c['score']) for c in p['prediction']['choices'])
            best_prediction = max(predictions, key=lambda x: x[1])
            predictions_in_this_file[qid] = best_prediction[0]
        predictions_per_file[prediction_file] = predictions_in_this_file

    def longest_prefix_length(strings: List[str]) -> int:
        return sum(itertools.takewhile(lambda x: x == 1, (len(set(x)) for x in zip(*strings))))
    def longest_suffix_length(strings: List[str]) -> int:
        strings = map(reversed, strings)
        return sum(itertools.takewhile(lambda x: x == 1, (len(set(x)) for x in zip(*strings))))

    out = sys.stdout
    out.write('\t')
    prediction_files_prefix_length = longest_prefix_length(prediction_files)
    prediction_files_suffix_length = longest_suffix_length(prediction_files)
    out.write('\t'.join((f[prediction_files_prefix_length:-prediction_files_suffix_length] for f in prediction_files)))
    out.write('\n')
    for qid, correct_answer in correct_answers.items():
        out.write(qid)
        for prediction_file in prediction_files:
            correct = predictions_per_file[prediction_file][qid] == correct_answer
            if correct:
                out.write('\t1')
            else:
                out.write('\t0')
        out.write('\n')

if __name__=="__main__":
    main()
