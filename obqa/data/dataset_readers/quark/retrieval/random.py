#!/usr/bin/python3


"""
Script to add support for each answer choice and question, based on the perfect support given in the
OpenBookQA set.
USAGE:
 python scripts/add_gold_retrieved_text.py qa_file output_file

JSONL format of files
questions:
  {
    "id": "Mercury_SC_415702",
    "question": {
      "stem": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?",
      "choices": [
        {
          "text": "dry palms",
          "label": "A"
        },
        {
          "text": "wet palms",
          "label": "B"
        },
        {
          "text": "palms covered with oil",
          "label": "C"
        },
        {
          "text": "palms covered with lotion",
          "label": "D"
        }
      ]
    },
    "answerKey": "A",
    "fact1": "palms are sweaty",
    "fact2": "knees weak arms are heavy"
  },
 ...

 2. output:
  {
    "id": "Mercury_SC_415702",
    "question": {
      "stem": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?",
      "choices": [
        {
          "text": "dry palms",
          "label": "A",
          "support": [
            { "text": "palms are sweaty", "type": "sentence" },
            { "text": "knees weak arms are heavy", "type": "sentence" },
          ]
        },
        {
          "text": "wet palms",
          "label": "B",
          "support": [
            { "text": "palms are sweaty", "type": "sentence" },
            { "text": "knees weak arms are heavy", "type": "sentence" },
          ]
        },
        {
          "text": "palms covered with oil",
          "label": "C",
          "support": [
            { "text": "palms are sweaty", "type": "sentence" },
            { "text": "knees weak arms are heavy", "type": "sentence" },
          ]
        },
        {
          "text": "palms covered with lotion",
          "label": "D",
          "support": [
            { "text": "palms are sweaty", "type": "sentence" },
            { "text": "knees weak arms are heavy", "type": "sentence" },
          ]
        }
      ]
    },
    "answerKey": "A"
  }
  ...

Every answer choice gets the same support. Every answer choice gets two supports, i.e., the two
facts we get from OpenBookQA.
"""

import typing
import json
from .. import utilities
from allennlp.common.util import JsonDict
import random
import logging
from tqdm import tqdm

def retrieve(questions_file: str, corpus: utilities.Corpus, sentence_count: int = 100) -> typing.Generator[JsonDict, None, None]:
    corpus_name = corpus.short_name()
    logging.info("Loading corpus %s", corpus_name)
    corpus = list(tqdm(corpus.unique_lines()))
    logging.info("Done loading corpus %s", corpus_name)

    for question in utilities.json_from_file(questions_file):
        for choice in question["question"]["choices"]:
            retrieved_sentences = set(random.sample(corpus, sentence_count))
            choice["support"] = [
                {
                    "text": s,
                    "type": "sentence"
                } for s in retrieved_sentences
            ]
        yield question

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Perform retrieval using the gold facts in the source data')
    parser.add_argument(
        '--questions', '-q',
        help='Filename of the question set to read')
    parser.add_argument(
        '--corpus', '-c',
        help="This parameter is ignored. The gold facts don't need a corpus.")
    parser.add_argument(
        '--output', '-o',
        help='Output results to this file. Default is stdout.',
        default=None)
    parser.add_argument(
        '--sentences', '-s',
        help='Number of random sentences to retrieve.',
        type=int,
        default=10)
    args = parser.parse_args()

    with utilities.file_or_stdout(args.output) as output:
        for q_with_text in retrieve(args.questions, utilities.Corpus(args.corpus), args.sentences):
            output.write(json.dumps(q_with_text))
            output.write("\n")

if __name__ == "__main__":
    main()
