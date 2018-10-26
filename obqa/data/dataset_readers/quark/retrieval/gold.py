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

F_CUTOFF = 1
K_CUTOFF = 1

def retrieve(questions_file: str, corpus: utilities.Corpus, distractors: int = 0) -> typing.Generator[JsonDict, None, None]:
    if distractors > 0:
        corpus_name = corpus.short_name()
        logging.info("Loading corpus %s", corpus_name)
        corpus = set(corpus.unique_lines())
        logging.info("Done loading corpus %s", corpus_name)
    else:
        corpus = set()

    for question in utilities.json_from_file(questions_file):
        (identifier, question, answer_key, fact1, fact2) = \
            utilities.parse_dict(question, "id", "question", "answerKey", "fact1", "fact2")
        (question_stem, question_choices) = \
            utilities.parse_dict(question, "stem", "choices")

        fact_sentences = {fact1, fact2}
        corpus_without_facts = list(corpus - fact_sentences)
        choices = []
        for choice in question_choices:
            distractor_sentences = set(random.sample(corpus_without_facts, distractors))
            sentences_for_this_choice = fact_sentences | distractor_sentences
            support = [
                {
                    "text": s,
                    "type": "sentence",
                    "score": 1.0
                } for s in sentences_for_this_choice
            ]
            choices.append({
                "text": choice["text"],
                "label": choice["label"],
                "support": support
            })
        result = {
            "id": identifier,
            "question": {
                "stem": question_stem,
                "choices": choices
            },
            "answerKey": answer_key
        }
        yield result

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
        '--distractors', '-d',
        help='Number of random distractor sentences to add to the known facts.',
        type=int,
        default=0)
    args = parser.parse_args()

    with utilities.file_or_stdout(args.output) as output:
        for q_with_text in retrieve(args.questions, utilities.Corpus(args.corpus), args.distractors):
            output.write(json.dumps(q_with_text))
            output.write("\n")

if __name__ == "__main__":
    main()
