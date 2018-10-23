#!/usr/bin/python3

import scoring
import importlib
import logging
import utilities
from allennlp.common.util import JsonDict


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    import argparse

    parser = argparse.ArgumentParser(description='Evaluate a retrieval/QA combination.')
    parser.add_argument(
        '--questions', '-q',
        help='Filename of the question set to try')
    parser.add_argument(
        '--corpus', '-c',
        help='Filename of the corpus to try')
    parser.add_argument(
        '--retrieval', '-r',
        help='Name of the retrieval system to try',
        type=str,
        default='lucene')
    parser.add_argument(
        '--no-cutoff',
        help="Don't apply the cutoff to the retrieval results",
        action='store_true')
    parser.add_argument(
        '--qa',
        help='Name of the QA system to try',
        default="chaining")
    args = parser.parse_args()

    # configure loggers
    logging.getLogger("retrieval").addHandler(logging.FileHandler("retrieval.log", mode="w"))
    logging.getLogger("retrieval").propagate = False
    logging.getLogger("qa").addHandler(logging.FileHandler("qa.log", mode="w"))
    logging.getLogger("qa").propagate = False

    retrieval_module = importlib.import_module("retrieval." + args.retrieval)
    qa_module = importlib.import_module("qa." + args.qa)

    corpus = utilities.Corpus(args.corpus)
    questions_with_text = retrieval_module.retrieve(args.questions, corpus)

    f_cutoff = retrieval_module.F_CUTOFF
    k_cutoff = retrieval_module.K_CUTOFF
    if not args.no_cutoff and f_cutoff is not None and k_cutoff is not None:
        f_corpus = utilities.Corpus("corpora/f.txt.gz")
        f_sentences = {sentence.lower() for sentence in f_corpus.unique_lines()}

        def apply_f_and_k_cutoff(q: JsonDict) -> JsonDict:
            choices = q['question']['choices']
            for choice in choices:
                support = choice['support']
                f_sentences_left = f_cutoff
                k_sentences_left = k_cutoff
                new_support = []
                for sentence in support:
                    if sentence['text'].lower() in f_sentences:
                        f_sentences_left -= 1
                        c = f_sentences_left
                    else:
                        k_sentences_left -= 1
                        c = k_sentences_left
                    if c >= 0:
                        new_support.append(sentence)
                choice['support'] = new_support
            return q
        questions_with_cutoff_applied = map(apply_f_and_k_cutoff, questions_with_text)
    else:
        questions_with_cutoff_applied = questions_with_text

    questions_with_predictions = qa_module.qa(questions_with_cutoff_applied)
    scores = scoring.scores_from_questions_with_predictions(questions_with_predictions)
    print("\t".join(map(str, list(scores))))

if __name__ == "__main__":
    main()
