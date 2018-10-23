#!/usr/bin/python3
from typing import *
import json

from allennlp.commands.elmo import ElmoEmbedder

import utilities
from allennlp.common.util import JsonDict
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import logging
import numpy as np
import sklearn.metrics.pairwise
import os
import gzip
from collections import Counter
from math import log
import Stemmer

_DEFAULT_CORPUS_STATISTICS_LOCATION = "models/vector-corpus-statistics.tsv.gz"

def _ensure_folder(folder: str):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

def _empty_folder(folder: str):
    for root, dirs, files in os.walk(folder):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            os.rmdir(os.path.join(root, d))

def retrieve(
    questions_file: str,
    corpus: utilities.Corpus
) -> Generator[JsonDict, None, None]:
    # load corpus statistics
    stem = Stemmer.Stemmer('english').stemWord
    statistics_sentence_count, df_tokens = \
        load_corpus_statistics(_DEFAULT_CORPUS_STATISTICS_LOCATION, stem)
    def idf(token: str) -> float:
        count = df_tokens.get(stem(token), 0)
        return log(1 + (statistics_sentence_count / (count + 1)))

    # make vectors out of the corpus
    elmo = ElmoEmbedder(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    )
    # Elmo likes to be warmed up.
    elmo.embed_sentences(map(word_tokenize, [
        "Sometimes I wave to people I don't know.",
        "It's very dangerous to wave to someone you don't know, because what if they don't have a hand?",
        "They'll think you're cocky.",
        "Look what I got...",
        "This thing is useful.",
        "I'm gonna go pick something up."
    ]))
    def postprocess_elmo(t: np.ndarray) -> np.ndarray:
        return t[2,:,:]
    @utilities.memoize(exceptions = [elmo])
    def embed_sentences(sentences: List[str]) -> Tuple[List[str], List[np.ndarray]]:
        tokenized = list(map(word_tokenize, sentences))
        embedded = list(map(
            postprocess_elmo,
            elmo.embed_sentences(tqdm(tokenized, desc="Embedding"))))
        return tokenized, embedded

    corpus = list(tqdm(corpus.unique_lines(), desc="Reading corpus"))
    tokenized_corpus, embedded_corpus = embed_sentences(corpus)
    float_type = embedded_corpus[0].dtype

    # make vectors out of the questions
    def question_text_from_question(question: JsonDict) -> str:
        return "%s %s" % (
            question["question"]["stem"],
            " ".join((choice["text"] for choice in question["question"]["choices"]))
        )
    questions = list(utilities.json_from_file(questions_file))
    questions_text = list(map(question_text_from_question, questions))
    tokenized_questions_text, embedded_questions_text = embed_sentences(questions_text)

    # process questions
    logging.info("Processing questions")
    MATRIXLOGS_DIR = "/tmp/matrixlogs-with-weights"
    _ensure_folder(MATRIXLOGS_DIR)
    _empty_folder(MATRIXLOGS_DIR)
    def process_question(question_index: int) -> JsonDict:
        question = questions[question_index]
        question_text = questions_text[question_index]
        tokenized_question_text = tokenized_questions_text[question_index]
        embedded_question_text = embedded_questions_text[question_index]
        question_text_weights = np.fromiter((idf(token) for token in tokenized_question_text), float_type)

        with gzip.open(os.path.join(MATRIXLOGS_DIR, question["id"] + ".txt.gz"), "wt") as matrixlog:
            matrixlog.write(f"QID\t{question['id']}\n")
            matrixlog.write(f"Q\t{question_text}\n")

            scored_sentences = []
            for sentence_index in range(len(corpus)):
                sentence = corpus[sentence_index]
                tokenized_sentence = tokenized_corpus[sentence_index]
                embedded_sentence = embedded_corpus[sentence_index]
                sentence_weights = np.fromiter((idf(token) for token in tokenized_sentence), float_type)

                matrix_similarity = 1 - sklearn.metrics.pairwise.cosine_distances(embedded_question_text, embedded_sentence)
                matrix_weights = np.tensordot(question_text_weights, sentence_weights, 0)
                weighted_similarity = matrix_similarity * matrix_weights
                score = weighted_similarity.sum() / matrix_weights.sum()

                matrixlog.write(f"\nS\t{sentence}\n")
                matrixlog.write("\t\t")
                matrixlog.write("\t".join(tokenized_sentence))
                matrixlog.write("\n")
                matrixlog.write("\t\t")
                matrixlog.write("\t".join(map(str, sentence_weights)))
                matrixlog.write("\n")
                for row, token in enumerate(tokenized_question_text):
                    matrixlog.write(token)
                    matrixlog.write("\t")
                    matrixlog.write(str(question_text_weights[row]))
                    matrixlog.write("\t")
                    matrixlog.write("\t".join((str(x) for x in weighted_similarity[row])))
                    matrixlog.write("\n")

                scored_sentences.append((sentence, score))

            scored_sentences.sort(key=lambda x: -x[1])
            support = [
                {
                    "text": sentence,
                    "type": "sentence",
                    "score": score
                } for sentence, score in scored_sentences
            ]

            for choice in question["question"]["choices"]:
                choice["support"] = support
            return question

    yield from map(process_question, range(len(questions)))

def load_corpus_statistics(
    filename: str,
    stemmer = lambda x: x
) -> Tuple[int, Counter]:
    sentence_count = None
    df_tokens = Counter()
    for line in utilities.text_from_file(filename, strip_lines=False):
        if sentence_count is None:
            line = line.strip()
            sentence_count = int(line)
        else:
            token, count = line.split("\t")
            count = int(count)

            token = stemmer(token)
            df_tokens[token] += count

    return sentence_count, df_tokens

def learn_corpus_statistics(corpus: utilities.Corpus) -> Tuple[int, Counter]:
    # We don't stem in this. We stem when we load this file. That way we can run with different
    # stemmers without having to recreate the statistics. Same goes for stopwords.

    df_tokens = Counter()
    def term_counter_from_sentences(sentences: Iterable[str]) -> Tuple[Counter, int]:
        counter = Counter()
        sentence_count = 0
        for sentence in sentences:
            for token in set(word_tokenize(sentence.lower())):
                counter[token] += 1
            sentence_count += 1
        return counter, sentence_count

    counters = utilities.mp_map(
        term_counter_from_sentences,
        utilities.grouped_iterator(
            tqdm(corpus.unique_lines(), desc="Loading corpus"),
            10000))
    total_sentence_count = 0
    for counter, sentence_count in tqdm(counters, desc="Counting tokens"):
        df_tokens += counter
        total_sentence_count += sentence_count

    return total_sentence_count, df_tokens

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Perform retrieval using the gold facts in the source data')
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="run retrieval")
    run_parser.add_argument(
        '--questions', '-q',
        help='Filename of the question set to read')
    run_parser.add_argument(
        '--corpus', '-c',
        help="The corpus to retrieve from")
    run_parser.add_argument(
        '--output', '-o',
        help='Output results to this file. Default is stdout.',
        default=None)

    train_parser = subparsers.add_parser("train", help="gather word statistics for the retrieval module")
    train_parser.add_argument(
        '--corpus', '-c',
        help="The corpus to extract the statistics from")
    train_parser.add_argument(
        '--output', '-o',
        help='Output results to this file. Default is stdout.',
        default=None)

    args = parser.parse_args()

    with utilities.file_or_stdout(args.output) as output:
        if args.command == "run":
            for q_with_text in retrieve(args.questions, utilities.Corpus(args.corpus)):
                output.write(json.dumps(q_with_text))
                output.write("\n")
        elif args.command == "train":
            raise NotImplementedError()
        else:
            raise ValueError

if __name__ == "__main__":
    main()
