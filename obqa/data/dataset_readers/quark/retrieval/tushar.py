#!/usr/bin/python3

import json
from .. import utilities
from allennlp.common.util import JsonDict
from collections import Counter
from math import log
from nltk.corpus import stopwords as nltk_stopwords
import Stemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import logging
from typing import *
from allennlp.common.file_utils import cached_path

DEFAULT_WORD_STATS_LOCATION = "http://pip-package.dev.ai2/models/quark/tushar-word-stats.tsv.gz"

# Determined on the dev set that these numbers give us the correct results for 90% of the questions.
F_CUTOFF = 63
K_CUTOFF = 72

def retrieve(
    questions_file: str,
    corpus: utilities.Corpus,
    word_stats_file: str = DEFAULT_WORD_STATS_LOCATION
) -> Generator[JsonDict, None, None]:
    logging.info("Determining vocab from questions")
    questions = list(utilities.json_from_file(questions_file))
    def question_text_from_question(question: JsonDict) -> str:
        return "%s %s" % (
            question["question"]["stem"],
            " ".join((choice["text"] for choice in question["question"]["choices"]))
        )

    min_score = 0.0
    stemmer = Stemmer.Stemmer('english')
    stem = stemmer.stemWord

    stopwords = set(nltk_stopwords.words('english'))
    stopwords |= {',', '?', '.'}

    question_vocab = set()
    for question_text in map(question_text_from_question, questions):
        question_vocab |= {
            stem(tok)
            for tok in word_tokenize(question_text.lower())
        }
    question_vocab -= stopwords

    # read corpus
    logging.info("Reading corpus and counting tokens")

    def get_tokens(sentence: str) -> List[str]:
        result = [
            stem(tok)
            for tok in word_tokenize(sentence.lower())
        ]
        return result

    def get_tokens_set(sentence: str) -> Set[str]:
        return set(get_tokens(sentence)) & question_vocab

    # add the statistics from our small corpus to the big-corpus statistics
    df_tokens, doc_count = load_stats(cached_path(word_stats_file), question_vocab, stem)
    corpus_as_list = []
    for sentence in tqdm(corpus.unique_lines()):
        sentence_toks = get_tokens_set(sentence)
        doc_count += 1
        for token in sentence_toks:
            df_tokens[token] += 1
        corpus_as_list.append((sentence, sentence_toks))
    corpus = corpus_as_list

    # process questions
    logging.info("Processing questions")
    def scored_tokens(qtoks: Set[str], ftoks: Set[str]) -> Generator[Tuple[str, float], None, None]:
        common_toks = qtoks.intersection(ftoks)
        for tok in common_toks:
            tf = 1
            idf = log(1 + (doc_count / df_tokens[tok]))
            yield (tok, tf * idf)

    def process_question(question: JsonDict) -> JsonDict:
        question_text = question_text_from_question(question)
        question_text_tokens = get_tokens_set(question_text)

        scored_sentences = []
        for sentence, sentence_toks in corpus:
            sentence_score = 0.0
            score_per_token = {}
            for scored_token, token_score in scored_tokens(question_text_tokens, sentence_toks):
                sentence_score += token_score
                score_per_token[scored_token] = token_score
            if sentence_score > min_score:
                scored_sentences.append((sentence, sentence_score, score_per_token))
        scored_sentences.sort(key=lambda x: -x[1])
        support = [
            {
                "text": sentence,
                "type": "sentence",
                "score": score,
                "debug_info": score_per_token
            } for sentence, score, score_per_token in scored_sentences
        ]

        for choice in question["question"]["choices"]:
            choice["support"] = support
        return question

    yield from utilities.map_in_chunks(process_question, 10, questions)

def train(corpus: utilities.Corpus) -> Counter:
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
        utilities.slices(
            10000,
            tqdm(corpus.unique_lines(), desc="Loading corpus")))
    total_sentence_count = 0
    for counter, sentence_count in tqdm(counters, desc="Counting tokens"):
        df_tokens += counter
        total_sentence_count += sentence_count
    df_tokens[""] = total_sentence_count

    return df_tokens

def load_stats(
    filename: str,
    vocab: Optional[Set[str]] = None,
    stemmer: Callable[[str], str] = lambda x: x
) -> Tuple[Counter, int]:
    df_tokens = Counter()
    sentence_count = 0
    for line in utilities.text_from_file(filename, strip_lines=False):
        token, count = line.split("\t")
        count = int(count)

        if len(token) <= 0:
            sentence_count = count
            continue

        token = stemmer(token)
        if vocab is not None and token not in vocab:
            continue

        df_tokens[token] += count

    if sentence_count <= 0:
        logging.warning("Did not find a valid sentence count in the corpus statistics")
    return df_tokens, sentence_count

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
    run_parser.add_argument(
        '--word-stats',
        help='File with word statistics',
        type=str,
        default=DEFAULT_WORD_STATS_LOCATION)

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
            stats = train(utilities.Corpus(args.corpus))
            for token, count in stats.items():
                output.write(f"{token}\t{count}\n")
        else:
            raise ValueError

if __name__ == "__main__":
    main()
