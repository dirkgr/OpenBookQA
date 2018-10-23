#!/usr/bin/python3
from typing import *
import json
import utilities
from allennlp.common.util import JsonDict
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import logging
from math import log
import Stemmer
from nltk.corpus import stopwords as nltk_stopwords
from . import tushar
import heapq
from functools import lru_cache
import multiprocessing as mp
import ctypes

# Determined on the dev set that these numbers give us the correct results for 90% of the questions.
F_CUTOFF = 34
K_CUTOFF = 88

def retrieve(
    questions_file: str,
    corpus: utilities.Corpus
) -> Generator[JsonDict, None, None]:
    stem: Callable[[str], str] = Stemmer.Stemmer('english').stemWord
    stopwords = set(nltk_stopwords.words('english'))
    stopwords |= {',', '?', '.', stem("it's")}
    @lru_cache(maxsize=None)
    def stemmed_token_set(sentence: str) -> FrozenSet[str]:
        return frozenset((
            stem(token)
            for token in word_tokenize(sentence.lower())
            if token not in stopwords
        ))

    logging.info("Loading word statistics")
    @utilities.memoize([stem], version=1)
    def load_word_statistics():
        return tushar.load_stats(tushar.DEFAULT_WORD_STATS_LOCATION, None, stem)
    df_tokens, doc_count = load_word_statistics()

    logging.info("Indexing corpus")
    index: Dict[str, Set[str]] = {}
    for sentence in tqdm(corpus.unique_lines(), desc="Reading corpus"):
        tokens = stemmed_token_set(sentence)
        for token in tokens:
            # add to index
            s = index.setdefault(token, set())
            s.add(sentence)

            # add to word stats
            c = df_tokens.get(token, 0)
            df_tokens[token] = c + 1
        doc_count += 1

    logging.info("Precomputing token scores")
    token_scores = {
        token: log(1 + (doc_count / df))
        for token, df in df_tokens.items()
    }
    @lru_cache(maxsize=None)
    def sum_of_token_scores(tokens: Iterable[str]) -> float:
        return sum((token_scores.get(token, 0.0) for token in tokens))

    logging.info("Processing questions")

    def prefix_follows_postfix_score_with_sets(prefix: FrozenSet[str], postfix: FrozenSet[str]) -> float:
        best_possible_score = min(sum_of_token_scores(postfix), sum_of_token_scores(prefix))
        if best_possible_score > 0:
            return sum_of_token_scores(prefix & postfix) / best_possible_score
        else:
            return 0.0

    def prefix_follows_postfix_score(prefix: str, postfix: str) -> float:
        prefix = stemmed_token_set(prefix)
        postfix = stemmed_token_set(postfix)
        return prefix_follows_postfix_score_with_sets(prefix, postfix)

    class Chain(object):
        OCCAMS_RAZOR_FACTOR = 1.0
        # Naively, adding a sentence that completely overlaps with the sentences
        # we already have has no penalty, so we would always add more of those
        # sentences. To counteract this, we multiply the score by this factor
        # every time we add a sentence.

        def __init__(self, sentence: str):
            self.sentences = [sentence]
            self.last_element_token_set = stemmed_token_set(sentence)
            self.chain_token_set = self.last_element_token_set
            self.scores = [1.0]

        def score(self) -> float:
            return self.scores[-1]

        def add_sentence(self, sentence: str):
            new_chain = Chain("")
            new_chain.sentences = self.sentences + [sentence]
            new_chain.last_element_token_set = stemmed_token_set(sentence)

            prefix = self.last_element_token_set
            if len(self.sentences) >= 2:
                prefix = prefix - stemmed_token_set(self.sentences[-2])
            new_chain.scores = self.scores + [
                self.score() *
                self.OCCAMS_RAZOR_FACTOR *
                prefix_follows_postfix_score_with_sets(
                    prefix,
                    new_chain.last_element_token_set)]

            new_chain.chain_token_set = self.chain_token_set | new_chain.last_element_token_set
            return new_chain

        def __str__(self):
            return f"Chain(len={len(self.sentences)}, score={self.score()})"

        def __cmp__(self, other):
            if self.score() < other.score():
                return 1
            if self.score() > other.score():
                return -1
            return 0

        def __gt__(self, other):
            return self.__cmp__(other) > 0

        def __lt__(self, other):
            return self.__cmp__(other) < 0

        def __ge__(self, other):
            return self.__cmp__(other) >= 0

        def __le__(self, other):
            return self.__cmp__(other) <= 0

        def __eq__(self, other):
            return self.__cmp__(other) == 0

        def __ne__(self, other):
            return self.__cmp__(other) != 0

    chain_logger = logging.getLogger("retrieval.chaining")
    tqdm_position = None
    worker_counter = mp.Value(ctypes.c_int)
    def process_question(q: JsonDict) -> JsonDict:
        unexplored_chains = [Chain(q["question"]["stem"])]
        done_chains_queue = []
        done_chains = []

        nonlocal tqdm_position
        if tqdm_position is None:
            with worker_counter.get_lock():
                tqdm_position = worker_counter.value + 1
                worker_counter.value = tqdm_position
        with tqdm(desc=f"Question {q['id']:>6}", position=tqdm_position) as t:
            t.total = len(unexplored_chains)
            while len(unexplored_chains) > 0:
                chain = heapq.heappop(unexplored_chains)
                t.update()

                # transfer chains from done_chains_queue to done_chains
                while len(done_chains_queue) > 0 and done_chains_queue[0].score() > chain.score():
                    done_chains.append(heapq.heappop(done_chains_queue))

                # check whether this chain ends in a choice
                if len(chain.sentences) > 1:    # We don't want to jump directly from question to answer.
                    for choice in q["question"]["choices"]:
                        chain_with_choice = chain.add_sentence(choice["text"])
                        if chain_with_choice.score() > 0:
                            heapq.heappush(done_chains_queue, chain_with_choice)

                # add further chains to the queue
                if len(chain.sentences) < 3:
                    unexplored_chains_before = len(unexplored_chains)
                    sentences_already_added = set()
                    for token_from_chain in chain.last_element_token_set:
                        for sentence in index.get(token_from_chain, set()):
                            if sentence not in chain.sentences and sentence not in sentences_already_added:
                                new_chain = chain.add_sentence(sentence)
                                if new_chain.score() >= 0.08:
                                    heapq.heappush(unexplored_chains, new_chain)
                                    sentences_already_added.add(sentence)
                    t.total = t.total + (len(unexplored_chains) - unexplored_chains_before)

        # transfer the chains from done_chains_queue to done_chains
        while len(done_chains_queue) > 0:
            done_chains.append(heapq.heappop(done_chains_queue))

        # print all the chains
        qid = q['id']
        log_message = [f"{qid}: {len(done_chains)} chains from {q['question']['stem']}"]
        for i, chain in enumerate(done_chains):
            sentences = [f"{s} ({score:.3f})" for s, score in zip(chain.sentences, chain.scores)]
            sentences = " -> ".join(sentences[1:])
            log_message.append(f"{' ' * len(qid)}  {chain.score():.3f}\t{sentences}")
        chain_logger.info("\n".join(log_message))

        # sum up all the chains
        sentence_to_scores = {}
        for chain in done_chains:
            for sentence in chain.sentences[1:-1]:
                score_rounded = sentence_to_scores.get(sentence, 0.0)
                sentence_to_scores[sentence] = score_rounded + chain.score()
        scored_sentences = [(score, sentence) for sentence, score in sentence_to_scores.items()]
        scored_sentences.sort(key=lambda x: (-x[0], len(x[1]), x[1]))

        # build support
        support = [
            {
                "text": sentence,
                "type": "sentence",
                "score": score
            } for score, sentence in scored_sentences
        ]
        for choice in q["question"]["choices"]:
            choice["support"] = support

        return q

    #yield from map(process_question, utilities.json_from_file(questions_file))
    yield from utilities.mp_map(process_question, utilities.json_from_file(questions_file))

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
