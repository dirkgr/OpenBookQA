#!/usr/bin/python3

from .. import utilities
from allennlp.common.util import JsonDict
import typing
import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm

try:
    from spacy.lang.en.stop_words import STOP_WORDS as stop_words
except:
    from spacy.en import STOP_WORDS as stop_words

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower())]

def retrieve(
    questions_file: str,
    corpus: utilities.Corpus,
    results_per_question_choice:int = 100
) -> typing.Generator[JsonDict, None, None]:
    # train TF-IDF on the questions # TODO: Why on the questions?
    logging.info("Learning Tfidf parameters")
    tfidf = TfidfVectorizer(
        strip_accents="unicode",
        stop_words=stop_words,
        decode_error='replace',
        tokenizer=LemmaTokenizer())
    def documents_from_questions() -> typing.Generator[str, None, None]:
        for question in utilities.json_from_file(questions_file):
            stem = question["question"]["stem"]
            yield from (
                "%s %s" % (stem, choice["text"])
                for choice in question["question"]["choices"]
            )
    tfidf.fit(tqdm(documents_from_questions()))

    logging.info("Reading corpus")
    corpus = [x.lower() for x in tqdm(corpus.unique_lines())]
    # TODO: strip punctuation?

    logging.info("Featurizing corpus")
    corpus_features = utilities.tfidf_parallel_transform(tfidf, corpus)

    # add results to the question
    logging.info("Processing questions")

    def process_question(question):
        stem = question["question"]["stem"]
        choices = question["question"]["choices"]

        choices_text = ["%s %s" % (stem, choice["text"]) for choice in choices]
        choices_features = tfidf.transform(choices_text)
        for choice, choice_features in zip(choices, choices_features):
            distances = pairwise_distances(choice_features, corpus_features, "cosine")
            distances = np.squeeze(distances, axis=0)
            support_indices = np.argsort(distances)[:results_per_question_choice]
            support_texts = [corpus[support_index] for support_index in support_indices]

            support = [
                {
                    "text": support_text,
                    "type": "sentence"
                } for support_text in support_texts
            ]
            choice["support"] = support

        return question

    yield from utilities.mp_map(process_question, utilities.json_from_file(questions_file))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Perform retrieval using Lucene')
    parser.add_argument(
        '--questions', '-q',
        help='Filename of the question set to read')
    parser.add_argument(
        '--corpus', '-c',
        help='Filename of the corpus to read')
    parser.add_argument(
        '--output', '-o',
        help='Output results to this file. Default is stdout.',
        default=None)
    args = parser.parse_args()

    with utilities.file_or_stdout(args.output) as output:
        for q_with_text in retrieve(args.questions, utilities.Corpus(args.corpus)):
            output.write(json.dumps(q_with_text))
            output.write("\n")

if __name__ == "__main__":
    main()
