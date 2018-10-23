#!/usr/bin/python3

import utilities
import json
import typing
from allennlp.common.util import JsonDict
import os
import elasticsearch.helpers
from elasticsearch import Elasticsearch
import re

class EsHit:
    def __init__(self, score: float, position: int, text: str, type: str):
        """
        Basic information about an ElasticSearch Hit
        :param score: score returned by the query
        :param position: position in the retrieved results (before any filters are applied)
        :param text: retrieved sentence
        :param type: type of the hit in the index (by default, only documents of type "sentence"
        will be retrieved from the index)
        """
        self.score = score
        self.position = position
        self.text = text
        self.type = type


class EsSearch:
    def __init__(
        self,
        es,
        index: str,
        max_question_length: int = 1000,
        max_hits_retrieved: int = 500,
        max_hit_length: int = 300,
        max_hits_per_choice: int = 100
    ):
        """
        Class to search over the text corpus using ElasticSearch
        :param es_client: Location of the ElasticSearch service
        :param indices: Comma-separated list of indices to search over
        :param max_question_length: Max number of characters used from the question for the
        query (for efficiency)
        :param max_hits_retrieved: Max number of hits requested from ElasticSearch
        :param max_hit_length: Max number of characters for accepted hits
        :param max_hits_per_choice: Max number of hits returned per answer choice
        """
        self._es = es
        self._indices = index
        self._max_question_length = max_question_length
        self._max_hits_retrieved = max_hits_retrieved
        self._max_hit_length = max_hit_length
        self._max_hits_per_choice = max_hits_per_choice
        # Regex for negation words used to ignore Lucene results with negation
        self._negation_regexes = [re.compile(r) for r in ["not\\s", "n't\\s", "except\\s"]]

    def get_hits_for_question(self, question: str, choices: typing.List[str]) -> typing.Dict[str, typing.List[EsHit]]:
        """
        :param question: Question text
        :param choices: List of answer choices
        :return: Dictionary of hits per answer choice
        """
        choice_hits = dict()
        for choice in choices:
            choice_hits[choice] = self.filter_hits(self.get_hits_for_choice(question, choice))
        return choice_hits

    # Constructs an ElasticSearch query from the input question and choice
    # Uses the last self._max_question_length characters from the question and requires that the
    # text matches the answer choice and the hit type is a "sentence"
    def construct_qa_query(self, question, choice):
        return {"from": 0, "size": self._max_hits_retrieved,
                "query": {
                    "bool": {
                        "must": [
                            {"match": {
                                "text": question[-self._max_question_length:] + " " + choice
                            }}
                        ],
                        "filter": [
                            {"match": {"text": choice}},
                            {"type": {"value": "sentence"}}
                        ]
                    }
                }}

    # Retrieve unfiltered hits for input question and answer choice
    def get_hits_for_choice(self, question, choice):
        res = self._es.search(index=self._indices, body=self.construct_qa_query(question, choice))
        hits = []
        for idx, es_hit in enumerate(res['hits']['hits']):
            es_hit = EsHit(score=es_hit["_score"],
                           position=idx,
                           text=es_hit["_source"]["text"],
                           type=es_hit["_type"])
            hits.append(es_hit)
        return hits

    # Remove hits that contain negation, are too long, are duplicates, are noisy.
    def filter_hits(self, hits: typing.List[EsHit]) -> typing.List[EsHit]:
        filtered_hits = []
        selected_hit_keys = set()
        for hit in hits:
            hit_sentence = hit.text
            hit_sentence = hit_sentence.strip().replace("\n", " ")
            if len(hit_sentence) > self._max_hit_length:
                continue
            for negation_regex in self._negation_regexes:
                if negation_regex.search(hit_sentence):
                    # ignore hit
                    continue
            if self.get_key(hit_sentence) in selected_hit_keys:
                continue
            if not self.is_clean_sentence(hit_sentence):
                continue
            filtered_hits.append(hit)
            selected_hit_keys.add(self.get_key(hit_sentence))
        return filtered_hits[:self._max_hits_per_choice]

    # Check if the sentence is not noisy
    def is_clean_sentence(self, s):
        # must only contain expected characters, should be single-sentence and only uses hyphens
        # for hyphenated words
        return (re.match("^[a-zA-Z0-9][a-zA-Z0-9;:,\(\)%\-\&\.'\"\s]+\.?$", s) and
                not re.match(".*\D\. \D.*", s) and
                not re.match(".*\s\-\s.*", s))

    # Create a de-duplication key for a HIT
    def get_key(self, hit):
        # Ignore characters that do not effect semantics of a sentence and URLs
        return re.sub('[^0-9a-zA-Z\.\-^;&%]+', '', re.sub('http[^ ]+', '', hit)).strip().rstrip(".")


def retrieve(questions_file: str, corpus: utilities.Corpus) -> typing.Generator[JsonDict, None, None]:
    # index the corpus if necessary
    index_name = corpus.short_name()
    es = Elasticsearch(hosts=[{"host": "localhost"}], retries = 3, timeout = 60)
    if es.indices.exists(index_name):
        # We set the read-only flag when the index has been created successfully. If it's not set,
        # we recreate the index.
        read_only = es.indices.get_settings(index=index_name, name="index.blocks.read_only")
        for key in [index_name, "settings", "index", "blocks", "read_only"]:
            try:
                read_only = read_only[key]
            except KeyError:
                read_only = False
                break
        read_only = read_only == "true"
        if not read_only:
            es.indices.delete(index_name)
    if not es.indices.exists(index_name):
        mapping = '''
        {
          "mappings": {
            "sentence": {
              "dynamic": "false",
              "properties": {
                "docId": {
                  "type": "keyword"
                },
                "text": {
                  "analyzer": "snowball",
                  "type": "text",
                  "fields": {
                    "raw": {
                      "type": "keyword"
                    }
                  }
                },
                "tags": {
                  "type": "keyword"
                }
              }
            }
          }
        }'''
        doc_type = "sentence"

        def make_documents():
            doc_id = 0
            for line in corpus.unique_lines():
                doc = {
                    '_op_type': 'create',
                    '_index': index_name,
                    '_type': doc_type,
                    '_id': doc_id,
                    '_source': {'text': line.strip()}
                }
                doc_id += 1
                yield doc

        es.indices.create(index=index_name, ignore=400, body=mapping)
        elasticsearch.helpers.bulk(es, make_documents())
        es.indices.put_settings(index=index_name, body={"index.blocks.read_only":True})

    # add sentences from lucene to each question
    def filter_hits_across_choices(
        hits_per_choice: typing.Dict[str, typing.List[EsHit]],
        top_k: int
    ):
        """
        Filter the hits from all answer choices(in-place) to the top_k hits based on the hit score
        """
        # collect ir scores
        ir_scores = [hit.score for hits in hits_per_choice.values() for hit in hits]
        # if more than top_k hits were found
        if len(ir_scores) > top_k:
            # find the score of the top_kth hit
            min_score = sorted(ir_scores, reverse=True)[top_k - 1]
            # filter hits below this score
            for choice, hits in hits_per_choice.items():
                hits[:] = [hit for hit in hits if hit.score >= min_score]

    MAX_HITS = 8
    es_search = EsSearch(es, max_hits_per_choice=MAX_HITS, max_hits_retrieved=100, index=index_name)
    for question in utilities.json_from_file(questions_file):
        question_text = question["question"]["stem"]
        choices = [choice["text"] for choice in question["question"]["choices"]]
        hits_per_choice = es_search.get_hits_for_question(question_text, choices)
        filter_hits_across_choices(hits_per_choice, MAX_HITS)
        for choice in question["question"]["choices"]:
            choice_text = choice["text"]
            hits = hits_per_choice[choice_text]

            support = [
                {
                    "text": hit.text,
                    "type": hit.type,
                    "ir_pos": hit.position,
                    "ir_score": hit.score
                } for hit in hits
            ]
            choice["support"] = support
        yield question

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
        for q_with_text in retrieve(args.questions, args.corpus):
            output.write(json.dumps(q_with_text))
            output.write("\n")

if __name__ == "__main__":
    main()
