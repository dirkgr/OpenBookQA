#!/usr/bin/python3

from .. import utilities
from allennlp.common.util import JsonDict
import typing
import json

from ....retrieval.knowledge.readers.simple_reader_arc_qa_question_choice import SimpleReaderARC_Question_Choice
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm

try:
    from spacy.lang.en.stop_words import STOP_WORDS
except:
    from spacy.en import STOP_WORDS

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower())]

def retrieve(questions_file: str, corpus: utilities.Corpus) -> typing.Generator[JsonDict, None, None]:
    paragraphs = [x.lower() for x in tqdm(corpus.unique_lines())]
    # TODO: strip punctuation?

    data_utils = SimpleReaderARC_Question_Choice("question")
    data_items, queries_index = data_utils.get_reader_items_field_text_index(questions_file)
    id_field = data_utils.id_field
    field_names = data_utils.field_names
    items_to_export = []
    queries_items_meta = []
    queries_items_fields = []

    for lid, item in enumerate(data_items):
        items_to_export.append({id_field: item[id_field], "ext_fact_global_ids": []})
        queries_items_fields.append([item[field_names[0]], item[field_names[1]]])
        queries_items_meta.append({"lid": lid, "obj": item})

    tfidf = TfidfVectorizer(
        strip_accents="unicode",
        stop_words=STOP_WORDS,
        decode_error='replace',
        tokenizer=LemmaTokenizer(),
        dtype=np.float32)
    queries_index_transformed = tfidf.fit_transform(queries_index)
    # Why are we learning corpus statistics on the questions instead of the knowledge?

    para_features = utilities.tfidf_parallel_transform(tfidf, paragraphs)
    similarity_matrix = pairwise_distances(queries_index_transformed, para_features, "cosine")

    max_facts_per_choice = 100 # None means no limit
    combine_feat_scores = "mul"

    comb_funcs = {
        "mul": np.multiply,
        "add": np.add,
        "2x+y": lambda x, y: 2*x+y
    }

    def combine_similarities(scores_per_feat, top:typing.Optional[int]=10, combine_feat_scores="mul"):
        """
        Get similarities based on multiple independent queries that are then combined using combine_feat_scores
        :param query_feats: Multiple vectorized text queries
        :param para_features: Multiple vectorized text paragraphs that will be scored against the queries
        :param top: Top N facts to keep
        :param combine_feat_scores: The way for combining the multiple scores
        :return: Ranked fact ids with scores List[tuple(id, weight)]
        """
        #scores_per_feat = [pairwise_distances(q_feat, para_features, "cosine").ravel() for q_feat in query_feats]  # this is distance - low is better!!!
        comb_func = comb_funcs[combine_feat_scores]

        smooting_val = 0.000001
        max_val = pow((1 + smooting_val), 2)
        dists = scores_per_feat[0] + smooting_val
        if len(scores_per_feat) > 1:
            for i in range(1, len(scores_per_feat)):
                dists = comb_func(scores_per_feat[i] + smooting_val, dists)

        max_val = max(np.max(dists), 1)
        sorted_ix = np.argsort(dists)
        scaled = (max_val - dists) / max_val
        sorted_ix = sorted_ix[:top]
        return list(zip(sorted_ix, scaled[sorted_ix]))

    res_dists = []
    for i, item_feats in enumerate(queries_items_fields):
        feat_similarities = similarity_matrix[item_feats]

        dists = combine_similarities(
            feat_similarities,
            max_facts_per_choice,
            combine_feat_scores=combine_feat_scores)
        res_dists.append(dists)

    lid_to_proc_id = {item["lid"]: i for i, item in enumerate(queries_items_meta)}
    for i, item in enumerate(items_to_export):
        if i in lid_to_proc_id:
            item["ext_fact_global_ids"] = res_dists[lid_to_proc_id[i]]

    # End of Todor's code. Start of Dirk's adaptation to the quark question format.

    id_to_results = {
        item["id"]: item["ext_fact_global_ids"]
        for item in items_to_export
    }
    for question in utilities.json_from_file(questions_file):
        choices = question["question"]["choices"]
        for choice_index, choice in enumerate(choices):
            result_id = "%s__ch_%d" % (question["id"], choice_index)

            support = [
                {
                    "text": paragraphs[para_index],
                    "type": "sentence",
                    "score": np.asscalar(score)
                } for para_index, score in id_to_results[result_id]
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
        for q_with_text in retrieve(args.questions, utilities.Corpus(args.corpus)):
            output.write(json.dumps(q_with_text))
            output.write("\n")

if __name__ == "__main__":
    main()
