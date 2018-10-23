#!/usr/bin/python3

import importlib
import utilities
import logging
import typing

def main():
    logging.getLogger().setLevel(logging.DEBUG)

    import argparse

    parser = argparse.ArgumentParser(description='Find out how well the woverlap solver does with distractors')
    parser.add_argument(
        '--questions', '-q',
        help='Filename of the question set to try',
        required=True)
    parser.add_argument(
        '--corpus', '-c',
        help='Filename of the corpus to try',
        required=True)
    parser.add_argument(
        '--retrieval', '-r',
        help='Name of the retrieval system to try',
        type=str,
        default='lucene')

    args = parser.parse_args()

    # configure loggers
    logging.getLogger("retrieval").addHandler(logging.FileHandler("retrieval.log", mode="w"))
    logging.getLogger("retrieval").propagate = False

    corpus = utilities.Corpus(args.corpus)
    retrieval_module = importlib.import_module("retrieval." + args.retrieval)
    questions_with_text = retrieval_module.retrieve(args.questions, corpus)

    f_corpus = utilities.Corpus("corpora/f.txt.gz")
    f_sentences = {sentence.lower() for sentence in f_corpus.unique_lines()}

    rank_to_qids_fact1: typing.Dict[int, set] = {}
    # rank_to_quid_fact1[2] = {"q1", "q3"}
    # This means that question 1 and question 3 have fact1 at rank 2. This rank
    # is the rank in the list of facts from F.

    rank_to_qids_fact2: typing.Dict[int, set] = {}
    # rank_to_quid_fact2[4] = {"q2"}
    # This means that question 2 has fact2 at rank 4. This rank is in the list
    # of facts from K. K is all facts that are not in F.

    number_of_questions = 0
    number_of_questions_with_both_facts_retrieved = 0
    for question in questions_with_text:
        (qid, question, answer_key, fact1, fact2) = \
            utilities.parse_dict(question, "id", "question", "answerKey", "fact1", "fact2")
        (question_stem, question_choices) = \
            utilities.parse_dict(question, "stem", "choices")
        number_of_questions += 1

        #
        # print an analysis of the question
        #
        choices_text = " ".join((
            "%s) %s" % (choice["label"], choice["text"]) for choice in question_choices
        ))
        print(f"Question {qid}: {question_stem} {choices_text}")
        print("Fact 1: %s" % fact1)
        print("Fact 2: %s" % fact2)

        for choice in question_choices:
            # We're only evaluating the correct answer right now.
            if choice["label"] != answer_key:
                continue

            fact1_index = -1
            fact2_index = -1
            for index, support in enumerate(choice["support"]):
                if support["type"] != "sentence":
                    continue

                sentence = support["text"]

                marker = "     "
                if fact1 == sentence:
                    marker = "Fact1"
                    fact1_index = index
                if fact2 == sentence:
                    marker = "Fact2"
                    fact2_index = index

                scored_tokens = support.get("debug_info")
                if scored_tokens is None:
                    scored_tokens = f"{support['score']:.3f}"
                else:
                    scored_tokens = ", ".join((
                        f"{token}: {score:.3f}" for token, score in scored_tokens.items()
                    ))
                    scored_tokens = f"{support['score']:.3f} ({scored_tokens})"

                print(f"{index+1}\t{marker}\t{sentence}\t{scored_tokens}")

                if fact1_index >= 0 and fact2_index >= 0:
                    break

            if fact1_index < 0:
                print(f"\tFact1\t{fact1}\tNOT FOUND")
            if fact2_index < 0:
                print(f"\tFact2\t{fact2}\tNOT FOUND")
            if fact1_index >= 0 and fact2_index >= 0:
                number_of_questions_with_both_facts_retrieved += 1

            print()

        #
        # do some book-keeping to compute our metric
        #

        for choice in question_choices:
            # We're only evaluating the correct answer right now.
            if choice["label"] != answer_key:
                continue

            rank_in_f = 0
            rank_in_k = 0
            for support in choice["support"]:
                if support["type"] != "sentence":
                    continue

                sentence = support["text"]
                sentence_is_in_f = sentence.lower() in f_sentences
                if sentence_is_in_f:
                    rank_in_f += 1
                else:
                    rank_in_k += 1

                if fact1 == sentence:
                    assert sentence_is_in_f
                    qids = rank_to_qids_fact1.setdefault(rank_in_f, set())
                    qids.add(qid)
                if fact2 == sentence:
                    qids = rank_to_qids_fact2.setdefault(rank_in_k, set())
                    qids.add(qid)

    print(f"Got both facts for {100.0 * number_of_questions_with_both_facts_retrieved / number_of_questions:.3f}%")

    # calculate the final numbers for l1 and l2
    max_l1 = max(rank_to_qids_fact1.keys())
    max_l2 = max(rank_to_qids_fact2.keys())

    qids_where_rank_is_enough_for_fact1: typing.Dict[int, typing.Set[str]] = {
        0: set()
    }
    for rank in range(1, max_l1 + 1):
        qids_where_rank_is_enough_for_fact1[rank] = \
            qids_where_rank_is_enough_for_fact1[rank - 1].union(rank_to_qids_fact1.get(rank, set()))

    qids_where_rank_is_enough_for_fact2: typing.Dict[int, typing.Set[str]] = {
        0: set()
    }
    for rank in range(1, max_l2 + 1):
        qids_where_rank_is_enough_for_fact2[rank] = \
            qids_where_rank_is_enough_for_fact2[rank - 1].union(rank_to_qids_fact2.get(rank, set()))

    def number_of_questions_with_both_facts(l1: int, l2: int) -> int:
        f1 = qids_where_rank_is_enough_for_fact1.get(l1, qids_where_rank_is_enough_for_fact1[max_l1])
        f2 = qids_where_rank_is_enough_for_fact2.get(l2, qids_where_rank_is_enough_for_fact2[max_l2])
        qids = f1 & f2
        return len(qids)

    target_fraction = 0.91   # We're determining l1 and l2 such that this fraction of questions has the right facts.

    best_l1_l2 = None
    best_fraction = None
    for current_metric_value in range(2, max_l1 + max_l2 + 1):
        # try all combinations of l1 and l2 that add up to this metric value
        successful_fraction = None
        for l1 in range(1, current_metric_value):
            l2 = current_metric_value - l1

            successful_fraction = \
                number_of_questions_with_both_facts(l1, l2) / number_of_questions_with_both_facts_retrieved
            if best_fraction is None or successful_fraction > best_fraction:
                best_fraction = successful_fraction
                best_l1_l2 = (l1, l2)
            if successful_fraction >= target_fraction:
                break
        if successful_fraction is not None and successful_fraction >= target_fraction:
            break

    print(f"Successful fraction: {best_fraction:.3f}")
    print(f"Best ls are {best_l1_l2}")
    print(f"sum(l) is {sum(best_l1_l2)}")

if __name__ == "__main__":
    main()
