#!/usr/bin/python3

from operator import itemgetter
import typing
from allennlp.common.util import JsonDict
import collections
import logging

Scores = collections.namedtuple(
    "Scores", [
        "total",
        "num_questions",
        "exam_score",
        "correct",
        "incorrect",
        "partial"
    ]
)

def scores_from_questions_with_predictions(qwps: typing.Generator[JsonDict, None, None]) -> Scores:
    total = 0
    num_questions = 0
    partial = 0
    correct = 0
    incorrect = 0
    for qwp in qwps:
        answer_choices = qwp["question"]["choices"]
        max_choice_score = max(answer_choices, key=itemgetter("score"))["score"]
        # Collect all answer choices with the same score
        selected_answers = [
            choice["label"]
            for choice in answer_choices
            if choice["score"] == max_choice_score
        ]
        answer_key = qwp["answerKey"]

        if answer_key in selected_answers:
            question_score = 1 / len(selected_answers)
            if question_score < 1:
                partial += 1
            else:
                correct += 1
        else:
            question_score = 0
            incorrect += 1

        if question_score < 1:
            question = qwp["question"]["stem"]
            choices = ["%s) %s" % (c["label"], c["text"]) for c in qwp["question"]["choices"]]
            question = " ".join([question] + choices)
            logging.info("Got %.2f points on this question: %s", question_score, question)
            logging.info("Selected answers were %r", selected_answers)
            # for choice in qwp["question"]["choices"]:
            #     label = choice["label"]
            #     support = [s["text"] for s in choice["support"]]
            #     support = " -- ".join(support)
            #     logging.info("Support for choice %s was \"%s\"", label, support)

        total += question_score
        num_questions += 1

    return Scores(total, num_questions, 100 * total / num_questions, correct, incorrect, partial)
