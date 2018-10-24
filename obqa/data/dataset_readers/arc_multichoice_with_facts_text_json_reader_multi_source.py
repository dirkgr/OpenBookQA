import importlib
from typing import Dict, List, Any
import json
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, ListField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from .quark import utilities

from obqa.data.dataset_readers.common import tokenizer_dict_from_params, token_indexer_dict_from_params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("arc-multi-choice-w-facts-txt-json-multi-source")
class ArcMultiChoiceWithFactsTextJsonReaderMultiSource(DatasetReader):
    """
    Reads a file from the AllenAI-V1-Feb2018 dataset in Json format.  This data is
    formatted as jsonl, one json-formatted instance per line.  An example of the json in the data is:

        {"id":"MCAS_2000_4_6",
        "question":{"stem":"Which technology was developed most recently?",
            "choices":[
                {"text":"cellular telephone","label":"A"},
                {"text":"television","label":"B"},
                {"text":"refrigerator","label":"C"},
                {"text":"airplane","label":"D"}
            ]},
        "answerKey":"A"
        }

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 corpus: str,
                 field_tokenizers: Dict[str, Tokenizer] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 choice_value_type: str = None,
                 question_value_type: str = None,
                 lazy: bool = False,
                 ) -> None:
        super().__init__(lazy)

        self._field_tokenizers = field_tokenizers or {"default": WordTokenizer()}
        self._default_tokenizer = self._field_tokenizers.get("default")
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._question_value_type = question_value_type
        self._choice_value_type = choice_value_type
        self._corpus = utilities.Corpus(cached_path(corpus))

    def get_question_text_from_item(self, item_json, question_value_type):
        question_text = item_json["question"]["stem"]
        if question_value_type == "fact1":
            question_text = item_json["fact1"]
        elif question_value_type == "fact2":
            question_text = item_json["fact2"]
        elif question_value_type == "fact1_fact2":
            question_text = item_json["fact1"] + " " + item_json["fact2"]
        elif question_value_type == "question_fact1":
            question_text = question_text + " " + item_json["fact1"]
        elif question_value_type == "question_fact2":
            question_text = question_text + " " + item_json["fact2"]
        elif question_value_type == "question_workerId":
            worker_id_token = "@%s@" % item_json["workerId"]
            question_text = worker_id_token + " " + question_text + " " + worker_id_token

        return question_text

    def get_choice_text_from_item(self, item_json, choice_id, choice_value_type):
        choice_text = item_json["question"]["choices"][choice_id]["text"]

        if choice_value_type == "question_choice":
            question_text = item_json["question"]["stem"]
            choice_text = question_text + " " + choice_text
        elif choice_value_type == "choice_fact1":
            choice_text = choice_text + " " + item_json["fact1"]
        elif choice_value_type == "choice_fact2":
            choice_text = choice_text + " " + item_json["fact2"]
        elif choice_value_type == "choice_workerId":
            worker_id_token = "@%s@" % item_json["workerId"]
            choice_text = worker_id_token + " " + choice_text + " " + worker_id_token

        return choice_text

    @overrides
    def _read(self, file_path: str):
        retrieval = "tushar"
        retrieval_module = importlib.import_module("obqa.data.dataset_readers.quark.retrieval." + retrieval)

        # Read knowledge facts to instances
        file_path = cached_path(file_path)
        logger.info("Reading ARC instances from jsonl dataset at: %s", file_path)
        for item_json in retrieval_module.retrieve(file_path, self._corpus):
            item_id = item_json["id"]
            question_text = self.get_question_text_from_item(item_json, self._question_value_type)

            gold_facts_text_meta = {"gold_facts":
                                        {
                                            "fact1": item_json.get("fact1", ""),
                                            "fact2": item_json.get("fact2", "")
                                        }
                                    }

            choice_label_to_id = {}
            choice_text_list = []

            for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
                choice_label = choice_item["label"]
                choice_label_to_id[choice_label] = choice_id

                choice_text = self.get_choice_text_from_item(item_json, choice_id, self._choice_value_type)
                choice_text_list.append(choice_text)

            answer_id = choice_label_to_id[item_json["answerKey"]]

            # loading the facts from different sources
            facts = {}
            for choice in item_json['question']['choices']:
                support = (
                    (support['text'], support['score'])
                    for support in choice['support']
                    if support['type'] == "sentence"
                )
                for text, score in support:
                    old_score = facts.get(text, 0.0)
                    facts[text] = old_score + score
            facts = list(facts.items())
            facts.sort(key=lambda x: (-x[1], len(x[0]), x[0]))  # sort by score desc, then length asc, then the text itself
            facts = [f[0] for f in facts[:100]]

            yield self.text_to_instance(item_id,
                                        question_text,
                                        choice_text_list,
                                        facts,
                                        answer_id,
                                        gold_facts_text_meta)

    def tokenize(self, text, tokenizer_name="default"):
        tokenizer = self._field_tokenizers.get(tokenizer_name, self._default_tokenizer)
        return tokenizer.tokenize(text)

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: Any,
                         question_text: str,
                         choice_text_list: List[str],
                         facts_text_list: List[str],
                         answer_id: int,
                         meta_fields: Dict = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        question_tokens = self.tokenize(question_text, "question")
        choices_tokens_list = [self.tokenize(x, "choice") for x in choice_text_list]
        facts_tokens_list = [self.tokenize(x, "fact") for x in facts_text_list]

        fields['question'] = TextField(question_tokens, self._token_indexers)
        fields['choices_list'] = ListField([TextField(x, self._token_indexers) for x in choices_tokens_list])
        fields['facts_list'] = ListField([TextField(x, self._token_indexers) for x in facts_tokens_list])

        fields['label'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question_text": question_text,
            "choice_text_list": choice_text_list,
            "facts_text_list": facts_text_list,
            "question_tokens": [x.text for x in question_tokens],
            "choice_tokens_list": [[x.text for x in ct] for ct in choices_tokens_list],
            "facts_tokens_list": [[x.text for x in ct] for ct in facts_tokens_list],
            "label_gold": answer_id,
        }

        if meta_fields is not None:
            for k, v in meta_fields.items():
                metadata[k] = v

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'ArcMultiChoiceWithFactsTextJsonReaderMultiSource':
        # read tokenizers
        field_tokenizers = tokenizer_dict_from_params(params.get('tokenizers', {}))
        token_indexers = token_indexer_dict_from_params(params.get('token_indexers', {}))

        corpus = params.get('corpus')
        choice_value_type = params.get('choice_value_type', None)
        question_value_type = params.get('question_value_type', None)

        lazy = params.pop('lazy', False)
        # params.assert_empty(cls.__name__)

        return ArcMultiChoiceWithFactsTextJsonReaderMultiSource(field_tokenizers=field_tokenizers,
                                                                token_indexers=token_indexers,
                                                                corpus=corpus,
                                                                choice_value_type=choice_value_type,
                                                                question_value_type=question_value_type,
                                                                lazy=lazy)

    @classmethod
    def config_example(self):
        config_json = {
            "dataset_reader": {
                "type": "arc-multi-choice-w-facts-txt-json",
                "token_indexers": {
                    "tokens": {
                        "type": "single_id",
                        "lowercase_tokens": True
                    }
                },
                "tokenizers": {
                    "default": {
                        "start_tokens": ["@start@"],
                        "end_tokens": ["@end@"]
                    }
                },
                "corpus": "corpora/arc.txt.gz"
            },
        }

        return json.dumps(config_json, indent=4, sort_keys=True)
