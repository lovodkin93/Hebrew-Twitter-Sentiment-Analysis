import os
import shlex
from copy import deepcopy
import numpy as np
from configargparse import ArgParser, Namespace, ArgumentDefaultsHelpFormatter, SUPPRESS
import ast


class FallbackNamespace(Namespace):
    def __init__(self, fallback, kwargs=None):
        super().__init__(**(kwargs or {}))
        self._fallback = fallback
        self._children = {}

    def __getattr__(self, item):
        if item.startswith("_"):
            return getattr(super(), item)
        return getattr(super(), item, getattr(self._fallback, item))

    def __getitem__(self, item):
        if item:
            name, _, rest = item.partition(SEPARATOR)
            return self._children.setdefault(name, FallbackNamespace(self))[rest]
        return self

    def vars(self):
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def items(self):
        return self.vars().items()

    def update(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def traverse(self, prefix=None):
        if prefix and self.vars():
            yield (prefix, self)
        for name, child in self._children.items():
            yield from child.traverse(SEPARATOR.join(filter(None, (prefix, name))))


class Config(object):
    def __init__(self, *args):
        self.arg_parser = ap = ArgParser(description="Sentiment Analysis of Tweets in Hebrew",
                                         formatter_class=ArgumentDefaultsHelpFormatter)

        ap.add_argument("--cross-validation", choices=[True, False], type=ast.literal_eval,
                        default=True)
        ap.add_argument("--print-info", choices=[True, False], type=ast.literal_eval,
                        default=True)
        ap.add_argument("--with-stat", choices=[True, False], type=ast.literal_eval,
                        default=True)
        ap.add_argument("--save-model", choices=[True, False], type=ast.literal_eval,
                        default=True)
        ap.add_argument("--train-data-path", type=str,
                        default='../data/train_morph_data.tsv')
        ap.add_argument("--test-data-path", type=str,
                        default='../data/test_morph_data.tsv')
        ap.add_argument("--all-data-path", type=str,
                        default='../data/all_morph_data.tsv')
        ap.add_argument("--best-no-cv-model-path", type=str,
                        default='best_models/best_no_cv_model.sav')
        ap.add_argument("--best-cv-model-path", type=str,
                        default='best_models/best_cv_model.sav')
        ap.add_argument("--best-cv-unbiased-model-path", type=str,
                        default='best_models/best_cv_unbiased_model.sav')
        ap.add_argument("--with-bert", choices=[True, False], type=ast.literal_eval,
                        default=False)
        ap.add_argument("--morph", choices=['Yap', 'Stanza', None],
                        default=None)
        ap.add_argument("--morphamized-data-train-path", type=str,
                        default='../data/train_morph_data.tsv')
        ap.add_argument("--morphamized-data-test-path", type=str,
                        default='../data/test_morph_data.tsv')
        self.args = FallbackNamespace(ap.parse_args(args if args else None))