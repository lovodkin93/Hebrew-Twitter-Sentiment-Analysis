import os
import shlex
from copy import deepcopy

import dynet_config
import numpy as np
from configargparse import ArgParser, Namespace, ArgumentDefaultsHelpFormatter, SUPPRESS
from logbook import Logger, FileHandler, StderrHandler


# Classifiers

SPARSE = "sparse"
MLP = "mlp"
BIRNN = "bilstm"
HIGHWAY_RNN = "highway"
HIERARCHICAL_RNN = "hbirnn"
NOOP = "noop"
NN_CLASSIFIERS = (MLP, BIRNN, HIGHWAY_RNN, HIERARCHICAL_RNN)
CLASSIFIERS = (SPARSE, MLP, BIRNN, HIGHWAY_RNN, HIERARCHICAL_RNN, NOOP)

FEATURE_PROPERTIES = "wmtudhencpqxyAPCIEMNT#^$"

# Swap types
REGULAR = "regular"
COMPOUND = "compound"

# Required number of edge labels per format
EDGE_LABELS_NUM = {"amr": 110, "sdp": 70, "conllu": 60}
SPARSE_ARG_NAMES = set()
NN_ARG_NAMES = set()
DYNET_ARG_NAMES = set()
RESTORED_ARGS = set()

SEPARATOR = "."


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


class Hyperparams:
    def __init__(self, parent, shared=None, **kwargs):
        self.shared = FallbackNamespace(parent, shared)
        self.specific = FallbackNamespace(parent)
        for name, args in kwargs.items():
            self.specific[name].update(args)

    def items(self):
        return ([("shared", self.shared)] if self.shared.vars() else []) + list(self.specific.traverse())


class HyperparamsInitializer:
    def __init__(self, name=None, *args, **kwargs):
        """
        :param name: name of hyperparams subset
        :param args: raw arg strings
        :param kwargs: parsed and initialized values
        """
        self.name = name
        self.str_args = list(args) + ["--%s %s" % (k.replace("_", "-"), v) for k, v in kwargs.items()]

    def __str__(self):
        return '"%s"' % " ".join([self.name] + list(self.str_args))

    def __bool__(self):
        return bool(self.str_args)

    @classmethod
    def action(cls, args):
        return cls(*args.replace("=", " ").split())


class Iterations:
    def __init__(self, args):
        try:
            epochs, *hyperparams = args.replace("=", " ").split()
        except (AttributeError, ValueError):
            epochs, *hyperparams = args,
        self.epochs, self.hyperparams = int(epochs), HyperparamsInitializer(str(epochs), *hyperparams)

    def __str__(self):
        return str(self.hyperparams or self.epochs)


class Config(object):
    def __init__(self, *args):
        self.arg_parser = ap = ArgParser(description="Sentiment Analysis of Tweets in Hebrew",
                                         formatter_class=ArgumentDefaultsHelpFormatter)

        ap.add_argument("--cross-validation", choices=[True, False],
                        default=True)
        ap.add_argument("--morph", choices=['Yap', 'Stanza', None],
                        default=None)
        ap.add_argument("--print-info", choices=[True, False],
                        default=True)
        ap.add_argument("--with-stat", choices=[True, False],
                        default=True)
        ap.add_argument("--save-model", choices=[True, False],
                        default=True)


        self.args = FallbackNamespace(ap.parse_args(args if args else None))

    def args_str(self, args):
        return ["--" + ("no-" if v is False else "") + k.replace("_", "-") +
                ("" if v is False or v is True else
                 (" " + str(" ".join(map(str, v)) if hasattr(v, "__iter__") and not isinstance(v, str) else v)))
                for (k, v) in sorted(args.items()) if
                v not in (None, (), "", self.arg_parser.get_default(k))
                and not k.startswith("_")
                and (args.node_labels or ("node_label" not in k and "node_categor" not in k))
                and (args.swap or "swap_" not in k)
                and (args.swap == COMPOUND or k != "max_swap")
                and (not args.require_connected or k != "orphan_label")
                and (args.classifier == SPARSE or k not in SPARSE_ARG_NAMES)
                and (args.classifier in NN_CLASSIFIERS or k not in NN_ARG_NAMES | DYNET_ARG_NAMES)
                and k != "passages"]

    def __str__(self):
        self.args.hyperparams = [HyperparamsInitializer(name, **args.vars()) for name, args in self.hyperparams.items()]
        return " ".join(list(self.args.passages) + self.args_str(self.args))
