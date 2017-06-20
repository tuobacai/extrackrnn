#!/usr/bin/python
# -*- coding:utf-8 -*-

"""Utilities for tokenizing text, create vocabulary and so on"""
#deepnlp textsum
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
from tensorflow.python.platform import gfile

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def prepare_headline_data(data_dir, vocabulary_size, tokenizer=None):
  train_path = os.path.join(data_dir, "train")
  src_train_path = os.path.join(train_path, "content-train.txt")
  dest_train_path = os.path.join(train_path, "title-train.txt")

  dev_path = os.path.join(data_dir, "dev")
  src_dev_path = os.path.join(dev_path, "content-dev.txt")
  dest_dev_path = os.path.join(dev_path, "title-dev.txt")

  # Create vocabularies of the appropriate sizes.
  vocab_path = os.path.join(data_dir, "vocab")
  create_vocabulary(vocab_path, src_train_path, vocabulary_size, tokenizer)

  # Create token ids for the training data.
  src_train_ids_path = os.path.join(train_path, "content_train_id")
  dest_train_ids_path = os.path.join(train_path, "title_train_id")
  data_to_token_ids(src_train_path, src_train_ids_path, vocab_path, tokenizer)
  data_to_token_ids(dest_train_path, dest_train_ids_path, vocab_path, tokenizer)

  # Create token ids for the development data.
  src_dev_ids_path = os.path.join(dev_path, "content_dev_id")
  dest_dev_ids_path = os.path.join(dev_path, "title_dev_id")
  data_to_token_ids(src_dev_path, src_dev_ids_path, vocab_path, tokenizer)
  data_to_token_ids(dest_dev_path, dest_dev_ids_path, vocab_path, tokenizer)

  return (src_train_ids_path, dest_train_ids_path,
          src_dev_ids_path, dest_dev_ids_path,
          vocab_path, vocab_path)
