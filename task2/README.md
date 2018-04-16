## Baseline system for the CoNLL-SIGMORPHON shared task 2018 Task 2

This is a simple baseline system for solving tasks 2. The system is an
encoder decoder on character sequences. It takes a lemma as input and
generates a word form. The process is conditioned on the context of
the lemma as explained below. The baseline system is implemented using
the [DyNet](http://dynet.readthedocs.io/en/latest/python_ref.html)
python library. Please, install DyNet if you want to run the baseline.

https://sigmorphon.github.io/sharedtasks/2018/

### Description

The general idea is to feed in the input lemma and a context embedding
into the encoder and then use the decoder together with an attention
mechanism to generate the output word form.

There are two tracks in task 2. In the first track, we are supposed to
generate word forms (`years` in this case) given lemmas and
morpho-syntactic descriptions (MSDs) of context words.

```
CHERNOBYL  Chernobyl      PROPN;SG
ACCIDENT   accident       N;SG
:          :              PUNCT
TEN        ten            NUM
_          year           _
ON         on             ADV
```

In the second track, we only get the word forms in the context

```
CHERNOBYL  _              _
ACCIDENT   _              _
:          _              _
TEN        _              _
_          year           _
ON         _              _
```

The baseline treats the lemma, word form and MSD of the previous
and following word as context in track 1. In track 2, the baseline only
considers the word forms of the previous and next word. In both tracks,
the input lemma is also part of the context.

The baseline system concatenates embeddings for context word forms,
lemmas and MSDs into a context vector. The baseline then
computes character embeddings for each character in the input
lemma. Each of these is concatenated with a copy of the context
vector. The resulting sequence of vectors is encoded using an LSTM
encoder. Subsequently, an LSTM decoder generates the characters in the
output word form using encoder states and an attention mechanism.

### How to run the baseline system?

  Example for English in the high setting:

  ```
  python3 baseline/baseline.py trainsets/en-track1-high devsets/en-track1-covered devsets/en-uncovered
  ```

  The baseline system will write a tagged output file
  (devsets/en-track1-covered-out in this case). The baseline system
  does not generate a model file.

  The is no batching in the baseline system. Therefore, it can be slow
  (up to 2 hours on CPU) to run depending on the language, your system
  and quantity of training data. We, therefore, provide tagged
  development data and a summary file for all tracks, languages and
  data settings in the directory `baselineresults`.

### How to evaluate?
  ```
  python3 baseline/eval.py devsets/en-track1-covered-out devsets/en-uncovered
  ```

  The evaluation script measures accuracy of word form generation and
  the average Levenshtein distance of output word forms and gold
  standard word forms.

### About the baseline

  The baseline code is heavily based on the DyNet example code for an
  encoder-decoder
  (https://github.com/clab/dynet/blob/master/examples/sequence-to-sequence/attention.py).

