## Task 2 of the CoNLL-SIGMORPHON shared task 2018

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

In the second track, we only get word forms of context words.

```
CHERNOBYL  _              _
ACCIDENT   _              _
:          _              _
TEN        _              _
_          year           _
ON         _              _
```

https://sigmorphon.github.io/sharedtasks/2018/

### Data

Training sets are located in the directory `trainsets` and development
sets in `devsets`. 

For each language (German, English, Spanish, Finnish, French, Russian
and Swedish), there are high, medium and low train sets containing
varying amounts of training examples. Moreover, we give separate training
sets for track 1 and track 2. You are allowed to use individual training
sets in any way you like but 

 * You are not allowed to use external data
sets in this task, and 
 * You are not allowed to mix training sets from
track 1 and track 2 or from high, medium and low settings, and
 * You cannot use development sets for training your final system. They
can only be used to tuning of hyperparameters.

Track 1 training sets contain word forms, lemmas and MSDs for all
words. Track 2 training sets contain word forms for context words and
lemmas and word forms for target words. You should learn a system to
predict the word form of a target word given its lemma.

We provide development sets separately for track 1 and track 2 (for
example `devsets/en-track1-covered` and
`devsets/en-track2-covered`). We also provide gold standard
development sets in (for example `devsets/en-uncovered`).

Both training and development sets contain exactly one gold standard
word form for each target lemma. These are the word forms from the
original Universal Dependency treebanks used to generate these data
sets. However, in the evaluation phase of task 2, we will take into
account the fact that several word forms can sometimes be correct in
the same context.

### Description

This is a simple baseline system for solving tasks 2. The system is an
encoder decoder on character sequences. It takes a lemma as input and
generates a word form. The process is conditioned on the context of
the lemma as explained below. The baseline system is implemented using
the [DyNet](http://dynet.readthedocs.io/en/latest/python_ref.html)
python library. Please, install DyNet if you want to run the baseline.

The general idea is to feed in the input lemma and a context embedding
into the encoder and then use the decoder together with an attention
mechanism to generate the output word form.

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
  python3 baseline/eval.py baselineresults/en-track1-covered-out devsets/en-uncovered
  ```

  The evaluation script measures accuracy of word form generation and
  the average Levenshtein distance of output word forms and gold
  standard word forms.

### About the baseline

  The baseline code is heavily based on the DyNet example code for an
  encoder-decoder
  (https://github.com/clab/dynet/blob/master/examples/sequence-to-sequence/attention.py).

