#!/usr/bin/env python
"""
Official Evaluation Script for the CoNLL-SIGMORPHON 2018 Shared Task
Task2. Returns accuracy and mean Levenhstein distance. 

Accuracy is given separately for the original word form in UD test
files and all plausible word forms as given by an annotator.

Levenshtein distance is given w.r.t. to the original UD word form
only.

Author: Miikka Silfverberg
Last Update: 07/19/2018

Based on the official evaluation script for the CoNLL-SIGMORPHON 2017
shared task by Ryan Cotterell.
"""

import numpy as np
import codecs

def distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    m = np.zeros([len(str2)+1, len(str1)+1])
    for x in xrange(1, len(str2) + 1):
        m[x][0] = m[x-1][0] + 1
    for y in xrange(1, len(str1) + 1):
        m[0][y] = m[0][y-1] + 1
    for x in xrange(1, len(str2) + 1):
        for y in xrange(1, len(str1) + 1):
            if str1[y-1] == str2[x-1]:
                dg = 0
            else:
                dg = 1
            m[x][y] = min(m[x-1][y] + 1, m[x][y-1] + 1, m[x-1][y-1] + dg)
    return int(m[len(str2)][len(str1)])

def read(fname):
    """ read file name """
    data = [[]]
    with codecs.open(fname, 'rb', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n\r')
            if line == '':
                data.append([])
            else:
                word, lemma, tag = line.split("\t")
                data[-1].append((word,lemma,tag))
            
    return [d for d in data if d]

def eval_form(gold, guess, ignore=set()):
    """ compute average accuracy and edit distance for task 2 """
    assert(len(gold) == len(guess))

    first_correct, any_correct, dist, total = 0., 0., 0., 0.
    for gold_s, guess_s in zip(gold,guess):
        assert(len(gold_s) == len(guess_s))

        for gold_e, guess_e in zip(gold_s, guess_s):
            gold_words, gold_lemma, gold_tag = gold_e
            guess_word, guess_lemma, guess_tag = guess_e

            # Lower-case because casing cannot be inferred from the
            # lemma.
            gold_words = [w.lower() for w in gold_words.split('/')]
            guess_word = guess_word.lower()

            if gold_tag == '_':                
                assert(gold_lemma == guess_lemma)

                first_correct += 1 if gold_words[0] == guess_word else 0
                any_correct += 1 if guess_word in gold_words else 0
                dist += distance(gold_words[0],guess_word)
                total += 1

    return (round(first_correct/total*100, 2), 
            round(any_correct/total*100, 2), 
            round(dist/total, 2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CoNLL-SIGMORPHON 2017 Shared Task Evaluation')
    parser.add_argument("--gold", help="Gold standard (uncovered)", required=True, type=str)
    parser.add_argument("--guess", help="Model output", required=True, type=str)
    args = parser.parse_args()    

    data_gold = read(args.gold)
    data_guess = read(args.guess)

    print "original form acccuracy:\t{0:.2f}\nplausible form acccuracy:\t{1:.2f}\nlevenshtein:\t{2:.2f}".format(*eval_form(data_gold, data_guess))
