#!/usr/bin/env python3

# This file illustrates how you might experiment with the HMM interface.
# You can paste these commands in at the Python prompt, or execute `test_en.py` directly.
# A notebook interface is nicer than the plain Python prompt, so we provide
# a notebook version of this file as `test_en.ipynb`, which you can open with
# `jupyter` or with Visual Studio `code` (run it with the `nlp-class` kernel).

import logging
import math
import os
from pathlib import Path
from typing import Callable

from corpus import TaggedCorpus
from eval import eval_tagging, model_cross_entropy, viterbi_error_rate
from hmm import HiddenMarkovModel
from lexicon import build_lexicon
import torch

# Set up logging.
log = logging.getLogger("test_en")       # For usage, see findsim.py in earlier assignment.
logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)  # could change INFO to DEBUG
# torch.autograd.set_detect_anomaly(True)    # uncomment to improve error messages from .backward(), but slows down

# Switch working directory to the directory where the data live.  You may need to edit this line.
os.chdir("../data")

entrain = TaggedCorpus(Path("ensup"), Path("enraw"))                               # all training
ensup =   TaggedCorpus(Path("ensup"), tagset=entrain.tagset, vocab=entrain.vocab)  # supervised training
endev =   TaggedCorpus(Path("endev"), tagset=entrain.tagset, vocab=entrain.vocab)  # evaluation
log.info(f"Tagset: f{list(entrain.tagset)}")
known_vocab = TaggedCorpus(Path("ensup")).vocab    # words seen with supervised tags; used in evaluation

# Make an HMM.  Let's do supervised pre-training to approximately
# maximize the regularized log-likelihood.  If you want to speed this
# up, you can increase the tolerance of training (using the
# `tolerance` argument), since we don't really have to train to
# convergence.
#
# We arbitrarily choose `reg=1`, but it would be better to search
# for the best regularization strength.

lexicon = build_lexicon(entrain, embeddings_file=Path('words-50.txt'))  # works better with more dims!
hmm = HiddenMarkovModel(entrain.tagset, entrain.vocab, lexicon)  # randomly initialized parameters
loss_sup = lambda model: model_cross_entropy(model, eval_corpus=ensup)
hmm.train(corpus=ensup, loss=loss_sup, 
          minibatch_size=30, evalbatch_size=10000, 
          reg=1, lr=0.0001, save_path="ensup_hmm.pkl") 

# Now let's throw in the unsupervised training data as well, and continue
# training to try to improve accuracy on held-out development data.
# We'll stop when this accuracy stops getting better.
# 
# This step is delicate, so we'll use a much smaller learning rate and
# pause to evaluate more often, in hopes that tagging accuracy will go
# up for a little bit before it goes down again (see Merialdo 1994).
# (Log-likelihood will continue to improve, just not accuracy.)

hmm = HiddenMarkovModel.load("ensup_hmm.pkl")  # reset to supervised model (in case you're re-executing this bit)
loss_dev = lambda model: viterbi_error_rate(model, eval_corpus=endev, 
                                            known_vocab=known_vocab)
hmm.train(corpus=entrain, loss=loss_dev,
          minibatch_size=30, evalbatch_size=len(entrain)//4, # evaluate 4 times per epoch
          reg=1, lr=0.000001, save_path="entrain_hmm.pkl")

# You can also retry the above workflow where you start with a worse
# supervised model (like Merialdo).  Replace `ensup` throughout the
# corpus setup with `ensup-tiny`, which is only 25 sentences (that
# cover all tags in `endev`).  And change the names of your saved
# models.

# More detailed look at the first 10 sentences in the held-out corpus,
# including Viterbi tagging.
for m, sentence in enumerate(endev):
    if m >= 10: break
    viterbi = hmm.viterbi_tagging(sentence.desupervise(), endev)
    counts = eval_tagging(predicted=viterbi, gold=sentence, 
                          known_vocab=known_vocab)
    num = counts['NUM', 'ALL']
    denom = counts['DENOM', 'ALL']
    
    log.info(f"Gold:    {sentence}")
    log.info(f"Viterbi: {viterbi}")
    log.info(f"Loss:    {denom - num}/{denom}")
    log.info(f"Prob:    {math.exp(hmm.log_prob(sentence, endev))}")
