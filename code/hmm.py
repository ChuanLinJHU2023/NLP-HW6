#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Implementation of Hidden Markov Models.

from __future__ import annotations
import logging
from math import inf, log, exp, sqrt
from pathlib import Path
from typing import Callable, List, Optional, Tuple, cast

import torch
from torch import Tensor as Tensor
from torch import tensor as tensor
from torch import optim as optim
from torch import nn as nn
from torch import cuda as cuda
from torch.nn import functional as F
from jaxtyping import Float
from typeguard import typechecked
from tqdm import tqdm # type: ignore
import pickle

from corpus import (BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag,
                    TaggedCorpus, Word)
from integerize import Integerizer

TorchScalar = Float[torch.Tensor, ""] # a torch.Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

###
# HMM tagger
###
class HiddenMarkovModel(nn.Module):
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 unigram: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The unigram
        flag says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended
        to support higher-order HMMs: trigram HMMs used to be popular.)"""

        super().__init__() # type: ignore # pytorch nn.Module does not have type annotations

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)
        # As usual in Python, attributes starting with _ are intended as private;
        # in this case, they might go away if you changed the parametrization of the model.

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        assert vocab[-2:] == [EOS_WORD, BOS_WORD]  # make sure these are the last two

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.d = lexicon.size(1)   # dimensionality of a word's embedding in attribute space
        self.unigram = unigram     # do we fall back to a unigram model?

        self.tagset = tagset
        self.vocab = vocab
        self._E = lexicon[:-2]  # embedding matrix; omits rows for EOS_WORD and BOS_WORD

        # Useful constants that are invoked in the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        assert self.bos_t is not None    # we need this to exist
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors

        self.init_params()     # create and initialize params

    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> List[Tuple[int,Optional[int]]]:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        # Make sure that the sentence comes from a corpus that this HMM knows
        # how to handle.
        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")
        # If so, go ahead and integerize it.
        # print("This is print on _integerize_sentence line 92 hmm.py")
        # print(sentence)
        # print(corpus.integerize_sentence(sentence))
        # print(corpus.vocab._objects)
        # print(corpus.tagset._objects)
        return corpus.integerize_sentence(sentence)

    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).
        However, we initialize the BOS_TAG column of _WA to -inf, to ensure that
        we have 0 probability of transitioning to BOS_TAG (see "Don't guess when you know").
        See the "Parametrization" section of the reading handout."""

        # See the reading handout section "Parametrization."
        # 

        # As in HW3's probs.py, our model instance's parameters are tensors
        # that have been wrapped in nn.Parameter.  That wrapper produces a
        # new view of each tensor with requires_grad=True.  It also ensures
        # that they are included in self.parameters(), which this class
        # inherits from nn.Module and which is used below for
        # regularization and training.

        ThetaB = 0.01*torch.rand(self.k, self.d)    
        self._ThetaB = nn.Parameter(ThetaB)    # params used to construct emission matrix

        WA = 0.01*torch.rand(1 if self.unigram # just one row if unigram model
                             else self.k,      # but one row per tag s if bigram model
                             self.k)           # one column per tag t
        WA[:, self.bos_t] = -inf               # correct the BOS_TAG column
        self._WA = nn.Parameter(WA)            # params used to construct transition matrix


    @typechecked
    def params_L2(self) -> TorchScalar:
        """What's the L2 norm of the current parameter vector?
        We consider only the finite parameters."""
        l2 = tensor(0.0)
        for x in self.parameters():
            x_finite = x[x.isfinite()]
            l2 = l2 + x_finite @ x_finite   # add ||x_finite||^2
        return l2


    def updateAB(self) -> None:
        """Set the transition and emission matrices A and B, based on the current parameters.
        See the "Parametrization" section of the reading handout."""
        
        A = F.softmax(self._WA, dim=1)       # run softmax on params to get transition distributions
                                             # note that the BOS_TAG column will be 0, but each row will sum to 1
        if self.unigram:
            # A is a row vector giving unigram probabilities p(t).
            # We'll just set the bigram matrix to use these as p(t | s)
            # for every row s.  This lets us simply use the bigram
            # code for unigram experiments, although unfortunately that
            # preserves the O(nk^2) runtime instead of letting us speed 
            # up to O(nk) in the unigram case.
            self.A = A.repeat(self.k, 1)
        else:
            # A is already a full matrix giving p(t | s).
            self.A = A

        WB = self._ThetaB @ self._E.t()  # inner products of tag weights and word embeddings
        B = F.softmax(WB, dim=1)         # run softmax on those inner products to get emission distributions
        self.B = B.clone()
        self.B[self.eos_t, :] = 0        # but don't guess: EOS_TAG can't emit any column's word (only EOS_WORD)
        self.B[self.bos_t, :] = 0        # same for BOS_TAG (although BOS_TAG will already be ruled out by other factors)


    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))

        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")


    def get_one_hot_vector(self, index_for_nonzero: int):
        if index_for_nonzero is None:
            return None
        res = torch.zeros(self.k)
        res[index_for_nonzero] = 1
        return res

    @typechecked
    def log_prob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Compute the log probability of a single sentence under the current
        model parameters.  ********* If the sentence is not fully tagged, the probability
        will marginalize over all possible tags. *****************

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""
        return self.log_forward(sentence, corpus)


    @typechecked
    def log_forward(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Run the forward algorithm from the handout on ********** a tagged, untagged,
        or partially tagged sentence. ************** Return <<<<<<<log Z>>>>>>> (the log of the forward
        probability).

        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're 
        integerizing correctly.

        Chuan:
        We utilize A.4 Matrix Formulas and equation 12 to enhance the performance
        Here is equation 12
        a(j) = [a(j-1) @ A ] * B_*wj
        However, to deal with the partially tagged sentence and the tag restriction tau in algorithm 1,
        we need to make the following changes to equation 12
        Assume that tau(j) is a 0/1 mask vector of length k (where k is the number of tags)
        Then we modify equation 12 into
        a(j) = ( [a(j-1) @ A ] * B_*wj ) * tau(j)
        """

        sent = self._integerize_sentence(sentence, corpus)
        assert sent[0][1] == self.bos_t
        assert sent[-1][1] == self.eos_t

        logger.debug(" We are running forward algo on the following sentence ")
        logger.debug(sentence)
        logger.debug(sent)

        alpha = [torch.empty(self.k) for _ in sent]
        alpha[0] = self.get_one_hot_vector(self.bos_t)
        logger.debug("On i = 0, we have alpha = ")
        logger.debug(alpha[0])

        for i in range(1, len(alpha)-1):  # skip i for bos and eos
            word_i, tag_i = sent[i]
            mask_vector = self.get_one_hot_vector(tag_i)
            alpha_i = (alpha[i - 1] @ self.A) * self.B[:, word_i]
            alpha[i] = alpha_i if tag_i is None else alpha_i * mask_vector
            logger.debug(f"On i = {i}, we have alpha = ")
            logger.debug(alpha[i])

        # Here is i for eos
        i = len(alpha)-1
        word_i, tag_i = sent[i]
        mask_vector = self.get_one_hot_vector(tag_i)
        alpha_i = (alpha[i - 1] @ self.A) # we don't need to consider the emission of word eos
        alpha[i] = alpha_i if tag_i is None else alpha_i * mask_vector
        logger.debug(f"On i = {i}, we have alpha = ")
        logger.debug(alpha[i])

        assert alpha[-1][self.eos_t] != 0
        z = alpha[-1][self.eos_t]
        log_z = torch.log(z)
        return log_z
        # raise NotImplementedError   # you fill this in!

        # The "nice" way to construct the sequence of vectors alpha[0],
        # alpha[1], ...  is by appending to a List[Tensor] at each step.
        # But to better match the notation in the handout, we'll instead
        # preallocate a list of length n+2 so that we can assign directly
        # to each alpha[j] in turn.

        # Alternatively, when running the forward algorithm or the Viterbi
        # algorithm, you don't really need to keep a whole list of vectors.
        # You could just have two variables: on step j through the loop,
        # alpha_new is computed from alpha_prev, where those represent the
        # vectors that the handout calls alpha[j] and alpha[j-1].

        # If you're tracking gradients, then alpha[j-1] will still
        # remember how it depends on alpha[j-2], which will still remember
        # how it depends on alpha[j-3].  That is how PyTorch supports
        # backprop.  But you don't need to store those earlier vectors in a
        # Python variable of your own.

        # The only reason to store the whole list of vectors is if you find
        # it personally convenient for your coding.  It will let your code
        # more closely match the pseudocode in the handout, and you might wish 
        # to inspect the list when debugging.


    def hstack(self, v: torch.Tensor, n = None):
        """
        horizontally stack column vectors!!!!! We have n such vectors
        See stack_methods.py for usage of this function
        """
        n = self.k if n is None else n
        result = torch.concat([v.unsqueeze(1)] * n, dim=1)  # unsqueeze(1) to get column vector
        assert result.shape == (len(v), n)
        return result

    def vstack(self, v: torch.Tensor, n = None):
        """
        vertically stack row vectors!!!!! We have n such vectors
        See stack_methods.py for usage of this function
        """
        n = self.k if n is None else n
        result = torch.concat([v.unsqueeze(0)] * n, dim=0)  # unsqueeze(0) to get row vector
        assert result.shape == (n, len(v))
        return result

    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model.

        Chuan:
        We utilize A.4 Matrix Formulas and equation 12 to enhance the performance
        Here is equation 12
        a(j) = [a(j-1) @ A ] * B_*wj
        However, to deal with algorithm 2,
        we need to make the following changes to equation 12
        Assume that tau(j) is a 0/1 mask vector of length k (where k is the number of tags)
        Then we modify equation 12 into
        a(j)           =    max( HStack(a(j-1)) * A * VStack(B_*wj) , dim = 0 ) * tau(j)
        backpointer(j) = argmax( HStack(a(j-1)) * A * VStack(B_*wj) , dim = 0 ) * tau(j)
        HStack(v) is a matrix by stacking multiple col vectors v horizontally
        VStack(v) is a matrix by stacking multiple row vectors v vertically
        """

        sent = self._integerize_sentence(sentence, corpus)
        logger.debug(" We are running viterbi algo on the following sentence ")
        logger.debug(sentence)
        logger.debug(sent)

        alpha = [torch.empty(self.k) for _ in sent]
        backpointer = [torch.empty(self.k, dtype=torch.int32) for _ in sent]
        alpha[0] = self.get_one_hot_vector(self.bos_t)
        logger.debug("On i = 0, we have alpha = ")
        logger.debug(alpha[0])

        for i in range(1, len(alpha)-1): # skip i for bos and eos
            word_i, tag_i = sent[i]
            mask_vector = self.get_one_hot_vector(tag_i)
            alpha_i, backpointer_i = torch.max(self.hstack(alpha[i-1]) * self.A * self.vstack(self.B[:, word_i]),
                                                 dim=0)
            alpha[i] = alpha_i if tag_i is None else alpha_i * mask_vector
            backpointer[i] = backpointer_i if tag_i is None else backpointer_i * mask_vector
            logger.debug(f"On i = {i}, we have alpha = ")
            logger.debug(alpha[i])

        # Here is i for eos
        i = len(alpha)-1
        word_i, tag_i = sent[i]
        mask_vector = self.get_one_hot_vector(tag_i)
        alpha_i, backpointer_i = torch.max(self.hstack(alpha[i - 1]) * self.A, # we don't need to consider the emission of word eos
                                           dim=0)
        alpha[i] = alpha_i if tag_i is None else alpha_i * mask_vector
        backpointer[i] = backpointer_i if tag_i is None else backpointer_i * mask_vector
        logger.debug(f"On i = {i}, we have alpha = ")
        logger.debug(alpha[i])

        tags = [None] * len(sentence)
        tags[len(sentence)-1] = self.eos_t
        for i in range(len(sentence)-1, 0, -1):
            # make sure that each tag is a int!!!!!!
            tags[i-1] = int(backpointer[i][tags[i]])
        assert tags[0] == self.bos_t
        tags_real = [self.tagset[tag] for tag in tags]
        words_real = [word for word, _ in sentence]
        return_sentence = Sentence(list(zip(words_real, tags_real)))
        return return_sentence
        # raise NotImplementedError   # you fill this in!

        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # The code continues to use the name alpha, rather than \hat{alpha}
        # as in the handout.

        # We'll start by integerizing the input Sentence.
        # But make sure you deintegerize the words and tags again when
        # constructing the return value, since the type annotation on
        # this method says that it returns a Sentence object, and
        # that's what downstream methods like eval_tagging will
        # expect.  (Running mypy on your code will check that your
        # code conforms to the type annotations ...)

    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[HiddenMarkovModel], float],
              tolerance: float = 0.001,
              minibatch_size: int = 1,
              evalbatch_size: int = 500,
              lr: float = 1.0,
              reg: float = 0.0,
              save_path: Path = Path("my_hmm.pkl")) -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        The minibatch size controls how often we do an update.
        (Recommended to be larger than 1 for speed; can be inf for the whole training corpus.)
        The evalbatch size controls how often we evaluate (e.g., on a development corpus).
        We will stop when the relative improvement of the evaluation loss,
        since the last evalbatch, is less than the tolerance; in particular,
        we will stop when the improvement is negative, i.e., the evaluation loss 
        is getting worse (overfitting).
        lr is the learning rate, and reg is an L2 batch regularization coefficient."""

        # This is relatively generic training code.  Notice however that the
        # updateAB step before each minibatch produces A, B matrices that are
        # then shared by all sentences in the minibatch.

        # All of the sentences in a minibatch could be treated in parallel,
        # since they use the same parameters.  The code below treats them
        # in series, but if you were using a GPU, you could get speedups
        # by writing the forward algorithm using higher-dimensional tensor 
        # operations that update alpha[j-1] to alpha[j] for all the sentences
        # in the minibatch at once, and then PyTorch could actually take
        # better advantage of hardware parallelism.

        logger.info(f"Training {type(self).__name__} with {sum(x.numel() for x in self.parameters())} parameters")
        
        assert minibatch_size > 0
        if minibatch_size > len(corpus):
            minibatch_size = len(corpus)  # no point in having a minibatch larger than the corpus
        assert reg >= 0

        old_dev_loss: Optional[float] = None    # we'll keep track of the dev loss here

        optimizer = optim.SGD(self.parameters(), lr=lr)  # optimizer knows what the params are
        self.updateAB()                                        # compute A and B matrices from current params
        log_likelihood = tensor(0.0)                           # accumulator for minibatch log_likelihood
        for m, sentence in tqdm(enumerate(corpus.draw_sentences_forever())):
            # Before we process the new sentence, we'll take stock of the preceding
            # examples.  (It would feel more natural to do this at the end of each
            # iteration instead of the start of the next one.  However, we'd also like
            # to do it at the start of the first time through the loop, to print out
            # the dev loss on the initial parameters before the first example.)

            # m is the number of examples we've seen so far.
            # If we're at the end of a minibatch, do an update.
            if m % minibatch_size == 0 and m > 0:
                logger.debug(f"Training log-likelihood per example: {log_likelihood.item()/minibatch_size:.3f} nats")
                optimizer.zero_grad()          # backward pass will add to existing gradient, so zero it
                objective = -log_likelihood + (minibatch_size/corpus.num_tokens()) * reg * self.params_L2()
                objective.backward()           # type: ignore # compute gradient of regularized negative log-likelihod
                length = sqrt(sum((x.grad*x.grad).sum().item() for x in self.parameters()))
                logger.debug(f"Size of gradient vector: {length}")  # should approach 0 for large minibatch at local min
                optimizer.step()               # SGD step
                self.updateAB()                # update A and B matrices from new params
                log_likelihood = tensor(0.0)   # reset accumulator for next minibatch

            # If we're at the end of an eval batch, or at the start of training, evaluate.
            if m % evalbatch_size == 0:
                with torch.no_grad():       # type: ignore # don't retain gradients during evaluation
                    dev_loss = loss(self)   # this will print its own log messages
                if old_dev_loss is not None and dev_loss >= old_dev_loss * (1-tolerance):
                    # we haven't gotten much better, so stop
                    self.save(save_path)  # Store this model, in case we'd like to restore it later.
                    break
                old_dev_loss = dev_loss            # remember for next eval batch

            # Finally, add likelihood of sentence m to the minibatch objective.
            log_likelihood = log_likelihood + self.log_prob(sentence, corpus)


    def save(self, model_path: Path) -> None:
        logger.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved model to {model_path}")


    @classmethod
    def load(cls, model_path: Path, device: str = 'cpu') -> HiddenMarkovModel:
        logger.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=device)
            # torch.load is similar to pickle.load but handles tensors too
            # map_location allows loading tensors on different device than saved
        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls} but got {type(model)} from saved file {model_path}.")
        logger.info(f"Loaded model from {model_path}")
        return model