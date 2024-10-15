#!/usr/bin/env python3
"""
Command-line interface for training and evaluating HMM and CRF taggers.
"""
import argparse
import logging
from pathlib import Path
import torch
from eval import model_cross_entropy, write_tagging
from hmm import HiddenMarkovModel
from lexicon import build_lexicon
from corpus import TaggedCorpus

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval", type=str, help="evaluation file")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="optional initial model file to load (will be trained further).  Loading a model overrides most of the other options."
    )
    parser.add_argument(
        "-l",
        "--lexicon",
        type=str,
        help="newly created model (if no model was loaded) should use this lexicon file",
    )
    parser.add_argument(
        "--crf",
        action="store_true",
        default=False,
        help="the newly created model (if no model was loaded) should be a CRF"
    )
    parser.add_argument(
        "-u",
        "--unigram",
        action="store_true",
        default=False,
        help="the newly created model (if no model was loaded) should only be a unigram HMM or CRF"
    )
    parser.add_argument(
        "-a",
        "--awesome",
        action="store_true",
        default=False,
        help="the newly created model (if no model was loaded) should use extra improvements"
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str,
        nargs="*",
        help="training files to train the model further"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=50000,
        help="maximum number of steps to train to prevent training for too long "
             "(this is an practical trick that you can choose to implement in the `train` method of hmm.py and crf.py)"
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=1.0,
        help="l2 regularization during further training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate during further training"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="tolerance for early stopping"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="tmp.model",
        help="where to save the trained model"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="where to save the prediction outputs"
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if not args.model and not args.lexicon:
        parser.error("Please provide lexicon file path when no model provided")
    if not args.model and not args.train:
        parser.error("Please provide at least one training file when no model provided")
    return args

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)

    train = None
    model = None
    if args.model is not None:
        if args.crf:
            raise NotImplementedError   # you fill this in!
        else:
            model = HiddenMarkovModel.load(Path(args.model), device=args.device)
        assert model is not None
        tagset = model.tagset
        vocab = model.vocab
        if args.train is not None:
            train = TaggedCorpus(*[Path(t) for t in args.train], tagset=tagset, vocab=vocab)
    else:
        train = TaggedCorpus(*[Path(t) for t in args.train])
        tagset = train.tagset
        vocab = train.vocab
        if args.crf:
            raise NotImplementedError   # you fill this in!
        else:
            lexicon = build_lexicon(train, embeddings_file=Path(args.lexicon), log_counts=args.awesome)
            model = HiddenMarkovModel(tagset, vocab, lexicon, unigram=args.unigram)

    dev = TaggedCorpus(Path(args.eval), tagset=tagset, vocab=vocab)
    if args.train is not None:
        assert train is not None and model is not None
        # you can instantiate a different development loss depending on the question / which one optimizes the evaluation loss
        dev_loss =  lambda x: model_cross_entropy(x, dev)
        model.train(corpus=train,
                    loss=dev_loss,
                    minibatch_size=args.train_batch_size,
                    evalbatch_size=args.eval_batch_size,
                    lr=args.lr,
                    reg=args.reg,
                    save_path=args.save_path,
                    tolerance=args.tolerance)
    write_tagging(model, dev, Path(args.eval+".output") if args.output_file is None else args.output_file)


if __name__ == "__main__":
    main()
