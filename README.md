# Named-Entity-Recognition

Requirements: Python 3.10+, PyTorch 2.0+, NumPy

Directory layout:
    data/train, data/dev, data/test
    eval/eval.py, eval/conll03eval
    glove.6B.100d (unzipped)

Training + prediction file generation:
    python blstm1.py
    python blstm2.py
    python blstm3.py

blstm1.py trains the Task 1 model, saves blstm1.pt, and writes dev1.out and test1.out.
blstm2.py trains the Task 2 model, saves blstm2.pt, and writes dev2.out and test2.out.
blstm3.py trains the Bonus model, saves blstm3.pt, and writes dev3.out and pred.

Evaluation with conll03eval + perl:
    python eval/eval.py -g data/dev -p dev1.out
    python eval/eval.py -g data/dev -p dev2.out
    python eval/eval.py -g data/dev -p dev3.out
