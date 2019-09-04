# shredder
A simple python implement for GBDT.

## GBDT Introduction


## Code
It's really simple and just used for further understanding the theory of gradient boost machine(GBM). The original paper of GBM is [Greedy Function Approximation A Gradient Boosting Machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf). All the implement in my code follow this paper.

The code is really easy and one can peek at it:
* data.py: deal with the data
* tree.py: a crude implement of classification and regression tree
* loss.py: loss functions
* criterion.py: split strategy for decision tree
* gbm.py: models of gradient boost machine

There is also a demo and just run `train.py`(need python3). It will output the loss, accuracy and the importance of each features.
