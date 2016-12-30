#coding: utf-8
import abc
import math
import numpy


class LossFuntion(metaclass=abc.ABCMeta):
  def __init__(self):
    pass

  @abc.abstractmethod
  def terminal_region_value(self, targets, region_sample_indices):
    """Calculate the predict value of the leaf.
    """
  @abc.abstractmethod
  def init_predict_value(self, predict_value, dataset):
    """Init F_0(xi) for every i in dataset. 
    """

  @abc.abstractmethod
  def compute_loss(self, targets, predicts):
    """Compute the loss for predict values.
    """


class BinomialLoss(LossFuntion):
  def __init__(self):
    super(BinomialLoss, self).__init__()

  def terminal_region_value(self, targets, region_sample_indices):
    numerator = 0.
    denominator = 0.
    for idx in region_sample_indices:
      y = targets[idx]
      numerator += y
      denominator += abs(y) * (2 - abs(y))
    return numerator / denominator

  def init_predict_value(self, predict_value, dataset):
    targets = dataset.get_targets()
    target_mean = numpy.mean(targets)
    sample_indices = dataset.get_sample_indices()
    for sample_idx in sample_indices:
      predict_value.append(target_mean)

  def compute_loss(self, targets, predicts):
    loss = 0.
    for (target, predict) in zip(targets, predicts):
      loss += math.log(1 + math.exp(-2 * target * predict))
    return loss / len(targets)
   
def loss_of_type(loss_type):
  if loss_type == 'binomial':
    return BinomialLoss()
