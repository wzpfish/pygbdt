#coding: utf-8
"""
criterion实现了决策树分裂的标准.
"""
import numpy

def least_square(left_targets, right_targets):
  """Greedy Function Approximation: A Gradient Boosting Machine中公式35
  且采用Lk_TreeBoost中的Unit Weights形式.
  """
  left_weight = len(left_targets)
  right_weight = len(right_targets)
  left_targets_mean = numpy.mean(left_targets)
  right_targets_mean = numpy.mean(right_targets)
  diff_square = ((left_targets_mean - right_targets_mean) * 
           (left_targets_mean - right_targets_mean))
  return (left_weight * right_weight * diff_square) / (left_weight + right_weight) 


#TODO: 实现更多的criterion函数
def criterion_of_type(type):
  if type == 'mse':
    return least_square
  return None
