#coding: utf-8
import math
import numpy

from . import tree
from . import loss

class GBDT(object):
	def __init__(self, num_trees, learning_rate, max_num_of_leaf_nodes, loss_type, criterion_type='mse'):
		self.num_trees = num_trees
		self.learning_rate = learning_rate
		self.max_num_of_leaf_nodes = max_num_of_leaf_nodes
		self.loss = loss.loss_of_type(loss_type)
		self.criterion_type = criterion_type

		self.predict_value = []
		self.trees = []

	
	def init_train(self, dataset):
		# Init the F0 value of every samples.
		self.loss.init_predict_value(self.predict_value, dataset)

	def train_iter(self, dataset):
		pseudo_targets = self._calc_residual(dataset)
		base_tree = tree.Tree(self.max_num_of_leaf_nodes, self.loss, self.criterion_type)
		base_tree.build(dataset, pseudo_targets)
		self.trees.append(base_tree)
		self.update_predict_value(base_tree)
		train_loss = self._compute_loss(dataset)
		return train_loss

	def update_predict_value(self, tree):
		leaf_nodes = tree.get_leaf_nodes()
		for node in leaf_nodes:
			sample_indices = node.get_samples()
			for sample_idx in sample_indices:
				self.predict_value[sample_idx] += self.learning_rate * node.get_predict_value()

	def feature_importance(self, dataset):
		importances = [0. for it in dataset.get_feature_indices()]
		for tree in self.trees:
			for node in tree.non_leaf_nodes:
				importances[node.split_feature_id] += node.split_improvement
		sum_import = sum(importances)
		norm = [it / sum_import for it in importances]
		return norm

	def _calc_residual(self, dataset):
		residuals = []
		sample_indices = dataset.get_sample_indices()
		for sample_idx in sample_indices:
			target = dataset.get_target_at(sample_idx)
			residual = 2. * target / (1 + math.exp(2 * target * self.predict_value[sample_idx]))
			residuals.append(residual)
		return residuals

	def _compute_loss(self, dataset):
		targets = dataset.get_targets()
		predicts = self.predict_value
		return self.loss.compute_loss(targets, predicts)

	def accuracy(self, dataset, threshold=0.5):
		targets = dataset.get_targets()
		features = dataset.get_features()
		mean = numpy.mean(targets)
		true_count = 0
		for idx, feature in enumerate(features):
			predict_value = mean
			for tree in self.trees:
				predict_value += tree.predict(feature)
			target = targets[idx]
			prob = 1.0 / (1 + math.exp(-2 * predict_value))
			predict_label = -1
			if prob >= threshold:
				predict_label = 1
			if predict_label == target:
				true_count += 1
		return true_count * 1.0 / len(targets)