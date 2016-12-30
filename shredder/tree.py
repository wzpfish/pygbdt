#coding: utf-8
import collections
import random

from . import data
from . import loss
from . import criterion


class TreeNode(object):
	def __init__(self, sample_indices):
		"""By default, one node is a leaf node. So it maintains samples.
		If it is split, it turns to an intermediate node which don't
		maintain samples.
		"""
		self.is_leaf = True
		self.sample_indices = sample_indices
		self.split_feature_id = None
		self.split_feature_value = None
		self.split_improvement = None
		self.left_node = None
		self.right_node = None
		self.predict_value = None

	def update(self,
			   is_leaf,
			   sample_indices,
			   split_feature_id,
			   split_feature_value,
			   split_improvement,
			   left_node,
			   right_node):
		self.is_leaf = is_leaf
		self.sample_indices = sample_indices
		self.split_feature_id = split_feature_id
		self.split_feature_value = split_feature_value
		self.split_improvement = split_improvement
		self.left_node = left_node
		self.right_node = right_node

	def describe(self, addtion_info=""):
		if not self.left_node or not self.right_node:
			return "{LeafNode:" + str(self.predict_value) + "}"
		left_info = self.left_node.describe()
		right_info = self.right_node.describe()
		info = (addtion_info + "{split_feature:" + str(self.split_feature_id) + ",split_value:" + 
				str(self.split_feature_value) + "[left_node:" + left_info + ",right_node:" + right_info + "]}")
		return info

	def get_samples(self):
		return self.sample_indices

	def get_predict_value(self):
		return self.predict_value

	def calc_predict_value(self, targets, loss):
		self.predict_value = loss.terminal_region_value(targets, self.sample_indices)


class Tree(object):
	def __init__(self, max_num_of_leaf_nodes, loss, criterion_type='mse'):
		self.root_node = None
		self.leaf_nodes = []
		self.non_leaf_nodes = []
		self.num_of_leaf_nodes = 0
		self.max_num_of_leaf_nodes = max_num_of_leaf_nodes
		self.criterion_func = criterion.criterion_of_type(criterion_type)
		self.loss = loss

	def build(self, dataset, targets):
		self.dataset = dataset
		self.root_node = TreeNode(dataset.get_sample_indices())
		self.num_of_leaf_nodes = 1
		leaf_nodes = collections.deque()
		leaf_nodes.append(self.root_node)
		while 0 < self.num_of_leaf_nodes < self.max_num_of_leaf_nodes:
			node = leaf_nodes.popleft()
			split_feature_id, split_feature_value, split_left_samples, split_right_samples, split_improvement = \
				self._find_split_point(node, targets)	
			if split_feature_id is not None:
				# every split increase num of leaf nodes by one.
				self.num_of_leaf_nodes += 1
				left_node = TreeNode(split_left_samples)
				right_node = TreeNode(split_right_samples)
				leaf_nodes.append(left_node)
				leaf_nodes.append(right_node)
				node.update(is_leaf = False,
							sample_indices = None,
							split_feature_id=split_feature_id,
							split_feature_value=split_feature_value,
							split_improvement=split_improvement,
							left_node=left_node,
							right_node=right_node)
				self.non_leaf_nodes.append(node)
			else:
				# If node can't be split, it must be leaf node.
				# So we update the predict value of this leaf.
				node.calc_predict_value(targets, self.loss)
				self.leaf_nodes.append(node)
		# For all left leaves, we update the predict value
		while len(leaf_nodes) > 0:
			node = leaf_nodes.popleft()
			node.calc_predict_value(targets, self.loss)
			self.leaf_nodes.append(node)

		return True

	def get_leaf_nodes(self):
		return self.leaf_nodes

	def predict(self, instance):
		def get_value(root, instance):
			if root.is_leaf is True:
				return root.get_predict_value()
			elif instance[root.split_feature_id] <= root.split_feature_value:
				return get_value(root.left_node, instance)
			else:
				return get_value(root.right_node, instance)
		return get_value(self.root_node, instance)

	def _find_split_point(self, node, targets, sample_count=10):
		max_improvement = -1
		split_feature_id = None
		split_feature_value = None
		split_left_samples = None
		split_right_samples = None

		dataset = self.dataset
		feature_indices = dataset.get_feature_indices()
		for feature_idx in feature_indices:
			feature_values = dataset.get_feature_values_of_samples(node.sample_indices, feature_idx)
			if sample_count < len(feature_values):
				feature_values = random.sample(feature_values, sample_count)
			for feature_value in feature_values:
				left_samples = []
				right_samples = []
				for sample_idx in node.sample_indices:
					sample_feature_value = dataset.get_feature_values_of_samples([sample_idx], feature_idx)[0]
					if sample_feature_value <= feature_value:
						left_samples.append(sample_idx)
					else:
						right_samples.append(sample_idx)
				# If can't split the node by this value, just skip it.
				if len(left_samples) == 0 or len(right_samples) == 0:
					continue
				left_targets = [targets[idx] for idx in left_samples]
				right_targets = [targets[idx] for idx in right_samples]
				improvement = self.criterion_func(left_targets, right_targets)
				if max_improvement < 0 or improvement > max_improvement:
					max_improvement = improvement
					split_feature_id = feature_idx
					split_feature_value = feature_value
					split_left_samples = left_samples
					split_right_samples = right_samples

		return split_feature_id, split_feature_value, split_left_samples, split_right_samples, max_improvement


def __gen_dummy_file():
	content = ('1\t0.1\t0.2\t0.3\t0.4\n'
			   '-1\t0.5\t0.8\t0.4\t0.9\n'
			   '-1\t0.9\t1.0\t1.1\t1.2\n')
	dummy_filename = './dummy.txt'
	with open(dummy_filename, 'w') as fout:
		fout.write(content)
	return dummy_filename
	
def test_test():
	dummy_filename = __gen_dummy_file()
	dataset = data.Dataset(dummy_filename)
	import os
	os.remove(dummy_filename)

	tree = Tree(3, loss.BinomialLoss())
	tree.build(dataset, dataset.get_targets())
	print(tree.root_node.describe())


if __name__ == '__main__':
	test_test()