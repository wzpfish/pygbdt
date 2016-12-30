#coding: utf-8
import sys
sys.path.append('..')
import time

from shredder import GBDT, Dataset

def start_train(model, dataset):
	model.init_train(dataset)
	for iter in range(model.num_trees):
		iter_loss = model.train_iter(dataset)
		acc = model.accuracy(dataset)
		cur_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
		print('{} iter {}, loss {}, accuracy {}'.format(cur_time, iter, iter_loss, acc))
	print(model.feature_importance(dataset))

def main():
	dataset = Dataset('data/train.txt')
	model = GBDT(num_trees=100,
				 learning_rate=0.02,
				 max_num_of_leaf_nodes=16,
				 loss_type='binomial')
	start_train(model, dataset)

if __name__ == '__main__':
	main()
