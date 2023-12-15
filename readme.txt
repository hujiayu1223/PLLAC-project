
This repository provides the implementation for "An Unbiased Risk Estimator for Learning with Augmented Classes" 

Requirements:
Python 3.6
numpy 1.14
Pytorch 1.1
torchvision 0.2

Demo:

	python main.py -dt benchmark -lo unbiased -mo mlp -ds mnist -lr 1e-4 -al unbiased -seed 0 
	python main.py  -dt uci -al unbiased -lo unbiased -ds msplice -mo linear -uci 1 -ep 150 -lr 0.01 -wd 1e-4 -gpu 0 -seed 0
	python main.py  -dt realworld -lo unbiased -ds lost -mo linear -uci 1 -ep 200 -lr 0.005 -wd 1e-4 -gpu 0 -al unbiased -seed 0