all:

test:
	python3 main.py --solver=pointer_net
	python3 main.py --solver=random
	python3 main.py --solver=fsb

train:
	python3 main.py --mode=train --solver=pointer_net

debug:
	python3 main.py --scenario=generate_mnist_property --benchmark=mnist