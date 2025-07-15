all:

test:
	python3 main.py --solver=pointer_net

train:
	python3 main.py --mode=train --solver=pointer_net

prepare:
	python3 main.py --scenario=prepare_subgraph
	# python3 main.py --scenario=prepare_properties --benchmark=mnist --eps=0.02

plot:
	python3 main.py --scenario=plot_line
