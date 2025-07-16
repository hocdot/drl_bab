all:

test:
	python3 main.py --solver=pointer_net

train:
	python3 main.py --mode=train --solver=pointer_net

prepare:
	python3 main.py --scenario=prepare_properties --eps=0.035 --clear
	python3 main.py --scenario=prepare_node_embedding --clear

plot:
	python3 main.py --scenario=plot_line
