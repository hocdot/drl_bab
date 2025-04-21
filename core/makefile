all: prepare test plot

prepare_sub_graph:
	python3 main.py --scenario=prepare_sub_graph --verbose

prepare_graph:
	python3 main.py --scenario=prepare_graph --verbose

prepare_data:
	python3 main.py --scenario=prepare_offline_buffer --mode=train --n_prepare_episode=1 --n_prepare_iter=1 --clear_buffer --verbose

train_offline:
	python3 main.py --solver=propose --mode=train_offline --n_train_offline_step=500 --gamma=0.0 --device=cuda:0 --batch_size=300

propose:
	python3 main.py --mode=test --solver=propose --verbose --device=cuda:0

heuristic:
	python3 main.py --mode=test --solver=heuristic --verbose --device=cuda:0
	# python3 main.py --mode=test --solver=greedy --verbose

boxplot:
	clear
	python3 main.py --scenario=plot_boxplot --mode=test --metric=n_step & \
	python3 main.py --scenario=plot_boxplot --mode=test --metric=n_visited & \
	wait

plot:
	clear
	python3 main.py --scenario=plot_line --mode=test --metric=n_visited & \
	python3 main.py --scenario=plot_line --mode=test --metric=n_step & \
	python3 main.py --scenario=plot_line --mode=train_offline --metric=actor_loss & \
	python3 main.py --scenario=plot_line --mode=train_offline --metric=critic_loss & \
	wait
