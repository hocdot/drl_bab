all: prepare test plot

prepare:
	# python3 main.py --scenario=split_train_test
	# python3 main.py --scenario=prepare_graph
	python3 main.py --scenario=prepare_train_offline_buffer --mode=train --n_prepare_episode=10 --top_k=3 --n_prepare_iter=5 --clear_buffer

train_offline:
	python3 main.py --solver=propose --mode=train_offline --gnn_class=Linear --embed_class=Linear --n_train_offline_step=5000

test:
	python3 main.py --mode=test --solver=propose --score=critic --gnn_class=Linear --embed_class=Linear --verbose
	# python3 main.py --mode=test --solver=heuristic --verbose
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
