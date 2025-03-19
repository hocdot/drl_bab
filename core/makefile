all: prepare test plot

prepare:
	# python3 main.py --scenario=split_train_test
	# python3 main.py --scenario=prepare_graph
	# python3 main.py --scenario=prepare_train_offline_buffer --mode=train --n_prepare_episode=113 --clear_buffer

train_offline:
	python3 main.py --solver=propose_2 --mode=train_offline --gnn_class=SSGConv --n_train_offline_step=2000 # fail to learn
	# python3 main.py --solver=propose --mode=train_offline --gnn_class=ChebConv --n_train_offline_step=800 # fail to learn
	# python3 main.py --solver=propose --mode=train_offline --gnn_class=TAGConv --n_train_offline_step=100 # ok
	# python3 main.py --solver=propose --mode=train_offline --gnn_class=ARMAConv # ok
	# python3 main.py --solver=propose --mode=train_offline --gnn_class=SSGConv # ok
	# python3 main.py --solver=propose --mode=train_offline --gnn_class=Linear --load_state_dict

test:
	python3 main.py --mode=test --solver=heuristic
	python3 main.py --mode=test --solver=greedy_2
	python3 main.py --mode=test --solver=propose --score=actor --gnn_class=Linear
	# python3 main.py --mode=test --solver=greedy_3
	# python3 main.py --mode=test --solver=greedy
	# python3 main.py --mode=test --solver=propose --score=critic --gnn_class=Linear
	# python3 main.py --mode=test --solver=propose --score=combined --gnn_class=Linear
	# python3 main.py --mode=test --solver=propose --score=actor --gnn_class=ChebConv

	# python3 main.py --mode=test --solver=heuristic --verbose --n_test_episode=1 --n_step 200
	# python3 main.py --mode=test --solver=propose_2 --score=actor --gnn_class=SSGConv --verbose --n_test_episode=1 --n_step 200

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
