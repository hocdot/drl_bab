train: 
	python3 train.py --mode train --config config/dnnv.yaml --load_checkpoint

infer:
	python3 infer.py --config config/dnnv.yaml --decision_topk_test 10 --checkpoint_postfix 4 --device cuda:1

plot:
	python3 -m plot.plot_dnnv --config config/dnnv.yaml 