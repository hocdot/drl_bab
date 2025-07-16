# TODO

# Install neuralsat
```sh
# git submodule add -b heuristic git@github.com:dynaroars/neuralsat.git neuralsat
git submodule update --init --recursive
cd neuralsat
pip3 install -e .
```

## Hai Duong
- [ ] CNN problems ONNX + VNNLIB
- [ ] Thinking customized loss function towards select better active neurons

## Thanh Le
- [x] Extract subgraph based on vnnlib
- [x] Cache list potential vnnlib
- [ ] Check if subgraph pruning is correct with CNN onnx
- [ ] Finally cache subgraph (sub edge index, edge weight)
- [ ] Idea: non-trainable embedding if it is faster

## Requirement

```bash
pip3 install -r requirements.txt
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
cd neuralsat && pip3 install -e .
```
