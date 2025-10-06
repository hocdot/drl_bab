import pandas as pd
import random
import torch
import tqdm
import os
import gc

from dnnv_solver.helper.spec.write_vnnlib import write_vnnlib_classify_single
from dnnv_solver.helper.misc.torch_cuda_memory import gc_cuda
from dnnv_solver.helper.network.read_onnx import parse_onnx
from dnnv_solver.helper.spec.objective import parse_vnnlib
from dnnv_solver.dnnv_bab import DnnvBabSolver


def extract_instance(net_path, vnnlib_path):
    model, input_shape, output_shape = parse_onnx(net_path)
    objectives = parse_vnnlib(vnnlib_path, input_shape)
    return model, input_shape, objectives


def instance_generator(csv_name,args):
    # load hard instances list
    root_dir = os.path.join(args.dataset_dir, args.dataset)
    path = os.path.join(root_dir, csv_name)
    df = pd.read_csv(path, header=None)
    for i, (onnx, vnnlib) in tqdm.tqdm(df.iterrows(), total=len(df), desc=f'{csv_name}', leave=False):
        yield os.path.join(root_dir, onnx), os.path.join(root_dir, vnnlib)
        
def seperate_vnnlib(args, input_csv_name, output_csv_name):
    output_dir = os.path.join(args.dataset_dir, args.dataset)
    if os.path.exists(os.path.join(output_dir, output_csv_name)):
        return
    
    vnnlib_dir_name = 'vnnlib_processed'
    os.makedirs(os.path.join(output_dir, vnnlib_dir_name), exist_ok=True)
    with open(os.path.join(output_dir, output_csv_name), 'w') as f:
        for (onnx, vnnlib) in instance_generator(input_csv_name, args):
            net, input_shape, objs = extract_instance(onnx, vnnlib)
            count = 0
            obj = objs.pop(1)
            
            x = ((obj.lower_bounds + obj.upper_bounds) * 0.5).view(input_shape)
            pred = net(x)
            spec_names = write_vnnlib_classify_single(
                spec_path=os.path.join(output_dir, vnnlib_dir_name, os.path.basename(os.path.splitext(vnnlib)[0])),
                data_lb=obj.lower_bounds,
                data_ub=obj.upper_bounds,
                prediction=pred,
                negate_spec=True,
            )
            for spec_name in spec_names:
                print(f'onnx/{os.path.basename(onnx)},{vnnlib_dir_name}/{spec_name}', file=f)
            count += 1

def process_instance(onnx, vnnlib, args):
    io_path = os.path.join(args.dataset_dir, args.dataset, 'io', f'{os.path.basename(onnx)}____{os.path.basename(vnnlib)}.pt')
    io_filtered_path = os.path.splitext(io_path)[0] + '.txt'
    if os.path.exists(io_path):
        return True

    if os.path.exists(io_filtered_path):
        return False
    
    net, input_shape, objs = extract_instance(onnx, vnnlib)
    assert len(objs) == 1, f'{len(objs)=}'
    
    # initialize dnn solver
    iv = DnnvBabSolver(net=net, input_shape=input_shape, objective=objs, args=args)
    # print(f'{onnx=} {vnnlib=} {iv.remaining_subproblem=}')
    if not iv.remaining_subproblem:
        with open(io_filtered_path, 'w') as f:
            print('filtered', file=f)
        return False
    
    # extract observation and score
    observation, score = iv.get_observation(return_fsb_score=True)
    print(f'    + save: {observation.shape=} {score.shape=}')
    # observation = torch.cat(observation, dim=1)
    
    # save io data
    score = score.cpu()
    observation = observation.cpu()
    torch.save({'observation': observation, 'score': score}, io_path)
    
    
    graph_path = os.path.join(args.dataset_dir, args.dataset, 'graph', f'{os.path.basename(onnx)}.pt')
    if not os.path.exists(graph_path):
        # save graph data
        obj = objs.pop(1)
        edge_weight, edge_index = iv.get_edge_data(obj)
        edge_weight = edge_weight.cpu()
        edge_index = edge_index.cpu()
        torch.save({'edge_weight': edge_weight, 'edge_index': edge_index}, graph_path)
        
    # collect garbage
    del iv
    gc_cuda()
    gc.collect()
    return True

def prepare_io(args):
    # create result dirs if not exist
    assert args.dataset == 'dnnv', f'{args.dataset=}'
    io_dir = os.path.join(args.dataset_dir, args.dataset, 'io')
    os.makedirs(io_dir, exist_ok=True)
    graph_dir = os.path.join(args.dataset_dir, args.dataset, 'graph')
    os.makedirs(graph_dir, exist_ok=True)
    # form a torch geometric dataset from hard instances list
    print(f'[+] seperate vnnlibs')
    seperate_vnnlib(args, input_csv_name='instances_raw.csv', output_csv_name='instances_seperated.csv')
    
    print(f'[+] processing onnxs and vnnlibs')
    # with parallel_config(backend='threading', n_jobs=os.cpu_count()//2):
    #     Parallel()(delayed(read_instance)(onnx, vnnlib, args) for onnx, vnnlib in instance_generator(args))
    all_lines = []
    for (onnx, vnnlib) in instance_generator('instances_seperated.csv', args):
        if process_instance(onnx, vnnlib, args):
            all_lines.append(f'onnx/{os.path.basename(onnx)},vnnlib_processed/{os.path.basename(vnnlib)}')
    
    random.shuffle(all_lines)
    train_lines = all_lines[:int(len(all_lines) * 0.8)]
    test_lines = all_lines[int(len(all_lines) * 0.8):]
    with open(os.path.join(args.dataset_dir, args.dataset, 'instances_train.csv'), 'w') as f:
        for line in train_lines:
            print(line, file=f)
    with open(os.path.join(args.dataset_dir, args.dataset, 'instances_test.csv'), 'w') as f:
        for line in test_lines:
            print(line, file=f)

if __name__ == '__main__':
    from arguments import get_args
    args = get_args()
    args.device = 'cpu'
    prepare_io(args)