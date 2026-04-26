import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import torchvision
import torch
import os

from dnnv_solver.helper.spec.write_vnnlib import write_vnnlib_classify
from dnnv_solver.helper.network.read_onnx import parse_onnx

from arguments import get_args

if __name__ == '__main__':
    
    args = get_args()
    assert args.dataset == 'dnnv', f'{args.dataset=}'
    
    print(f'[+] {args.dataset_dir=} {args.dataset=}')
        
    if args.clean:
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/io')
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/log')
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/graph')
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/embed')
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/vnnlib')
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/vnnlib_processed')
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/instances_*.csv')
    
    if args.input_dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = torchvision.datasets.MNIST(
            root=args.dataset_dir,
            train=False,
            transform=transform,
            download=True,
        )
    elif args.input_dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=False,
            transform=transform,
            download=True,
        )
    else:
        raise NotImplementedError(f'{args.input_dataset=}')
    
    root_dir = os.path.join(args.dataset_dir, args.dataset)
    onnx_dir = os.path.join(root_dir, 'onnx')
    vnnlib_dir = os.path.join(root_dir, 'vnnlib')
    os.makedirs(vnnlib_dir, exist_ok=True)
    
    all_onnx_eps_settings = pd.read_csv(os.path.join(root_dir, 'onnx_eps_setting.csv'), header=None)
    with open(os.path.join(root_dir, 'instances_raw.csv'), 'w') as f:
        for seed, (onnx_name, eps) in enumerate(tqdm(all_onnx_eps_settings.values)):
            onnx_path = os.path.join(onnx_dir, onnx_name)
            model, input_shape, _ = parse_onnx(onnx_path)
            count = 0
            seed += args.seed
            torch.manual_seed(seed)
            loader = DataLoader(dataset, batch_size=1, shuffle=True)
            for i, (x, label) in enumerate(loader):
                pred = model(x.view(input_shape))
                if pred.argmax(-1).item() != label:
                    continue
                
                vnnlib_name = f'prop_{os.path.splitext(onnx_name)[0]}_{count}_{eps:.2f}_seed_{seed}.vnnlib'
                vnnlib_path = os.path.join(vnnlib_dir, vnnlib_name)
                if not os.path.exists(vnnlib_path):
                    write_vnnlib_classify(
                        spec_path=vnnlib_path,
                        data_lb=x - eps / 2.0,
                        data_ub=x + eps / 2.0,
                        prediction=pred,
                        negate_spec=True,
                        single_output=False,
                    )
                print(f'onnx/{onnx_name},vnnlib/{vnnlib_name}', file=f)
                count += 1
                print(f'[+] {count=} {onnx_name=} {vnnlib_name=}')
                if count >= args.n_vnnlib_per_onnx:
                    break