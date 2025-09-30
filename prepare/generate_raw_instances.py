import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import os

from dnnv_solver.helper.spec.write_vnnlib import write_vnnlib_classify
from dnnv_solver.helper.network.read_onnx import parse_onnx

from arguments import get_args

if __name__ == '__main__':
    
    args = get_args()
    assert args.dataset == 'dnnv', f'{args.dataset=}'
        
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
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    root_dir = os.path.join(args.dataset_dir, args.dataset)
    onnx_dir = os.path.join(root_dir, 'onnx')
    vnnlib_dir = os.path.join(root_dir, 'vnnlib')
    eps = 0.04
    all_onnx_names = os.listdir(onnx_dir)
    with open(os.path.join(root_dir, 'instances_raw.csv'), 'w') as f:
        for onnx_name in tqdm(all_onnx_names):
            onnx_path = os.path.join(onnx_dir, onnx_name)
            model, input_shape, _ = parse_onnx(onnx_path)
            count = 0
            for i, (x, label) in enumerate(loader):
                pred = model(x.view(input_shape))
                if pred.argmax(-1).item() != label:
                    continue
                
                vnnlib_name = f'prop_{os.path.splitext(onnx_name)[0]}_{count}_{eps:.2f}.vnnlib'
                vnnlib_path = os.path.join(vnnlib_dir, vnnlib_name)
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
                if count >= args.n_vnnlib_per_onnx:
                    break