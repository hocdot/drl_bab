import torchvision.transforms as transforms
import neuralsat as ns
import torchvision
import torch
import onnx
import os


from .util import write_vnnlib_classify

def generate_mnist_property(args):
    onnx_dir = os.path.join(args.benchmark_dir, args.benchmark, 'onnx')
    vnnlib_dir = os.path.join(args.benchmark_dir, args.benchmark, 'vnnlib')
    csv_dir = os.path.join(args.benchmark_dir, args.benchmark)
    
    if args.benchmark == 'mnist':
        dataset = torchvision.datasets.MNIST(
            root=args.data_dir, 
            train=False, 
            download=True, 
            transform=transforms.ToTensor(),
        )
    elif args.benchmark == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, 
            train=False, 
            download=True, 
            transform=transforms.ToTensor(),
        )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    count = 0
    with open(os.path.join(csv_dir, 'instances.csv'), 'w') as f:
        for onnx_name in os.listdir(onnx_dir):
            onnx_path = os.path.join(onnx_dir, onnx_name)
            
            model, input_shape = ns.util.network.read_onnx.parse_onnx(onnx_path)[:2]
            model.to(args.device)
            model.eval()
        
            for idx, (x, y) in enumerate(dataloader):
                x = x.to(args.device)
                y = y.to(args.device)
                prediction = model(x.view(input_shape))
                if prediction.argmax(-1) != y:
                    print(f'[!] Skip {idx=} {y=} {prediction.argmax(-1)=}')
                    continue
                
                vnnlib_path = os.path.join(vnnlib_dir, f'{idx}_{onnx_name[:-5]}.vnnlib')
                
                write_vnnlib_classify(
                    spec_path=vnnlib_path,
                    data_lb=x - args.eps,
                    data_ub=x + args.eps,
                    prediction=prediction,
                    negate_spec=True,
                    single_output=True,
                )
                
                # TODO: check attack
                f.write(f'{count},{onnx_path},{vnnlib_path}\n')
                count += 1
                if count >= args.n_instance:
                    # TODO: next network
                    return
        
    