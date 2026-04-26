import os
import random

import numpy as np
import onnxruntime
import torch
from torchvision import datasets

from arguments import get_args
from dnnv_solver.helper.spec.write_vnnlib import write_vnnlib_classify

DELTAS = [0.1, 0.15, 0.2, 0.25]

def get_mu_std(prior_onnx: str, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    session = onnxruntime.InferenceSession(prior_onnx, None)
    input_name = session.get_inputs()[0].name
    output_name1 = session.get_outputs()[0].name
    output_name2 = session.get_outputs()[1].name
    mu, log_var = session.run([output_name1, output_name2], {input_name: image})
    std = np.exp(0.5 * log_var)
    return mu[0], std[0]


def get_concat_logits(concat_onnx: str, net_input: np.ndarray) -> np.ndarray:
    session = onnxruntime.InferenceSession(concat_onnx, None)
    image_input = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {image_input: net_input})[0][0]


def main(args):
    assert args.dataset == "dnnv", f"{args.dataset=}"
    assert args.input_dataset == "mnist", f"{args.input_dataset=}"

    if args.clean:
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/io')
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/log')
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/graph')
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/embed')
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/vnnlib')
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/vnnlib_processed')
        os.system(f'rm -rf {args.dataset_dir}/{args.dataset}/instances_*.csv')
    
    root_dir = os.path.join(args.dataset_dir, args.dataset)
    onnx_dir = os.path.join(root_dir, "onnx")
    vnnlib_dir = os.path.join(root_dir, "vnnlib")
    os.makedirs(vnnlib_dir, exist_ok=True)

    prior_onnx = os.path.join(onnx_dir, "mnist_prior.onnx")
    concat_onnx = os.path.join(onnx_dir, "mnist_concat.onnx")

    image_size = (1, 1, 28, 28)
    mnist_test = datasets.MNIST(
        root=args.dataset_dir,
        train=False,
        download=True,
        transform=None,
    )

    max_per_delta = args.n_vnnlib_per_onnx
    all_indices = list(range(len(mnist_test)))

    instances_path = os.path.join(root_dir, "instances_raw.csv")
    with open(instances_path, "w"):
        pass

    concat_basename = os.path.basename(concat_onnx)
    for delta in DELTAS:
        indices = all_indices.copy()
        random.shuffle(indices)
        count = 0
        for index in indices:
            if count >= max_per_delta:
                break

            image = np.array(mnist_test[index][0]).reshape(image_size).astype(np.float32) / 255.0
            label = mnist_test[index][1]
            mu, std = get_mu_std(prior_onnx, image)

            flat = image.flatten()
            net_input = np.array([list(flat) + list(mu)], dtype=np.float32)
            logits = get_concat_logits(concat_onnx, net_input)
            top_label = int(np.argmax(logits))
            if top_label != label:
                continue

            pred = torch.from_numpy(logits.astype(np.float32)).view(1, -1)

            pixel_lb = torch.from_numpy(flat.copy())
            pixel_ub = torch.from_numpy(flat.copy())
            latent_lb = torch.from_numpy(mu - delta * std)
            latent_ub = torch.from_numpy(mu + delta * std)
            data_lb = torch.cat([pixel_lb.flatten(), latent_lb.flatten()])
            data_ub = torch.cat([pixel_ub.flatten(), latent_ub.flatten()])

            vnnlib_name = f"prop_mnist_concat_index{index}_delta{delta:.2f}_seed{args.seed}.vnnlib"
            vnnlib_path = os.path.join(vnnlib_dir, vnnlib_name)
            write_vnnlib_classify(
                spec_path=vnnlib_path,
                data_lb=data_lb,
                data_ub=data_ub,
                prediction=pred,
                negate_spec=True,
                single_output=False,
            )
            with open(instances_path, "a") as f:
                f.write(f"onnx/{concat_basename},vnnlib/{vnnlib_name}\n")
            count += 1


if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)
    main(args)
