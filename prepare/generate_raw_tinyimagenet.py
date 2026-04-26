import os
import random

import numpy as np
import torch

from arguments import get_args
from dnnv_solver.helper.network.read_onnx import parse_onnx
from dnnv_solver.helper.spec.write_vnnlib import write_vnnlib_classify

DELTAS = [1 / 255]


def create_input_bounds(
    img: torch.Tensor,
    eps: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    mean_t = mean.view(-1, 1, 1).to(device=img.device, dtype=img.dtype)
    std_t = std.view(-1, 1, 1).to(device=img.device, dtype=img.dtype)

    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32, device=img.device)
    bounds[..., 0] = (torch.clamp(img - eps, 0, 1) - mean_t) / std_t
    bounds[..., 1] = (torch.clamp(img + eps, 0, 1) - mean_t) / std_t

    return bounds.view(-1, 2)


def main(args):
    assert args.dataset == "dnnv", f"{args.dataset=}"
    assert args.input_dataset == "tinyimagenet", f"{args.input_dataset=}"

    if args.clean:
        os.system(f"rm -rf {args.dataset_dir}/{args.dataset}/io")
        os.system(f"rm -rf {args.dataset_dir}/{args.dataset}/log")
        os.system(f"rm -rf {args.dataset_dir}/{args.dataset}/graph")
        os.system(f"rm -rf {args.dataset_dir}/{args.dataset}/embed")
        os.system(f"rm -rf {args.dataset_dir}/{args.dataset}/vnnlib")
        os.system(f"rm -rf {args.dataset_dir}/{args.dataset}/vnnlib_processed")
        os.system(f"rm -rf {args.dataset_dir}/{args.dataset}/instances_*.csv")

    root_dir = os.path.join(args.dataset_dir, args.dataset)
    onnx_dir = os.path.join(root_dir, "onnx")
    vnnlib_dir = os.path.join(root_dir, "vnnlib")
    os.makedirs(vnnlib_dir, exist_ok=True)

    onnx_name = "TinyImageNet_resnet_medium.onnx"
    onnx_path = os.path.join(onnx_dir, onnx_name)
    assert os.path.isfile(onnx_path), f"Missing ONNX: {onnx_path}"
    model, input_shape, _ = parse_onnx(onnx_path)
    model.eval()

    dataset_pt = os.path.join(root_dir, "TinyImageNet_resnet_medium.pt")
    assert os.path.isfile(dataset_pt), f"Missing dataset: {dataset_pt}"

    instances_path = os.path.join(root_dir, "instances_raw.csv")
    with open(instances_path, "w"):
        pass

    mu = torch.tensor([0.4802, 0.4481, 0.3975]).view(3, 1, 1)
    std = torch.tensor([0.2302, 0.2265, 0.2262]).view(3, 1, 1)

    full_list = torch.load(dataset_pt, weights_only=False)
    all_indices = list(range(len(full_list)))
    max_per_delta = args.n_vnnlib_per_onnx

    for delta in DELTAS:
        indices = all_indices.copy()
        random.shuffle(indices)
        count = 0
        for index in indices:
            if count >= max_per_delta:
                break

            image, label, sidx, idx = full_list[index]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if isinstance(sidx, torch.Tensor):
                sidx = sidx.item()
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            x = image.unsqueeze(0).float() if image.dim() == 3 else image.float()
            with torch.no_grad():
                pred = model(x.view(input_shape))
            # if pred.argmax(-1).item() != label:
            #     continue

            image_b = x
            print(f"[{count}/{max_per_delta}] delta={delta:.6f} {index=} {sidx=} {idx=} {label=} pred={pred.argmax(-1).item()}") 

            input_bounds = create_input_bounds(image_b, delta, mean=mu, std=std)
            data_lb = input_bounds[:, 0].reshape(x.shape)
            data_ub = input_bounds[:, 1].reshape(x.shape)
            vnnlib_name = (
                f"TinyImageNet_resnet_medium_prop_idx_{idx}_sidx_{sidx}_delta_{delta:.6f}.vnnlib"
            )
            spec_path = os.path.join(vnnlib_dir, vnnlib_name)
            write_vnnlib_classify(
                spec_path=spec_path,
                data_lb=data_lb,
                data_ub=data_ub,
                prediction=pred,
                negate_spec=True,
                single_output=False,
            )

            with open(instances_path, "a") as f:
                f.write(f"onnx/{onnx_name},vnnlib/{vnnlib_name}\n")
            count += 1


if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)
    main(args)
