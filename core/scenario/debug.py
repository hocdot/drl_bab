from neuralsat.heuristic.util import compute_masks
import neuralsat as ns
import torch

def debug(args):
    #
    onnx_path = '../data/benchmark/cifar/onnx/0.onnx'
    vnnlib_path = '../data/benchmark/cifar/vnnlib/0.vnnlib'
    net, input_shape, objs = ns.test.extract_instance(onnx_path, vnnlib_path)
    iv = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device=args.device)
    obj = objs.pop(4)
    obj = objs.pop(1)
    obj_id = 4

    #######################
    # EXTRACT NODE FEATURES
    #######################
    # 1. Input Features
    # -----------------
    sample = iv.get_sample_observation(obj)
    lb = sample.input_lowers.cpu()
    ub = sample.input_uppers.cpu()
    bias = torch.zeros_like(lb).cpu()
    mask = torch.zeros_like(lb).cpu()
    x = torch.stack([lb, ub, bias, mask], dim=-1)[0]
    print(f'input: {x.shape=}')

    # 2. Hidden Features
    # ------------------
    # ordered name of layers
    ordered_names = [_.name for _ in iv.abstractor.net.split_nodes]
    # extract mask
    masks = compute_masks(sample.lower_bounds, sample.upper_bounds, device='cpu')
    # extract hidden
    for i, name in enumerate(ordered_names):
        lb = sample.lower_bounds[name].cpu()
        ub = sample.upper_bounds[name].cpu()
        bias = iv.abstractor.net[name].inputs[-1].param.detach().cpu()
        if lb.ndim == 4: # conv layer
            bs, c, h, w = lb.shape
            bias = bias[None, : , None, None]
            bias = bias.repeat(bs, 1, h, w)
        elif lb.ndim == 2: # fc layer
            bias = bias[None]
        else:
            raise NotImplementedError
        mask = masks[name].view(lb.shape)
        assert (lb[torch.where(mask == 1)] < 0).all()
        assert (ub[torch.where(mask == 1)] > 0).all()
        x = torch.stack([lb, ub, bias, mask], dim=-1)[0]
        print(f'hidden: {name=} {x.shape=}')
        # print(f'{i=} {name=} {lb.shape=} {ub.shape=} {bias.shape=} {mask.shape=}')

    # 3. Output Features
    # ------------------
    lb = sample.output_lbs.cpu()
    ub = torch.zeros_like(lb).cpu()
    bias = iv.abstractor.net.final_node().inputs[-1].param.detach().cpu()[obj_id][None, None]
    mask = torch.zeros_like(lb).cpu()
    x = torch.stack([lb, ub, bias, mask], dim=-1)[0]
    print(f'output: {name=} {x.shape=}')
