from . import pgd

def create(net, objectives, input_shape, args):
    return eval(args.attacker).Attacker(net, objectives, input_shape, args.device)
