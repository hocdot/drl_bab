from beartype import beartype
import torch

from ..util import check_solution
from .general import attack

class Attacker:

    @beartype
    def __init__(self, net: torch.nn.Module, objective, input_shape: tuple, device: str = 'cpu') -> None:
        self.net = net
        self.objective = objective
        self.input_shape = input_shape
        self.device = device

    @beartype
    def run(self, iterations: int = int(1e6), restarts: int = 20, timeout: float = 2.0) -> tuple[bool, torch.Tensor | None]:
        timeout = timeout / 2 # since PGD run twice
        data_min = self.objective.lower_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        data_max = self.objective.upper_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)

        x = (data_max[:, 0] - data_min[:, 0]) * torch.rand(data_min[:, 0].shape, device=self.device) + data_min[:, 0]
        assert torch.all(x <= data_max[:, 0])
        assert torch.all(x >= data_min[:, 0])

        cs = self.objective.cs.to(self.device)
        rhs = self.objective.rhs.to(self.device)

        self.net.to(cs.dtype)
        is_attacked, attack_images = attack(
            model=self.net,
            x=x.to(cs.dtype),
            data_min=data_min,
            data_max=data_max,
            cs=cs,
            rhs=rhs,
            attack_iters=iterations,
            num_restarts=restarts,
            timeout=timeout,
        )

        if is_attacked:
            with torch.no_grad():
                for i in range(attack_images.shape[1]): # restarts
                    for j in range(attack_images.shape[2]): # props
                        adv = attack_images[:, i, j]
                        if check_solution(self.net, adv, cs=cs, rhs=rhs, data_min=data_min, data_max=data_max):
                            return True, adv

        self.net.to(cs.dtype)
        return False, None

    def __repr__(self):
        return f'PGDAttack(device={self.device})'
