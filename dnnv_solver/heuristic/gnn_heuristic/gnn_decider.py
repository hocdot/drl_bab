from __future__ import annotations

import os
from typing import List

import torch
from torch import nn

from .graphnet.graph_conv import GraphNet


DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "models/cifar_trained_gnn/best_snapshot_None_0_val_acc_0.826_loss_val_0.1036_epoch_57.pt")


class GnnDecider(nn.Module):
    """
    Loads the CIFAR GraphNet checkpoint and returns a single [dec_lay, dec_idx] decision
    (same convention as GraphChoice.decision).
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.model = GraphNet(2, 64)
        ckpt = os.path.abspath(self.checkpoint_path)
        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _cumulative_lengths(self, mask_2d: List[torch.Tensor]) -> torch.Tensor:
        trans_len: list[int] = []
        t = 0
        for m in mask_2d:
            t += m.size(1)
            trans_len.append(t)
        return torch.tensor(trans_len, device=self.device)

    def _normalize_masks(self, mask: List[torch.Tensor]) -> tuple[List[torch.Tensor], torch.Tensor]:
        # decision_heuristics/domain_params convention: unstable mask from util.compute_masks
        # (typically float 0/1), with [N] or [B, N] layout.
        mask_2d: List[torch.Tensor] = []
        batch = None
        for m in mask:
            mm = m.to(device=self.device)
            if mm.ndim == 1:
                mm = mm.unsqueeze(0)
            assert mm.ndim == 2, f"Expected mask dim 1/2, got {mm.ndim}"
            batch = mm.size(0) if batch is None else batch
            assert mm.size(0) == batch, "All layer masks must share batch dimension"
            mask_2d.append(mm)
        mask_1d = torch.cat(mask_2d, dim=1)
        return mask_2d, mask_1d

    @torch.no_grad()
    def forward(
        self,
        abstractor,
        domain_params,
    ) -> list[list]:
        """
        Args:
            abstractor: network abstractor (same interface used by DecisionHeuristic).
            domain_params: AbstractResults with lower/upper bounds and masks.

        Returns:
            List of length batch:
            [[layer_name, neuron_idx, split_point], ...]
        """
        split_node_names = [n.name for n in abstractor.net.split_nodes]
        split_node_points = {k: abstractor.net.split_activations[k][0][0].get_split_point() for k in split_node_names}

        # Input + split-node bounds only (no final logits layer): branching never targets the output node.
        lower_bounds_all = [domain_params.input_lowers] + [
            domain_params.lower_bounds[k] for k in split_node_names
        ]
        upper_bounds_all = [domain_params.input_uppers] + [
            domain_params.upper_bounds[k] for k in split_node_names
        ]
        mask = [domain_params.masks[k] for k in split_node_names]
        layers = {"fixed_layers": [], "prop_layers": []}

        device = self.device
        dtype = lower_bounds_all[0].dtype

        lb = [x.to(device=device, dtype=dtype) for x in lower_bounds_all]
        ub = [x.to(device=device, dtype=dtype) for x in upper_bounds_all]
        primal_input = 0.5 * (lb[0] + ub[0])

        mask_2d, mask_1d = self._normalize_masks(mask)
        trans_len = self._cumulative_lengths(mask_2d)

        layers_dev = layers

        scores = self.model(
            lower_bounds_all=lb,
            upper_bounds_all=ub,
            dual_vars=[],
            primals=[],
            primal_inputs=primal_input,
            layers=layers_dev,
            masks=mask_1d,
            include_output_embedding=False,
        )

        decisions: list[list] = []
        for b, sb in enumerate(scores):
            if sb.numel() == 0:
                selected = False
                for lname in split_node_names:
                    candidates = (domain_params.masks[lname][b] != 0).nonzero(as_tuple=False)
                    if len(candidates):
                        decisions.append([lname, candidates[0].item(), split_node_points[lname]])
                        selected = True
                        break
                if not selected:
                    raise RuntimeError("No unstable neuron available for GNN decision")
                continue
            _, choice = torch.max(sb, 0)
            flat_idx = mask_1d[b].nonzero()[choice].item()
            dec_lay = (trans_len > flat_idx).nonzero()[0].item()
            if dec_lay == 0:
                dec_idx = flat_idx
            else:
                dec_idx = (flat_idx - trans_len[dec_lay - 1]).item()

            lname = split_node_names[dec_lay]
            decisions.append([lname, dec_idx, split_node_points[lname]])

        return decisions
