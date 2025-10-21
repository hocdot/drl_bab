import torch.nn.functional as F
import torch
import os

from .replay_buffer import TrainingData

def print_gradients(self, network_name, networks):
    """Print gradient statistics to check for vanishing/exploding gradients."""
    return

    if self.global_step % 5000 or self.global_step == 0:
        return
    
    print(f"\n=== {network_name} Gradient Stats (Step {self.global_step}) ===")
    for i, network in enumerate(networks):
        network_suffix = f"_{i+1}" if len(networks) > 1 else ""
        print(f"{network_name}{network_suffix}:")
        
        grad_norms = []
        for name, param in network.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if grad_norm < 1e-6:
                    print(f"  {name}: {grad_norm:.2e} (VANISHING)")
                elif grad_norm > 10.0:
                    print(f"  {name}: {grad_norm:.2e} (EXPLODING)")
                else:
                    print(f"  {name}: {grad_norm:.6f}")
            else:
                print(f"  {name}: None (no gradient)")
        
        if grad_norms:
            mean_grad = sum(grad_norms) / len(grad_norms)
            max_grad = max(grad_norms)
            min_grad = min(grad_norms)
            print(f"  Summary - Mean: {mean_grad:.6f}, Max: {max_grad:.6f}, Min: {min_grad:.6f}")
        print()
        
def optimize_qf(self, data):
    # train
    with torch.no_grad():
        # 1st branch
        _, next_log_prob, next_action_probs = self.actor.decode_and_get_action(data.next_observation[:, 0], data.action[:, 0])
        qv1_target      = self.qf1_target.decode_and_forward(data.next_observation[:, 0], data.action[:, 0])
        qv2_target      = self.qf2_target.decode_and_forward(data.next_observation[:, 0], data.action[:, 0])
        min_qv_target_1 = next_action_probs * (torch.min(qv1_target, qv2_target) - self.alpha * next_log_prob)
        # min_qv_target_1 = next_action_probs * (torch.min(qv1_target, qv2_target))
        min_qv_target_1 = min_qv_target_1.sum(dim=1)
        # 2nd branch
        _, next_log_prob, next_action_probs = self.actor.decode_and_get_action(data.next_observation[:, 1], data.action[:, 1])
        qv1_target      = self.qf1_target.decode_and_forward(data.next_observation[:, 1], data.action[:, 1])
        qv2_target      = self.qf2_target.decode_and_forward(data.next_observation[:, 1], data.action[:, 1])
        min_qv_target_2 = next_action_probs * (torch.min(qv1_target, qv2_target) - self.alpha * next_log_prob)
        # min_qv_target_2 = next_action_probs * (torch.min(qv1_target, qv2_target))
        min_qv_target_2 = min_qv_target_2.sum(dim=1)
        # add to next qv
        next_qv = data.reward + (1 - data.done[:, 0]) * self.args.gamma * (min_qv_target_1) + (1 - data.done[:, 1]) * self.args.gamma * (min_qv_target_2)
    # print(f'{data.observation.shape=} {data.prev_action.shape=}')
    qv1 = self.qf1.decode_and_forward(data.observation, data.prev_action)
    qv2 = self.qf2.decode_and_forward(data.observation, data.prev_action)
    # print(f'{qv1.shape=} {qv2.shape=} {data.action.shape=}')
    qv1_a = qv1.gather(1, data.action[:, 0][:, None]).squeeze()
    qv2_a = qv2.gather(1, data.action[:, 0][:, None]).squeeze()

    # compute loss
    qf1_loss = F.mse_loss(qv1_a, next_qv)
    qf2_loss = F.mse_loss(qv2_a, next_qv)
    qf_loss = qf1_loss + qf2_loss
    self.q_optimizer.zero_grad()
    qf_loss.backward()
    # print grad to check gradient vanishing/exploding
    self.print_gradients("Q-Networks", [self.qf1, self.qf2])
    
    self.q_optimizer.step()
    return qf_loss.item()

def optimize_actor(self, data):
    alpha_loss = 0.0
    # train
    _, log_prob, action_probs = self.actor.decode_and_get_action(data.observation, data.prev_action)
    with torch.no_grad():
        qv1 = self.qf1.decode_and_forward(data.observation, data.prev_action)
        qv2 = self.qf2.decode_and_forward(data.observation, data.prev_action)
        min_qv = torch.min(qv1, qv2)
    actor_loss = (action_probs * ((self.alpha * log_prob) - min_qv)).mean()
    # print(f'{action_probs.shape=} {min_qv.shape=}')
    # actor_loss = (action_probs * (- min_qv)).mean()
    self.actor_optimizer.zero_grad()
    self.actor_scaler.scale(actor_loss).backward()
    self.actor_scaler.step(self.actor_optimizer)
    self.actor_scaler.update()
    
    # print grad to check gradient vanishing/exploding
    self.print_gradients("Actor", [self.actor])
        
    actor_loss = actor_loss.item()
    # train
    if self.args.autotune_entropy:
        # reuse action probabilities for temperature loss
        alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_prob + self.target_entropy).detach())).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        alpha_loss = alpha_loss.item()
    return actor_loss, alpha_loss

def update_target(self):
    # update
    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

def sample_batch(self):
    transitions = self.replay_buffer.sample(self.args.batch_size)
    data = TrainingData(transitions, self.reward_scaler, self.args)
    
    if self.args.save_offline_data:
        offline_data_dir = os.path.join(self.args.dataset_dir, self.args.dataset, 'offline_data')
        # offline_data_dir = os.path.join('/media/hai/offline_data')
        
        os.makedirs(offline_data_dir, exist_ok=True)
        data.export(os.path.join(offline_data_dir, f'step_{self.global_step}_batch_{self.args.batch_size}.pt'))
    return data

def optimize(self, data=None, n_qf_optimize=1):
    self.actor.train()
    self.qf1.train()
    self.qf2.train()
    # print(f'[+] training data')
    # print(f'    - {data.prev_action.shape=}')      # n_subproblem
    # print(f'    - {data.observation.shape=}')      # n_subproblem, n_node, n_feature
    # print(f'    - {data.action.shape=}')           # n_subproblem, 2
    # print(f'    - {data.next_observation.shape=}') # n_subproblem, 2, n_node, n_feature
    # print(f'    - {data.done.shape=}')             # n_subproblem, 2
    # print(f'    - {data.reward.shape=}')           # n_subproblem
    flag = data is None
    with torch.autograd.set_detect_anomaly(True):
        qf_loss = 0.0
        for _ in range(n_qf_optimize):
            if flag:
                data = self.sample_batch()
            qf_loss += self.optimize_qf(data)
        qf_loss /= n_qf_optimize
        actor_loss, alpha_loss = self.optimize_actor(data)
    if self.global_step % self.args.n_target_update == 0:
        self.update_target()
    return actor_loss, qf_loss, alpha_loss

