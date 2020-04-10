import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter


class PPO():
    def __init__(self,
                 actor_critic,
                 agent_i,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 model_dir=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.agent_i = agent_i
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.training_step = 0
        self.writer = SummaryWriter("/home/chenjy/new_version/logs"+model_dir)
        

    def update(self, rollouts, now_agent_num):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        #print(rollouts.rewards.mean())
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)
            for sample in data_generator:
                share_obs_batch, obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    share_obs_batch,
                    obs_batch, now_agent_num, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                self.training_step += 1
                #if rollouts.rewards.max() > 2:
                #    import pdb; pdb.set_trace()
                self.writer.add_scalars('agent%i/mean_episode_reward' % self.agent_i,
                    {'reward': rollouts.rewards.mean()},
                    self.training_step)
                self.writer.add_scalars('agent%i/value_loss' % self.agent_i,
                    {'value_loss': value_loss},
                    self.training_step)
                self.writer.add_scalars('agent%i/action_loss' % self.agent_i,
                    {'action_loss': action_loss},
                    self.training_step)
                #print("agent num: ", self.agent_i, " value_loss: ", value_loss)
                if((self.training_step+1) % 100 == 0 and self.agent_i == 0):
                    print("training_steps: " + str(self.training_step+1) + " mean rewards: " + str(rollouts.rewards.mean()))

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        #print(self.training_step)
        #import pdb; pdb.set_trace()

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
