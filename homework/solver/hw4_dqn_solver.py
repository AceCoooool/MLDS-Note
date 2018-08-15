import torch
import random
from collections import deque
from model import DQN
from .base_rl_solver import RLSolverBase
from utils import pre_process


class DQNSolver(RLSolverBase):
    def __init__(self, env, model, optimizer, criterion, reward_func, config):
        super(DQNSolver, self).__init__(env, model, optimizer, criterion, reward_func, config)
        self.init_eps, self.final_eps, self.eps_step = config.init_eps, config.final_eps, config.eps_step
        self.target = DQN(in_c=config.in_c, num_actions=config.num_actions).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.batch_size, self.num_actions = config.batch_size, config.num_actions
        self.reward_mean, self.reward_list = None, deque(maxlen=config.display_interval)
        self.epsilon = self.init_eps
        if config.visdom:
            self._build_visdom()

    def _train_episode(self, episode):
        """
        training phase
        """
        state = self.env.reset()
        state = pre_process(state).repeat(self.config.in_c, 1, 1)
        loss, reward_sum, done = 0, 0, False
        while not done:
            action = random.randint(0, self.num_actions - 1) if random.random() >= 1 - self.epsilon \
                else self._make_action(state, False)
            next_state, reward, done, _ = self.env.step(action)
            next_state = pre_process(next_state)
            next_state = torch.cat([state[:3], next_state], dim=0)
            reward_sum += self.reward_func(reward)
            self.step += 1
            self.memory.append((state, next_state, torch.LongTensor([action]),
                                torch.FloatTensor([reward]), torch.FloatTensor([done])))
            state = next_state
            if self.step >= self.config.observate_time:
                loss = self._update_param()
                self.update_step += 1
                if self.update_step % self.config.update_target:
                    self.target.load_state_dict(self.model.state_dict())
            if self.step <= self.eps_step:
                self.epsilon -= (self.init_eps - self.final_eps) / self.eps_step

        self.reward_list.append(reward_sum)
        self.reward_mean = reward_sum if self.reward_mean is None else self.reward_mean * 0.99 + reward_sum * 0.01
        if self.config.visdom:
            self.visual.update_vis_line(episode - 1, [self.reward_mean], 'train', 'append')
        log = {'Episode': episode, 'Reward_cur': reward_sum, 'Reward_mean': self.reward_mean, 'Loss': loss,
               'Reward_{}'.format(self.config.display_interval): sum(self.reward_list)}
        return log

    def _update_param(self):
        if len(self.memory) < self.config.batch_size:
            return 0
        batch = random.sample(self.memory, self.batch_size)
        batch_state, batch_next, batch_action, batch_reward, batch_done = zip(*batch)
        batch_state = torch.stack(batch_state).to(self.device)
        batch_next = torch.stack(batch_next).to(self.device)
        batch_action = torch.stack(batch_action).to(self.device)
        batch_reward = torch.stack(batch_reward).to(self.device)
        batch_done = torch.stack(batch_done).to(self.device)

        current_q = self.model(batch_state).gather(1, batch_action)
        if self.config.use_double:
            next_q = batch_reward + (1 - batch_done) * self.config.gamma * self.target(batch_next).detach(). \
                gather(1, self.model(batch_next).max(-1, keepdim=True)[1].detach())
        else:
            next_q = batch_reward + (1 - batch_done) * self.config.gamma * \
                     self.target(batch_next).detach().max(-1, keepdim=True)[0]

        self.optimizer.zero_grad()
        loss = self.criterion(current_q, next_q)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _make_action(self, state, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: torch.FloatTensor
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        action = self.model(state.unsqueeze(0).to(self.device)).max(-1)[1].item()
        return action

    def _build_visdom(self):
        from utils import Visdom
        self.visual = Visdom(1)
        self.visual.create_vis_line('episode', 'reward', 'average reward with episode', ['reward'], 'train')
