import random

import torch
from torchvision import transforms

from envs import VectorEnv
import networks


class Policy:
    def step(self, state):
        raise NotImplementedError

class RandomPolicy(Policy):
    def __init__(self, cfg, random_seed=None):
        self.cfg = cfg
        self.robot_group_types = [next(iter(g.keys())) for g in self.cfg.robot_config]
        if random_seed is not None:
            random.seed(random_seed)

    def step(self, state):
        action = [[None for _ in g] for g in state]
        for i, g in enumerate(state):
            robot_type = self.robot_group_types[i]
            for j, _ in enumerate(g):
                action[i][j] = random.randrange(VectorEnv.get_action_space(robot_type))
        return action

class DQNPolicy(Policy):
    def __init__(self, cfg, train=False, suffix=None, random_seed=None):
        self.cfg = cfg
        self.robot_group_types = [next(iter(g.keys())) for g in self.cfg.robot_config]
        self.train = train
        self.suffix = suffix
        if random_seed is not None:
            random.seed(random_seed)

        self.num_robot_groups = len(self.robot_group_types)
        self.transform = transforms.ToTensor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_nets = self.build_policy_nets()

        # Resume if applicable
        if self.cfg.policy_path is not None:
            self.policy_checkpoint = torch.load(self.cfg.policy_path, map_location=self.device)
            for i in range(self.num_robot_groups):
                key = 'state_dicts' if self.suffix is None else f'state_dicts_{self.suffix}'
                self.policy_nets[i].load_state_dict(self.policy_checkpoint[key][i])
                if self.train:
                    self.policy_nets[i].train()
                else:
                    self.policy_nets[i].eval()
            print("=> loaded policy '{}'".format(self.cfg.policy_path))

    def build_policy_nets(self):
        policy_nets = []
        for robot_type in self.robot_group_types:
            num_output_channels = VectorEnv.get_num_output_channels(robot_type)
            policy_nets.append(torch.nn.DataParallel(
                networks.FCN(num_input_channels=self.cfg.num_input_channels, num_output_channels=num_output_channels)
            ).to(self.device))
        return policy_nets

    def apply_transform(self, s):
        return self.transform(s).unsqueeze(0)

    def step(self, state, exploration_eps=None, debug=False):
        if exploration_eps is None:
            exploration_eps = self.cfg.final_exploration

        action = [[None for _ in g] for g in state]
        output = [[None for _ in g] for g in state]
        with torch.no_grad():
            for i, g in enumerate(state):
                robot_type = self.robot_group_types[i]
                self.policy_nets[i].eval()
                for j, s in enumerate(g):
                    if s is not None:
                        #from PIL import Image; Image.fromarray(utils.to_uint8_image(s[:, :, -1])).show()
                        s = self.apply_transform(s).to(self.device)
                        o = self.policy_nets[i](s).squeeze(0)
                        if random.random() < exploration_eps:
                            a = random.randrange(VectorEnv.get_action_space(robot_type))
                        else:
                            a = o.view(1, -1).max(1)[1].item()
                        action[i][j] = a
                        output[i][j] = o.cpu().numpy()
                if self.train:
                    self.policy_nets[i].train()

        if debug:
            info = {'output': output}
            return action, info

        return action

class MultiFreqPolicy(Policy):
    def __init__(self, cfg, policy_high=None, policy_mid=None, policy_low=None, train=False, random_seed=None):
        self.cfg = cfg
        self.policy_high = policy_high
        self.policy_mid = policy_mid
        self.policy_low = policy_low
        self.robot_group_types = [next(iter(g.keys())) for g in self.cfg.robot_config]
        self.num_robot_groups = len(self.robot_group_types)
        self.state_width = VectorEnv.get_state_width()

        num_robots = sum(sum(g.values()) for g in self.cfg.robot_config)
        assert num_robots == 1  # Multi-agent not implemented

        # Create policies if not passed in
        if self.policy_high is None:
            self.policy_high = DQNPolicy(self.cfg, train=train, suffix='high', random_seed=random_seed)
            self.policy_mid = DQNPolicy(self.cfg, train=train, suffix='mid', random_seed=(None if random_seed is None else random_seed + 1))
            self.policy_low = DQNPolicy(self.cfg, train=train, suffix='low', random_seed=(None if random_seed is None else random_seed + 2))

        self.mid_level_count = None
        self.low_level_count = None

    def step(self, state, exploration_eps=None, debug=False):
        if debug:
            info = {'levels': []}

        # First try to use policy_low
        if self.low_level_count is not None:
            if self.low_level_count == self.cfg.num_low_steps_per_mid_step:
                self.low_level_count = None
            else:
                if debug:
                    action, info_new = self.policy_low.step(state, exploration_eps=exploration_eps, debug=True)
                    info.update(info_new)
                else:
                    action = self.policy_low.step(state, exploration_eps=exploration_eps, debug=False)

                self.low_level_count += 1

                if debug:
                    info['levels'].append('l')
                    return action, info
                else:
                    return action

        # If low_level_count is None, then try to use policy_mid
        assert self.low_level_count is None
        if self.mid_level_count is not None:
            if self.mid_level_count == self.cfg.num_mid_steps_per_high_step:
                self.mid_level_count = None
            else:
                if debug:
                    action, info_new = self.policy_mid.step(state, exploration_eps=exploration_eps, debug=True)
                    info.update(info_new)
                else:
                    action = self.policy_mid.step(state, exploration_eps=exploration_eps, debug=False)

                self.mid_level_count += 1

                # Hand off to low-level
                if self.cfg.num_low_steps_per_mid_step > 0:
                    self.low_level_count = 0

                if debug:
                    info['levels'].append('m')
                    return action, info
                else:
                    return action

        # If mid_level_count and low_level_count are both None, then use policy_high
        assert self.mid_level_count is None
        if debug:
            action, info_new = self.policy_high.step(state, exploration_eps=exploration_eps, debug=True)
            info.update(info_new)
        else:
            action = self.policy_high.step(state, exploration_eps=exploration_eps, debug=False)

        # Hand off to mid-level
        if self.cfg.num_mid_steps_per_high_step > 0:
            self.mid_level_count = 0

        if debug:
            info['levels'].append('h')
            return action, info
        else:
            return action

    def reset(self):
        self.mid_level_count = None
        self.low_level_count = None
