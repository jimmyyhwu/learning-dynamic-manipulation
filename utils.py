import inspect
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from matplotlib import cm
from munch import Munch
from PIL import Image
from prompt_toolkit.shortcuts import radiolist_dialog
from skimage.draw import circle_perimeter

from envs import VectorEnv

################################################################################
# Experiment management

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = Munch.fromYAML(f)
    return cfg

def save_config(config_path, cfg):
    with open(config_path, 'w') as f:
        f.write(cfg.toYAML())

def get_logs_dir():
    return Path('logs')

def get_checkpoints_dir():
    return Path('checkpoints')

def get_eval_dir():
    return Path('eval')

def setup_run(config_path):
    cfg = load_config(config_path)

    if cfg.log_dir is not None:
        # Run has already been set up
        return config_path

    assert cfg.experiment_name == Path(config_path).with_suffix('').name

    # Get root directories
    logs_dir = get_logs_dir() if cfg.logs_dir is None else Path(cfg.logs_dir)
    checkpoints_dir = get_checkpoints_dir() if cfg.checkpoints_dir is None else Path(cfg.checkpoints_dir)

    # Set up run_name, log_dir, and checkpoint_dir
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S%f')
    cfg.run_name = '{}-{}'.format(timestamp, cfg.experiment_name)
    log_dir = logs_dir / cfg.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir = str(log_dir)
    cfg.checkpoint_dir = str(checkpoints_dir / cfg.run_name)

    # Save config file for the new run
    config_path = log_dir / 'config.yml'
    save_config(config_path, cfg)

    return config_path

def select_run():
    logs_dir = get_logs_dir()
    log_dirs = [x for x in sorted(logs_dir.iterdir()) if x.is_dir()]
    if len(log_dirs) == 0:
        return None

    grouped_config_paths = {}
    for log_dir in log_dirs:
        parts = log_dir.name.split('-')
        experiment_name = '-'.join(parts[1:])
        if experiment_name not in grouped_config_paths:
            grouped_config_paths[experiment_name] = []
        grouped_config_paths[experiment_name].append(log_dir / 'config.yml')

    if len(grouped_config_paths) > 1:
        config_paths = radiolist_dialog(
            values=[(value, key) for key, value in sorted(grouped_config_paths.items())],
            text='Please select an experiment:').run()
        if config_paths is None:
            return None
    else:
        config_paths = next(iter(grouped_config_paths.values()))

    selected_config_path = radiolist_dialog(
        values=[(path, path.parent.name) for path in config_paths],
        text='Please select a run:').run()
    if selected_config_path is None:
        return None

    return selected_config_path

################################################################################
# Visualization

JET = np.array([list(cm.jet(i)[:3]) for i in range(256)], dtype=np.float32)
np.seterr(all='raise')  # Raise exception when divide by zero occurs

def to_uint8_image(image):
    return np.round(255.0 * image).astype(np.uint8)

def scale_min_max(image):
    assert image.dtype == np.float32
    return (image - image.min()) / (image.max() - image.min() + 1e-6)

def get_state_visualization(state):
    assert state.dtype == np.float32
    if state.shape[2] == 1:
        return np.stack((state[:, :, 0], state[:, :, 0], state[:, :, 0]), axis=2)  # (overhead map, overhead map, overhead map)
    if state.shape[2] == 2:
        return np.stack((state[:, :, 1], state[:, :, 0], state[:, :, 0]), axis=2)  # (robot map, overhead map, overhead map)
    return np.stack((state[:, :, 1], state[:, :, 0], state[:, :, -1]), axis=2)  # (robot map, overhead map, last added channel)

def get_overhead_image(state):
    assert state.dtype == np.float32
    return np.stack([state[:, :, 0], state[:, :, 0], state[:, :, 0]], axis=2)

def get_output_visualization(overhead_image, output, alpha=0.5):
    assert overhead_image.dtype == np.float32
    assert output.dtype == np.uint8
    return (1 - alpha) * overhead_image + alpha * JET[output, :]  # np.float32

def get_state_output_visualization(state, output):
    panels = []
    vertical_bar = np.zeros((state.shape[1], 1, 3), dtype=np.float32)

    # State
    panels.append(get_state_visualization(state))
    panels.append(vertical_bar)

    # Output
    overhead_image = get_overhead_image(state)
    output = to_uint8_image(scale_min_max(output))
    for i, channel in enumerate(output):
        panels.append(get_output_visualization(overhead_image, channel))
        if i < len(output) - 1:
            panels.append(vertical_bar)
    return np.concatenate(panels, axis=1)  # np.float32

def enlarge_image(image, scale_factor=4):
    return image.resize((scale_factor * image.size[0], scale_factor * image.size[1]), resample=Image.NEAREST)  # PIL.Image

def get_reward_image(reward, state_width, reward_image_height=12):
    reward_image = np.zeros((reward_image_height, state_width, 3), dtype=np.float32)
    text = '{:+.02f}'.format(reward)
    cv2.putText(reward_image, text, (state_width - 5 * len(text), 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (1, 1, 1))
    return reward_image  # np.float32

def get_transition_visualization(state=None, action=None, reward=0):
    state_width = VectorEnv.get_state_width()
    if state is None:
        state = np.zeros((state_width, state_width, 3), dtype=np.float32)
    state_vis = get_state_visualization(state)
    if action is not None:
        i, j = np.unravel_index(action % (state.shape[0] * state.shape[1]), (state.shape[0], state.shape[1]))
        color = (1, 0, 0) if action < state_width * state_width else (0.5, 0, 0)
        rr, cc = circle_perimeter(i, j, 2)
        state_vis[rr, cc, :] = color
    reward_image = get_reward_image(reward, state_vis.shape[1])
    return np.concatenate((reward_image, state_vis), axis=0)

################################################################################
# Environment

def apply_misc_env_modifications(cfg_or_kwargs, env_name):
    # Also add assertions in env code for redundancy

    # Room size
    if env_name.startswith('large'):
        cfg_or_kwargs['room_length'] = 1.0
        cfg_or_kwargs['room_width'] = 1.0
        cfg_or_kwargs['num_objects'] = 100
    else:
        cfg_or_kwargs['room_length'] = 1.0
        cfg_or_kwargs['room_width'] = 0.5
        cfg_or_kwargs['num_objects'] = 50

def sanity_check_config(cfg):
    # Discount factors
    assert len(cfg.discount_factors) == len(cfg.robot_config)

    # Number of input channels
    expected_num_input_channels = 1
    if cfg.use_robot_map:
        expected_num_input_channels += 1
    if cfg.use_distance_to_receptacle_map:
        expected_num_input_channels += 1
    if cfg.use_shortest_path_map:
        expected_num_input_channels += 1
    if cfg.use_shortest_path_to_receptacle_map:
        assert not cfg.use_distance_to_receptacle_map
        expected_num_input_channels += 1
    assert cfg.num_input_channels == expected_num_input_channels, (cfg.num_input_channels, expected_num_input_channels)

def get_env_from_cfg(cfg, **kwargs):
    sanity_check_config(cfg)

    args_to_ignore = {'self', 'random_seed',
        'show_trajectories', 'show_debug_annotations', 'show_occupancy_maps',
        'real', 'real_robot_indices', 'real_debug'}
    final_kwargs = {}
    for arg_name in inspect.getfullargspec(VectorEnv.__init__).args:
        if arg_name in args_to_ignore:
            continue
        if arg_name in cfg:
            final_kwargs[arg_name] = cfg[arg_name]
        else:
            raise Exception('kwarg {} not found in config'.format(arg_name))
    final_kwargs.update(kwargs)

    # Additional modifications for real robot
    if 'real' in final_kwargs:
        final_kwargs['show_gui'] = True
        final_kwargs['show_debug_annotations'] = True

        # Remove randomness from obstacle placement
        if final_kwargs['env_name'] in {'small_divider', 'large_doors', 'large_tunnels', 'large_rooms'}:
            final_kwargs['env_name'] = '{}_norand'.format(final_kwargs['env_name'])

    return VectorEnv(**final_kwargs)
