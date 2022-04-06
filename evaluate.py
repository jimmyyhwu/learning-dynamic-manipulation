import argparse

# Prevent numpy from using up all cpu
import os
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position

import numpy as np
import utils
from policies import MultiFreqPolicy

def run_eval(cfg, num_episodes=20):
    # Check that output dir exists
    eval_dir = utils.get_eval_dir()
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True, exist_ok=True)

    random_seed = 0

    # Create env
    env = utils.get_env_from_cfg(cfg, random_seed=random_seed)

    # Create policy
    policy = MultiFreqPolicy(cfg, random_seed=random_seed)

    # Run policy
    data = [[] for _ in range(num_episodes)]
    episode_count = 0
    state = env.reset()
    while True:
        action, policy_info = policy.step(state, debug=True)
        state, _, done, info = env.step(action)
        data[episode_count].append({
            'simulation_steps': info['simulation_steps'],
            'objects': info['total_objects'],
            'policy_levels': policy_info['levels'],
        })

        if done:
            episode_count += 1
            print(f'Completed {episode_count}/{num_episodes} episodes')
            if episode_count >= num_episodes:
                break
            state = env.reset()
            policy.reset()

    env.close()

    eval_path = eval_dir / f'{cfg.run_name}.npy'
    np.save(eval_path, np.array(data, dtype=object))
    print(eval_path)

def main(args):
    config_path = args.config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is not None:
        cfg = utils.load_config(config_path)
        run_eval(cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path')
    parser.add_argument('--env-name')
    main(parser.parse_args())
