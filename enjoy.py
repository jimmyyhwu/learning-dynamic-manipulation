import argparse
import cv2
import utils
from policies import MultiFreqPolicy

def main(args):
    config_path = args.config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is None:
        return
    print(config_path)
    cfg = utils.load_config(config_path)

    if args.debug:
        cv2.namedWindow('out', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('out', 3 * 384, 384)

    # Create env
    if args.real:
        real_robot_indices = list(map(int, args.real_robot_indices.split(',')))
        env = utils.get_env_from_cfg(cfg, real=True, real_robot_indices=real_robot_indices)
    else:
        env = utils.get_env_from_cfg(cfg, show_gui=True)

    # Create policy
    policy = MultiFreqPolicy(cfg)

    # Run policy
    state = env.reset()
    try:
        while True:
            if args.debug:
                action, info = policy.step(state, debug=True)
                cv2.imshow('out', utils.get_state_output_visualization(state[0][0], info['output'][0][0])[:, :, ::-1])
                cv2.waitKey(1)
            else:
                action = policy.step(state)

            state, _, done, _ = env.step(action)
            if done:
                state = env.reset()
                policy.reset()
    finally:
        env.close()

parser = argparse.ArgumentParser()
parser.add_argument('--config-path')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--real', action='store_true')
parser.add_argument('--real-robot-indices', default='0')
main(parser.parse_args())
