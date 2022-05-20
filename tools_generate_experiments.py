from pathlib import Path
import utils

def generate_experiment(experiment_name, template_experiment_name, modify_cfg_fn, output_dir, template_dir='config/experiments/base'):
    # Ensure output dir exists
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Read template config
    cfg = utils.load_config(Path(template_dir) / '{}.yml'.format(template_experiment_name))

    # Apply modifications
    cfg.experiment_name = experiment_name
    num_fields = len(cfg)
    modify_cfg_fn(cfg)
    assert num_fields == len(cfg), experiment_name  # New fields should not have been added

    # Save new config
    utils.save_config(output_dir / '{}.yml'.format(experiment_name), cfg)

def get_discount_factors(robot_config, offset=0):
    discount_factor_list = [0.2, 0.35, 0.5, 0.65, 0.75, 0.85, 0.9]  # Raise to power 1.5
    start_indices = {
        'pushing_robot': 4,
        'blowing_robot': 4,
        'moving_blowing_robot': 4,
        'side_blowing_robot': 4,
    }
    num_robots = sum(next(iter(g.values())) for g in robot_config)
    robot_group_types = [next(iter(g.keys())) for g in robot_config]
    discount_factors = []
    for robot_type in robot_group_types:
        idx = start_indices[robot_type]
        if num_robots > 1:
            idx += 1
        idx += offset
        discount_factors.append(discount_factor_list[idx])
    return discount_factors

assert get_discount_factors([{'pushing_robot': 1}]) == [0.75]
assert get_discount_factors([{'blowing_robot': 1}]) == [0.75]

def main():
    ################################################################################
    # Robot types

    def modify_cfg_pushing_to_pushing(cfg):
        cfg.env_name = 'small_empty'
        cfg.discount_factors = get_discount_factors(cfg.robot_config)
        cfg.total_timesteps = 60000
        cfg.obstacle_collision_penalty = 0

    def modify_cfg_pushing_to_blowing(cfg):
        cfg.robot_config = [{'blowing_robot': 1}]
        cfg.env_name = 'small_empty'
        cfg.discount_factors = get_discount_factors(cfg.robot_config)
        cfg.total_timesteps = 20000

    def modify_cfg_pushing_to_moving_blowing(cfg):
        modify_cfg_pushing_to_blowing(cfg)
        cfg.robot_config = [{'moving_blowing_robot': 1}]

    def modify_cfg_pushing_to_side_blowing(cfg):
        modify_cfg_pushing_to_blowing(cfg)
        cfg.robot_config = [{'side_blowing_robot': 1}]

    output_dir = 'config/experiments/base'
    generate_experiment('pushing_1-small_empty-base', 'pushing_1-small_empty', modify_cfg_pushing_to_pushing, output_dir, template_dir='config/templates')
    generate_experiment('blowing_1-small_empty-base', 'pushing_1-small_empty', modify_cfg_pushing_to_blowing, output_dir, template_dir='config/templates')
    generate_experiment('moving_blowing_1-small_empty-base', 'pushing_1-small_empty', modify_cfg_pushing_to_moving_blowing, output_dir, template_dir='config/templates')
    generate_experiment('side_blowing_1-small_empty-base', 'pushing_1-small_empty', modify_cfg_pushing_to_side_blowing, output_dir, template_dir='config/templates')

    ################################################################################
    # Config for local development

    def modify_cfg_to_local(cfg):
        cfg.logs_dir = 'logs'
        cfg.checkpoints_dir = 'checkpoints'
        cfg.batch_size = 4
        cfg.replay_buffer_size = 1000
        cfg.learning_starts_frac = 0.00025
        cfg.inactivity_cutoff_per_robot = 5
        cfg.show_gui = True
        cfg.num_parallel_collectors = None

    output_dir = 'config/local'
    for template_experiment_name in [
            'pushing_1-small_empty-base',
            'blowing_1-small_empty-base',
        ]:
        experiment_name = template_experiment_name.replace('base', 'local')
        generate_experiment(experiment_name, template_experiment_name, modify_cfg_to_local, output_dir)

    ################################################################################
    # Environments

    def modify_cfg_env_name(cfg, env_name):
        cfg.env_name = env_name
        utils.apply_misc_env_modifications(cfg, env_name)

    output_dir = 'config/experiments/base'
    for template_experiment_name in [
            'pushing_1-small_empty-base',
            'blowing_1-small_empty-base',
            'moving_blowing_1-small_empty-base',
            'side_blowing_1-small_empty-base',
        ]:
        for env_name in ['large_empty', 'large_columns', 'large_door', 'large_center']:
            experiment_name = template_experiment_name.replace('small_empty', env_name)
            generate_experiment(experiment_name, template_experiment_name, lambda x: modify_cfg_env_name(x, env_name), output_dir)

    ################################################################################
    # Multi-frequency

    def modify_cfg_to_singlefreq(cfg, num_mid_steps_per_high_step=4, num_low_steps_per_mid_step=0):
        factor = max(1, num_mid_steps_per_high_step, num_mid_steps_per_high_step * num_low_steps_per_mid_step)
        cfg.target_update_freq *= factor
        cfg.total_timesteps *= factor
        cfg.train_freq *= factor

    def modify_cfg_to_multifreq(cfg, num_mid_steps_per_high_step=4, num_low_steps_per_mid_step=0):
        cfg.num_mid_steps_per_high_step = num_mid_steps_per_high_step
        cfg.num_low_steps_per_mid_step = num_low_steps_per_mid_step
        if num_low_steps_per_mid_step > 0:
            cfg.total_timesteps *= num_mid_steps_per_high_step * num_low_steps_per_mid_step
            cfg.target_update_freq *= num_mid_steps_per_high_step * num_low_steps_per_mid_step
            cfg.train_freq *= num_mid_steps_per_high_step * num_low_steps_per_mid_step
        else:
            assert num_low_steps_per_mid_step == 0
            cfg.total_timesteps *= num_mid_steps_per_high_step
            cfg.target_update_freq *= num_mid_steps_per_high_step
            cfg.train_freq *= num_mid_steps_per_high_step

    for template_experiment_name in [
            'pushing_1-small_empty-base',
            'pushing_1-large_empty-base',
            'pushing_1-large_columns-base',
            'pushing_1-large_door-base',
            'pushing_1-large_center-base',
            'blowing_1-small_empty-base',
            'blowing_1-large_empty-base',
            'blowing_1-large_columns-base',
            'blowing_1-large_door-base',
            'blowing_1-large_center-base',
            'moving_blowing_1-small_empty-base',
            'moving_blowing_1-large_empty-base',
            'moving_blowing_1-large_columns-base',
            'moving_blowing_1-large_door-base',
            'moving_blowing_1-large_center-base',
            'side_blowing_1-small_empty-base',
            'side_blowing_1-large_empty-base',
            'side_blowing_1-large_columns-base',
            'side_blowing_1-large_door-base',
            'side_blowing_1-large_center-base',
        ]:
        experiment_name = template_experiment_name.replace('base', 'singlefreq_4')
        generate_experiment(experiment_name, template_experiment_name, modify_cfg_to_singlefreq, 'config/experiments/singlefreq')
        if not template_experiment_name.startswith('pushing'):
            experiment_name = template_experiment_name.replace('base', 'multifreq_4')
            generate_experiment(experiment_name, template_experiment_name, modify_cfg_to_multifreq, 'config/experiments/multifreq')

    ################################################################################
    # 3-level multi-frequency

    for template_experiment_name in [
            'blowing_1-small_empty-base',
            'blowing_1-large_empty-base',
            'blowing_1-large_columns-base',
            'blowing_1-large_door-base',
            'blowing_1-large_center-base',
        ]:
        experiment_name = template_experiment_name.replace('base', 'multifreq_4_4')
        generate_experiment(experiment_name, template_experiment_name, lambda x: modify_cfg_to_multifreq(x, 4, 4), 'config/experiments/multifreq-3level')

    ################################################################################
    # Blowing force

    def modify_cfg_blowing_force(cfg, blowing_force):
        cfg.blowing_force = blowing_force

    for template_experiment_name in [
            'blowing_1-small_empty-multifreq_4',
            'blowing_1-large_empty-multifreq_4',
            'blowing_1-large_columns-multifreq_4',
            'blowing_1-large_door-multifreq_4',
            'blowing_1-large_center-multifreq_4',
        ]:
        for blowing_force in [0.2, 0.5, 0.65]:
            experiment_name = f'{template_experiment_name}-blowforce_{blowing_force}'
            generate_experiment(experiment_name, template_experiment_name, lambda x: modify_cfg_blowing_force(x, blowing_force), 'config/experiments/blowforce', template_dir='config/experiments/multifreq')

main()
