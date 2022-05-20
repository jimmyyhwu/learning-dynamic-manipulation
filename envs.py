import math
from abc import ABC, abstractmethod
from multiprocessing.connection import Client
from pathlib import Path
from pprint import pprint

import anki_vector
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
from scipy.ndimage import rotate as rotate_image
from scipy.ndimage.morphology import distance_transform_edt
from shapely.geometry import box
from shapely.ops import unary_union
from skimage.draw import line
from skimage.morphology import binary_dilation, dilation
from skimage.morphology.selem import disk

import vector_utils
from shortest_paths.shortest_paths import GridGraph


class VectorEnv:
    WALL_HEIGHT = 0.1
    RECEPTACLE_WIDTH = 0.15
    IDENTITY_QUATERNION = (0, 0, 0, 1)
    REMOVED_BODY_Z = -1000  # Hide removed bodies 1000 m below
    OBJECT_COLOR = (237.0 / 255, 201.0 / 255, 72.0 / 255, 1)  # Yellow
    DEBUG_LINE_COLORS = [
        (78.0 / 255, 121.0 / 255, 167.0 / 255),  # Blue
        (89.0 / 255, 169.0 / 255, 79.0 / 255),  # Green
        (176.0 / 255, 122.0 / 255, 161.0 / 255),  # Purple
        (242.0 / 255, 142.0 / 255, 43.0 / 255),  # Orange
    ]

    def __init__(
        # This comment is here to make code folding work
            self,
            robot_config=None, room_length=1.0, room_width=0.5,
            num_objects=10, object_type=None, object_width=0.012, object_mass=0.00009,
            env_name='small_empty',
            slowing_sim_step_target=50, blowing_sim_step_target=100,
            blowing_fov=15, blowing_num_wind_particles=40, blowing_wind_particle_sparsity=2,
            blowing_wind_particle_radius=0.003, blowing_wind_particle_mass=0.001, blowing_force=0.35,
            overhead_map_scale=1.0,
            use_robot_map=True, robot_map_scale=1.0,
            use_distance_to_receptacle_map=False, distance_to_receptacle_map_scale=0.25,
            use_shortest_path_to_receptacle_map=True, use_shortest_path_map=True, shortest_path_map_scale=0.25,
            use_shortest_path_partial_rewards=True, success_reward=1.0,
            partial_rewards_scale=2.0,
            obstacle_collision_penalty=0.25, robot_collision_penalty=1.0,
            use_shortest_path_movement=True, use_partial_observations=True,
            inactivity_cutoff_per_robot=100,
            random_seed=None,
            show_gui=False, show_trajectories=False, show_debug_annotations=False, show_occupancy_maps=False,
            real=False, real_robot_indices=None, real_debug=False,
        ):

        ################################################################################
        # Arguments

        # Room configuration
        self.robot_config = robot_config
        self.room_length = room_length
        self.room_width = room_width
        self.num_objects = num_objects
        self.object_type = object_type
        self.object_width = object_width
        self.object_mass = object_mass
        self.env_name = env_name

        # Robot configuration
        self.slowing_sim_step_target = slowing_sim_step_target
        self.blowing_sim_step_target = blowing_sim_step_target
        self.blowing_fov = blowing_fov
        self.blowing_num_wind_particles = blowing_num_wind_particles
        self.blowing_wind_particle_sparsity = blowing_wind_particle_sparsity
        self.blowing_wind_particle_radius = blowing_wind_particle_radius
        self.blowing_wind_particle_mass = blowing_wind_particle_mass
        self.blowing_force = blowing_force

        # State representation
        self.overhead_map_scale = overhead_map_scale
        self.use_robot_map = use_robot_map
        self.robot_map_scale = robot_map_scale
        self.use_distance_to_receptacle_map = use_distance_to_receptacle_map
        self.distance_to_receptacle_map_scale = distance_to_receptacle_map_scale
        self.use_shortest_path_to_receptacle_map = use_shortest_path_to_receptacle_map
        self.use_shortest_path_map = use_shortest_path_map
        self.shortest_path_map_scale = shortest_path_map_scale

        # Rewards
        self.use_shortest_path_partial_rewards = use_shortest_path_partial_rewards
        self.success_reward = success_reward
        self.partial_rewards_scale = partial_rewards_scale
        self.obstacle_collision_penalty = obstacle_collision_penalty
        self.robot_collision_penalty = robot_collision_penalty

        # Misc
        self.use_shortest_path_movement = use_shortest_path_movement
        self.use_partial_observations = use_partial_observations
        self.inactivity_cutoff_per_robot = inactivity_cutoff_per_robot
        self.random_seed = random_seed

        # Debugging
        self.show_gui = show_gui
        self.show_trajectories = show_trajectories
        self.show_debug_annotations = show_debug_annotations
        self.show_occupancy_maps = show_occupancy_maps

        # Real environment
        self.real = real
        self.real_robot_indices = real_robot_indices
        self.real_debug = real_debug

        pprint(self.__dict__)

        ################################################################################
        # Set up pybullet

        if self.show_gui:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
            self.p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self.p.resetDebugVisualizerCamera(
            0.47 + (5.25 - 0.47) / (10 - 0.7) * (self.room_length - 0.7), 0, -70,
            (0, -(0.07 + (1.5 - 0.07) / (10 - 0.7) * (self.room_width - 0.7)), 0))

        # Used to determine whether robot poses are out of date
        self.step_simulation_count = 0

        ################################################################################
        # Robots and room configuration

        # Random placement of robots, objects, and obstacles
        self.room_random_state = np.random.RandomState(self.random_seed)
        self.robot_spawn_bounds = None
        self.object_spawn_bounds = None

        # Robots
        if self.robot_config is None:
            self.robot_config = [{'pushing_robot': 1}]
        self.num_robots = sum(sum(g.values()) for g in self.robot_config)
        self.robot_group_types = [next(iter(g.keys())) for g in self.robot_config]
        self.robot_ids = None
        self.robots = None
        self.robot_groups = None
        self.last_robot_index = None
        self.robot_random_state = np.random.RandomState(self.random_seed + 1 if self.random_seed is not None else None)  # Add randomness to blowing

        # Room
        self.obstacle_ids = None
        self.object_ids = None
        self.receptacle_id = None
        if self.env_name == 'large_center':
            self.receptacle_position = (0, 0, 0)
        else:
            self.receptacle_position = (self.room_length / 2 - VectorEnv.RECEPTACLE_WIDTH / 2, self.room_width / 2 - VectorEnv.RECEPTACLE_WIDTH / 2, 0)

        # Collections for keeping track of environment state
        self.obstacle_collision_body_b_ids_set = None  # For collision detection
        self.robot_collision_body_b_ids_set = None  # For collision detection
        self.available_object_ids_set = None  # Excludes removed objects
        self.removed_object_ids_set = None  # Objects that have been removed

        ################################################################################
        # Misc

        # End an episode after too many steps of inactivity
        self.inactivity_cutoff = self.num_robots * self.inactivity_cutoff_per_robot

        # Stats
        self.steps = None
        self.simulation_steps = None
        self.inactivity_steps = None

        ################################################################################
        # Real environment

        if self.real:
            assert len(self.real_robot_indices) == self.num_robots
            self.real_robot_indices_map = None
            self.num_objects = 0  # When running in real environment, do not simulate objects in PyBullet
            self.object_mask = None  # Mask showing locations of objects in the real environment

            # Connect to aruco server for pose estimates
            address = 'localhost'
            if self.env_name.startswith('large'):
                # Left camera, right camera
                self.conns = [Client((address, 6001), authkey=b'secret password'), Client((address, 6002), authkey=b'secret password')]
            else:
                self.conns = [Client((address, 6000), authkey=b'secret password')]

    def reset(self):
        # Disconnect robots
        if self.real:
            self._disconnect_robots()

        # Reset pybullet
        self.p.resetSimulation()
        self.p.setRealTimeSimulation(0)
        self.p.setGravity(0, 0, -9.8)

        # Create env
        self._create_env()
        if self.real:
            self.real_robot_indices_map = dict(zip(self.robot_ids, self.real_robot_indices))

        # Reset poses
        if self.real:
            self.update_poses()
        else:
            self._reset_poses()
        self._step_simulation_until_still()

        # Set awaiting new action for first robot
        self._set_awaiting_new_action()

        # State representation
        for robot in self.robots:
            robot.update_map()

        # Stats
        self.steps = 0
        self.simulation_steps = 0
        self.inactivity_steps = 0

        return self.get_state()

    def store_new_action(self, action):
        for robot_group, robot_group_actions in zip(self.robot_groups, action):
            for robot, a in zip(robot_group, robot_group_actions):
                if a is not None:
                    robot.store_new_action(a)

    def step(self, action):
        ################################################################################
        # Setup before action execution

        self.store_new_action(action)

        # Store initial object positions for pushing or blowing partial rewards
        if any(isinstance(robot, (PushingRobot, BlowingRobot)) for robot in self.robots):
            initial_object_positions = {}
            for object_id in self.available_object_ids_set:
                initial_object_positions[object_id] = self.get_object_position(object_id)

        ################################################################################
        # Execute actions

        if self.real:
            sim_steps = self._execute_actions_real()
        else:
            sim_steps = self._execute_actions()
        self._set_awaiting_new_action()

        ################################################################################
        # Process objects after action execution

        for object_id in self.available_object_ids_set.copy():
            object_position = self.get_object_position(object_id)

            # Reset out-of-bounds objects
            if (object_position[2] > VectorEnv.WALL_HEIGHT + 0.49 * self.object_width or  # On top of obstacle
                    object_position[2] < 0.4 * self.object_width):  # Inside obstacle (0.4 since dropped objects can temporarily go into the ground)
                pos_x, pos_y, heading = self._get_random_object_pose()
                self.reset_object_pose(object_id, pos_x, pos_y, heading)
                continue

            if self.receptacle_id is not None:
                closest_robot = self.robots[np.argmin([distance(robot.get_position(), object_position) for robot in self.robots])]

                # Process final object position for pushing partial rewards
                if isinstance(closest_robot, (PushingRobot, BlowingRobot)):
                    closest_robot.process_object_position(object_id, initial_object_positions)

                # Process objects that are in the receptacle (objects were pushed in)
                if self.object_position_in_receptacle(object_position):
                    closest_robot.process_object_success()
                    self.remove_object(object_id)
                    self.available_object_ids_set.remove(object_id)

        # Robots that are awaiting new action need an up-to-date map
        for robot in self.robots:
            if robot.awaiting_new_action:
                robot.update_map()

        ################################################################################
        # Compute rewards and stats

        # Increment counters
        self.steps += 1
        self.simulation_steps += sim_steps
        if sum(robot.objects for robot in self.robots) > 0:
            self.inactivity_steps = 0
        else:
            self.inactivity_steps += 1

        # Episode ends after too many steps of inactivity
        if self.real:
            done = False  # When running on real robot, objects are not simulated, so environment needs to be manually reset
        else:
            done = len(self.removed_object_ids_set) == self.num_objects or self.inactivity_steps >= self.inactivity_cutoff

        # Compute per-robot rewards and stats
        for robot in self.robots:
            if robot.awaiting_new_action or done:
                robot.compute_rewards_and_stats(done=done)

        ################################################################################
        # Compute items to return

        state = [[None for _ in g] for g in self.robot_groups] if done else self.get_state()
        reward = [[robot.reward if (robot.awaiting_new_action or done) else None for robot in robot_group] for robot_group in self.robot_groups]
        info = {
            'steps': self.steps,
            'simulation_steps': self.simulation_steps,
            'distance': [[robot.distance if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_objects': [[robot.cumulative_objects if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_distance': [[robot.cumulative_distance if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_reward': [[robot.cumulative_reward if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_obstacle_collisions': [[robot.cumulative_obstacle_collisions if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_robot_collisions': [[robot.cumulative_robot_collisions if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'total_objects': sum(robot.cumulative_objects for robot in self.robots),
            'total_obstacle_collisions': sum(robot.cumulative_obstacle_collisions for robot in self.robots),
        }

        return state, reward, done, info

    def get_state(self, all_robots=False, save_figures=False):
        return [[robot.get_state(save_figures=save_figures) if robot.awaiting_new_action or all_robots else None for robot in robot_group] for robot_group in self.robot_groups]

    def close(self):
        self.p.disconnect()
        if self.real:
            self._disconnect_robots()

    def step_simulation(self):
        self.p.stepSimulation()
        #import time; time.sleep(1.0 / 180)
        self.step_simulation_count += 1

    def get_object_pose(self, object_id):
        return self.p.getBasePositionAndOrientation(object_id)

    def get_object_position(self, object_id):
        position, _ = self.get_object_pose(object_id)
        return position

    def reset_object_pose(self, object_id, pos_x, pos_y, heading):
        position = (pos_x, pos_y, self.object_width / 2)
        self.p.resetBasePositionAndOrientation(object_id, position, heading_to_orientation(heading))

    def remove_object(self, object_id):
        self.p.resetBasePositionAndOrientation(object_id, (0, 0, VectorEnv.REMOVED_BODY_Z), VectorEnv.IDENTITY_QUATERNION)
        self.removed_object_ids_set.add(object_id)

    def object_position_in_receptacle(self, object_position):
        assert self.receptacle_id is not None

        half_width = (VectorEnv.RECEPTACLE_WIDTH - self.object_width) / 2
        if (self.receptacle_position[0] - half_width < object_position[0] < self.receptacle_position[0] + half_width and
                self.receptacle_position[1] - half_width < object_position[1] < self.receptacle_position[1] + half_width):
            return True
        return False

    def get_robot_group_types(self):
        return self.robot_group_types

    @staticmethod
    def get_state_width():
        return Mapper.LOCAL_MAP_PIXEL_WIDTH

    @staticmethod
    def get_num_output_channels(robot_type):
        return Robot.get_robot_cls(robot_type).NUM_OUTPUT_CHANNELS

    @staticmethod
    def get_action_space(robot_type):
        return VectorEnv.get_num_output_channels(robot_type) * Mapper.LOCAL_MAP_PIXEL_WIDTH * Mapper.LOCAL_MAP_PIXEL_WIDTH

    def get_camera_image(self, image_width=1024, image_height=768):
        assert self.show_gui
        return self.p.getCameraImage(image_width, image_height, flags=pybullet.ER_NO_SEGMENTATION_MASK, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)[2]

    def start_video_logging(self, video_path):
        assert self.show_gui
        return self.p.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, video_path)

    def stop_video_logging(self, log_id):
        self.p.stopStateLogging(log_id)

    def update_poses(self):
        assert self.real

        # Get new pose estimates
        for conn in self.conns:
            if self.real_debug:
                debug_data = [(robot.waypoint_positions, robot.target_end_effector_position, robot.controller.debug_data) for robot in self.robots]
                conn.send(debug_data)
            else:
                conn.send(None)

        for conn in self.conns:
            robot_poses, object_mask = conn.recv()

            for robot in self.robots:
                # Update robot poses
                if robot_poses is not None:
                    robot_pose = robot_poses.get(self.real_robot_indices_map[robot.id], None)
                    if robot_pose is not None:
                        robot.reset_pose(robot_pose['position'][0], robot_pose['position'][1], robot_pose['heading'])

            # Store object mask
            self.object_mask = object_mask

        self.step_simulation()

    def _create_env(self):
        # Assertions
        assert self.room_length >= self.room_width
        if not self.real:
            assert self.num_objects > 0
        assert all(len(g) == 1 for g in self.robot_config)  # Each robot group should be homogeneous
        assert not len(self.robot_group_types) > 4  # More than 4 groups not supported

        # Create floor
        floor_thickness = 10
        wall_thickness = 1.4
        room_length_with_walls = self.room_length + 2 * wall_thickness
        room_width_with_walls = self.room_width + 2 * wall_thickness
        floor_half_extents = (room_length_with_walls / 2, room_width_with_walls / 2, floor_thickness / 2)
        floor_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_BOX, halfExtents=floor_half_extents)
        floor_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_BOX, halfExtents=floor_half_extents)
        self.p.createMultiBody(0, floor_collision_shape_id, floor_visual_shape_id, (0, 0, -floor_thickness / 2))

        # Create obstacles (including walls)
        obstacle_color = (0.9, 0.9, 0.9, 1)
        rounded_corner_path = str(Path(__file__).parent / 'assets' / 'rounded_corner.obj')
        self.obstacle_ids = []
        for obstacle in self._get_obstacles(wall_thickness):
            if obstacle['type'] == 'corner':
                obstacle_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_MESH, fileName=rounded_corner_path)
                obstacle_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_MESH, fileName=rounded_corner_path, rgbaColor=obstacle_color)
            else:
                obstacle_half_extents = (obstacle['x_len'] / 2, obstacle['y_len'] / 2, VectorEnv.WALL_HEIGHT / 2)
                obstacle_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_BOX, halfExtents=obstacle_half_extents)
                obstacle_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_BOX, halfExtents=obstacle_half_extents, rgbaColor=obstacle_color)

            obstacle_id = self.p.createMultiBody(
                0, obstacle_collision_shape_id, obstacle_visual_shape_id,
                (obstacle['position'][0], obstacle['position'][1], VectorEnv.WALL_HEIGHT / 2), heading_to_orientation(obstacle['heading']))
            self.obstacle_ids.append(obstacle_id)

        # Create target receptacle
        receptacle_color = (1, 87.0 / 255, 89.0 / 255, 1)  # Red
        receptacle_visual_shape_id = self.p.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=(VectorEnv.RECEPTACLE_WIDTH / 2, VectorEnv.RECEPTACLE_WIDTH / 2, 0.0001),
            rgbaColor=receptacle_color, visualFramePosition=(0, 0, 0.0001))
        self.receptacle_id = self.p.createMultiBody(0, baseVisualShapeIndex=receptacle_visual_shape_id, basePosition=self.receptacle_position)

        # Create robots
        self.robot_collision_body_b_ids_set = set()
        self.robot_ids = []
        self.robots = []  # Flat list
        self.robot_groups = [[] for _ in range(len(self.robot_config))]  # Grouped list
        for robot_group_index, g in enumerate(self.robot_config):
            robot_type, count = next(iter(g.items()))
            for _ in range(count):
                if self.real:
                    real_robot_index = self.real_robot_indices[len(self.robots)]
                    robot = Robot.get_robot(robot_type, self, robot_group_index, real=True, real_robot_index=real_robot_index)
                else:
                    robot = Robot.get_robot(robot_type, self, robot_group_index)
                self.robots.append(robot)
                self.robot_groups[robot_group_index].append(robot)
                self.robot_ids.append(robot.id)
        self.last_robot_index = 0

        # Create objects
        if self.object_type == 'mixed_sizes':
            default_object_radius = self.object_width / 2
            self.object_ids = []
            for _ in range(self.num_objects):
                scale = self.room_random_state.uniform(1, 2)
                object_radius = scale * default_object_radius
                object_mass = self.object_mass * scale**3
                object_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_SPHERE, radius=object_radius)
                color = list(VectorEnv.OBJECT_COLOR)
                color[2] += (scale - 1)
                object_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_SPHERE, radius=object_radius, rgbaColor=color)
                object_id = self.p.createMultiBody(object_mass, object_collision_shape_id, object_visual_shape_id)
                self.p.changeDynamics(object_id, -1, lateralFriction=0.5, rollingFriction=0.001)
                self.object_ids.append(object_id)

        elif self.object_type == 'mixed_shapes':
            self.object_ids = []
            for _ in range(self.num_objects):
                shape_idx = self.room_random_state.randint(4)
                color = [
                    (242.0 / 255, 142.0 / 255, 43.0 / 255, 1),  # Orange
                    (89.0 / 255, 169.0 / 255, 79.0 / 255, 1),  # Green
                    (78.0 / 255, 121.0 / 255, 167.0 / 255, 1),  # Blue
                    VectorEnv.OBJECT_COLOR,  # Yellow
                ][shape_idx]
                if shape_idx == 0:
                    # Cube
                    object_half_extents = (self.object_width / 2, self.object_width / 2, self.object_width / 2)
                    object_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_BOX, halfExtents=object_half_extents)
                    object_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_BOX, halfExtents=object_half_extents, rgbaColor=color)
                    object_id = self.p.createMultiBody(self.object_mass, object_collision_shape_id, object_visual_shape_id)
                elif shape_idx == 1:
                    # Rectangular cuboid
                    object_half_extents = (self.object_width, self.object_width / 2, self.object_width / 2)
                    object_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_BOX, halfExtents=object_half_extents)
                    object_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_BOX, halfExtents=object_half_extents, rgbaColor=color)
                    object_id = self.p.createMultiBody(self.object_mass, object_collision_shape_id, object_visual_shape_id)
                elif shape_idx == 2:
                    # Cylinder
                    object_radius = self.object_width / 2
                    object_length = 2 * self.object_width
                    object_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_CYLINDER, radius=object_radius, height=object_length)
                    object_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_CYLINDER, radius=object_radius, length=object_length, rgbaColor=color)
                    object_id = self.p.createMultiBody(self.object_mass, object_collision_shape_id, object_visual_shape_id)
                else:
                    # Sphere
                    object_radius = self.object_width / 2
                    object_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_SPHERE, radius=object_radius)
                    object_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_SPHERE, radius=object_radius, rgbaColor=color)
                    object_id = self.p.createMultiBody(self.object_mass, object_collision_shape_id, object_visual_shape_id)
                self.p.changeDynamics(object_id, -1, lateralFriction=0.5, rollingFriction=0.001)
                self.object_ids.append(object_id)

        else:
            object_radius = self.object_width / 2
            object_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_SPHERE, radius=object_radius)
            object_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_SPHERE, radius=object_radius, rgbaColor=VectorEnv.OBJECT_COLOR)
            self.object_ids = []
            for _ in range(self.num_objects):
                object_id = self.p.createMultiBody(self.object_mass, object_collision_shape_id, object_visual_shape_id)
                self.p.changeDynamics(object_id, -1, lateralFriction=0.5, rollingFriction=0.001)
                self.object_ids.append(object_id)

        # Initialize collections
        self.obstacle_collision_body_b_ids_set = set(self.obstacle_ids)
        self.robot_collision_body_b_ids_set.update(self.robot_ids)
        self.available_object_ids_set = set(self.object_ids)
        self.removed_object_ids_set = set()

    def _get_obstacles(self, wall_thickness):
        # Assertions
        if self.env_name.startswith('small'):
            assert math.isclose(self.room_length, 1.0)
            assert math.isclose(self.room_width, 0.5)
            expected_num_objects = 50
        elif self.env_name.startswith('large'):
            assert math.isclose(self.room_length, 1.0)
            assert math.isclose(self.room_width, 1.0)
            expected_num_objects = 100
        assert math.isclose(self.object_width, 0.012)
        assert math.isclose(self.object_mass, 0.00009)
        if self.real:
            expected_num_objects = 0
        assert self.num_objects == expected_num_objects, (self.num_objects, expected_num_objects)

        def get_obstacle_box(obstacle, buffer_width=0.08):
            x, y = obstacle['position']
            x_len, y_len = obstacle['x_len'], obstacle['y_len']
            b = box(x - x_len / 2, y - y_len / 2, x + x_len / 2, y + y_len / 2)
            if buffer_width > 0:
                b = b.buffer(buffer_width)
            return b

        def get_receptacle_box():
            obstacle = {'position': self.receptacle_position[:2], 'heading': 0, 'x_len': VectorEnv.RECEPTACLE_WIDTH, 'y_len': VectorEnv.RECEPTACLE_WIDTH}
            return get_obstacle_box(obstacle, buffer_width=0)

        def draw_polygons(polygons):
            import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
            for polygon in polygons:
                for coords in [polygon.exterior.coords] + [interior.coords for interior in polygon.interiors]:
                    coords = np.asarray(coords)
                    plt.plot(coords[:, 0], coords[:, 1])
            padding = Robot.RADIUS
            plt.axis([-self.room_length / 2 - padding, self.room_length / 2 + padding, -self.room_width / 2 - padding, self.room_width / 2 + padding])
            plt.show()

        def add_random_columns(max_num_columns):
            num_columns = self.room_random_state.randint(max_num_columns) + 1
            column_x_len, column_y_len = 0.1, 0.1
            buffer_width = 0.08

            polygons = [get_receptacle_box()] + [get_obstacle_box(obstacle) for obstacle in obstacles]
            for _ in range(10):
                new_obstacles = []
                new_polygons = []
                polygon_union = unary_union(polygons)
                for _ in range(num_columns):
                    for _ in range(100):
                        x = self.room_random_state.uniform(
                            -self.room_length / 2 + 2 * buffer_width + column_x_len / 2,
                            self.room_length / 2 - 2 * buffer_width - column_x_len / 2
                        )
                        y = self.room_random_state.uniform(
                            -self.room_width / 2 + 2 * buffer_width + column_y_len / 2,
                            self.room_width / 2 - 2 * buffer_width - column_y_len / 2
                        )
                        obstacle = {'type': 'column', 'position': (x, y), 'heading': 0, 'x_len': column_x_len, 'y_len': column_y_len}
                        b = get_obstacle_box(obstacle)
                        if not polygon_union.intersects(b):
                            new_obstacles.append(obstacle)
                            new_polygons.append(b)
                            polygon_union = unary_union(polygons + new_polygons)
                            break
                if len(new_polygons) == num_columns:
                    break
            obstacles.extend(new_obstacles)
            #draw_polygons(polygons + new_polygons)

        def add_horiz_doorway(opening_width=0.30, x_offset=0, y_offset=0):
            divider_width = 0.05
            divider_len = (self.room_width - opening_width) / 2
            divider_x = self.room_length / 2 - divider_len / 2
            obstacles.append({'type': 'divider', 'position': (-divider_x + x_offset / 2, y_offset), 'heading': 0, 'x_len': divider_len + x_offset, 'y_len': divider_width})
            obstacles.append({'type': 'divider', 'position': (divider_x + x_offset / 2, y_offset), 'heading': 0, 'x_len': divider_len - x_offset, 'y_len': divider_width})
            self.robot_spawn_bounds = (None, None, y_offset + divider_width / 2, None)
            self.object_spawn_bounds = (None, None, None, y_offset - divider_width / 2)

        # Walls
        obstacles = []
        for x, y, x_len, y_len in [
                (-self.room_length / 2 - wall_thickness / 2, 0, wall_thickness, self.room_width),
                (self.room_length / 2 + wall_thickness / 2, 0, wall_thickness, self.room_width),
                (0, -self.room_width / 2 - wall_thickness / 2, self.room_length + 2 * wall_thickness, wall_thickness),
                (0, self.room_width / 2 + wall_thickness / 2, self.room_length + 2 * wall_thickness, wall_thickness),
            ]:
            obstacles.append({'type': 'wall', 'position': (x, y), 'heading': 0, 'x_len': x_len, 'y_len': y_len})

        # Other obstacles
        if self.env_name in {'small_empty', 'large_empty'}:
            pass

        elif self.env_name == 'large_columns':
            add_random_columns(6)

        elif self.env_name == 'large_door':
            add_horiz_doorway(x_offset=self.room_random_state.uniform(-0.05, 0.05), y_offset=self.room_random_state.uniform(-0.05, 0.05))

        elif self.env_name == 'large_center':
            self.robot_spawn_bounds = (-0.5, 0.5, -0.5, 0.5)
            self.object_spawn_bounds = (-0.5, 0.5, -0.5, 0.5)

        else:
            raise Exception(self.env_name)

        ################################################################################
        # Rounded corners

        rounded_corner_width = 0.1006834873
        # Room corners
        for i, (x, y) in enumerate([
                (-self.room_length / 2, self.room_width / 2),
                (self.room_length / 2, self.room_width / 2),
                (self.room_length / 2, -self.room_width / 2),
                (-self.room_length / 2, -self.room_width / 2),
            ]):
            if distance((x, y), self.receptacle_position) > (1 + 1e-6) * (VectorEnv.RECEPTACLE_WIDTH / 2) * math.sqrt(2):
                heading = -math.radians(i * 90)
                offset = rounded_corner_width / math.sqrt(2)
                adjusted_position = (x + offset * math.cos(heading - math.radians(45)), y + offset * math.sin(heading - math.radians(45)))
                obstacles.append({'type': 'corner', 'position': adjusted_position, 'heading': heading})

        # Corners between walls and dividers
        new_obstacles = []
        for obstacle in obstacles:
            if obstacle['type'] == 'divider':
                position, length, width = obstacle['position'], obstacle['x_len'], obstacle['y_len']
                x, y = position
                corner_positions = None
                if math.isclose(x - length / 2, -self.room_length / 2):
                    corner_positions = [(-self.room_length / 2, y - width / 2), (-self.room_length / 2, y + width / 2)]
                    corner_headings = [0, 90]
                elif math.isclose(x + length / 2, self.room_length / 2):
                    corner_positions = [(self.room_length / 2, y - width / 2), (self.room_length / 2, y + width / 2)]
                    corner_headings = [-90, 180]
                elif math.isclose(y - width / 2, -self.room_width / 2):
                    corner_positions = [(x - length / 2, -self.room_width / 2), (x + length / 2, -self.room_width / 2)]
                    corner_headings = [180, 90]
                elif math.isclose(y + width / 2, self.room_width / 2):
                    corner_positions = [(x - length / 2, self.room_width / 2), (x + length / 2, self.room_width / 2)]
                    corner_headings = [-90, 0]
                elif 'snap_y' in obstacle:
                    snap_y = obstacle['snap_y']
                    corner_positions = [(x - length / 2, snap_y), (x + length / 2, snap_y)]
                    corner_headings = [-90, 0] if snap_y > y else [180, 90]
                if corner_positions is not None:
                    for position, heading in zip(corner_positions, corner_headings):
                        heading = math.radians(heading)
                        offset = rounded_corner_width / math.sqrt(2)
                        adjusted_position = (
                            position[0] + offset * math.cos(heading - math.radians(45)),
                            position[1] + offset * math.sin(heading - math.radians(45))
                        )
                        obstacles.append({'type': 'corner', 'position': adjusted_position, 'heading': heading})
        obstacles.extend(new_obstacles)

        return obstacles

    def _reset_poses(self):
        # Reset robot poses
        for robot in self.robots:
            pos_x, pos_y, heading = self._get_random_robot_pose(padding=robot.RADIUS, bounds=self.robot_spawn_bounds)
            robot.reset_pose(pos_x, pos_y, heading)

        # Reset object poses
        if self.object_type == 'mixed_shapes':
            for object_id in self.object_ids:
                pos_x, pos_y, _ = self._get_random_object_pose()
                position = (pos_x, pos_y, self.object_width / 2)
                # See https://en.wikipedia.org/wiki/Rotation_matrix#Uniform_random_rotation_matrices
                orientation = self.room_random_state.randn(4)
                orientation /= np.linalg.norm(orientation)
                self.p.resetBasePositionAndOrientation(object_id, position, orientation)
        else:
            for object_id in self.object_ids:
                pos_x, pos_y, heading = self._get_random_object_pose()
                self.reset_object_pose(object_id, pos_x, pos_y, heading)

        # Check if any robots need another pose reset
        done = False
        while not done:
            done = True
            self.step_simulation()
            for robot in self.robots:
                reset_robot_pose = False

                # Check if robot is stacked on top of a object
                if robot.get_position(set_z_to_zero=False)[2] > 0.001:  # 1 mm
                    reset_robot_pose = True

                # Check if robot is inside an obstacle or another robot
                for contact_point in self.p.getContactPoints(robot.id):
                    if contact_point[2] in self.obstacle_collision_body_b_ids_set or contact_point[2] in self.robot_collision_body_b_ids_set:
                        reset_robot_pose = True
                        break

                if reset_robot_pose:
                    done = False
                    pos_x, pos_y, heading = self._get_random_robot_pose(padding=robot.RADIUS, bounds=self.robot_spawn_bounds)
                    robot.reset_pose(pos_x, pos_y, heading)

    def _get_random_object_pose(self):
        done = False
        while not done:
            pos_x, pos_y = self._get_random_position(padding=self.object_width / 2, bounds=self.object_spawn_bounds)

            # Only spawn objects outside of the receptacle
            if self.receptacle_id is None or not self.object_position_in_receptacle((pos_x, pos_y)):
                done = True
        heading = self.room_random_state.uniform(-math.pi, math.pi)
        return pos_x, pos_y, heading

    def _get_random_robot_pose(self, padding=0, bounds=None):
        position_x, position_y = self._get_random_position(padding=padding, bounds=bounds)
        heading = self.room_random_state.uniform(-math.pi, math.pi)
        return position_x, position_y, heading

    def _get_random_position(self, padding=0, bounds=None):
        low_x = -self.room_length / 2 + padding
        high_x = self.room_length / 2 - padding
        low_y = -self.room_width / 2 + padding
        high_y = self.room_width / 2 - padding
        if bounds is not None:
            x_min, x_max, y_min, y_max = bounds
            if x_min is not None:
                low_x = x_min + padding
            if x_max is not None:
                high_x = x_max - padding
            if y_min is not None:
                low_y = y_min + padding
            if y_max is not None:
                high_y = y_max - padding
        position_x, position_y = self.room_random_state.uniform((low_x, low_y), (high_x, high_y))
        return position_x, position_y

    def _step_simulation_until_still(self):
        # Kick-start gravity
        for _ in range(2):
            self.step_simulation()

        movable_body_ids = self.robot_ids + self.object_ids
        prev_positions = []
        sim_steps = 0
        done = False
        while not done:
            # Check whether any bodies moved since last step
            positions = [self.p.getBasePositionAndOrientation(body_id)[0] for body_id in movable_body_ids]
            if len(prev_positions) > 0:
                done = True
                for prev_position, position in zip(prev_positions, positions):
                    change = distance(prev_position, position)
                    # Ignore removed objects (negative z)
                    if position[2] > -0.0001 and change > 0.0005:  # 0.5 mm
                        done = False
                        break
            prev_positions = positions

            self.step_simulation()
            sim_steps += 1

            if sim_steps > 800:
                break

    def _set_awaiting_new_action(self):
        if sum(robot.awaiting_new_action for robot in self.robots) == 0:
            for robot in [self.robots[(i +self.last_robot_index) % self.num_robots] for i in range(self.num_robots)]:
                if robot.is_idle():
                    robot.awaiting_new_action = True
                    break
            self.last_robot_index = (self.last_robot_index + 1) % self.num_robots

    def _execute_actions(self):
        sim_steps = 0
        while True:
            if any(robot.is_idle() for robot in self.robots):
                break

            self.step_simulation()
            sim_steps += 1
            for robot in self.robots:
                robot.step()

        return sim_steps

    def _execute_actions_real(self):
        assert self.real

        # If debug mode is enabled, all robots will pause and resume actions during any robot's action selection
        if self.real_debug:
            for robot in self.robots:
                robot.controller.resume()

        sim_steps = 0
        any_idle = False
        while True:
            if not any_idle and any(robot.is_idle() for robot in self.robots):
                any_idle = True
                if self.real_debug:
                    for robot in self.robots:
                        robot.controller.pause()

            if any_idle:
                # If debug mode is enabled, do not exit loop until all robots have actually stopped moving
                if not self.real_debug or all((robot.is_idle() or robot.controller.state == 'paused') for robot in self.robots):
                    break

            self.update_poses()
            sim_steps += 1
            for robot in self.robots:
                robot.step()

        return sim_steps

    def _disconnect_robots(self):
        assert self.real
        if self.robots is not None:
            for robot in self.robots:
                robot.controller.disconnect()

class Robot(ABC):
    HALF_WIDTH = 0.03
    BACKPACK_OFFSET = -0.0135
    BASE_LENGTH = 0.065  # Does not include the hooks
    TOP_LENGTH = 0.057  # Leaves 1 mm gap for lifted object
    END_EFFECTOR_LOCATION = BACKPACK_OFFSET + BASE_LENGTH
    RADIUS = math.sqrt(HALF_WIDTH**2 + END_EFFECTOR_LOCATION**2)
    HEIGHT = 0.07
    NUM_OUTPUT_CHANNELS = 1
    COLOR = (0.3529, 0.3529, 0.3529, 1)  # Gray
    CONSTRAINT_MAX_FORCE = 10

    @abstractmethod  # Should not be instantiated directly
    def __init__(self, env, group_index, real=False, real_robot_index=None):
        self.env = env
        self.group_index = group_index
        self.real = real
        self.id = self._create_multi_body()
        self.cid = self.env.p.createConstraint(self.id, -1, -1, -1, pybullet.JOINT_FIXED, None, (0, 0, 0), (0, 0, 0))
        self._last_step_simulation_count = -1  # Used to determine whether pose is out of date
        self._position_raw = None  # Most current position, not to be directly accessed (use self.get_position())
        self._position = None  # Most current position (with z set to 0), not to be directly accessed (use self.get_position())
        self._heading = None  # Most current heading, not to be directly accessed (use self.get_heading())

        # Movement
        self.action = None
        self.target_end_effector_position = None
        self.waypoint_positions = None
        self.waypoint_headings = None
        self.controller = RealRobotController(self.env, self, real_robot_index, debug=self.env.real_debug) if real else RobotController(self.env, self)

        # Collision detection
        self.collision_body_a_ids_set = set([self.id])

        # State representation
        self.mapper = Mapper(self.env, self)

        # Step variables and stats
        self.awaiting_new_action = False  # Only one robot at a time can be awaiting new action
        self.objects = 0
        self.reward = None
        self.objects_with_reward = 0
        self.distance = 0
        self.prev_waypoint_position = None  # For tracking distance traveled over the step
        self.collided_with_obstacle = False
        self.collided_with_robot = False

        # Episode stats (robots are recreated every episode)
        self.cumulative_objects = 0
        self.cumulative_distance = 0
        self.cumulative_reward = 0
        self.cumulative_obstacle_collisions = 0
        self.cumulative_robot_collisions = 0

    def store_new_action(self, action):
        # Action is specified as an index specifying an end effector action, along with (row, col) of the selected pixel location
        self.action = tuple(np.unravel_index(action, (self.NUM_OUTPUT_CHANNELS, Mapper.LOCAL_MAP_PIXEL_WIDTH, Mapper.LOCAL_MAP_PIXEL_WIDTH)))  # Immutable tuple

        # Get current robot pose
        current_position, current_heading = self.get_position(), self.get_heading()

        # Compute distance from front of robot (not center of robot), which is used to find the
        # robot position and heading that would place the end effector over the specified location
        dx, dy = Mapper.pixel_indices_to_position(self.action[1], self.action[2], (Mapper.LOCAL_MAP_PIXEL_WIDTH, Mapper.LOCAL_MAP_PIXEL_WIDTH))
        dist = math.sqrt(dx**2 + dy**2)
        theta = current_heading + math.atan2(-dx, dy)
        self.target_end_effector_position = (current_position[0] + dist * math.cos(theta), current_position[1] + dist * math.sin(theta), 0)

        ################################################################################
        # Waypoints
        self.waypoint_positions, self.waypoint_headings = self._compute_waypoints(
            current_position, current_heading, self.target_end_effector_position, use_shortest_path_movement=self.env.use_shortest_path_movement)

        ################################################################################
        # Step variables and stats

        # Reset controller
        self.controller.reset()
        self.controller.new_action()

        # Reset step variables and stats
        self.awaiting_new_action = False
        self.objects = 0
        self.reward = None
        self.objects_with_reward = 0
        self.distance = 0
        self.prev_waypoint_position = current_position
        self.collided_with_obstacle = False
        self.collided_with_robot = False

    def _compute_waypoints(self, current_position, current_heading, target_end_effector_position, use_shortest_path_movement=True):
        # Compute waypoint positions
        if use_shortest_path_movement:
            waypoint_positions = self.mapper.shortest_path(current_position, target_end_effector_position)
        else:
            waypoint_positions = [current_position, target_end_effector_position]

        # Compute waypoint headings
        waypoint_headings = [current_heading]
        for i in range(1, len(waypoint_positions)):
            dx = waypoint_positions[i][0] - waypoint_positions[i - 1][0]
            dy = waypoint_positions[i][1] - waypoint_positions[i - 1][1]
            waypoint_headings.append(restrict_heading_range(math.atan2(dy, dx)))

        # Compute target position and heading for the robot. This involves applying an
        # offset to shift the final waypoint from end effector position to robot position.
        signed_dist = distance(waypoint_positions[-2], waypoint_positions[-1]) - (self.END_EFFECTOR_LOCATION + self.env.object_width / 2)
        target_heading = waypoint_headings[-1]
        target_position = (
            waypoint_positions[-2][0] + signed_dist * math.cos(target_heading),
            waypoint_positions[-2][1] + signed_dist * math.sin(target_heading),
            0
        )
        waypoint_positions[-1] = target_position

        # Avoid awkward backing up to reach the last waypoint
        if len(waypoint_positions) > 2 and signed_dist < 0:
            waypoint_positions[-2] = waypoint_positions[-1]
            dx = waypoint_positions[-2][0] - waypoint_positions[-3][0]
            dy = waypoint_positions[-2][1] - waypoint_positions[-3][1]
            waypoint_headings[-2] = restrict_heading_range(math.atan2(dy, dx))

        return waypoint_positions, waypoint_headings

    def step(self):
        self.controller.step()

    def update_map(self):
        self.mapper.update()

    def get_state(self, save_figures=False):
        return self.mapper.get_state(save_figures=save_figures)

    def process_object_success(self):
        self.objects += 1

    def compute_rewards_and_stats(self, done=False):
        # Ways a step can end
        # - Successfully completed action
        # - Collision
        # - Step limit exceeded
        # - Episode ended (no objects left or too many steps of inactivity)

        if done:
            self.update_distance()
            self.controller.reset()

        # Calculate final reward
        success_reward = self.env.success_reward * self.objects_with_reward
        obstacle_collision_penalty = -self.env.obstacle_collision_penalty * self.collided_with_obstacle
        robot_collision_penalty = -self.env.robot_collision_penalty * self.collided_with_robot
        self.reward = success_reward + obstacle_collision_penalty + robot_collision_penalty

        # Update cumulative stats
        self.cumulative_objects += self.objects
        self.cumulative_reward += self.reward
        self.cumulative_distance += self.distance
        self.cumulative_obstacle_collisions += self.collided_with_obstacle
        self.cumulative_robot_collisions += self.collided_with_robot

    def reset(self):
        self.action = None
        self.target_end_effector_position = None
        self.waypoint_positions = None
        self.waypoint_headings = None
        self.controller.reset()

    def is_idle(self):
        return self.controller.state == 'idle'

    def get_position(self, set_z_to_zero=True):
        # Returned position is immutable tuple
        if self._last_step_simulation_count < self.env.step_simulation_count:
            self._update_pose()
        if not set_z_to_zero:
            return self._position_raw
        return self._position

    def get_heading(self):
        if self._last_step_simulation_count < self.env.step_simulation_count:
            self._update_pose()
        return self._heading

    def reset_pose(self, position_x, position_y, heading):
        # Reset robot pose
        position = (position_x, position_y, 0)
        orientation = heading_to_orientation(heading)
        self.env.p.resetBasePositionAndOrientation(self.id, position, orientation)
        self.env.p.changeConstraint(self.cid, jointChildPivot=position, jointChildFrameOrientation=orientation, maxForce=Robot.CONSTRAINT_MAX_FORCE)
        self._last_step_simulation_count = -1

    def check_for_collisions(self):
        for body_a_id in self.collision_body_a_ids_set:
            for contact_point in self.env.p.getContactPoints(body_a_id):
                body_b_id = contact_point[2]
                if body_b_id in self.collision_body_a_ids_set:
                    continue
                if body_b_id in self.env.obstacle_collision_body_b_ids_set:
                    self.collided_with_obstacle = True
                if body_b_id in self.env.robot_collision_body_b_ids_set:
                    self.collided_with_robot = True
                if self.collided_with_obstacle or self.collided_with_robot:
                    break

    def update_distance(self):
        current_position = self.get_position()
        self.distance += distance(self.prev_waypoint_position, current_position)
        if self.env.show_trajectories or self.env.show_debug_annotations:
            self.env.p.addUserDebugLine(
                (self.prev_waypoint_position[0], self.prev_waypoint_position[1], 0.001),
                (current_position[0], current_position[1], 0.001),
                VectorEnv.DEBUG_LINE_COLORS[self.group_index]
            )
        self.prev_waypoint_position = current_position

    def _update_pose(self):
        position, orientation = self.env.p.getBasePositionAndOrientation(self.id)
        self._position_raw = position
        self._position = (position[0], position[1], 0)  # Use immutable tuples to represent positions
        self._heading = orientation_to_heading(orientation)
        self._last_step_simulation_count = self.env.step_simulation_count

    def _create_multi_body(self):
        base_height = 0.035
        mass = 0.180
        shape_types = [pybullet.GEOM_CYLINDER, pybullet.GEOM_BOX, pybullet.GEOM_BOX]
        radii = [Robot.HALF_WIDTH, None, None]
        half_extents = [
            None,
            (self.BASE_LENGTH / 2, Robot.HALF_WIDTH, base_height / 2),
            (Robot.TOP_LENGTH / 2, Robot.HALF_WIDTH, Robot.HEIGHT / 2),
        ]
        lengths = [Robot.HEIGHT, None, None]
        rgba_colors = [self.COLOR, None, None]  # pybullet seems to ignore all colors after the first
        frame_positions = [
            (Robot.BACKPACK_OFFSET, 0, Robot.HEIGHT / 2),
            (Robot.BACKPACK_OFFSET + self.BASE_LENGTH / 2, 0, base_height / 2),
            (Robot.BACKPACK_OFFSET + Robot.TOP_LENGTH / 2, 0, Robot.HEIGHT / 2),
        ]
        collision_shape_id = self.env.p.createCollisionShapeArray(
            shapeTypes=shape_types, radii=radii, halfExtents=half_extents, lengths=lengths, collisionFramePositions=frame_positions)
        visual_shape_id = self.env.p.createVisualShapeArray(
            shapeTypes=shape_types, radii=radii, halfExtents=half_extents, lengths=lengths, rgbaColors=rgba_colors, visualFramePositions=frame_positions)
        return self.env.p.createMultiBody(mass, collision_shape_id, visual_shape_id)

    @staticmethod
    def get_robot_cls(robot_type):
        if robot_type == 'pushing_robot':
            return PushingRobot
        if robot_type == 'blowing_robot':
            return BlowingRobot
        if robot_type == 'moving_blowing_robot':
            return MovingBlowingRobot
        if robot_type == 'side_blowing_robot':
            return SideBlowingRobot
        raise Exception(robot_type)

    @staticmethod
    def get_robot(robot_type, *args, real=False, real_robot_index=None):
        return Robot.get_robot_cls(robot_type)(*args, real=real, real_robot_index=real_robot_index)

class PushingRobot(Robot):
    BASE_LENGTH = Robot.BASE_LENGTH + 0.005  # 5 mm blade
    END_EFFECTOR_LOCATION = Robot.BACKPACK_OFFSET + BASE_LENGTH
    RADIUS = math.sqrt(Robot.HALF_WIDTH**2 + END_EFFECTOR_LOCATION**2)
    COLOR = (0.1765, 0.1765, 0.1765, 1)  # Dark gray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_dist_closer = 0

    def store_new_action(self, action):
        super().store_new_action(action)
        self.object_dist_closer = 0

    def process_object_success(self):
        super().process_object_success()
        self.objects_with_reward += 1

    def compute_rewards_and_stats(self, done=False):
        super().compute_rewards_and_stats(done=done)
        partial_rewards = self.env.partial_rewards_scale * self.object_dist_closer
        self.reward += partial_rewards
        self.cumulative_reward += partial_rewards

    def process_object_position(self, object_id, initial_object_positions):
        if object_id not in initial_object_positions:
            return
        object_position = self.env.get_object_position(object_id)
        dist_closer = self.mapper.distance_to_receptacle(initial_object_positions[object_id]) - self.mapper.distance_to_receptacle(object_position)
        self.object_dist_closer += dist_closer

class BlowingRobot(Robot):
    NUM_OUTPUT_CHANNELS = 2
    BLOWER_THICKNESS = 0.015
    TUBE_LENGTH = 0.025
    TUBE_BASE_HEIGHT = 0.019
    TUBE_OFFSET = 0.028

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_dist_closer = 0
        self.blower_fov = math.radians(self.env.blowing_fov)
        self.num_particles = self.env.blowing_num_wind_particles
        self.blower_id = self._create_blower_shape()

        # Collision detection
        self.collision_body_a_ids_set.add(self.blower_id)
        self.env.robot_collision_body_b_ids_set.add(self.blower_id)

        self.blower_cid = None
        self._attach_blower_shape()

        # Create wind particles
        particle_collision_shape_id = self.env.p.createCollisionShape(pybullet.GEOM_SPHERE, radius=self.env.blowing_wind_particle_radius)  # Half of object width
        particle_visual_shape_id = self.env.p.createVisualShape(pybullet.GEOM_SPHERE, radius=self.env.blowing_wind_particle_radius, rgbaColor=(0, 0, 0, 0))
        self.particle_ids = []
        for i in range(self.num_particles):
            self.particle_ids.append(self.env.p.createMultiBody(
                self.env.blowing_wind_particle_mass, particle_collision_shape_id, particle_visual_shape_id, (0, 0, VectorEnv.REMOVED_BODY_Z - i)))
        self.particle_counter = 0

    def reset_pose(self, *args):
        super().reset_pose(*args)

        # Reset pose of blower shape
        self.env.p.removeConstraint(self.blower_cid)
        self._attach_blower_shape()

    def store_new_action(self, action):
        super().store_new_action(action)
        self.object_dist_closer = 0

    def _compute_waypoints(self, *args, use_shortest_path_movement=True):
        if self.action[0] == 1:
            return super()._compute_waypoints(*args, use_shortest_path_movement=False)
        return super()._compute_waypoints(*args, use_shortest_path_movement=use_shortest_path_movement)

    def process_object_success(self):
        super().process_object_success()
        if self.action is not None and self.action[0] > 0:  # Channel 0 is just movement, no blowing
            self.objects_with_reward += 1

    def compute_rewards_and_stats(self, done=False):
        super().compute_rewards_and_stats(done=done)
        if self.action is not None and self.action[0] > 0:  # Channel 0 is just movement, no blowing
            partial_rewards = self.env.partial_rewards_scale * self.object_dist_closer
            self.reward += partial_rewards
            self.cumulative_reward += partial_rewards

    def process_object_position(self, object_id, initial_object_positions):
        if object_id not in initial_object_positions:
            return
        object_position = self.env.get_object_position(object_id)
        dist_closer = self.mapper.distance_to_receptacle(initial_object_positions[object_id]) - self.mapper.distance_to_receptacle(object_position)
        self.object_dist_closer += dist_closer

    def _get_wind_particle(self):
        particle_id = None
        if self.particle_counter % self.env.blowing_wind_particle_sparsity == 0:
            particle_id = self.particle_ids[(self.particle_counter // self.env.blowing_wind_particle_sparsity) % self.num_particles]
        self.particle_counter += 1
        return particle_id

    def blow_objects(self):
        particle_id = self._get_wind_particle()
        if particle_id is not None:
            if self.action[0] > 0:
                x_offset = Robot.BACKPACK_OFFSET + self.BASE_LENGTH
                y_offset = -Robot.HALF_WIDTH - 0.001 - self.BLOWER_THICKNESS / 2
                tube_height = 0.002 + self.TUBE_BASE_HEIGHT / 2
                current_position, current_heading = self.get_position(), self.get_heading()
                blower_position = (
                    current_position[0] + x_offset * math.cos(current_heading) - y_offset * math.sin(current_heading),
                    current_position[1] + x_offset * math.sin(current_heading) + y_offset * math.cos(current_heading),
                    tube_height
                )
                theta = self.env.robot_random_state.uniform(-self.blower_fov / 2, self.blower_fov / 2)
                self.env.p.resetBasePositionAndOrientation(particle_id, blower_position, heading_to_orientation(current_heading + theta))
                self.env.p.applyExternalForce(particle_id, -1, (self.env.blowing_force, 0, 0), (0, 0, 0), flags=pybullet.LINK_FRAME)
            else:
                self.env.p.resetBasePositionAndOrientation(particle_id, (0, 0, VectorEnv.REMOVED_BODY_Z), VectorEnv.IDENTITY_QUATERNION)

    def _create_blower_shape(self):
        mass = 0.027
        shape_types = [pybullet.GEOM_CYLINDER, pybullet.GEOM_CYLINDER, pybullet.GEOM_BOX]
        radii = [0.024, 0.007, None]
        half_extents = [None, None, (self.TUBE_OFFSET / 2, self.BLOWER_THICKNESS / 2, self.TUBE_BASE_HEIGHT / 2)]
        lengths = [self.BLOWER_THICKNESS, self.TUBE_LENGTH, None]
        rgba_colors = [self.COLOR, None, None]  # pybullet seems to ignore all colors after the first
        frame_positions = [
            (0, 0, 0),
            (self.TUBE_OFFSET + self.TUBE_LENGTH / 2, 0, -self.TUBE_OFFSET + self.TUBE_BASE_HEIGHT / 2),
            (self.TUBE_OFFSET / 2, 0, -self.TUBE_OFFSET + self.TUBE_BASE_HEIGHT / 2),
        ]
        frame_orientations = [
            pybullet.getQuaternionFromEuler((math.radians(90), 0, 0)),
            pybullet.getQuaternionFromEuler((0, math.radians(90), 0)),
            VectorEnv.IDENTITY_QUATERNION,
        ]
        collision_shape_id = self.env.p.createCollisionShapeArray(
            shapeTypes=shape_types, radii=radii, halfExtents=half_extents, lengths=lengths,
            collisionFramePositions=frame_positions, collisionFrameOrientations=frame_orientations)
        visual_shape_id = self.env.p.createVisualShapeArray(
            shapeTypes=shape_types, halfExtents=half_extents, radii=radii, lengths=lengths, rgbaColors=rgba_colors,
            visualFramePositions=frame_positions, visualFrameOrientations=frame_orientations)
        return self.env.p.createMultiBody(mass, collision_shape_id, visual_shape_id)

    def _attach_blower_shape(self):
        # Attach to side of robot
        x_offset = Robot.BACKPACK_OFFSET + self.BASE_LENGTH - (self.TUBE_OFFSET + self.TUBE_LENGTH)
        y_offset = -Robot.HALF_WIDTH - 0.001 - self.BLOWER_THICKNESS / 2  # 1 mm buffer
        height = self.TUBE_OFFSET + 0.002  # 2 mm clearance
        current_position, current_heading = self.get_position(), self.get_heading()
        parent_frame_position_world = (
            current_position[0] + x_offset * math.cos(current_heading) - y_offset * math.sin(current_heading),
            current_position[1] + x_offset * math.sin(current_heading) + y_offset * math.cos(current_heading),
            height
        )
        self.env.p.resetBasePositionAndOrientation(self.blower_id, parent_frame_position_world, heading_to_orientation(current_heading))

        # Create constraint
        parent_frame_position = (x_offset, y_offset, height)
        self.blower_cid = self.env.p.createConstraint(self.id, -1, self.blower_id, -1, pybullet.JOINT_FIXED, None, parent_frame_position, (0, 0, 0))

class MovingBlowingRobot(BlowingRobot):
    pass

class SideBlowingRobot(BlowingRobot):
    END_EFFECTOR_LOCATION = Robot.BACKPACK_OFFSET + Robot.BASE_LENGTH + 0.001 + BlowingRobot.BLOWER_THICKNESS / 2  # 1 mm buffer

    def blow_objects(self):
        particle_id = self._get_wind_particle()
        if particle_id is not None:
            if self.action[0] > 0:
                x_offset = self.END_EFFECTOR_LOCATION
                y_offset = -(self.TUBE_OFFSET + self.TUBE_LENGTH)
                tube_height = 0.002 + self.TUBE_BASE_HEIGHT / 2
                current_position, current_heading = self.get_position(), self.get_heading()
                blower_position = (
                    current_position[0] + x_offset * math.cos(current_heading) - y_offset * math.sin(current_heading),
                    current_position[1] + x_offset * math.sin(current_heading) + y_offset * math.cos(current_heading),
                    tube_height
                )
                theta = self.env.robot_random_state.uniform(-self.blower_fov / 2, self.blower_fov / 2)
                self.env.p.resetBasePositionAndOrientation(particle_id, blower_position, heading_to_orientation(current_heading - math.radians(90) + theta))
                self.env.p.applyExternalForce(particle_id, -1, (self.env.blowing_force, 0, 0), (0, 0, 0), flags=pybullet.LINK_FRAME)
            else:
                self.env.p.resetBasePositionAndOrientation(particle_id, (0, 0, VectorEnv.REMOVED_BODY_Z), VectorEnv.IDENTITY_QUATERNION)

    def _attach_blower_shape(self):
        # Attach to front of robot, points to the right side
        x_offset = self.END_EFFECTOR_LOCATION
        y_offset = 0
        height = self.TUBE_OFFSET + 0.002  # 2 mm clearance
        current_position, current_heading = self.get_position(), self.get_heading()
        parent_frame_position_world = (
            current_position[0] + x_offset * math.cos(current_heading) - y_offset * math.sin(current_heading),
            current_position[1] + x_offset * math.sin(current_heading) + y_offset * math.cos(current_heading),
            height
        )
        self.env.p.resetBasePositionAndOrientation(self.blower_id, parent_frame_position_world, heading_to_orientation(current_heading - math.radians(90)))

        # Create constraint
        parent_frame_position = (x_offset, y_offset, height)
        self.blower_cid = self.env.p.createConstraint(self.id, -1, self.blower_id, -1, pybullet.JOINT_FIXED, None, parent_frame_position, (0, 0, 0), heading_to_orientation(-math.radians(90)))

class RobotController:
    DRIVE_STEP_SIZE = 0.005  # 5 mm results in exactly 1 mm per simulation step
    TURN_STEP_SIZE = math.radians(5)  # 5 deg results in exactly 1 deg per simulation step
    ROTATION_TURN_THRESHOLD = math.radians(1)  # For blowing robots

    def __init__(self, env, robot):
        self.env = env
        self.robot = robot
        self.state = 'idle'
        self.next_state = None
        self.waypoint_index = None  # Index of waypoint we are currently headed towards
        self.prev_position = None  # Position before call to p.stepSimulation()
        self.prev_heading = None
        self.sim_steps = 0
        self.consecutive_turning_sim_steps = None  # Used to detect if robot is stuck and oscillating
        self.slowing_sim_step_target = 0
        self.slowing_sim_steps = 0

    def reset(self):
        self.state = 'idle'
        self.next_state = None
        self.waypoint_index = 1
        self.prev_position = None
        self.prev_heading = None
        self.sim_steps = 0
        self.consecutive_turning_sim_steps = 0

    def new_action(self):
        self.state = 'turning'

    def step(self):
        # States: idle, turning, driving, slowing

        assert not self.state == 'idle'
        self.sim_steps += 1

        # Periodically update the map
        if self.sim_steps % 200 == 0:
            self.robot.update_map()

        if self.state in {'turning', 'driving'}:
            current_position, current_heading = self.robot.get_position(), self.robot.get_heading()

            # First check change after sim step
            if self.prev_position is not None:

                # Detect if robot is still moving
                driving = distance(self.prev_position, current_position) > 0.0005  # 0.5 mm
                turning = abs(heading_difference(self.prev_heading, current_heading)) > math.radians(0.5)  # 0.5 deg
                self.consecutive_turning_sim_steps = (self.consecutive_turning_sim_steps + 1) if turning else 0
                stuck_oscillating = self.consecutive_turning_sim_steps > 240  # About 180 sim steps is sufficient for turning 180 deg
                not_moving = (not driving and not turning) or stuck_oscillating

                # Check for collisions
                if distance(self.robot.waypoint_positions[0], current_position) > RobotController.DRIVE_STEP_SIZE or not_moving:
                    self.robot.check_for_collisions()

                # Check if step limit exceeded (expect this won't ever happen, but just in case)
                step_limit_exceeded = self.sim_steps > 3200

                if self.robot.collided_with_obstacle or self.robot.collided_with_robot or step_limit_exceeded:
                    self.robot.update_distance()
                    self.state = 'idle'

                if self.state == 'driving' and not_moving:
                    # Reached current waypoint, move on to next waypoint
                    self.robot.update_distance()
                    if self.waypoint_index == len(self.robot.waypoint_positions) - 1:
                        self.state = 'slowing'
                        self.next_state = 'idle'
                        self.slowing_sim_step_target = self.env.slowing_sim_step_target
                    else:
                        self.waypoint_index += 1
                        self.state = 'turning'

            # If still moving, set constraint for new pose
            if self.state in {'turning', 'driving'}:
                new_position, new_heading = current_position, current_heading

                if self.state == 'turning':
                    # Determine whether to keep turning or start driving
                    next_waypoint_heading = self.robot.waypoint_headings[self.waypoint_index]
                    heading_diff = heading_difference(current_heading, next_waypoint_heading)
                    if abs(heading_diff) > RobotController.TURN_STEP_SIZE:
                        # Turn to face next waypoint
                        new_heading += math.copysign(1, heading_diff) * RobotController.TURN_STEP_SIZE
                    elif (isinstance(self.robot, BlowingRobot) and not isinstance(self.robot, MovingBlowingRobot) and self.robot.action[0] == 1):
                        # Rotation-only action
                        new_heading = next_waypoint_heading
                        if abs(heading_diff) < RobotController.ROTATION_TURN_THRESHOLD:
                            self.state = 'slowing'
                            self.next_state = 'idle'
                            self.slowing_sim_step_target = self.env.slowing_sim_step_target + self.env.blowing_sim_step_target
                    else:
                        if self.waypoint_index == 1:
                            # Only the first turn requires slowing before driving
                            self.state = 'slowing'
                            self.next_state = 'driving'
                            self.slowing_sim_step_target = self.env.slowing_sim_step_target
                        else:
                            self.state = 'driving'

                if self.state == 'driving':
                    # Drive forward
                    curr_waypoint_position = self.robot.waypoint_positions[self.waypoint_index]
                    dx = curr_waypoint_position[0] - current_position[0]
                    dy = curr_waypoint_position[1] - current_position[1]
                    if distance(current_position, curr_waypoint_position) < RobotController.DRIVE_STEP_SIZE:
                        new_position = curr_waypoint_position
                    else:
                        move_sign = math.copysign(1, distance(current_position, self.robot.target_end_effector_position) - self.robot.END_EFFECTOR_LOCATION)
                        # Note: To be consistent with rest of code, the line above should include an offset
                        #move_sign = math.copysign(1, distance(current_position, self.robot.target_end_effector_position) - (self.robot.END_EFFECTOR_LOCATION + self.env.object_width / 2))
                        new_heading = math.atan2(move_sign * dy, move_sign * dx)
                        new_position = (
                            new_position[0] + move_sign * RobotController.DRIVE_STEP_SIZE * math.cos(new_heading),
                            new_position[1] + move_sign * RobotController.DRIVE_STEP_SIZE * math.sin(new_heading),
                            new_position[2]
                        )

                # Set constraint
                self.env.p.changeConstraint(
                    self.robot.cid, jointChildPivot=new_position, jointChildFrameOrientation=heading_to_orientation(new_heading), maxForce=Robot.CONSTRAINT_MAX_FORCE)

            if self.state != 'slowing':
                self.prev_position, self.prev_heading = current_position, current_heading

        elif self.state == 'slowing':
            self.slowing_sim_steps += 1
            if self.slowing_sim_steps >= self.slowing_sim_step_target:
                self.slowing_sim_step_target = 0
                self.slowing_sim_steps = 0
                self.state = self.next_state
                self.next_state = None

        if isinstance(self.robot, BlowingRobot):
            self.robot.blow_objects()

class RealRobotController:
    LOOKAHEAD_DISTANCE = 0.1  # 10 cm
    TURN_THRESHOLD = math.radians(5)  # 5 deg

    def __init__(self, env, robot, real_robot_index, debug=False):
        self.env = env
        self.robot = robot
        self.real_robot_name = vector_utils.get_robot_name(real_robot_index)
        self.debug = debug
        self.real_robot = anki_vector.AsyncRobot(serial=vector_utils.get_robot_serial(real_robot_index), default_logging=False, behavior_control_level=anki_vector.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY)
        self.real_robot.connect()
        battery_state = self.real_robot.get_battery_state().result()
        battery_volts = '{:.2f}'.format(battery_state.battery_volts) if battery_state else '?'
        print('Connected to {} ({} V)'.format(self.real_robot_name, battery_volts))
        self._reset_motors()

        self.state = 'idle'
        self.resume_state = None  # For pausing
        self.waypoint_index = None  # Index of waypoint we are currently headed towards
        self.prev_position = None
        self.prev_heading = None
        self.sim_steps = 0
        self.not_driving_sim_steps = None
        self.not_turning_sim_steps = None

        if self.debug:
            self.debug_data = None

    def reset(self):
        self.real_robot.motors.set_wheel_motors(0, 0)
        if not isinstance(self.robot, BlowingRobot):
            self.real_robot.behavior.set_lift_height(0)

        self.state = 'idle'
        self.resume_state = None
        self.waypoint_index = 1
        self.prev_position = None
        self.prev_heading = None
        self.sim_steps = 0
        self.not_driving_sim_steps = 0
        self.not_turning_sim_steps = 0

        if self.debug:
            self.debug_data = None

    def new_action(self):
        self.state = 'turning'
        if isinstance(self.robot, BlowingRobot):
            if self.robot.action[0] > 0:
                # Note: If lift gets stuck on an obstacle when it is almost all the way down,
                # it will use that position as the new reference point for height 0.
                self.real_robot.behavior.set_lift_height(0)  # Lower the blower
            else:
                self.real_robot.behavior.set_lift_height(0.65)

    def step(self):
        # States: idle, stopping, turning, driving, slowing

        if self.state == 'idle':
            return

        self.sim_steps += 1

        # Periodically update the map (map updates are slow)
        if self.sim_steps % 20 == 0:
            self.robot.update_map()

        if self.state == 'stopping':
            self.real_robot.motors.set_wheel_motors(0, 0)
            if not self.real_robot.status.are_wheels_moving:
                self._done_stopping()

        elif self.state in {'turning', 'driving', 'slowing'}:
            current_position, current_heading = self.robot.get_position(), self.robot.get_heading()

            lookahead_position = self._get_lookahead_position()
            dx = lookahead_position[0] - current_position[0]
            dy = lookahead_position[1] - current_position[1]
            heading_diff = heading_difference(current_heading, math.atan2(dy, dx))

            if self.debug:
                self.debug_data = (lookahead_position, None, None, None, None)

            if self.state == 'turning':
                if abs(heading_diff) < RealRobotController.TURN_THRESHOLD:
                    self.real_robot.motors.set_wheel_motors(0, 0)
                    if not self.real_robot.status.are_wheels_moving:
                        if isinstance(self.robot, BlowingRobot) and self.robot.action[0] == 1:
                            self.real_robot.motors.set_wheel_motors(0, 0)
                            self.state = 'stopping'
                        else:
                            self.state = 'driving'
                else:
                    #speed = max(20, min(100, 100 * abs(heading_diff)))  # Must be at least 20 for marker detection to detect changes
                    speed = max(20, min(50, 100 * abs(heading_diff)))  # Must be at least 20 for marker detection to detect changes

                    if self.prev_heading is not None:
                        # Detect if robot is turning more slowly than expected
                        if abs(heading_difference(self.prev_heading, current_heading)) < speed / 2000:
                            self.not_turning_sim_steps += 1
                        else:
                            self.not_turning_sim_steps = 0
                        if self.not_turning_sim_steps > 20:
                            self.real_robot.motors.set_wheel_motors(0, 0)
                            self.state = 'stopping'

                    if self.state == 'turning':
                        sign = math.copysign(1, heading_diff)
                        self.real_robot.motors.set_wheel_motors(-1 * sign * speed, sign * speed)

            elif self.state in {'driving', 'slowing'}:
                signed_dist = distance(current_position, self.robot.target_end_effector_position) - (self.robot.END_EFFECTOR_LOCATION + self.env.object_width / 2)
                speed = max(20, min(100, 2000 * abs(signed_dist))) if self.state == 'slowing' else 100  # Must be at least 20 for marker detection to detect changes

                if self.prev_position is not None:
                    # Detect if robot is driving more slowly than expected
                    if distance(self.prev_position, current_position) < speed / 40000:
                        self.not_driving_sim_steps += 1
                    else:
                        self.not_driving_sim_steps = 0

                    # Check for collisions (It would be nice to have collision detection while turning too, but that is not currently implemented)
                    if distance(self.robot.waypoint_positions[0], current_position) > 0.01 or self.not_driving_sim_steps > 20:
                        self.robot.check_for_collisions()

                if self.robot.collided_with_obstacle or self.robot.collided_with_robot or self.not_driving_sim_steps > 20:
                    self.real_robot.motors.set_wheel_motors(0, 0)
                    self.state = 'stopping'

                elif self.state == 'driving' and signed_dist < 0.044:  # Hardcode 44 mm since it works well
                    self.state = 'slowing'

                elif self.state == 'slowing' and abs(signed_dist) < 0.002:  # 2 mm
                    self._done_slowing()

                else:
                    # Pure pursuit
                    lookahead_dist = math.sqrt(dx**2 + dy**2)
                    signed_radius = lookahead_dist / (2 * math.sin(heading_diff))
                    sign = math.copysign(1, signed_dist)
                    wheel_width = 0.1  # 10 cm (larger than actual width due to tread slip)
                    left_wheel_speed = sign * speed * (signed_radius - sign * wheel_width / 2) / signed_radius
                    right_wheel_speed = sign * speed * (signed_radius + sign * wheel_width / 2) / signed_radius

                    # Turn more forcefully if stuck
                    if isinstance(self.robot, PushingRobot) and abs(heading_diff) > RealRobotController.TURN_THRESHOLD and self.not_driving_sim_steps > 10:
                        if left_wheel_speed > right_wheel_speed:
                            right_wheel_speed = -left_wheel_speed
                        else:
                            left_wheel_speed = -right_wheel_speed

                    self.real_robot.motors.set_wheel_motors(left_wheel_speed, right_wheel_speed)

                    if self.debug:
                        self.debug_data = (lookahead_position, signed_radius, heading_diff, current_position, current_heading)

            self.prev_position, self.prev_heading = current_position, current_heading

    def pause(self):
        if self.state != 'idle':
            self.resume_state = self.state
            self.real_robot.motors.set_wheel_motors(0, 0)
            self.state = 'stopping'

    def resume(self):
        if self.resume_state is not None:
            self.state = self.resume_state
            self.resume_state = None

    def disconnect(self):
        self._reset_motors()
        self.real_robot.disconnect()
        print('Disconnected from {}'.format(self.real_robot_name))

    def _done_stopping(self):
        self.robot.update_distance()
        self.state = 'paused' if self.resume_state is not None else 'idle'

    def _done_slowing(self):
        self.real_robot.motors.set_wheel_motors(0, 0)
        self.state = 'stopping'

    def _reset_motors(self):
        self.real_robot.motors.set_wheel_motors(0, 0)
        if isinstance(self.robot, BlowingRobot):
            self.real_robot.behavior.set_lift_height(0.65)  # Using 1.0 causes the marker on top of the robot to be occluded
        else:
            self.real_robot.behavior.set_lift_height(0)
        self.real_robot.behavior.set_head_angle(anki_vector.util.degrees(0))

    def _get_lookahead_position(self):
        current_position = self.robot.get_position()
        while True:
            start = self.robot.waypoint_positions[self.waypoint_index - 1]
            end = self.robot.waypoint_positions[self.waypoint_index]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - current_position[0], start[1] - current_position[1])
            t2 = self._intersect(d, f, RealRobotController.LOOKAHEAD_DISTANCE)
            if t2 is not None:
                return (start[0] + t2 * d[0], start[1] + t2 * d[1])
            if self.waypoint_index == len(self.robot.waypoint_positions) - 1:
                return self.robot.target_end_effector_position
            self.robot.update_distance()
            self.waypoint_index += 1

    @staticmethod
    def _intersect(d, f, r, use_t1=False):
        # https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm/1084899%231084899
        a = dot(d, d)
        b = 2 * dot(f, d)
        c = dot(f, f) - r * r
        discriminant = (b * b) - (4 * a * c)
        if discriminant >= 0:
            if use_t1:
                t1 = (-b - math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t1 <= 1:
                    return t1
            else:
                t2 = (-b + math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t2 <= 1:
                    return t2
        return None

class Camera(ABC):
    NEAR = None
    FAR = None
    ASPECT = None
    FOV = 60  # Vertical FOV
    SEG_VALUES = {
        'floor': 1.0 / 8,
        'obstacle': 2.0 / 8,
        'receptacle': 3.0 / 8,
        'object': 4.0 / 8,
        'robot_group_1': 5.0 / 8,
        'robot_group_2': 6.0 / 8,
        'robot_group_3': 7.0 / 8,
        'robot_group_4': 8.0 / 8,
    }

    @abstractmethod  # Should not be instantiated directly
    def __init__(self, env):
        self.env = env
        self.image_pixel_height = int(1.63 * Mapper.LOCAL_MAP_PIXEL_WIDTH)
        self.image_pixel_width = int(self.ASPECT * self.image_pixel_height)
        self.projection_matrix = self.env.p.computeProjectionMatrixFOV(Camera.FOV, self.ASPECT, self.NEAR, self.FAR)
        self._initialized = False

        # Body ids for constructing the segmentation
        self.min_obstacle_id = None
        self.max_obstacle_id = None
        self.receptacle_id = None
        self.min_object_id = None
        self.max_object_id = None

    def _ensure_initialized(self):
        if self._initialized:
            return

        # Note: This should be called after the environment is fully created
        self.min_obstacle_id = min(self.env.obstacle_ids)
        self.max_obstacle_id = max(self.env.obstacle_ids)
        self.receptacle_id = self.env.receptacle_id
        self.min_object_id = None if len(self.env.object_ids) == 0 else min(self.env.object_ids)
        self.max_object_id = None if len(self.env.object_ids) == 0 else max(self.env.object_ids)
        self._initialized = True

    def capture_image(self, robot_position, robot_heading):
        self._ensure_initialized()

        # Capture images
        camera_position, camera_target, camera_up = self._get_camera_params(robot_position, robot_heading)
        view_matrix = self.env.p.computeViewMatrix(camera_position, camera_target, camera_up)
        images = self.env.p.getCameraImage(self.image_pixel_width, self.image_pixel_height, view_matrix, self.projection_matrix)
        #self.env.p.addUserDebugLine(camera_position, (np.array(camera_position) + Robot.HEIGHT * np.array(camera_up)).tolist(), (1, 0, 0))
        #self.env.p.addUserDebugLine(camera_position, camera_target, (1, 0, 0))

        # Compute depth
        depth_buffer = np.reshape(images[3], (self.image_pixel_height, self.image_pixel_width))
        depth = self.FAR * self.NEAR / (self.FAR - (self.FAR - self.NEAR) * depth_buffer)

        # Construct point cloud
        camera_position = np.array(camera_position, dtype=np.float32)
        principal = np.array(camera_target, dtype=np.float32) - camera_position
        principal = principal / np.linalg.norm(principal)
        camera_up = np.array(camera_up, dtype=np.float32)
        up = camera_up - np.dot(camera_up, principal) * principal
        up = up / np.linalg.norm(up)
        right = np.cross(principal, up)
        right = right / np.linalg.norm(right)
        limit_y = math.tan(math.radians(Camera.FOV / 2))
        limit_x = limit_y * self.ASPECT
        pixel_x = (2 * limit_x) * (np.arange(self.image_pixel_width, dtype=np.float32) / self.image_pixel_width - 0.5)
        pixel_y = (2 * limit_y) * (0.5 - (np.arange(self.image_pixel_height, dtype=np.float32) + 1) / self.image_pixel_height)
        pixel_xv, pixel_yv = np.meshgrid(pixel_x, pixel_y)
        points = camera_position + depth[:, :, np.newaxis] * (principal + pixel_xv[:, :, np.newaxis] * right + pixel_yv[:, :, np.newaxis] * up)
        #for point in points[0, ::5, :]:
        #    self.env.p.addUserDebugLine(camera_position, point, (1, 0, 0))
        #for point in points[-1, ::5, :]:
        #    self.env.p.addUserDebugLine(camera_position, point, (1, 0, 0))

        # Construct segmentation
        seg_raw = np.reshape(images[4], (self.image_pixel_height, self.image_pixel_width))
        seg = Camera.SEG_VALUES['floor'] * (seg_raw == 0).astype(np.float32)
        seg += Camera.SEG_VALUES['obstacle'] * np.logical_and(seg_raw >= self.min_obstacle_id, seg_raw <= self.max_obstacle_id).astype(np.float32)
        if self.receptacle_id is not None:
            seg += Camera.SEG_VALUES['receptacle'] * (seg_raw == self.receptacle_id).astype(np.float32)
        if self.min_object_id is not None:
            seg += Camera.SEG_VALUES['object'] * np.logical_and(seg_raw >= self.min_object_id, seg_raw <= self.max_object_id).astype(np.float32)
        #from PIL import Image; import utils; Image.fromarray(utils.to_uint8_image(seg)).show()

        return points, seg

    def get_seg_value(self, body_type):
        self._ensure_initialized()
        return Camera.SEG_VALUES[body_type]

    @abstractmethod
    def _get_camera_params(self, robot_position, robot_heading):
        pass

class OverheadCamera(Camera):
    HEIGHT = 1  # 1 m
    ASPECT = 1
    NEAR = 0.1  # 10 cm
    FAR = 10  # 10 m

    def __init__(self, env):
        super().__init__(env)

    def _get_camera_params(self, robot_position, robot_heading):
        camera_position = (robot_position[0], robot_position[1], OverheadCamera.HEIGHT)
        camera_target = (robot_position[0], robot_position[1], 0)
        camera_up = (math.cos(robot_heading), math.sin(robot_heading), 0)
        return camera_position, camera_target, camera_up

class ForwardFacingCamera(Camera):
    HEIGHT = Robot.HEIGHT
    PITCH = -30
    ASPECT = 16.0 / 9  # 60 deg vertical FOV, 90 deg horizontal FOV
    NEAR = 0.001  # 1 mm
    FAR = 1  # 1 m

    def __init__(self, env):
        super().__init__(env)

    def _get_camera_params(self, robot_position, robot_heading):
        camera_position_offset = Robot.BACKPACK_OFFSET + Robot.TOP_LENGTH + 0.002  # Move forward additional 2 mm (past lifted object)
        camera_position = (
            robot_position[0] + camera_position_offset * math.cos(robot_heading),
            robot_position[1] + camera_position_offset * math.sin(robot_heading),
            ForwardFacingCamera.HEIGHT
        )
        camera_target_offset = ForwardFacingCamera.HEIGHT * math.tan(math.radians(90 + ForwardFacingCamera.PITCH))
        camera_target = (
            camera_position[0] + camera_target_offset * math.cos(robot_heading),
            camera_position[1] + camera_target_offset * math.sin(robot_heading),
            0
        )
        camera_up = (
            math.cos(math.radians(90 + ForwardFacingCamera.PITCH)) * math.cos(robot_heading),
            math.cos(math.radians(90 + ForwardFacingCamera.PITCH)) * math.sin(robot_heading),
            math.sin(math.radians(90 + ForwardFacingCamera.PITCH))
        )
        return camera_position, camera_target, camera_up

class Mapper:
    LOCAL_MAP_PIXEL_WIDTH = 96
    LOCAL_MAP_WIDTH = 1  # 1 meter
    LOCAL_MAP_PIXELS_PER_METER = LOCAL_MAP_PIXEL_WIDTH / LOCAL_MAP_WIDTH

    def __init__(self, env, robot):
        self.env = env
        self.robot = robot

        # Camera
        if self.env.use_partial_observations:
            self.camera = ForwardFacingCamera(self.env)
        else:
            self.camera = OverheadCamera(self.env)

        # Overhead map
        self.global_overhead_map_without_robots = self._create_padded_room_zeros()

        # Occupancy map
        self.global_occupancy_map = OccupancyMap(self.robot, self.env.room_length, self.env.room_width, show_map=self.env.show_occupancy_maps)

        # Robot masks for overhead map and robot map
        self.robot_masks = {}
        for g in self.env.robot_config:
            robot_type = next(iter(g))
            robot_cls = Robot.get_robot_cls(robot_type)
            self.robot_masks[robot_cls] = self._create_robot_mask(robot_cls)

        # Precompute global distance to receptacle map
        if self.env.use_distance_to_receptacle_map:
            self.global_distance_to_receptacle_map = self._create_global_distance_to_receptacle_map()

        # Assertions
        if self.env.use_distance_to_receptacle_map or self.env.use_shortest_path_to_receptacle_map:
            assert self.env.receptacle_id is not None

        if self.env.real:
            # For real robot only, mask the overhead map in case robot drives outside of the room
            selem = disk(4)
            self.room_mask = dilation(self.global_occupancy_map._create_room_mask(), selem)

    def update(self):
        # Get new observation
        points, seg = self.camera.capture_image(self.robot.get_position(), self.robot.get_heading())
        augmented_points = np.concatenate((points, seg[:, :, np.newaxis]), axis=2).reshape(-1, 4)
        augmented_points = augmented_points[np.argsort(augmented_points[:, 2])]

        # Incorporate new observation into overhead map
        pixel_i, pixel_j = Mapper.position_to_pixel_indices(augmented_points[:, 0], augmented_points[:, 1], self.global_overhead_map_without_robots.shape)
        self.global_overhead_map_without_robots[pixel_i, pixel_j] = self.env.overhead_map_scale * augmented_points[:, 3]
        if self.env.real:
            mask_pixel_i, mask_pixel_j = Mapper.position_to_pixel_indices(augmented_points[:, 0], augmented_points[:, 1], self.env.object_mask.shape)
            points_mask = self.env.object_mask[mask_pixel_i, mask_pixel_j] > 0
            self.global_overhead_map_without_robots[pixel_i[points_mask], pixel_j[points_mask]] = self.env.overhead_map_scale * self.camera.get_seg_value('object')
            # Note: objects in receptacle are not masked out, so they need to be removed quickly

            # Mask out regions beyond the wall in case robot drives outside of the room
            self.global_overhead_map_without_robots *= self.room_mask

        # Update occupancy map
        if self.global_occupancy_map is not None:
            self.global_occupancy_map.update(points, seg, self.camera.get_seg_value('obstacle'))

    def get_state(self, save_figures=False):
        channels = []

        # Overhead map
        global_overhead_map = self._create_global_overhead_map()
        local_overhead_map = self._get_local_map(global_overhead_map)
        channels.append(local_overhead_map)

        # Robot map
        if self.env.use_robot_map:
            global_robot_map = self._create_global_robot_map(seg=False)
            local_robot_map = self._get_local_map(global_robot_map)
            channels.append(local_robot_map)

        # Distance to receptacle map
        if self.env.use_distance_to_receptacle_map:
            channels.append(self._get_local_distance_map(self.global_distance_to_receptacle_map))

        # Shortest path distance to receptacle map
        if self.env.use_shortest_path_to_receptacle_map:
            global_shortest_path_to_receptacle_map = self._create_global_shortest_path_to_receptacle_map()
            local_shortest_path_to_receptacle_map = self._get_local_distance_map(global_shortest_path_to_receptacle_map)
            channels.append(local_shortest_path_to_receptacle_map)

        # Shortest path distance map
        if self.env.use_shortest_path_map:
            global_shortest_path_map = self._create_global_shortest_path_map()
            local_shortest_path_map = self._get_local_distance_map(global_shortest_path_map)
            channels.append(local_shortest_path_map)

        if save_figures:
            from PIL import Image; import utils  # pylint: disable=import-outside-toplevel
            output_dir = Path('figures') / 'robot_id_{}'.format(self.robot.id)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)

            def global_map_room_only(global_map):
                crop_width = Mapper.round_up_to_even((self.env.room_length + 2 * Robot.HALF_WIDTH) * Mapper.LOCAL_MAP_PIXELS_PER_METER)
                crop_height = Mapper.round_up_to_even((self.env.room_width + 2 * Robot.HALF_WIDTH) * Mapper.LOCAL_MAP_PIXELS_PER_METER)
                start_i = global_map.shape[0] // 2 - crop_height // 2
                start_j = global_map.shape[1] // 2 - crop_width // 2
                return global_map[start_i:start_i + crop_height, start_j:start_j + crop_width]

            # Environment
            Image.fromarray(self.env.get_camera_image()).save(output_dir / 'env.png')

            def visualize_overhead_map(global_overhead_map, local_overhead_map):
                brightness_scale_factor = 1.33
                global_overhead_map_vis = brightness_scale_factor * global_map_room_only(global_overhead_map)
                local_overhead_map_vis = brightness_scale_factor * local_overhead_map
                return global_overhead_map_vis, local_overhead_map_vis

            # Overhead map
            global_overhead_map_vis, local_overhead_map_vis = visualize_overhead_map(global_overhead_map, local_overhead_map)
            utils.enlarge_image(Image.fromarray(utils.to_uint8_image(global_overhead_map_vis))).save(output_dir / 'global-overhead-map.png')
            utils.enlarge_image(Image.fromarray(utils.to_uint8_image(local_overhead_map_vis))).save(output_dir / 'local-overhead-map.png')

            def visualize_map(overhead_map_vis, distance_map):
                overhead_map_vis = np.stack(3 * [overhead_map_vis], axis=2)
                distance_map_vis = utils.JET[utils.to_uint8_image(distance_map), :]
                return 0.5 * overhead_map_vis + 0.5 * distance_map_vis

            def save_map_visualization(global_map, local_map, suffix, brightness_scale_factor=1):
                global_map_vis = global_map_room_only(global_map)
                global_map_vis = visualize_map(global_overhead_map_vis, brightness_scale_factor * global_map_vis)
                utils.enlarge_image(Image.fromarray(utils.to_uint8_image(global_map_vis))).save(output_dir / 'global-{}.png'.format(suffix))
                local_map = visualize_map(local_overhead_map_vis, brightness_scale_factor * local_map)
                utils.enlarge_image(Image.fromarray(utils.to_uint8_image(local_map))).save(output_dir / 'local-{}.png'.format(suffix))

            # Robot map
            if self.env.use_robot_map:
                save_map_visualization(global_robot_map, local_robot_map, 'robot-map')

            # Shortest path distance to receptacle map
            if self.env.use_shortest_path_to_receptacle_map:
                save_map_visualization(global_shortest_path_to_receptacle_map, local_shortest_path_to_receptacle_map, 'shortest-path-to-receptacle-map', brightness_scale_factor=2)

            # Shortest path distance map
            if self.env.use_shortest_path_map:
                save_map_visualization(global_shortest_path_map, local_shortest_path_map, 'shortest-path-map', brightness_scale_factor=2)

            # Occupancy map
            if self.env.show_occupancy_maps:
                self.global_occupancy_map.save_figure(output_dir / 'global-occupancy-map.png')

        assert all(channel.dtype == np.float32 for channel in channels)
        return np.stack(channels, axis=2)

    def shortest_path(self, source_position, target_position):
        return self.global_occupancy_map.shortest_path(source_position, target_position)

    def distance_to_receptacle(self, position):
        # Note: Initial and final distances should be computed with the same configuration space
        assert self.env.receptacle_id is not None
        if self.env.use_shortest_path_partial_rewards:
            # Use receptacle as shortest path source for better caching
            return self._shortest_path_distance(self.env.receptacle_position, position)
        return distance(position, self.env.receptacle_position)

    def _shortest_path_distance(self, source_position, target_position):
        return self.global_occupancy_map.shortest_path_distance(source_position, target_position)

    def _get_local_map(self, global_map):
        robot_position, robot_heading = self.robot.get_position(), self.robot.get_heading()
        crop_width = Mapper.round_up_to_even(math.sqrt(2) * Mapper.LOCAL_MAP_PIXEL_WIDTH)
        rotation_angle = 90 - math.degrees(robot_heading)
        pixel_i, pixel_j = Mapper.position_to_pixel_indices(robot_position[0], robot_position[1], global_map.shape)
        crop = global_map[pixel_i - crop_width // 2:pixel_i + crop_width // 2, pixel_j - crop_width // 2:pixel_j + crop_width // 2]
        rotated_crop = rotate_image(crop, rotation_angle, order=0)
        local_map = rotated_crop[
            rotated_crop.shape[0] // 2 - Mapper.LOCAL_MAP_PIXEL_WIDTH // 2:rotated_crop.shape[0] // 2 + Mapper.LOCAL_MAP_PIXEL_WIDTH // 2,
            rotated_crop.shape[1] // 2 - Mapper.LOCAL_MAP_PIXEL_WIDTH // 2:rotated_crop.shape[1] // 2 + Mapper.LOCAL_MAP_PIXEL_WIDTH // 2
        ]
        #from PIL import Image; import utils; Image.fromarray(utils.to_uint8_image(local_map)).show()
        return local_map

    def _get_local_distance_map(self, global_map):
        local_map = self._get_local_map(global_map)
        local_map -= local_map.min()
        return local_map

    @staticmethod
    def _create_robot_mask(robot_cls):
        robot_pixel_width = math.ceil(2 * robot_cls.RADIUS * Mapper.LOCAL_MAP_PIXELS_PER_METER)
        robot_mask = np.zeros((Mapper.LOCAL_MAP_PIXEL_WIDTH, Mapper.LOCAL_MAP_PIXEL_WIDTH), dtype=np.float32)
        start = math.floor(Mapper.LOCAL_MAP_PIXEL_WIDTH / 2 - robot_pixel_width / 2)

        for i in range(start, start + robot_pixel_width):
            for j in range(start, start + robot_pixel_width):
                position_x, position_y = Mapper.pixel_indices_to_position(i, j, robot_mask.shape)
                # Rectangular base
                in_base = abs(position_x) <= Robot.HALF_WIDTH and 0 <= position_y - Robot.BACKPACK_OFFSET <= robot_cls.BASE_LENGTH
                in_backpack = position_x**2 + (position_y - Robot.BACKPACK_OFFSET)**2 <= Robot.HALF_WIDTH ** 2  # Circular backpack
                if in_base or in_backpack:
                    robot_mask[i, j] = 1

        #from PIL import Image; Image.fromarray(255 * robot_mask).show()
        return robot_mask

    def _create_global_overhead_map(self):
        global_overhead_map = self.global_overhead_map_without_robots.copy()
        global_robot_map_seg = self._create_global_robot_map(seg=True)
        global_overhead_map[global_robot_map_seg > 0] = global_robot_map_seg[global_robot_map_seg > 0]
        assert global_overhead_map.max() <= max(self.env.robot_map_scale, self.env.overhead_map_scale)
        #from PIL import Image; import utils; Image.fromarray(utils.to_uint8_image(global_overhead_map)).show()
        return global_overhead_map

    def _create_global_robot_map(self, seg=True):
        global_robot_map = self._create_padded_room_zeros()
        for robot in self.env.robots:
            # Create robot visualization
            robot_vis = self.robot_masks[robot.__class__].copy()
            if seg:
                robot_vis *= self.camera.get_seg_value('robot_group_{}'.format(robot.group_index + 1))
            else:
                robot_vis *= self.env.robot_map_scale

            # Rotate based on robot heading
            rotation_angle = math.degrees(robot.get_heading()) - 90
            rotated = rotate_image(robot_vis, rotation_angle, order=0)

            # Place into global robot map
            robot_position = robot.get_position()
            pixel_i, pixel_j = Mapper.position_to_pixel_indices(robot_position[0], robot_position[1], global_robot_map.shape)
            start_i, start_j = pixel_i - rotated.shape[0] // 2, pixel_j - rotated.shape[1] // 2
            global_robot_map[start_i:start_i + rotated.shape[0], start_j:start_j + rotated.shape[1]] = np.maximum(
                global_robot_map[start_i:start_i + rotated.shape[0], start_j:start_j + rotated.shape[1]], rotated)
        #from PIL import Image; import utils; Image.fromarray(utils.to_uint8_image(global_robot_map)).show()
        return global_robot_map

    def _create_global_distance_to_receptacle_map(self):
        assert self.env.receptacle_id is not None
        global_map = self._create_padded_room_zeros()
        for i in range(global_map.shape[0]):
            for j in range(global_map.shape[1]):
                pos_x, pos_y = Mapper.pixel_indices_to_position(i, j, global_map.shape)
                global_map[i, j] = distance((pos_x, pos_y), self.env.receptacle_position)
        global_map *= self.env.distance_to_receptacle_map_scale
        #from PIL import Image; import utils; Image.fromarray(utils.to_uint8_image(global_map)).show()
        return global_map

    def _create_global_shortest_path_to_receptacle_map(self):
        assert self.env.receptacle_id is not None
        global_map = self.global_occupancy_map.shortest_path_image(self.env.receptacle_position)
        global_map[global_map < 0] = global_map.max()
        global_map *= self.env.shortest_path_map_scale
        #from PIL import Image; import utils; Image.fromarray(utils.to_uint8_image(global_map)).show()
        return global_map

    def _create_global_shortest_path_map(self):
        robot_position = self.robot.get_position()
        global_map = self.global_occupancy_map.shortest_path_image(robot_position)
        global_map[global_map < 0] = global_map.max()
        global_map *= self.env.shortest_path_map_scale
        #from PIL import Image; import utils; Image.fromarray(utils.to_uint8_image(global_map)).show()
        return global_map

    def _create_padded_room_zeros(self):
        return Mapper.create_padded_room_zeros(self.env.room_width, self.env.room_length)

    @staticmethod
    def create_padded_room_zeros(room_width, room_length):
        # Ensure dimensions are even
        return np.zeros((
            Mapper.round_up_to_even(room_width * Mapper.LOCAL_MAP_PIXELS_PER_METER + math.sqrt(2) * Mapper.LOCAL_MAP_PIXEL_WIDTH),
            Mapper.round_up_to_even(room_length * Mapper.LOCAL_MAP_PIXELS_PER_METER + math.sqrt(2) * Mapper.LOCAL_MAP_PIXEL_WIDTH)
        ), dtype=np.float32)

    @staticmethod
    def position_to_pixel_indices(position_x, position_y, image_shape):
        pixel_i = np.floor(image_shape[0] / 2 - position_y * Mapper.LOCAL_MAP_PIXELS_PER_METER).astype(np.int32)
        pixel_j = np.floor(image_shape[1] / 2 + position_x * Mapper.LOCAL_MAP_PIXELS_PER_METER).astype(np.int32)
        pixel_i = np.clip(pixel_i, 0, image_shape[0] - 1)
        pixel_j = np.clip(pixel_j, 0, image_shape[1] - 1)
        return pixel_i, pixel_j

    @staticmethod
    def pixel_indices_to_position(pixel_i, pixel_j, image_shape):
        position_x = ((pixel_j + 0.5) - image_shape[1] / 2) / Mapper.LOCAL_MAP_PIXELS_PER_METER
        position_y = (image_shape[0] / 2 - (pixel_i + 0.5)) / Mapper.LOCAL_MAP_PIXELS_PER_METER
        return position_x, position_y

    @staticmethod
    def round_up_to_even(x):
        return 2 * math.ceil(x / 2)

class OccupancyMap:
    def __init__(self, robot, room_length, room_width, show_map=False):
        self.robot = robot
        self.room_length = room_length
        self.room_width = room_width
        self.show_map = show_map

        # Binary map showing where obstacles are
        self.occupancy_map = self._create_padded_room_zeros().astype(np.uint8)

        # Configuration space for computing shortest paths
        self.configuration_space = None
        self.selem = disk(math.floor(self.robot.RADIUS * Mapper.LOCAL_MAP_PIXELS_PER_METER))
        self.closest_cspace_indices = None

        # Grid graph for computing shortest paths
        self.grid_graph = None

        # Configuration space checking for straight line paths
        self.cspace_thin = None
        self.selem_thin = disk(math.ceil(Robot.HALF_WIDTH * Mapper.LOCAL_MAP_PIXELS_PER_METER))

        # Precompute room mask, which is used to mask out the wall pixels
        self.room_mask = self._create_room_mask()

        if self.show_map:
            import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
            self.plt = plt
            self.plt.ion()
            self.fig_width = self.room_length + 2 * Robot.HALF_WIDTH
            self.fig_height = self.room_width + 2 * Robot.HALF_WIDTH
            figsize = (4 * self.fig_width, 4 * self.fig_height)
            self.fig = self.plt.figure(self.robot.id, figsize=figsize)
            self.free_space_map = self._create_padded_room_zeros().astype(np.uint8)
            self._update_map_visualization()

    def update(self, points, seg, obstacle_seg_value):
        # Incorporate new observation into occupancy map
        augmented_points = np.concatenate([points, np.isclose(seg[:, :, np.newaxis], obstacle_seg_value)], axis=2).reshape(-1, 4)
        obstacle_points = augmented_points[np.isclose(augmented_points[:, 3], 1)]
        pixel_i, pixel_j = Mapper.position_to_pixel_indices(obstacle_points[:, 0], obstacle_points[:, 1], self.occupancy_map.shape)
        self.occupancy_map[pixel_i, pixel_j] = 1
        assert self.occupancy_map.dtype == np.uint8

        # Update configuration space
        self.configuration_space = 1 - np.maximum(1 - self.room_mask, binary_dilation(self.occupancy_map, self.selem).astype(np.uint8))
        self.closest_cspace_indices = distance_transform_edt(1 - self.configuration_space, return_distances=False, return_indices=True)
        self.cspace_thin = 1 - binary_dilation(np.minimum(self.room_mask, self.occupancy_map), self.selem_thin).astype(np.uint8)  # No walls
        assert self.configuration_space.dtype == np.uint8
        #from PIL import Image; Image.fromarray(255 * self.configuration_space).show()
        #from PIL import Image; Image.fromarray(255 * self.cspace_thin).show()

        # Create a new grid graph with updated configuration space
        self.grid_graph = GridGraph(self.configuration_space)

        if self.show_map:
            free_space_points = augmented_points[np.isclose(augmented_points[:, 3], 0)]
            pixel_i, pixel_j = Mapper.position_to_pixel_indices(free_space_points[:, 0], free_space_points[:, 1], self.free_space_map.shape)
            self.free_space_map[pixel_i, pixel_j] = 1
            self._update_map_visualization()

    def _create_room_mask(self):
        room_mask = self._create_padded_room_zeros().astype(np.uint8)
        room_length_pixels = Mapper.round_up_to_even((self.room_length - 2 * Robot.HALF_WIDTH) * Mapper.LOCAL_MAP_PIXELS_PER_METER)
        room_width_pixels = Mapper.round_up_to_even((self.room_width - 2 * Robot.HALF_WIDTH) * Mapper.LOCAL_MAP_PIXELS_PER_METER)
        start_i = int(room_mask.shape[0] / 2 - room_width_pixels / 2)
        start_j = int(room_mask.shape[1] / 2 - room_length_pixels / 2)
        room_mask[start_i:start_i + room_width_pixels, start_j:start_j + room_length_pixels] = 1
        #from PIL import Image; Image.fromarray(255 * room_mask).show()
        assert room_mask.dtype == np.uint8
        return room_mask

    def shortest_path(self, source_position, target_position):
        # Convert positions to pixel indices
        source_i, source_j = Mapper.position_to_pixel_indices(source_position[0], source_position[1], self.configuration_space.shape)
        target_i, target_j = Mapper.position_to_pixel_indices(target_position[0], target_position[1], self.configuration_space.shape)

        # Check if there is a straight line path
        rr, cc = line(source_i, source_j, target_i, target_j)
        if (1 - self.cspace_thin[rr, cc]).sum() == 0:
            return [source_position, target_position]

        # Run SPFA
        source_i, source_j = self._closest_valid_cspace_indices(source_i, source_j)
        target_i, target_j = self._closest_valid_cspace_indices(target_i, target_j)
        path_pixel_indices = self.grid_graph.shortest_path((source_i, source_j), (target_i, target_j))

        # Convert pixel indices back to positions
        path = []
        for i, j in path_pixel_indices:
            position_x, position_y = Mapper.pixel_indices_to_position(i, j, self.configuration_space.shape)
            path.append((position_x, position_y, 0))

        if len(path) < 2:
            path = [source_position, target_position]
        else:
            path[0] = source_position
            path[-1] = target_position

        return path

    def shortest_path_distance(self, source_position, target_position):
        source_i, source_j = Mapper.position_to_pixel_indices(source_position[0], source_position[1], self.configuration_space.shape)
        target_i, target_j = Mapper.position_to_pixel_indices(target_position[0], target_position[1], self.configuration_space.shape)
        source_i, source_j = self._closest_valid_cspace_indices(source_i, source_j)
        target_i, target_j = self._closest_valid_cspace_indices(target_i, target_j)
        return self.grid_graph.shortest_path_distance((source_i, source_j), (target_i, target_j)) / Mapper.LOCAL_MAP_PIXELS_PER_METER

    def shortest_path_image(self, position):
        target_i, target_j = Mapper.position_to_pixel_indices(position[0], position[1], self.configuration_space.shape)
        target_i, target_j = self._closest_valid_cspace_indices(target_i, target_j)
        return self.grid_graph.shortest_path_image((target_i, target_j)) / Mapper.LOCAL_MAP_PIXELS_PER_METER

    def save_figure(self, output_path):
        assert self.show_map
        self.fig.savefig(output_path, bbox_inches='tight', pad_inches=0)

    def _closest_valid_cspace_indices(self, i, j):
        return self.closest_cspace_indices[:, i, j]

    def _create_padded_room_zeros(self):
        return Mapper.create_padded_room_zeros(self.room_width, self.room_length)

    def _update_map_visualization(self):
        # Create map visualization
        occupancy_map_vis = self._create_padded_room_zeros() + 0.5
        occupancy_map_vis[self.free_space_map == 1] = 1
        occupancy_map_vis[self.occupancy_map == 1] = 0

        # Show map visualization
        self.fig.clf()
        self.fig.add_axes((0, 0, 1, 1))
        ax = self.fig.gca()
        ax.axis('off')
        ax.axis([-self.fig_width / 2, self.fig_width / 2, -self.fig_height / 2, self.fig_height / 2])
        height, width = occupancy_map_vis.shape
        height, width = height / Mapper.LOCAL_MAP_PIXELS_PER_METER, width / Mapper.LOCAL_MAP_PIXELS_PER_METER
        ax.imshow(255.0 * occupancy_map_vis, extent=(-width / 2, width / 2, -height / 2, height / 2), cmap='gray', vmin=0, vmax=255.0)

        # Show waypoint positions
        if self.robot.waypoint_positions is not None:
            waypoint_positions = np.array(self.robot.waypoint_positions)
            ax.plot(waypoint_positions[:, 0], waypoint_positions[:, 1], color='r', marker='.')

        # Show target end effector position
        if self.robot.target_end_effector_position is not None:
            ax.plot(self.robot.target_end_effector_position[0], self.robot.target_end_effector_position[1], color='r', marker='x')

        # Update display
        self.plt.pause(0.001)

def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def orientation_to_heading(o):
    # Note: Only works for z-axis rotations
    return 2 * math.acos(math.copysign(1, o[2]) * o[3])

def heading_to_orientation(h):
    return pybullet.getQuaternionFromEuler((0, 0, h))

def restrict_heading_range(h):
    return (h + math.pi) % (2 * math.pi) - math.pi

def heading_difference(h1, h2):
    return restrict_heading_range(h2 - h1)

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]
