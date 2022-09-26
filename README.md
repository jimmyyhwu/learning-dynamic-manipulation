# learning-dynamic-manipulation

This code release accompanies the following paper:

### Learning Pneumatic Non-Prehensile Manipulation with a Mobile Blower

Jimmy Wu, Xingyuan Sun, Andy Zeng, Shuran Song, Szymon Rusinkiewicz, Thomas Funkhouser

*IEEE Robotics and Automation Letters (RA-L)*, 2022

*IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2022

[Project Page](https://learning-dynamic-manipulation.cs.princeton.edu) | [PDF](https://learning-dynamic-manipulation.cs.princeton.edu/paper.pdf) | [arXiv](https://arxiv.org/abs/2204.02390) | [Video](https://youtu.be/TsY83Kg0KAk)

**Abstract:** We investigate pneumatic non-prehensile manipulation (i.e., blowing) as a means of efficiently moving scattered objects into a target receptacle. Due to the chaotic nature of aerodynamic forces, a blowing controller must (i) continually adapt to unexpected changes from its actions, (ii) maintain fine-grained control, since the slightest misstep can result in large unintended consequences (e.g., scatter objects already in a pile), and (iii) infer long-range plans (e.g., move the robot to strategic blowing locations). We tackle these challenges in the context of deep reinforcement learning, introducing a multi-frequency version of the spatial action maps framework. This allows for efficient learning of vision-based policies that effectively combine high-level planning and low-level closed-loop control for dynamic mobile manipulation. Experiments show that our system learns efficient behaviors for the task, demonstrating in particular that blowing achieves better downstream performance than pushing, and that our policies improve performance over baselines. Moreover, we show that our system naturally encourages emergent specialization between the different subpolicies spanning low-level fine-grained control and high-level planning. On a real mobile robot equipped with a miniature air blower, we show that our simulation-trained policies transfer well to a real environment and can generalize to novel objects.

![](https://user-images.githubusercontent.com/6546428/161405574-e4d1cca9-3242-4625-b629-0bdd09efa054.gif) | ![](https://user-images.githubusercontent.com/6546428/161405575-0903f4c9-eeca-4420-997c-eb1dbf66c1eb.gif) | ![](https://user-images.githubusercontent.com/6546428/161405576-13f446bd-47ca-4d70-b6eb-387d61f337d8.gif)
:---: | :---: | :---:
![](https://user-images.githubusercontent.com/6546428/169499688-b5585212-2fe5-4a99-bd73-c41668abbe8a.gif) | ![](https://user-images.githubusercontent.com/6546428/169499691-8aa03784-052b-4946-862d-1022b8982810.gif) | ![](https://user-images.githubusercontent.com/6546428/169499681-891e1be1-4428-4f7a-b017-7f9f7cc3d293.gif)

## Installation

We recommend using a [`conda`](https://docs.conda.io/en/latest/miniconda.html) environment for this codebase. The following commands will set up a new conda environment with the correct requirements (tested on Ubuntu 18.04.6 LTS):

```bash
# Create and activate new conda env
conda create -y -n my-conda-env python=3.7.10
conda activate my-conda-env

# Install mkl numpy
conda install -y numpy==1.20.2

# Install pytorch
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# Install pip requirements
pip install -r requirements.txt

# Install shortest paths module (used in simulation environment)
cd shortest_paths
python setup.py build_ext --inplace
```

## Quickstart

We provide pretrained policies for each test environment. The `download-pretrained.sh` script will download the pretrained policies and save their configs and network weights into the `logs` and `checkpoints` directories, respectively. Use the following command to run it:

```bash
./download-pretrained.sh
```

You can then use `enjoy.py` to run a pretrained policy in the simulation environment. Here are a few examples you can try:

```bash
python enjoy.py --config-path logs/pretrained-blowing_1-small_empty-multifreq_4/config.yml
python enjoy.py --config-path logs/pretrained-blowing_1-large_empty-multifreq_4/config.yml
python enjoy.py --config-path logs/pretrained-blowing_1-large_columns-multifreq_4/config.yml
python enjoy.py --config-path logs/pretrained-blowing_1-large_door-multifreq_4/config.yml
python enjoy.py --config-path logs/pretrained-blowing_1-large_center-multifreq_4/config.yml
```

You should see the pretrained policy running in the PyBullet GUI that pops up.

You can also run `enjoy.py` without specifying a config path, and it will list all policies in the `logs` directory and allow you to pick one to run:

```bash
python enjoy.py
```

## Training in the Simulation Environment

The [`config/experiments`](config/experiments) directory contains the template config files used for all experiments in the paper. To start a training run, you can provide one of the template config files to the `train.py` script. For example, the following will train a multi-frequency blowing policy in the `SmallEmpty` environment:

```bash
python train.py --config-path config/experiments/multifreq/blowing_1-small_empty-multifreq_4.yml
```

The training script will create a log directory and checkpoint directory for the new training run inside `logs/` and `checkpoints/`, respectively. Inside the log directory, it will also create a new config file called `config.yml`, which stores training run config variables and can be used to resume training or to load a trained policy for evaluation.

### Simulation Environment

To interactively explore the simulation environment using our dense action space (spatial action maps), you can use `tools_interactive_gui.py`, which will load an environment and allow you to click on the agent's local overhead map to select navigational endpoints (each pixel is an action). Since the blowing robot has a 2-channel action space, the GUI will show two copies of the overhead map, one per action space channel. The first channel is for the `move-without-blowing` action type, and the second channel is for the `turn-while-blowing` action type.

```bash
python tools_interactive_gui.py
```

### Evaluation

Trained policies can be evaluated using the `evaluate.py` script, which takes in the config path for the training run. For example, to evaluate the pretrained multi-frequency blowing policy in the `SmallEmpty` environment, you can run:

```
python evaluate.py --config-path logs/pretrained-blowing_1-small_empty-multifreq_4/config.yml
```

This will load the trained policy from the specified training run, and run evaluation on it. The results are saved to an `.npy` file in the `eval` directory. You can then run `jupyter notebook` and navigate to [`eval_summary.ipynb`](eval_summary.ipynb) to load the `.npy` files and generate tables and plots of the results.

## Running in the Real Environment

We train policies in simulation and run them directly on the real robot by mirroring the real environment inside the simulation. To do this, we first use [ArUco](https://docs.opencv.org/4.4.0/d5/dae/tutorial_aruco_detection.html) markers to estimate 2D pose of the robot in the real environment, and then use the estimated pose to update the simulation. Note that setting up the real environment, particularly the marker pose estimation, can take a fair amount of time and effort.

### Vector SDK Setup

If you previously ran `pip install -r requirements.txt` following the installation instructions above, the `anki_vector` library should already be installed. Run the following command to set up the robot:

```bash
python -m anki_vector.configure
```

After the setup is complete, you can open the Vector config file located at `~/.anki_vector/sdk_config.ini` to verify that your robot is present.

You can also run some of the [official examples](https://developer.anki.com/vector/docs/downloads.html#sdk-examples) to verify that the setup procedure worked. For further reference, please see the [Vector SDK documentation](https://developer.anki.com/vector/docs/index.html).

### Connecting to the Vector

The following command will try to connect to all the robots in your Vector config file and keep them still. It will print out a message for each robot it successfully connects to, and can be used to verify that the Vector SDK can connect to all of your robots.

```bash
python vector_keep_still.py
```

**Note:** If you get the following error, you will need to make a small fix to the `anki_vector` library.

```
AttributeError: module 'anki_vector.connection' has no attribute 'CONTROL_PRIORITY_LEVEL'
```

Locate the `anki_vector/behavior.py` file inside your installed conda libraries. The full path should be in the error message. At the bottom of `anki_vector/behavior.py`, change `connection.CONTROL_PRIORITY_LEVEL.RESERVE_CONTROL` to `connection.ControlPriorityLevel.RESERVE_CONTROL`.

---

Sometimes the IP addresses of your robots will change. To update the Vector config file with new IP addresses, you can run the following command:

```bash
python vector_run_mdns.py
```

The script uses mDNS to find all Vector robots on the local network, and will automatically update their IP addresses in the Vector config file. It will also print out the hostname, IP address, and MAC address of every robot found. Make sure `zeroconf` is installed (`pip install zeroconf==0.24.0`) or mDNS may not work well. Alternatively, you can just open the Vector config file at `~/.anki_vector/sdk_config.ini` in a text editor and manually update the IP addresses.

### Controlling the Vector

The `vector_keyboard_controller.py` script is adapted from the [remote control example](https://github.com/anki/vector-python-sdk/blob/master/examples/apps/remote_control/remote_control.py) in the official SDK, and can be used to verify that you are able to control the robot using the Vector SDK. Use it as follows:

```bash
python vector_keyboard_controller.py --robot-index ROBOT_INDEX
```

The `--robot-index` argument specifies the robot you wish to control and refers to the index of the robot in the Vector config file (`~/.anki_vector/sdk_config.ini`).

### Building the Real Environment

Please reference the videos on the [project page](https://learning-dynamic-manipulation.cs.princeton.edu) when building the real environment setup.

We built the walls using 50 cm x 44 mm strips of Elmer's Foam Board. We also use several 3D printed parts, which we printed using the [Sindoh 3DWOX 1](https://www.amazon.com/Sindoh-3DWOX-Printer-New-Model/dp/B07C79C9RB) 3D printer (with PLA filament). All 3D model files are in the [`stl`](stl) directory.

Here are the different parts to 3D print for the real environment setup:
* `wall-support.stl`: triangular supports used to secure the walls to the tabletop
* `rounded-corner.stl`: rounded blocks installed in corners of the environment to allow pushing through corners
* `board-corner.stl`: used for robot pose estimation with ArUco markers
* `blower-tube.stl`: attach to front of the air blower, allows the blowing robot to blow more accurately
* `pushing-attachment.stl`: attach to front of Vector's lift, allows the pushing robot to push objects more predictably

Note that all robot attachments need to be secured to the robot (using tape, for example). The robots will not be able to reliably execute their end effector action with loose attachments.

To represent the target receptacle, the [`receptacle.pdf`](printouts/receptacle.pdf) file in the `printouts` directory can be printed out and installed in the top right corner of the room.

### Running Trained Policies on the Real Robot

First see the [`aruco`](aruco) directory for instructions on setting up pose estimation with ArUco markers.

Once the setup is completed, make sure the pose estimation server is started before proceeding:

```bash
cd aruco
python server.py
```

---

We can use `tools_interactive_gui.py` from before to manually control a robot in the real environment too, which will allow us to verify that all components of the real setup are working properly, including pose estimation and robot control. See the `main` function in `tools_interactive_gui.py` for the appropriate arguments. You will need to enable `real` and provide the appropriate robot index for `real_robot_indices`. You can then run the same command from before to start the GUI:

```bash
python tools_interactive_gui.py
```

You should see that the simulation environment in the PyBullet GUI mirrors the real setup with millimeter-level precision. If the poses in the simulation do not look correct, you can restart the pose estimation server with the `--debug` flag to enable debug visualizations:

```bash
cd aruco
python server.py --debug
```

---

Once you have verified that manual control with `tools_interactive_gui.py` works, you can then run a trained policy using `enjoy.py` from before. For example, to run the pretrained multi-frequency blowing policy in the `SmallEmpty` environment, you can run:

```bash
python enjoy.py --config-path logs/pretrained-blowing_1-small_empty-multifreq_4/config.yml --real --real-robot-indices 0
```

For debugging and visualization, `tools_interactive_gui.py` allows you to load a trained policy and interactively run the policy one step at a time while showing Q-value maps and transitions. For example:

```bash
python tools_interactive_gui.py --config-path logs/pretrained-blowing_1-small_empty-multifreq_4/config.yml
```

## Citation

If you find this work useful for your research, please consider citing:

```
@article{wu2022learning,
  title = {Learning Pneumatic Non-Prehensile Manipulation with a Mobile Blower},
  author = {Wu, Jimmy and Sun, Xingyuan and Zeng, Andy and Song, Shuran and Rusinkiewicz, Szymon and Funkhouser, Thomas},
  journal = {IEEE Robotics and Automation Letters},
  year = {2022}
}
```
