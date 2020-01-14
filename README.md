# Multi-Arm RRT with UR5
Tested under python 3.7.4 and pybullet 2.5.6

## Usage
Please specify the initial configuration of UR5 arm(s) and obstacles in a `json` file. 
An example is [`worlds/single_arm_with_obstacles`](worlds/single_arm_with_obstacles.json).

The [`demo.py`](demo.py) file takes the following arguments
- `world_file`: path to the `json` file specifying the world configuration
- `planner`: one of 'rrt' or 'birrt'
- `smooth`: number of smoothing iterations in postprocessing. This is currently only supported in 
birrt planner. 0 means no smoothing.
- `greedy`: whether the planner extends greedily to the random sampled configuration

For a complete and detailed list of parameters that you can play with, 
see the function [`plan_motion`](https://github.com/jingxixu/multi-arm-rrt/blob/acc203b5f5004b04928cd07b641596ae73758ab0/ur5_group.py#L109)
under [`ur5_group.py`](ur5_group.py).

__Examples__

Single arm with obstacles
```
python demo.py --world_file worlds/single_arm_with_obstacles.json
```

Two arms without any obstacles
```
python demo.py --world_file worlds/two_arms_no_obstacles.json
```