from smoothing import smooth_path
from rrt import TreeNode, configs
from rrt_utils import irange, argmin, RRT_ITERATIONS, RRT_RESTARTS, RRT_SMOOTHING
import pybullet_utils as pu


def rrt_connect(robot, joints, q1, q2, distance, sample, extend, collision, iterations=RRT_ITERATIONS, visualize=False):
    if collision(q1) or collision(q2):
        return None
    root1, root2 = TreeNode(q1), TreeNode(q2)
    nodes1, nodes2 = [root1], [root2]
    if visualize:
        color1, color2 = [0, 1, 0], [0, 0, 1]
    for _ in irange(iterations):
        if len(nodes1) > len(nodes2):
            nodes1, nodes2 = nodes2, nodes1
            if visualize:
                color1, color2 = color2, color1
        s = sample()

        last1 = argmin(lambda n: distance(n.config, s), nodes1)
        for q in extend(last1.config, s):
            if collision(q):
                break
            if visualize:
                p_now, _ = pu.forward_kinematics(robot, joints, q)
                p_prev, _ = pu.forward_kinematics(robot, joints, last1.config)
                pu.add_line(p_prev, p_now, color=color1)
            last1 = TreeNode(q, parent=last1)
            nodes1.append(last1)
            break


        last2 = argmin(lambda n: distance(n.config, last1.config), nodes2)
        for q in extend(last2.config, last1.config):
            if collision(q):
                break
            if visualize:
                p_now, _ = pu.forward_kinematics(robot, joints, q)
                p_prev, _ = pu.forward_kinematics(robot, joints, last2.config)
                pu.add_line(p_prev, p_now, color=color2)
            last2 = TreeNode(q, parent=last2)
            nodes2.append(last2)
            break
        if last2.config == last1.config:
            path1, path2 = last1.retrace(), last2.retrace()
            if path1[0] != root1:
                path1, path2 = path2, path1
            return configs(path1[:-1] + path2[::-1])
    return None

# TODO: version which checks whether the segment is valid

def direct_path(q1, q2, extend, collision):
    if collision(q1) or collision(q2):
        return None
    path = [q1]
    for q in extend(q1, q2):
        if collision(q):
            return None
        path.append(q)
    return path


def birrt(robot, joints, q1, q2, distance, sample, extend, collision,
          restarts=RRT_RESTARTS, iterations=RRT_ITERATIONS, smooth=RRT_SMOOTHING, visualize=False):
    if collision(q1) or collision(q2):
        return None
    path = direct_path(q1, q2, extend, collision)
    if path is not None:
        return path
    for _ in irange(restarts + 1):
        path = rrt_connect(robot, joints, q1, q2, distance, sample, extend,
                           collision, iterations=iterations, visualize=visualize)
        if path is not None:
            if smooth is None:
                return path
            return smooth_path(path, extend, collision, iterations=smooth)
    return None
