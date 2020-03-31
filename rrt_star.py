from random import random
from time import time
import pybullet_utils as pu

from rrt_utils import INF, argmin


class OptimalNode(object):

    def __init__(self, config, parent=None, d=0, path=[], iteration=None, visualize=True, group=True, fk=None, rgb_color=(0, 1, 0)):
        self.config = config
        self.parent = parent
        self.children = set()
        self.d = d
        self.path = path    # the path does not include itself or its parent
        self.visualize = visualize
        self.fk = fk
        self.group = group
        self.marker_ids = []
        self.solution_marker_ids = []
        self.visualize = visualize
        self.rgb_color = rgb_color
        if self.visualize and parent is not None:
            self.draw_path()
        if parent is not None:
            self.cost = parent.cost + d
            self.parent.children.add(self)
        else:
            self.cost = d
        self.solution = False
        self.creation = iteration
        self.last_rewire = iteration

    def set_solution(self, solution):
        if self.solution is solution:
            return
        self.solution = solution
        # visualize
        if self.visualize and self.parent is not None:
            if solution is True:
                self.draw_solution_path()
            else:
                # used to a solution, now is not a solution node
                self.remove_solution_path()
        if self.parent is not None:
            self.parent.set_solution(solution)

    def retrace(self):
        if self.parent is None:
            return self.path + [self.config]
        return self.parent.retrace() + self.path + [self.config]

    def rewire(self, parent, d, path, iteration=None):
        if self.solution:
            self.parent.set_solution(False)
        self.parent.children.remove(self)
        self.parent = parent
        self.parent.children.add(self)
        if self.solution:
            self.parent.set_solution(True)
        self.d = d
        self.path = path
        # visualize
        if self.visualize and parent is not None:
            self.remove_path()
            self.draw_path()
        self.update()
        self.last_rewire = iteration

    def update(self):
        self.cost = self.parent.cost + self.d
        for n in self.children:
            n.update()

    def clear(self):
        self.node_handle = None
        self.edge_handle = None

    def remove_solution_path(self):
        for i in self.solution_marker_ids:
            pu.remove_marker(i)
        self.solution_marker_ids = []

    def remove_path(self):
        for i in self.marker_ids:
            pu.remove_marker(i)
        self.marker_ids = []

    def draw_path(self):
        assert self.fk is not None, 'please provide a fk when visualizing'
        if self.group:
            for q_prev, q_now in zip([self.parent.config] + self.path, self.path + [self.config]):
                for pose_now, pose_prev in zip(self.fk(q_prev), self.fk(q_now)):
                    self.marker_ids.append(pu.draw_line(pose_prev[0], pose_now[0], rgb_color=self.rgb_color, width=1))
        else:
            for q_prev, q_now in zip([self.parent.config] + self.path, self.path + [self.config]):
                p_now = self.fk(q_prev)[0]
                p_prev = self.fk(q_now.config)[0]
                self.marker_ids.append(pu.draw_line(p_prev, p_now, rgb_color=self.rgb_color, width=1))

    def draw_solution_path(self):
        assert self.fk is not None, 'please provide a fk when visualizing'
        if self.group:
            for q_prev, q_now in zip([self.parent.config] + self.path, self.path + [self.config]):
                for pose_now, pose_prev in zip(self.fk(q_prev), self.fk(q_now)):
                    self.solution_marker_ids.append(pu.draw_line(pose_prev[0], pose_now[0], rgb_color=(1, 0, 0), width=6))
        else:
            for q_prev, q_now in zip([self.parent.config] + self.path, self.path + [self.config]):
                p_now = self.fk(q_prev)[0]
                p_prev = self.fk(q_now.config)[0]
                self.solution_marker_ids.append(pu.draw_line(p_prev, p_now, rgb_color=(1, 0, 0), width=width))

    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.config) + ')'

    __repr__ = __str__


def safe_path(sequence, collision):
    path = []
    for q in sequence:
        if collision(q):
            break
        path.append(q)
        # break
    return path


def rrt_star(start,
             goal,
             distance,
             sample,
             extend,
             collision,
             radius,
             goal_probability,
             informed,
             visualize,
             fk,
             group,
             max_time=INF,
             max_iterations=INF):
    if collision(start) or collision(goal):
        return None
    nodes = [OptimalNode(start)]
    goal_n = None
    t0 = time()
    it = 0
    while (time() - t0) < max_time and it < max_iterations:
        do_goal = goal_n is None and (it == 0 or random() < goal_probability)
        s = goal if do_goal else sample()
        # Informed RRT*
        if informed and goal_n is not None and distance(start, s) + distance(s, goal) >= goal_n.cost:
            continue
        # if it % 100 == 0:
        #     print(it, time() - t0, goal_n is not None, do_goal, (goal_n.cost if goal_n is not None else INF))
        it += 1

        nearest = argmin(lambda n: distance(n.config, s), nodes)
        path = safe_path(extend(nearest.config, s), collision)
        if len(path) == 0:
            continue
        new = OptimalNode(path[-1], parent=nearest, d=distance(
            nearest.config, path[-1]), path=path[:-1], iteration=it,
                          visualize=visualize, group=group, fk=fk)
        # if safe and do_goal:
        if do_goal and distance(new.config, goal) < 1e-6:
            goal_n = new
            goal_n.set_solution(True)
        # TODO - k-nearest neighbor version
        neighbors = list(filter(lambda n: distance(
            n.config, new.config) < radius, nodes))
        nodes.append(new)

        # check if we should n_new's parent to a neighbor
        for n in neighbors:
            d = distance(n.config, new.config)
            if n.cost + d < new.cost:
                path = safe_path(extend(n.config, new.config), collision)
                if len(path) != 0 and distance(new.config, path[-1]) < 1e-6:
                    new.rewire(n, d, path[:-1], iteration=it)

        # check if we should change a neighbor's parent to n_new
        for n in neighbors:  # TODO - avoid repeating work
            d = distance(new.config, n.config)
            if new.cost + d < n.cost:
                path = safe_path(extend(new.config, n.config), collision)
                if len(path) != 0 and distance(n.config, path[-1]) < 1e-6:
                    n.rewire(new, d, path[:-1], iteration=it)
    if goal_n is None:
        return None
    return goal_n.retrace()


def rrt_star_connect(start,
                     goal,
                     distance,
                     sample,
                     extend,
                     collision,
                     radius,
                     visualize,
                     fk,
                     group,
                     max_time=INF,
                     max_iterations=INF):
    min_cost = INF
    if collision(start) or collision(goal):
        return None
    if visualize:
        color1, color2 = [0, 1, 0], [0, 0, 1]
        tree1, tree2 = [OptimalNode(start, rgb_color=color1)], [OptimalNode(goal, rgb_color=color2)]
    else:
        tree1, tree2 = [OptimalNode(start)], [OptimalNode(goal)]
    goal_n_1 = None
    goal_n_2 = None
    t0 = time()
    it = 0
    while (time() - t0) < max_time and it < max_iterations:
        if len(tree1) > len(tree2):
            tree1, tree2 = tree2, tree1
            if visualize:
                color1, color2 = color2, color1
        s = sample()
        it += 1

        # tree 1
        nearest1 = argmin(lambda n: distance(n.config, s), tree1)
        path = safe_path(extend(nearest1.config, s), collision)
        if len(path) == 0:
            continue
        new1 = OptimalNode(path[-1], parent=nearest1, d=distance(
            nearest1.config, path[-1]), path=path[:-1], iteration=it,
                           visualize=visualize, group=group, fk=fk, rgb_color=color1)
        neighbors1 = list(filter(lambda n: distance(
            n.config, new1.config) < radius, tree1))
        tree1.append(new1)

        for n in neighbors1:
            d = distance(n.config, new1.config)
            if n.cost + d < new1.cost:
                path = safe_path(extend(n.config, new1.config), collision)
                if len(path) != 0 and distance(new1.config, path[-1]) < 1e-6:
                    new1.rewire(n, d, path[:-1], iteration=it)
        for n in neighbors1:  # TODO - avoid repeating work
            d = distance(new1.config, n.config)
            if new1.cost + d < n.cost:
                path = safe_path(extend(new1.config, n.config), collision)
                if len(path) != 0 and distance(n.config, path[-1]) < 1e-6:
                    n.rewire(new1, d, path[:-1], iteration=it)

        # tree 2
        nearest2 = argmin(lambda n: distance(n.config, new1.config), tree2)
        path = safe_path(extend(nearest2.config, new1.config), collision)
        if len(path) == 0:
            continue
        new2 = OptimalNode(path[-1], parent=nearest2, d=distance(
            nearest2.config, path[-1]), path=path[:-1], iteration=it,
                           visualize=visualize, group=group, fk=fk, rgb_color=color2)
        neighbors2 = list(filter(lambda n: distance(
            n.config, new2.config) < radius, tree2))
        tree2.append(new2)

        for n in neighbors2:
            d = distance(n.config, new2.config)
            if n.cost + d < new2.cost:
                path = safe_path(extend(n.config, new2.config), collision)
                if len(path) != 0 and distance(new2.config, path[-1]) < 1e-6:
                    new2.rewire(n, d, path[:-1], iteration=it)
        for n in neighbors2:  # TODO - avoid repeating work
            d = distance(new2.config, n.config)
            if new2.cost + d < n.cost:
                path = safe_path(extend(new2.config, n.config), collision)
                if len(path) != 0 and distance(n.config, path[-1]) < 1e-6:
                    n.rewire(new2, d, path[:-1], iteration=it)

        # solved
        if new1.config == new2.config:
            if new1.cost + new2.cost < min_cost:
                if goal_n_1 is not None and goal_n_2 is not None:
                    goal_n_1.set_solution(False)
                    goal_n_2.set_solution(False)
                    new1.set_solution(True)
                    new2.set_solution(True)
                    goal_n_1 = new1
                    goal_n_2 = new2
                else:
                    goal_n_1 = new1
                    goal_n_2 = new2
                    goal_n_1.set_solution(True)
                    goal_n_2.set_solution(True)
                min_cost = new1.cost + new2.cost

    if goal_n_1 is None and goal_n_2 is None:
        return None
    path1, path2 = goal_n_1.retrace(), goal_n_2.retrace()
    if path1[0] != start:
        path1, path2 = path2, path1
    return path1[:-1] + path2[::-1]
