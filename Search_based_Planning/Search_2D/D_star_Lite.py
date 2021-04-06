"""
D_star_Lite 2D
@author: huiming zhou
"""

import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env


class DStar:
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start, self.s_goal = s_start, s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env()  # class Env
        self.Plot = plotting.Plotting(s_start, s_goal)

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles
        self.x = self.Env.x_range
        self.y = self.Env.y_range

        self.g, self.rhs, self.U = {}, {}, {}
        self.km = 0

        for i in range(1, self.Env.x_range - 1):
            for j in range(1, self.Env.y_range - 1):
                self.rhs[(i, j)] = float("inf")
                self.g[(i, j)] = float("inf")

        self.rhs[self.s_goal] = 0.0
        self.U[self.s_goal] = self.CalculateKey(self.s_goal)
        self.visited = set()
        self.count = 0
        self.fig = plt.figure()

    def is_on_path(self, rob_traj, obs_pos):
        return(obs_pos in rob_traj)

    def calc_dist(self, v1, v2):
        return math.hypot(v1[0] - v2[0], v1[1] - v2[1])

    def dyn_col_checker(self, obs_traj, obs_idx, rob_traj, rob_idx, rob_step, pause_time):
        obs_curr_pos = np.asarray(obs_traj[obs_idx])
        obs_prev_pos = np.asarray(obs_traj[obs_idx-1])
        rob_curr_pos = np.asarray(rob_traj[rob_idx])
        # rob_prev_pos = np.asarray(rob_traj[rob_idx - rob_step])
        dist = self.calc_dist(obs_curr_pos, rob_curr_pos)
        obs_vel = np.true_divide(np.subtract(obs_curr_pos, obs_prev_pos), pause_time)
        # rob_vel = np.true_divide(np.subtract(rob_curr_pos, rob_prev_pos), pause_time)
        rob_vel = np.true_divide(np.subtract(rob_curr_pos, rob_traj[rob_idx - 1]), pause_time/rob_step)
        rob_speed = self.calc_dist(rob_vel, (0,0))
        radius = 8
        dist_tol = 4
        approaching_vel_tol = 7
        speed_inc = 10
        on_path = self.is_on_path(rob_traj, obs_traj[obs_idx])
        # New static obstacle on path outside bubble
        if (dist > radius and obs_curr_pos.all() == obs_prev_pos.all() and on_path):
            print("Outside bubble")
            return (True, rob_speed)  # How to deal with case when rob reaches obs before replanning is complete?
        # Inside bubble
        if (dist < radius):
            print("Inside bubble")
            # Stop and replan
            if (obs_curr_pos.all() == obs_prev_pos.all() and on_path):
                print ("Static")
                return (True, 0)
            # Stop or change speed
            else:
                rel_pos = np.subtract(obs_curr_pos, rob_curr_pos)
                rel_vel = np.subtract(rob_vel, obs_vel)
                approaching_vel = np.dot(rel_pos, rel_vel)
                far_fast = False
                close_slow = False
                close_fast = False
                print("dist, approachin_vel =", dist, approaching_vel)
                # Far away, but approaching fast
                if (dist > dist_tol and approaching_vel > approaching_vel_tol):
                    far_fast = True
                # Close by and approaching slow
                if (dist < dist_tol and approaching_vel > 0 and approaching_vel <= approaching_vel_tol):
                    close_slow = True
                # Close by and approaching fast
                if (dist < dist_tol and approaching_vel > approaching_vel_tol):
                    close_fast = True
                # Danger, stop
                if (close_slow == True or close_fast == True):
                    print("Danger")
                    print("close_slow = ", close_slow)
                    print("close_fast = ", close_fast)
                    return (False, 0)
                # Approaching, speed up
                elif (far_fast == True):
                    print("Approaching")
                    new_rob_speed = rob_speed + speed_inc # Can make smarter using rel_vel
                    return (False, new_rob_speed)
        return (False, rob_speed)

    def create_traj(self, o_start, o_goal):
        traj_x = []
        traj_y = []
        if (o_start[0] == o_goal[0]):
            # vertical case
            traj_x = [o_start[0]]*(abs(o_goal[1]-o_start[1]) + 1)
            if o_start[1] > o_goal[1]:
                step = -1
            else:
                step = 1
            traj_y = list(range(o_start[1], o_goal[1] + step, step))
        else:
            # horizontal case
            if o_start[0] > o_goal[0]:
                step = -1
            else:
                step = 1
            traj_x = list(range(o_start[0], o_goal[0] + step, step))
            traj_y = [o_start[1]]*(abs(o_goal[0]-o_start[0]) + 1)
        obs_traj = list(zip(traj_x, traj_y))
        return obs_traj

    def visualisation(self, rob_traj, obs_traj, rob_step, obs_step, rob_idx, itr, rob_done, obs_done, time_step=0.5):
        # vel = steps/time_step
        # note: rob_idx is the idex of the robot location at the PREVIOUS timestep

        if itr == 0:
            plt.plot(rob_traj[rob_idx][0], rob_traj[rob_idx][1], 'bs')
            plt.plot(obs_traj[itr*obs_step][0], obs_traj[itr*obs_step][1], 'sk')
        else:
            if rob_idx < len(rob_traj) - 1:
                rob_step = min(int(self.calc_dist(rob_traj[rob_idx], rob_traj[-1])), rob_step)
                rob_idx += rob_step
                plt.plot(rob_traj[rob_idx - rob_step][0], rob_traj[rob_idx - rob_step][1], marker = 's', color = 'white')
                plt.plot(rob_traj[rob_idx][0], rob_traj[rob_idx][1], 'bs')
            else:
                rob_done = True
            if itr*obs_step < len(obs_traj):
                plt.plot(obs_traj[itr*obs_step - obs_step][0], obs_traj[itr*obs_step - obs_step][1], marker = 's', color = 'white')
                plt.plot(obs_traj[itr*obs_step][0], obs_traj[itr*obs_step][1], 'sk')
            else:
                obs_done = True
        plt.pause(time_step)

        return rob_idx, rob_done, obs_done

    def run_dynamic(self):
        self.Plot.plot_grid("D* Lite")
        self.ComputePath()
        self.plot_path(self.extract_path())
        pause_time = 0.5
        # rob_step = rob_vel * time_step
        rob_step = 2
        # obs_step = obs_vel * time_step
        obs_step = 1
        rob_done = False
        obs_done = False

        ############################ CASES ###########################################
        # Outside bubble static, rob_step = 2, obs_step = 1
        static_start = (38, 29)
        static_goal = (38, 25)
        # Inside bubble static, rob_step = 2, obs_step = 1
        # static_start = (25, 29)
        # static_goal = (25, 14)
        # Inside bubble, danger close_fast, rob_step = 2, obs_step = 1
        # dyn_start = (25, 29)
        # dyn_goal = (25, 0)
        # Inside bubble, danger close_slow, rob_step = 2, obs_step = 1
        # dyn_start = (22, 29)
        # dyn_goal = (22, 0)
        # Inside bubble, approaching, rob_step = 2, obs_step = 1
        dyn_start = (24, 29)
        dyn_goal = (24, 0)

        ##############################################################################

        static_obs_traj = self.create_traj(static_start, static_goal)
        static_obs_traj.append(static_goal)
        dyn_obs_traj = self.create_traj(dyn_start, dyn_goal)
        obs_traj = static_obs_traj
        rob_traj = self.extract_path()
        rob_idx = 0
        itr = 0
        while not (rob_done and obs_done):
            rob_idx, rob_done, obs_done = self.visualisation(rob_traj, obs_traj, rob_step, obs_step, rob_idx, itr, rob_done, obs_done, pause_time)
            itr += 1
            if (itr % 2 == 0):
                if (itr < len(obs_traj)):
                    replan, new_rob_speed = self.dyn_col_checker(obs_traj, itr, rob_traj, rob_idx, rob_step, pause_time)
                    rob_step = int(new_rob_speed * pause_time)
                    print(replan, new_rob_speed)
        plt.show()

    # def run(self):
    #     self.Plot.plot_grid("D* Lite")
    #     self.ComputePath()
    #     self.plot_path(self.extract_path())
    #     static_start = (38, 29)
    #     static_goal = (38, 21)
    #     dyn_start = (25, 1)
    #     dyn_goal = (25, 30)
    #     traj = self.create_traj(dyn_start, dyn_goal)
    #     self.plot_obs(traj)
    #     self.fig.canvas.mpl_connect('button_press_event', self.on_press)
    #     plt.show()

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        if x < 0 or x > self.x - 1 or y < 0 or y > self.y - 1:
            print("Please choose right area!")
        else:
            x, y = int(x), int(y)
            print("Change position: s =", x, ",", "y =", y)

            s_curr = self.s_start
            s_last = self.s_start
            i = 0
            path = [self.s_start]

            while s_curr != self.s_goal:
                s_list = {}

                for s in self.get_neighbor(s_curr):
                    s_list[s] = self.g[s] + self.cost(s_curr, s)
                s_curr = min(s_list, key=s_list.get)
                path.append(s_curr)

                if i < 1:
                    self.km += self.h(s_last, s_curr)
                    s_last = s_curr
                    if (x, y) not in self.obs:
                        self.obs.add((x, y))
                        plt.plot(x, y, 'sk')
                        self.g[(x, y)] = float("inf")
                        self.rhs[(x, y)] = float("inf")
                    else:
                        self.obs.remove((x, y))
                        plt.plot(x, y, marker='s', color='white')
                        self.UpdateVertex((x, y))
                    for s in self.get_neighbor((x, y)):
                        self.UpdateVertex(s)
                    i += 1

                    self.count += 1
                    self.visited = set()
                    self.ComputePath()

            self.plot_visited(self.visited)
            self.plot_path(path)
            self.fig.canvas.draw_idle()

    def ComputePath(self):
        while True:
            s, v = self.TopKey()
            if v >= self.CalculateKey(self.s_start) and \
                    self.rhs[self.s_start] == self.g[self.s_start]:
                break

            k_old = v
            self.U.pop(s)
            self.visited.add(s)

            if k_old < self.CalculateKey(s):
                self.U[s] = self.CalculateKey(s)
            # Over consistent
            elif self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                for x in self.get_neighbor(s):
                    self.UpdateVertex(x)
            # Under consistent
            else:
                self.g[s] = float("inf")
                self.UpdateVertex(s)
                for x in self.get_neighbor(s):
                    self.UpdateVertex(x)

    def UpdateVertex(self, s):
        if s != self.s_goal:
            self.rhs[s] = float("inf")
            for x in self.get_neighbor(s):
                self.rhs[s] = min(self.rhs[s], self.g[x] + self.cost(s, x))
        if s in self.U:
            self.U.pop(s)

        if self.g[s] != self.rhs[s]:
            self.U[s] = self.CalculateKey(s)

    def CalculateKey(self, s):
        return [min(self.g[s], self.rhs[s]) + self.h(self.s_start, s) + self.km,
                min(self.g[s], self.rhs[s])]

    def TopKey(self):
        """
        :return: return the min key and its value.
        """

        s = min(self.U, key=self.U.get)
        return s, self.U[s]

    def h(self, s_start, s_goal):
        heuristic_type = self.heuristic_type  # heuristic type

        if heuristic_type == "manhattan":
            return abs(s_goal[0] - s_start[0]) + abs(s_goal[1] - s_start[1])
        else:
            return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return float("inf")

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def get_neighbor(self, s):
        nei_list = set()
        for u in self.u_set:
            s_next = tuple([s[i] + u[i] for i in range(2)])
            if s_next not in self.obs:
                nei_list.add(s_next)

        return nei_list

    def extract_path(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_start]
        s = self.s_start

        for k in range(100):
            g_list = {}
            for x in self.get_neighbor(s):
                if not self.is_collision(s, x):
                    g_list[x] = self.g[x]
            s = min(g_list, key=g_list.get)
            path.append(s)
            if s == self.s_goal:
                break

        return list(path)

    def plot_path(self, path):
        px = [x[0] for x in path]
        py = [x[1] for x in path]
        plt.plot(px, py, linewidth=2)
        plt.plot(self.s_start[0], self.s_start[1], "bs")
        plt.plot(self.s_goal[0], self.s_goal[1], "gs")

    def plot_visited(self, visited):
        color = ['gainsboro', 'lightgray', 'silver', 'darkgray',
                 'bisque', 'navajowhite', 'moccasin', 'wheat',
                 'powderblue', 'skyblue', 'lightskyblue', 'cornflowerblue']

        if self.count >= len(color) - 1:
            self.count = 0

        for x in visited:
            plt.plot(x[0], x[1], marker='s', color=color[self.count])


def main():
    s_start = (5, 5)
    s_goal = (45, 25)

    dstar = DStar(s_start, s_goal, "euclidean")
    dstar.run_dynamic()


if __name__ == '__main__':
    main()
