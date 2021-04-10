"""
D_star_Lite 2D
@author: huiming zhou
"""

import os
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.patches as p
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env

bubble_rad = 8
dist_tol = 5
approaching_vel_tol = 7
speed_inc = 3
initial_rob_step = 2

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

    def make_textbox(self, message, text_box):
        text_box.set_text(message)
        
    def dyn_col_checker(self, obs_traj, obs_idx, rob_traj, rob_idx, rob_step, pause_time, text_box):
        # Current and previous positions of obs and rob in their trajs
        obs_curr_pos = np.asarray(obs_traj[obs_idx])
        obs_prev_pos = np.asarray(obs_traj[obs_idx - 1])
        rob_curr_pos = np.asarray(rob_traj[rob_idx])
        rob_prev_pos = np.asarray(rob_traj[rob_idx - 1])
        # Current velocity vectors of obs and rob
        obs_vel = np.true_divide(np.subtract(obs_curr_pos, obs_prev_pos), pause_time)
        print (obs_vel)
        if rob_step == 0:
            rob_vel = np.zeros(2)
        else:
            rob_vel = np.true_divide(np.subtract(rob_curr_pos, rob_prev_pos), pause_time/rob_step)
        # Current rob speed
        rob_speed = self.calc_dist(rob_vel, (0,0))
        # Current distance between obs and rob
        dist = self.calc_dist(obs_curr_pos, rob_curr_pos)  
    
        on_path = self.is_on_path(rob_traj, obs_traj[obs_idx])
        rel_pos = np.subtract(obs_curr_pos, rob_curr_pos)
        rel_vel = np.subtract(rob_vel, obs_vel)
        approaching_vel = np.dot(rel_pos, rel_vel)

        # print(rob_curr_pos, rob_prev_pos, dist, approaching_vel)

        # New static obstacle on path outside bubble
        if (dist > bubble_rad and np.array_equal(obs_curr_pos, obs_prev_pos) and on_path): 
            print("Outside bubble, static")
            self.make_textbox("New Static Obstacle: Replan", text_box)
            return (True, rob_speed)  # How to deal with case when rob reaches obs before replanning is complete?
        # Inside bubble
        if (dist <= bubble_rad):
            print("Inside bubble")
            # Stop and replan
            if (np.array_equal(obs_curr_pos, obs_prev_pos) and on_path):
                self.make_textbox("New Static Obstacle: Replan", text_box) 
                print ("Static")
                return (True, 0)
            # Stop or speed up
            else:
                far_fast = False
                close = False 
                print("dist, approachin_vel =", dist, approaching_vel)
                # Far away, but approaching fast
                if (dist > dist_tol and approaching_vel > approaching_vel_tol):
                    far_fast = True
                # Close by and approaching 
                if (dist <= dist_tol and approaching_vel > 0):
                    close = True
                # Danger, stop
                if (close == True):
                    self.make_textbox("Danger: Stop", text_box)
                    print("Danger")
                    return (False, 0)
                # Approaching, speed up
                elif (far_fast == True):
                    self.make_textbox("Approaching Obstacle: Speed Up", text_box)
                    print("Approaching")
                    new_rob_speed = rob_speed + speed_inc # Can make smarter using rel_vel
                    return (False, new_rob_speed)
        # Post danger
        if (rob_speed == 0 and approaching_vel < 0 and dist > dist_tol):
            rob_speed = initial_rob_step/pause_time

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

    def create_complex_traj(self, o_start, o_goal, change_ind):
        traj_x = []
        traj_y = []
        y_pos = o_start[1]
        traj_x.append(o_start[0])
        traj_y.append(y_pos)
        i = 0
        while (y_pos != o_goal[1]):
            if (i < change_ind):
                step = -1
            else:
                step = -2
            y_pos = y_pos + step
            traj_x.append(o_start[0])
            traj_y.append(y_pos)
            i = i + 1
        
        obs_traj = list(zip(traj_x, traj_y))
        return obs_traj

    def visualisation(self, rob_traj, obs_traj, rob_step, obs_step, rob_idx, c1_prev, c2_prev, itr, rob_done, obs_done, time_step=0.5):
        # vel = steps/time_step
        # note rob_idx is the idex of the robot location at the PREVIOUS timestep
        if itr == 0:
            c1_prev.remove()
            c2_prev.remove()
            plt.plot(rob_traj[rob_idx][0], rob_traj[rob_idx][1], 'bs')
            plt.plot(obs_traj[itr*obs_step][0], obs_traj[itr*obs_step][1], 'sk')
            c1 = plt.Circle((rob_traj[rob_idx][0], rob_traj[rob_idx][1]), radius = bubble_rad, edgecolor = 'b', lw = 1, facecolor = 'none')
            c2 = plt.Circle((rob_traj[rob_idx][0], rob_traj[rob_idx][1]), radius = dist_tol, edgecolor = 'r', lw = 1, facecolor = 'none')
            plt.gca().add_patch(c1)
            plt.gca().add_patch(c2)
            c1_prev = c1
            c2_prev = c2
        else:
            if rob_idx < len(rob_traj) - 1:
                c1_prev.remove()
                c2_prev.remove()
                rob_step = min(int(self.calc_dist(rob_traj[rob_idx], rob_traj[-1])), rob_step)
                rob_idx += rob_step
                plt.plot(rob_traj[rob_idx - rob_step][0], rob_traj[rob_idx - rob_step][1], marker = 's', color = 'cyan')
                plt.plot(rob_traj[rob_idx][0], rob_traj[rob_idx][1], 'bs')
                c1 = plt.Circle((rob_traj[rob_idx][0], rob_traj[rob_idx][1]), radius = bubble_rad, edgecolor = 'b', lw = 1, facecolor = 'none')
                c2 = plt.Circle((rob_traj[rob_idx][0], rob_traj[rob_idx][1]), radius = dist_tol, edgecolor = 'r', lw = 1, facecolor = 'none')
                plt.gca().add_patch(c1)
                plt.gca().add_patch(c2)
                c1_prev = c1
                c2_prev = c2
            else:
                rob_done = True
            if itr*obs_step < len(obs_traj):
                plt.plot(obs_traj[itr*obs_step - obs_step][0], obs_traj[itr*obs_step - obs_step][1], marker = 's', color = 'red')
                plt.plot(obs_traj[itr*obs_step][0], obs_traj[itr*obs_step][1], 'sk')
            else:
                obs_done = True

        plt.pause(time_step)

        return rob_idx, c1_prev, c2_prev, rob_done, obs_done

    def find_nearest(self, path, pos):
        min = 100
        for i in range(len(path)):
            dist = self.calc_dist(path[i], pos)
            if (dist < min):
                min = dist
                i_min = i
        return i_min

    def run_dynamic(self):
        self.Plot.plot_grid("D* Lite")
        self.ComputePath()
        self.plot_path(self.extract_path())
        # Initialize textbox object
        label_patch = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        text_box = plt.text(0, 35, "", fontsize=11,
                verticalalignment='top', bbox = label_patch)
    
        pause_time = 0.5
        rob_step = 2
        obs_step = 1
        rob_done = False
        obs_done = False

        ############################ CASES ###########################################
        # Outside bubble static
        # static_start = (34, 29)
        # static_goal = (34, 22)
        # Inside bubble static
        # static_start = (24, 29)
        # static_goal = (24, 14)
        # rob pos not on new path case
        static_start = (26, 29)
        static_goal = (26, 14)
        # Inside bubble, approaching
        # dyn_start = (22, 29)
        # dyn_goal = (22, 0)
        # Inside bubble, danger
        # dyn_start = (24, 29)
        # dyn_goal = (24, 0)
        # Change from approaching to danger
        dyn_start = (22, 29)
        dyn_goal = (22, 0)
        comp_traj = self.create_complex_traj(dyn_start, dyn_goal, 11)
        
        ##############################################################################

        static_obs_traj = self.create_traj(static_start, static_goal)
        static_obs_traj.append(static_goal)
        dyn_obs_traj = self.create_traj(dyn_start, dyn_goal)
        obs_traj = comp_traj
        rob_traj = self.extract_path()
        rob_idx = 0
        itr = 0 
        # Random previous circles (invisible) for itr = 0
        c1_prev = plt.Circle((1,1), radius = 1)
        c2_prev = plt.Circle((1,1), radius = 1)
        plt.gca().add_patch(c1_prev)
        plt.gca().add_patch(c2_prev)
        c1_prev.set_visible(False)
        c2_prev.set_visible(False)
        while not (rob_done and obs_done):
            rob_idx, c1_prev, c2_prev, rob_done, obs_done = self.visualisation(rob_traj, obs_traj, rob_step, obs_step, rob_idx, c1_prev, c2_prev, itr, rob_done, obs_done, pause_time)
            itr += 1
            if (itr % 2 == 0):
                if (itr < len(obs_traj)):
                    is_replan, new_rob_speed = self.dyn_col_checker(obs_traj, itr, rob_traj, rob_idx, rob_step, pause_time, text_box)
                    rob_step = int(new_rob_speed * pause_time)
                    print(is_replan, new_rob_speed)
                    if is_replan == True:
                        new_path = self.replan(obs_traj[itr*obs_step]) 
                        # if curr rob pos is on new path, switch to new path  
                        if rob_traj[rob_idx] in new_path:
                            rob_idx = new_path.index(rob_traj[rob_idx])
                            rob_traj = new_path
                            if (rob_step == 0):
                                rob_step = initial_rob_step
                        # else find connecting path
                        else:
                            nearest_ind = self.find_nearest(new_path, rob_traj[rob_idx])
                            self.myComputePath(rob_traj[rob_idx], new_path[nearest_ind])
                            connect_path = self.my_extract_path(rob_traj[rob_idx], new_path[nearest_ind])
                            self.my_plot_path(connect_path)
                            new_path[nearest_ind:nearest_ind] = connect_path[0:-1]
                            rob_idx = new_path.index(rob_traj[rob_idx])
                            rob_traj = new_path
                            if (rob_step == 0):
                                rob_step = initial_rob_step
        plt.show()

    def replan(self, static_obs_pos):
        x, y = static_obs_pos
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
                self.obs.add((x, y))
                # plt.plot(x, y, 'sk')
                self.g[(x, y)] = float("inf")
                self.rhs[(x, y)] = float("inf")
                for s in self.get_neighbor((x, y)):
                    self.UpdateVertex(s)
                i += 1

                self.count += 1
                self.visited = set()
                self.ComputePath()

        # self.plot_visited(self.visited)
        self.plot_path(path)
        self.fig.canvas.draw_idle()
        return path

    def myComputePath(self, my_start, my_goal):
        while True:
            s, v = self.TopKey()
            if v >= self.myCalculateKey(my_start, my_start) and \
                    self.rhs[my_start] == self.g[my_start]:
                break

            k_old = v
            self.U.pop(s)
            self.visited.add(s)

            if k_old < self.myCalculateKey(s, my_start):
                self.U[s] = self.myCalculateKey(s, my_start)
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

    def myCalculateKey(self, s, my_start):
        return [min(self.g[s], self.rhs[s]) + self.h(my_start, s) + self.km,
                min(self.g[s], self.rhs[s])]

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

    def my_extract_path(self, my_start, my_goal):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [my_start]
        s = my_start

        for k in range(100):
            g_list = {}
            for x in self.get_neighbor(s):
                if not self.is_collision(s, x):
                    g_list[x] = self.g[x]
            s = min(g_list, key=g_list.get)
            path.append(s)
            if s == my_goal:
                break

        return list(path)

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

    def my_plot_path(self, path):
        px = [x[0] for x in path]
        py = [x[1] for x in path]
        plt.plot(px, py, linewidth=2, color="magenta")

    def plot_path(self, path):
        px = [x[0] for x in path]
        py = [x[1] for x in path]
        plt.plot(px, py, linewidth=2)
        # plt.plot(self.s_start[0], self.s_start[1], "bs")
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
