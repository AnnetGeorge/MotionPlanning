
class Env:
    def __init__(self):
        self.x_range = 51  # size of background
        self.y_range = 31
        # self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
        #                 (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.motions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        obs = set()

        # top and bottom boundary
        for i in range(x):
            obs.add((i, 0))
        for i in range(x):
            obs.add((i, y - 1))

        # left and right boundary
        for i in range(y):
            obs.add((0, i))
        for i in range(y):
            obs.add((x - 1, i))

        # Maze walls
        # for i in range(10, 21):
        #     obs.add((i, 15))
        # for i in range(15):
        #     obs.add((20, i))
        # for i in range(15, 30):
        #     obs.add((30, i))
        # for i in range(16):
        #     obs.add((40, i))

       # Exp2 walls
        for i in range(10, 21):
            obs.add((i, 15))
        for i in range(15):
            obs.add((20, i))
        for i in range(5, 30):
            obs.add((30, i))

        for i in range(31, 47):
            obs.add((i, 15))

        for i in range(5, 16):
            obs.add((47, i))

        # top line
        for i in range(33, 36):
            obs.add((i, 17))

        for i in range(37, 48):
            obs.add((i, 17))

        return obs
