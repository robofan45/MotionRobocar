"""

Grid based Dijkstra planning

Modified by: Mahim Trivedi (heap-based open set, plot legend)

"""

import heapq
import itertools
import math

import matplotlib.pyplot as plt

show_animation = True


class DijkstraPlanner:

    def __init__(self, ox, oy, resolution, robot_radius):
        """
        Initialize map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        if not resolution > 0 or not math.isfinite(resolution):
            raise ValueError("resolution must be a positive number")

        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.x_width = None
        self.y_width = None
        self.obstacle_map = None

        self.resolution = resolution
        self.robot_radius = robot_radius
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index  # index of previous Node

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        dijkstra path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gx: goal x position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
            (both lists are empty when the goal cannot be reached or the
            start or goal is outside the map)
        """
        if not all(math.isfinite(v) for v in (sx, sy, gx, gy)):
            raise ValueError("start and goal must be finite numbers")

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        if not (self.is_on_grid(start_node) and self.is_on_grid(goal_node)):
            print("Start or goal is outside the map..")
            return [], []

        open_set, closed_set = dict(), dict()
        # The open set is mirrored in a binary heap, so the cheapest node is
        # found in O(log n) instead of scanning every open node. Entries are
        # (cost, first-seen order, push count, grid index, node): the
        # first-seen order breaks ties exactly like the old min() scan, so
        # the paths do not change. Outdated entries are skipped when popped.
        open_heap = []
        first_seen = {}
        push_count = itertools.count()

        def push(index, node):
            order = first_seen.setdefault(index, len(first_seen))
            heapq.heappush(open_heap,
                           (node.cost, order, next(push_count), index, node))

        start_index = self.calc_index(start_node)
        open_set[start_index] = start_node
        push(start_index, start_node)
        goal_found = False

        if show_animation:  # pragma: no cover
            # for stopping simulation with the esc key (registered once).
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            # pop the cheapest open node, skipping outdated heap entries
            while True:
                *_, c_id, current = heapq.heappop(open_heap)
                if open_set.get(c_id) is current:
                    break

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_position(current.x, self.min_x),
                         self.calc_position(current.y, self.min_y), "xc")
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                goal_found = True
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand search grid based on motion model
            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)
                n_id = self.calc_index(node)

                if n_id in closed_set:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # Discover a new node
                    push(n_id, node)
                else:
                    if open_set[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        open_set[n_id] = node
                        push(n_id, node)

        if not goal_found:
            return [], []

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_position(goal_node.x, self.min_x)], [
            self.calc_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_position(n.x, self.min_x))
            ry.append(self.calc_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    def calc_position(self, index, minp):
        pos = index * self.resolution + minp
        return pos

    def calc_xy_index(self, position, minp):
        return round((position - minp) / self.resolution)

    def calc_index(self, node):
        return node.y * self.x_width + node.x

    def is_on_grid(self, node):
        return 0 <= node.x < self.x_width and 0 <= node.y < self.y_width

    def verify_node(self, node):
        px = self.calc_position(node.x, self.min_x)
        py = self.calc_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        if py < self.min_y:
            return False
        if px >= self.max_x:
            return False
        if py >= self.max_y:
            return False
        if node.x >= self.x_width or node.y >= self.y_width:
            return False  # outside the obstacle map

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        if len(ox) == 0 or len(ox) != len(oy):
            raise ValueError("ox and oy must be non-empty and the same length")
        if not all(math.isfinite(v) for v in itertools.chain(ox, oy)):
            raise ValueError("obstacle positions must be finite numbers")

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        # round up, so every position below max_x / max_y has a cell
        self.x_width = math.ceil((self.max_x - self.min_x) / self.resolution)
        self.y_width = math.ceil((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = -5.0  # [m]
    sy = -5.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(float(i))
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(float(i))
    for i in range(-10, 61):
        ox.append(float(i))
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(float(i))
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(float(i))
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k", label="Obstacle")
        plt.plot(sx, sy, "og", label="Start")
        plt.plot(gx, gy, "xb", label="Goal")
        plt.plot([], [], "xc", label="Searched cell")
        plt.plot([], [], "-r", label="Path")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.axis("equal")

    dijkstra = DijkstraPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = dijkstra.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    main()
