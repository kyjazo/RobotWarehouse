import math

from mesa import Model
import mesa
from mesa.space import MultiGrid, PropertyLayer
import json
from mesa.datacollection import DataCollector
import numpy as np
from agents import Robot, Package, Obstacle, Pheromones, QLearning

with open("config.json", "r") as f:
    config = json.load(f)
class WarehouseModel(Model):

    def __init__(self, width=config["width"], height=config["height"], num_robot=config["num_robot"],
                 num_package=config["num_package"], num_obstacle=config["num_obstacle"], pheromone_evaporation=config["pheromone_evaporation"],
                 pheromone_added=config["pheromone_added"], diffusion_rate=config["diffusion_rate"],
                 treshold=config["treshold"], render_pheromone=False, seed=None, learning=True, q_learning=None, q_table_file="q_table_avg.json",
                 max_steps=config["max_steps"], testing=False, max_weight=config["max_weight"]

                 ):

        super().__init__(seed=seed)

        self.max_weight = max_weight
        self.max_steps = max_steps
        self.testing = testing
        self.pheromone_treshold = treshold

        self.learning = learning
        self.q_learning = q_learning
        self.q_table_file = q_table_file

        self.render_pheromone = render_pheromone
        self.width = width
        self.height = height
        self.num_robot = num_robot
        self.num_package = num_package
        self.num_obstacle = num_obstacle
        self.pheromone_added = pheromone_added
        self.pheromone_evaporation = pheromone_evaporation
        self.diffusion_rate = diffusion_rate

        self.datacollector = DataCollector(
            agenttype_reporters={
                Robot: {
                    "Package_delivered": lambda a: int(a.package_delivered) if hasattr(a,
                                                                                        'package_delivered') else None,
                    "Reward": lambda a: float(np.mean(a.rewards))  # Usa la media delle reward
                    if hasattr(a, 'rewards') and a.rewards else None,
                    "Action_0": lambda a: a.action_counts[0] if hasattr(a, "action_counts") else None,
                    "Action_1": lambda a: a.action_counts[1] if hasattr(a, "action_counts") else None,
                    "Action_2": lambda a: a.action_counts[2] if hasattr(a, "action_counts") else None,
                    "Action_3": lambda a: a.action_counts[3] if hasattr(a, "action_counts") else None,
                }

            },


        )



        self.robot_pheromone_layer = PropertyLayer("robot_pheromone_layer", height=height, width=width,
                                                  default_value=0.000, dtype=float)
        self.package_pheromone_layer = PropertyLayer("package_pheromone_layer", height=height, width=width,
                                                   default_value=0.000, dtype=float)

        self.grid = MultiGrid(width, height, torus=False)
        self.running = True



        self.place_agents(Obstacle, num_obstacle)
        self.place_agents(Package, num_package)
        self.place_agents(Robot, num_robot)


        if render_pheromone:
            pheromones = Pheromones.create_agents(model=self, n=width * height)
            positions = np.array(np.meshgrid(np.arange(width), np.arange(height))).T.reshape(-1, 2)
            for agent, pos in zip(pheromones, positions):
                self.grid.place_agent(agent, tuple(pos))




    def place_agents(self, agent_class, num_agents):

        if agent_class.__name__ == "Robot":
            agents = agent_class.create_agents(model=self, n=num_agents)
            positions = []
            while len(positions) < num_agents:
                x = self.random.randint(0, self.grid.width - 1)
                y = self.random.randint(0, self.grid.height - 1)
                if (x, y) not in positions and self.grid.is_cell_empty((x, y)):
                    positions.append((x, y))
            positions = positions[:num_agents]

        elif agent_class.__name__ == "Package":
            agents = agent_class.create_agents(model=self, n=num_agents)
            positions = []
            while len(positions) < num_agents:
                x = self.random.randint(0, self.grid.width - 1)
                y = self.random.randint(0, self.grid.height - 1)
                if (x, y) not in positions and self.grid.is_cell_empty((x, y)):
                    positions.append((x, y))
            positions = positions[:num_agents]


        elif agent_class.__name__ == "Obstacle":

            agents = agent_class.create_agents(model=self, n=num_agents)

            obstacle_pattern = [
                # Blocchi 2x2
                (3, 3), (3, 4), (4, 3), (4, 4),
                (8, 8), (8, 9), (9, 8), (9, 9),
                (14, 14), (14, 15), (15, 14), (15, 15),
                (18, 2), (18, 3), (19, 2), (19, 3),
#
                # Blocchi 1x3 orizzontali
                (2, 10), (2, 11), (2, 12),
                (10, 18), (10, 19),
#
                # Blocchi 1x3 verticali
                (12, 2), (13, 2), (14, 2),
                (6, 16), (7, 16), (8, 16),
#
                # Blocchi a L
                (5, 15), (5, 16), (6, 15),  # L in alto a destra
                (15, 5), (16, 5), (15, 6),  # L in basso a sinistra
                (11, 11), (11, 12), (12, 11),  # L centrale
#
                # Ostacoli sparsi per raggiungere il 30%
                (1, 1), (1, 18), (18, 1), (18, 18),  # Angoli
                (7, 3), (7, 4), (12, 17), (13, 17),
                (3, 12), (4, 12), (16, 8), (16, 9),
            ]


            #obstacle_pattern = [
            #    # Blocchi 2x2
            #    (5, 5), (5, 6), (6, 5), (6, 6),
            #    (10, 10), (10, 11), (11, 10), (11, 11),
            #    (15, 15), (15, 16), (16, 15), (16, 16),
            #    (20, 20), (20, 21), (21, 20), (21, 21),
            #    (25, 25), (25, 26), (26, 25), (26, 26),
            #    (30, 30), (30, 31), (31, 30), (31, 31),
            #    (35, 35), (35, 36), (36, 35), (36, 36),
            #    (40, 40), (40, 41), (41, 40), (41, 41),
#
            #    # Blocchi 1x3 (orizzontali)
            #    (8, 12), (8, 13), (8, 14),
            #    (18, 22), (18, 23), (18, 24),
            #    (28, 32), (28, 33), (28, 34),
            #    (38, 42), (38, 43), (38, 44),
#
            #    # Blocchi 1x3 (verticali)
            #    (12, 8), (13, 8), (14, 8),
            #    (22, 18), (23, 18), (24, 18),
            #    (32, 28), (33, 28), (34, 28),
            #    (42, 38), (43, 38), (44, 38),
#
            #    # Blocchi a L (3 celle)
            #    (3, 10), (3, 11), (4, 10),  # L in alto a destra
            #    (10, 3), (11, 3), (10, 4),  # L in basso a sinistra
            #    (17, 24), (17, 25), (18, 24),  # L in alto a sinistra
            #    (24, 17), (25, 17), (24, 18),  # L in basso a destra
            #    (31, 38), (31, 39), (32, 38),  # L in alto a destra
            #    (38, 31), (39, 31), (38, 32),  # L in basso a sinistra
#
            #    # Ostacoli sparsi per raggiungere il 30%
            #    (2, 20), (2, 21), (3, 20), (3, 21),
            #    (7, 15), (7, 16), (8, 15), (8, 16),
            #    (13, 25), (13, 26), (14, 25), (14, 26),
            #    (19, 35), (19, 36), (20, 35), (20, 36),
            #    (25, 5), (25, 6), (26, 5), (26, 6),
            #    (30, 15), (30, 16), (31, 15), (31, 16),
            #    (35, 25), (35, 26), (36, 25), (36, 26),
            #    (40, 35), (40, 36), (41, 35), (41, 36),
            #]

            positions = [

                (x, y) for x, y in obstacle_pattern
                if 0 <= x < self.grid.width and 0 <= y < self.grid.height

            ]

            #if len(positions) < num_agents:
#
            #    extra_positions = []
            #    while len(positions) + len(extra_positions) < num_agents:
            #        x = self.random.randint(0, self.grid.width - 1)
            #        y = self.random.randint(0, self.grid.height - 1)
            #        if (x, y) not in positions and (x, y) not in extra_positions:
            #            extra_positions.append((x, y))
#
            #    positions.extend(extra_positions)

            positions = positions[:num_agents]


        for agent, pos in zip(agents, positions):
            self.grid.place_agent(agent, tuple(pos))

    def diffuse_pheromones(self):
        new_package_layer = self.package_pheromone_layer.data.copy()

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]

        fraction_per_direction = self.diffusion_rate / len(directions)

        for x in range(self.grid.width):
            for y in range(self.grid.height):

                current_pheromone = self.package_pheromone_layer.data[x, y]
                if current_pheromone < self.pheromone_treshold:
                    continue
                total_diffused = 0

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                        amount_diffused = current_pheromone * fraction_per_direction

                        new_package_layer[nx, ny] += amount_diffused
                        total_diffused += amount_diffused

                new_package_layer[x, y] -= total_diffused

        self.package_pheromone_layer.data = new_package_layer

    def evaporate_pheromones(self):

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                current_package = self.package_pheromone_layer.data[x, y]

                if current_package < self.pheromone_treshold:
                    continue

                self.package_pheromone_layer.set_cell((x, y), current_package * (1 - self.pheromone_evaporation))




    def check_packages(self):
        for agent in self.agents:
            if isinstance(agent, Package):
                if agent.pos != agent.destination:
                    return False
        return True

    def get_closest_package_distance(self, pos, radius=6):

        neighbors = self.grid.get_neighbors(pos, moore=True, include_center=False, radius=radius)
        package = [agent for agent in neighbors if isinstance(agent, Package) and not agent.collected]

        if not package:
            return radius + 1

        return min(self.get_distance(pos, s.pos) for s in package)

    def get_distance(self, pos1, pos2):

        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        dx = min(dx, self.grid.width - dx)
        dy = min(dy, self.grid.height - dy)

        return math.sqrt(dx ** 2 + dy ** 2)

    def decay_epsilon(self):
        if self.learning:
            self.q_learning.epsilon = max(self.q_learning.min_epsilon, self.q_learning.epsilon * self.q_learning.epsilon_decay)

    def save_q_tables(self):
        if not self.learning or not hasattr(self, 'q_table_file') or not self.q_table_file:
            return

        q_tables = {}
        for agent in self.agents:
            if isinstance(agent, Robot):
                for state, actions in agent.q_learning.q_table.items():
                    if state not in q_tables:
                        q_tables[state] = actions
                    else:
                        for action, value in actions.items():
                            if action in q_tables[state]:
                                q_tables[state][action] = (q_tables[state][action] + value) / 2
                            else:
                                q_tables[state][action] = value

        if q_tables:
            temp_q_learning = QLearning(actions=[0, 1, 2, 3])
            temp_q_learning.q_table = q_tables
            temp_q_learning.save_q_table(self.q_table_file)
            #print("Saved: ", self.q_table_file)
    def step(self):



        if self.check_packages() or self.steps >= self.max_steps:
            #for agent in self.agents:
            #    if isinstance(agent, Robot):
            #        print("Numero di package delivered: ", agent.package_delivered)
            self.running = False
            self.datacollector.collect(self)
            self.decay_epsilon()
            self.save_q_tables()

        else:

            for agent in self.agents:
                if isinstance(agent, Package):
                    agent.step()

            for agent in self.agents:
                if isinstance(agent, Robot):
                    agent.step()

            if self.render_pheromone:
                for agent in self.agents:
                    if isinstance(agent, Pheromones):
                        agent.step()
                for agent in self.agents:
                    if isinstance(agent, Pheromones):
                        agent.apply_diffusion()

            self.evaporate_pheromones()
            self.diffuse_pheromones()







