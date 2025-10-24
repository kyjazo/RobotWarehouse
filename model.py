import math
import os
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
                 max_steps=config["max_steps"], testing=False, max_weight=config["max_weight"], min_weight=config["min_weight"]

                 ):

        super().__init__(seed=seed)

        self.min_weight = min_weight
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
            q_tables_dir = None
            if getattr(self, "q_table_file", None):
                q_tables_dir = os.path.join(os.path.dirname(os.path.abspath(self.q_table_file)), "q_tables")

            agents = []
            if q_tables_dir and os.path.isdir(q_tables_dir):
                for i in range(num_agents):
                    candidate = os.path.join(q_tables_dir, f"q_table_{i}.json")
                    if os.path.exists(candidate):
                        #print(f"[DEBUG] Robot {i} carica q_table da: {candidate}")
                        agent = Robot(self, q_table_file=candidate, q_learning=self.q_learning)
                    else:
                        #print(f"[DEBUG] Robot {i} parte con q_table vuota (nessun file trovato)")
                        agent = Robot(self, q_table_file=None, q_learning=self.q_learning)
                    agents.append(agent)
            else:
                for i in range(num_agents):
                    agent = Robot(self, q_table_file=None, q_learning=self.q_learning)
                    agents.append(agent)
            positions = []
            while len(positions) < num_agents:
                x = self.random.randint(0, self.grid.width - 1)
                y = self.random.randint(0, self.grid.height - 1)
                if (x, y) not in positions and self.grid.is_cell_empty((x, y)):
                    positions.append((x, y))
            positions = positions[:num_agents]


        elif agent_class.__name__ == "Package":
            # Definisci la distribuzione fissa dei pesi
            weights_distribution = [2, 3, 4, 5]  # Pesi possibili
            num_each_weight = max(1, num_agents // len(weights_distribution))  # Quanti per peso

            agents = []
            for weight in weights_distribution:
                for _ in range(num_each_weight):
                    if len(agents) < num_agents:
                        pkg = Package(self, weight)
                        #pkg.weight = weight  # Imposta il peso fisso
                        agents.append(pkg)

            # Se avanzano posti, riempi con pesi casuali dalla distribuzione
            while len(agents) < num_agents:
                pkg = Package(self)
                pkg.weight = self.random.choice(weights_distribution)
                agents.append(pkg)

            positions = []
            while len(positions) < num_agents:
                x = self.random.randint(0, self.grid.width - 1)
                y = self.random.randint(0, self.grid.height - 1)
                if (x, y) not in positions and self.grid.is_cell_empty((x, y)):
                    positions.append((x, y))
            positions = positions[:num_agents]


        elif agent_class.__name__ == "Obstacle":

            agents = agent_class.create_agents(model=self, n=num_agents)

            #obstacle_pattern = [
            #    # Blocchi 2x2
            #    (3, 3), (3, 4), (4, 3), (4, 4),
            #    (8, 8), (8, 9), (9, 8), (9, 9),
            #    (14, 14), (14, 15), (15, 14), (15, 15),
            #    (18, 2), (18, 3), (19, 2), (19, 3),
####
            #    # Blocchi 1x3 orizzontali
            #    (2, 10), (2, 11), (2, 12),
            #    (10, 18), (10, 19),
####
            #    # Blocchi 1x3 verticali
            #    (12, 2), (13, 2), (14, 2),
            #    (6, 16), (7, 16), (8, 16),
####
            #    # Blocchi a L
            #    (5, 15), (5, 16), (6, 15),  # L in alto a destra
            #    (15, 5), (16, 5), (15, 6),  # L in basso a sinistra
            #    (11, 11), (11, 12), (12, 11),  # L centrale
####
            #    # Ostacoli sparsi per raggiungere il 30%
            #    (1, 1), (1, 18), (18, 1), (18, 18),  # Angoli
            #    (7, 3), (7, 4), (12, 17), (13, 17),
            #    (3, 12), (4, 12), (16, 8), (16, 9),
            #]

            obstacle_pattern = [
                # --- Blocchi 2x2 (esistenti) ---
                (2, 2), (2, 3), (3, 2), (3, 3),
                (2, 12), (2, 13), (3, 12), (3, 13),
                (2, 22), (2, 23), (3, 22), (3, 23),
                (2, 32), (2, 33), (3, 32), (3, 33),
                (12, 2), (12, 3), (13, 2), (13, 3),
                (12, 12), (12, 13), (13, 12), (13, 13),
                (12, 22), (12, 23), (13, 22), (13, 23),
                (12, 32), (12, 33), (13, 32), (13, 33),
                (22, 2), (22, 3), (23, 2), (23, 3),
                (22, 12), (22, 13), (23, 12), (23, 13),
                (22, 22), (22, 23), (23, 22), (23, 23),
                (22, 32), (22, 33), (23, 32), (23, 33),
                (32, 2), (32, 3), (33, 2), (33, 3),
                (32, 12), (32, 13), (33, 12), (33, 13),
                (32, 22), (32, 23), (33, 22), (33, 23),
                (32, 32), (32, 33), (33, 32), (33, 33),
                (42, 2), (42, 3), (43, 2), (43, 3),  # Nuovo blocco 2x2 al bordo destro
                (42, 12), (42, 13), (43, 12), (43, 13),
                (42, 22), (42, 23), (43, 22), (43, 23),
                (42, 32), (42, 33), (43, 32), (43, 33),
##
                # --- Blocchi 1x3 orizzontali (esistenti + estesi) ---
                (6, 6), (6, 7), (6, 8),
                (6, 16), (6, 17), (6, 18),
                (6, 26), (6, 27), (6, 28),
                (6, 36), (6, 37), (6, 38),  # Nuovo blocco orizzontale
                (16, 6), (16, 7), (16, 8),
                (16, 16), (16, 17), (16, 18),
                (16, 26), (16, 27), (16, 28),
                (16, 36), (16, 37), (16, 38),
                (26, 6), (26, 7), (26, 8),
                (26, 16), (26, 17), (26, 18),
                (26, 26), (26, 27), (26, 28),
                (26, 36), (26, 37), (26, 38),
                (36, 6), (36, 7), (36, 8),
                (36, 16), (36, 17), (36, 18),
                (36, 26), (36, 27), (36, 28),
                (36, 36), (36, 37), (36, 38),
##
                # --- Blocchi 1x3 verticali (esistenti + estesi) ---
                (6, 10), (7, 10), (8, 10),
                (6, 20), (7, 20), (8, 20),
                (6, 30), (7, 30), (8, 30),
                (6, 40), (7, 40), (8, 40),  # Nuovo blocco verticale
                (16, 10), (17, 10), (18, 10),
                (16, 20), (17, 20), (18, 20),
                (16, 30), (17, 30), (18, 30),
                (16, 40), (17, 40), (18, 40),
                (26, 10), (27, 10), (28, 10),
                (26, 20), (27, 20), (28, 20),
                (26, 30), (27, 30), (28, 30),
                (26, 40), (27, 40), (28, 40),
                (36, 10), (37, 10), (38, 10),
                (36, 20), (37, 20), (38, 20),
                (36, 30), (37, 30), (38, 30),
                (36, 40), (37, 40), (38, 40),
##
                # --- Blocchi a L (esistenti + nuovi) ---
                (5, 5), (5, 6), (6, 5),
                (5, 15), (5, 16), (6, 15),
                (5, 25), (5, 26), (6, 25),
                (5, 35), (5, 36), (6, 35),  # Nuovo blocco L
                (15, 5), (16, 5), (15, 6),
                (15, 15), (16, 15), (15, 16),
                (15, 25), (16, 25), (15, 26),
                (15, 35), (16, 35), (15, 36),
                (25, 5), (26, 5), (25, 6),
                (25, 15), (26, 15), (25, 16),
                (25, 25), (26, 25), (25, 26),
                (25, 35), (26, 35), (25, 36),
                (35, 5), (36, 5), (35, 6),
                (35, 15), (36, 15), (35, 16),
                (35, 25), (36, 25), (35, 26),
                (35, 35), (36, 35), (35, 36),
##
                # --- Extra cluster sparsi (nuovi per coprire bordi e angoli) ---
                # Angoli superiori/inferiori
                (0, 0), (0, 1), (1, 0), (1, 1),  # Angolo in alto a sinistra
                (0, 43), (0, 44), (1, 43), (1, 44),  # Angolo in alto a destra
                (43, 0), (43, 1), (44, 0), (44, 1),  # Angolo in basso a sinistra
                (43, 43), (43, 44), (44, 43), (44, 44),  # Angolo in basso a destra
                # Bordi verticali
                (0, 20), (0, 21), (1, 20), (1, 21),  # Bordo sinistro
                (44, 20), (44, 21), (43, 20), (43, 21),  # Bordo destro
                # Bordi orizzontali
                (20, 0), (21, 0), (20, 1), (21, 1),  # Bordo superiore
                (20, 44), (21, 44), (20, 43), (21, 43),  # Bordo inferiore
                # Altri cluster centrali per densità
                (10, 40), (11, 40), (10, 41), (11, 41),
                (20, 40), (21, 40), (20, 41), (21, 41),
                (30, 40), (31, 40), (30, 41), (31, 41),
                (40, 10), (41, 10), (40, 11), (41, 11),
                (40, 20), (41, 20), (40, 21), (41, 21),
                (40, 30), (41, 30), (40, 31), (41, 31),
            ]
            positions = [
#
                (x, y) for x, y in obstacle_pattern
                if 0 <= x < self.grid.width and 0 <= y < self.grid.height
#
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
        # Crea copie di entrambi gli strati di feromoni
        new_package_layer = self.package_pheromone_layer.data.copy()
        new_robot_layer = self.robot_pheromone_layer.data.copy()

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]

        fraction_per_direction = self.diffusion_rate / len(directions)

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                # Diffusione feromoni package
                current_package_pheromone = self.package_pheromone_layer.data[x, y]
                if current_package_pheromone >= self.pheromone_treshold:
                    total_package_diffused = 0
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                            amount_diffused = current_package_pheromone * fraction_per_direction
                            new_package_layer[nx, ny] += amount_diffused
                            total_package_diffused += amount_diffused
                    new_package_layer[x, y] -= total_package_diffused

                # Diffusione feromoni robot
                current_robot_pheromone = self.robot_pheromone_layer.data[x, y]
                if current_robot_pheromone >= self.pheromone_treshold:
                    total_robot_diffused = 0
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                            amount_diffused = current_robot_pheromone * fraction_per_direction
                            new_robot_layer[nx, ny] += amount_diffused
                            total_robot_diffused += amount_diffused
                    new_robot_layer[x, y] -= total_robot_diffused

        # Aggiorna entrambi gli strati
        self.package_pheromone_layer.data = new_package_layer
        self.robot_pheromone_layer.data = new_robot_layer

    def evaporate_pheromones(self):

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                current_package = self.package_pheromone_layer.data[x, y]
                current_robot = self.robot_pheromone_layer.data[x, y]

                if current_package > self.pheromone_treshold:
                    self.package_pheromone_layer.set_cell((x, y), current_package * (1 - self.pheromone_evaporation))

                if current_robot > self.pheromone_treshold:
                    self.robot_pheromone_layer.set_cell((x, y), current_robot * (1 - self.pheromone_evaporation))





    def check_packages(self):
        for agent in self.agents:
            if isinstance(agent, Package):
                if not agent.delivered:
                    return False
        return True

    def get_closest_package_distance(self, pos, radius=20):

        neighbors = self.grid.get_neighbors(pos, moore=True, include_center=False, radius=radius)
        package = [agent for agent in neighbors if isinstance(agent, Package) and not agent.collected]

        if not package:
            return radius + 1

        return min(self.get_distance(pos, s.pos) for s in package)
    def get_closest_robot_distance(self, pos, robot = None, radius=20):

        neighbors = self.grid.get_neighbors(pos, moore=True, include_center=True, radius=radius)
        robot = [agent for agent in neighbors if isinstance(agent, Robot) and agent is not robot]

        if not robot:
            return radius + 1

        return min(self.get_distance(pos, s.pos) for s in robot)

    def get_distance(self, pos1, pos2):

        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return math.sqrt(dx ** 2 + dy ** 2)

    def decay_epsilon(self):
        if self.learning:
            self.q_learning.epsilon = max(self.q_learning.min_epsilon, self.q_learning.epsilon * self.q_learning.epsilon_decay)

    def save_q_tables(self):
        if not self.learning or not hasattr(self, 'q_table_file') or not self.q_table_file:
            return

        q_tables_dir = os.path.join(os.path.dirname(os.path.abspath(self.q_table_file)), "q_tables")
        os.makedirs(q_tables_dir, exist_ok=True)

        robot_agents = [a for a in self.agents if isinstance(a, Robot)]

        for idx, robot in enumerate(robot_agents):
            if hasattr(robot, 'q_learning') and robot.q_learning:
                filename = os.path.join(q_tables_dir, f"q_table_{idx}.json")
                try:
                    robot.q_learning.save_q_table(filename)
                except Exception as e:
                    print(f"⚠️ Error saving q_table for robot idx={idx}: {e}")
                #print(f"[DEBUG] Q-table del Robot {idx} salvata in: {filename}")

    def __del__(self):
        self.save_q_tables()
    def step(self):

        #print(self.package_pheromone_layer.data)

        if self.check_packages() or self.steps >= self.max_steps:
            #for agent in self.agents:
            #    if isinstance(agent, Robot):
            #        print("Numero di package delivered: ", agent.package_delivered)
            #print("Epsilon: ", self.q_learning.epsilon)
            self.running = False
            self.datacollector.collect(self)
            self.decay_epsilon()
            self.save_q_tables()

        else:
            #print(self.robot_pheromone_layer.data)

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

            ###per la visualizzazione delle azioni in un singolo episodio
            #self.datacollector.collect(self)






