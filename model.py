from mesa import Model
import mesa
from mesa.space import MultiGrid, PropertyLayer
import json
from mesa.datacollection import DataCollector
import numpy as np
from agents import Robot, Package, Obstacle, Pheromones


with open("config.json", "r") as f:
    config = json.load(f)
class WarehouseModel(Model):

    def __init__(self, width=config["width"], height=config["height"], num_robot=config["num_robot"],
                 num_package=config["num_package"], num_obstacle=config["num_obstacle"], pheromone_evaporation=config["pheromone_evaporation"],
                 pheromone_added=config["pheromone_added"], diffusion_rate=config["diffusion_rate"],
                 treshold=config["treshold"], render_pheromone=False, seed=None
                 ):

        super().__init__(seed=seed)



        self.pheromone_treshold = treshold

        self.render_pheromone = render_pheromone
        self.width = width
        self.height = height
        self.num_robot = num_robot
        self.num_package = num_package
        self.num_obstacle = num_obstacle
        self.pheromone_added = pheromone_added
        self.pheromone_evaporation = pheromone_evaporation
        self.diffusion_rate = diffusion_rate



        self.robot_pheromone_layer = PropertyLayer("robot_pheromone_layer", height=height, width=width,
                                                  default_value=0.000)
        self.package_pheromone_layer = PropertyLayer("package_pheromone_layer", height=height, width=width,
                                                   default_value=0.000)

        self.grid = MultiGrid(width, height, torus=False)
        self.running = True

        self.datacollector = DataCollector()


        self.place_agents(Robot, num_robot)
        self.place_agents(Package, num_package)
        self.place_agents(Obstacle, num_obstacle)
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
                if (x, y) not in positions:
                    positions.append((x, y))
            positions = positions[:num_agents]

        elif agent_class.__name__ == "Package":
            agents = agent_class.create_agents(model=self, n=num_agents)
            positions = []
            while len(positions) < num_agents:
                x = self.random.randint(0, self.grid.width - 1)
                y = self.random.randint(0, self.grid.height - 1)
                if (x, y) not in positions:
                    positions.append((x, y))
            positions = positions[:num_agents]

        elif agent_class.__name__ == "Obstacle":
            agents = agent_class.create_agents(model=self, n=num_agents)
            positions = []
            while len(positions) < num_agents:
                x = self.random.randint(0, self.grid.width - 1)
                y = self.random.randint(0, self.grid.height - 1)
                if (x, y) not in positions:
                    positions.append((x, y))
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

    def step(self):



        if self.check_packages():
            self.running = False

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

            self.datacollector.collect(self)





