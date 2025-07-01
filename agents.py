from dataclasses import dataclass

from mesa import Agent
import numpy as np


@dataclass
class Pheromone:
    robot_pheromone: float = 0.0
    package_pheromone: float = 0.0

class Robot(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.carrying_package = None
        self.target_location = None
        return

    def get_best_step(self, possible_steps):

        valid_steps = []
        pheromone_levels = []

        for step in possible_steps:

            cell_contents = self.model.grid.get_cell_list_contents([step])
            is_free = not any(isinstance(obj, (Obstacle, Robot, Package)) for obj in cell_contents)

            if is_free:
                pheromone = self.model.package_pheromone_layer.data[step[0]][step[1]]
                if pheromone < self.model.pheromone_treshold:
                    pheromone = np.float64(0.00)
                valid_steps.append(step)
                pheromone_levels.append(pheromone)


        if not valid_steps:
            return None

        if self.carrying_package:
            min_distance = float('inf')

            for step in valid_steps:
                distance = abs(step[0] - self.target_location[0]) + abs(step[1] - self.target_location[1])

                if distance < min_distance:
                    min_distance = distance
                    best_steps = [step]
                elif distance == min_distance:
                    best_steps.append(step)

        else:
            max_pheromone = max(pheromone_levels)

            best_steps = [step for step, ph in zip(valid_steps, pheromone_levels) if ph == max_pheromone]

        best_step = self.model.random.choice(best_steps) if best_steps else None

        return best_step
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )

        if not self.carrying_package:
            self.get_package(possible_steps)

        best_step = self.get_best_step(possible_steps)
        if best_step:
            self.model.grid.move_agent(self, best_step)

    def get_package(self, possible_steps):

        for step in possible_steps:
            cell_contents = self.model.grid.get_cell_list_contents([step])
            for obj in cell_contents:
                if isinstance(obj, Package) and not obj.collected:
                    #print("Pacco raccolto")

                    self.carrying_package = obj
                    obj.collected = True
                    self.model.grid.remove_agent(obj)
                    self.target_location = obj.destination
                    return True
        return False

    def release_package(self):


        self.model.grid.place_agent(self.carrying_package, tuple(self.pos))

        self.carrying_package = None
        self.target_location = None


    def step(self):
        if self.pos == self.target_location:
            self.release_package()
        self.move()



class Package(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.collected = False
        self.destination = self.assign_random_destination()

    def assign_random_destination(self):
        while True:
            x = self.random.randrange(self.model.width)
            y = self.random.randrange(self.model.height)
            dest = (x, y)

            cell_contents = self.model.grid.get_cell_list_contents([dest])
            is_occupied = any(isinstance(obj, Obstacle) for obj in cell_contents)

            if not is_occupied:
                return dest

    def step(self):
        self.releasePheromone()

    def releasePheromone(self):
        if not self.collected:


            self.model.package_pheromone_layer.set_cell(self.pos, self.model.package_pheromone_layer.data[self.pos]
                                                    + self.model.pheromone_added)

            if self.model.render_pheromone:
                x, y = self.pos
                cell_contents = self.model.grid.get_cell_list_contents((x, y))
                for obj in cell_contents:
                    if isinstance(obj, Pheromones):
                        obj.pheromone.package_pheromone += self.model.pheromone_added
class Obstacle(Agent):
    def __init__(self, model):
        super().__init__(model)
        return



class Pheromones(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.pheromone = Pheromone()
        self.next_robot = 0.0
        self.next_package = 0.0

    def prepare_diffusion(self):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]

        fraction_per_direction = self.model.diffusion_rate / len(directions)
        x, y = self.pos

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < self.model.grid.width and 0 <= ny < self.model.grid.height:
                neighbors = self.model.grid.get_cell_list_contents((nx, ny))
                for neighbor in neighbors:
                    if isinstance(neighbor, Pheromones):

                        wolf_diffused = self.pheromone.robot_pheromone * fraction_per_direction
                        sheep_diffused = self.pheromone.package_pheromone * fraction_per_direction

                        neighbor.next_robot += wolf_diffused
                        neighbor.next_package += sheep_diffused

                        self.next_robot -= wolf_diffused
                        self.next_package -= sheep_diffused

    def apply_diffusion(self):
        self.pheromone.robot_pheromone += self.next_robot
        self.pheromone.package_pheromone += self.next_package
        self.next_robot = 0.0
        self.next_package = 0.0

    def step(self):
        #if self.pheromone.package_pheromone < self.model.pheromone_treshold:
        #    self.pheromone.package_pheromone = 0
        #if self.pheromone.robot_pheromone < self.model.pheromone_treshold:
        #    self.pheromone.robot_pheromone = 0

        self.pheromone.package_pheromone *= (1 - self.model.pheromone_evaporation)
        self.pheromone.robot_pheromone *= (1 - self.model.pheromone_evaporation)
        self.prepare_diffusion()

