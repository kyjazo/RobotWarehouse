import json
import os
from dataclasses import dataclass
import random

from mesa import Agent
import numpy as np


@dataclass
class Pheromone:
    robot_pheromone: float = 0.0
    package_pheromone: float = 0.0


class QLearning:
    def __init__(self, actions=[0, 1, 2, 3], alpha=0.1, gamma=0.9, epsilon=0.5, epsilon_decay=0.995,
                 min_epsilon=0.01,
                 q_table_file=None, q_learning=None):
        if q_learning:
            self.actions = q_learning.actions
            self.alpha = q_learning.alpha
            self.gamma = q_learning.gamma
            self.epsilon = q_learning.epsilon
            self.epsilon_decay = q_learning.epsilon_decay
            self.min_epsilon = q_learning.min_epsilon

        else:
            self.actions = actions
            self.alpha = alpha  # learning rate
            self.gamma = gamma  # discount factor
            self.epsilon = epsilon  # exploration rate iniziale
            self.epsilon_decay = epsilon_decay  # fattore di decadimento
            self.min_epsilon = min_epsilon  # valore minimo di exploration

        self.q_table = {}
        self.training = True
        # print(os.path.exists(q_table_file))
        if q_table_file and os.path.exists(q_table_file):
            self.load_q_table(q_table_file)
            # print("Ho caricato i pesi del file: ", q_table_file)

        # print("Creato q_learning con azioni: ", self.actions)

    def decay_epsilon(self):
        if self.training:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filename):
        serializable_q_table = {
            str(k): {int(ak): float(av) for ak, av in v.items()}
            for k, v in self.q_table.items()
        }

        with open(filename, 'w') as f:
            json.dump(serializable_q_table, f, indent=2)

        # print(f"Salvato q_table in {filename} (dimensione: {len(serializable_q_table)} stati)")

    def load_q_table(self, filename):
        if not os.path.exists(filename):
            return
        else:
            with open(filename, 'r') as f:
                serializable_q_table = json.load(f)

            self.q_table = {
                eval(k): {int(ak): float(av) for ak, av in v.items()}
                for k, v in serializable_q_table.items()
            }

        # print(f"Caricato q_table da {filename} (dimensione: {len(self.q_table)} stati)")

    def get_state(self, robot, pheromones, package_presence=None):
        threshold = robot.model.pheromone_treshold
        filtered_pheromones = []
        for ph in pheromones:
            robot_conc = ph.robot_pheromone if ph.robot_pheromone >= threshold else 0
            package_conc = ph.package_pheromone if ph.robot_pheromone >= threshold else 0
            filtered_pheromones.append(Pheromone(robot_pheromone=robot_conc, package_pheromone=package_conc))

        pheromones = filtered_pheromones

        possible_steps = robot.model.grid.get_neighborhood(
            robot.pos, moore=True, include_center=False, radius=1)
        robot_presence = any(
            any(isinstance(obj, Robot) for obj in robot.model.grid.get_cell_list_contents(step))
            for step in possible_steps
        )
        if package_presence is None:
            package_presence = any(
                any(isinstance(obj, Package) for obj in robot.model.grid.get_cell_list_contents(step))
                for step in possible_steps
            )


        accumulation = [ph.package_pheromone + ph.robot_pheromone for ph in pheromones]
        max_index = int(np.argmax(accumulation))
        #print(package_presence)

        package_presence = int(package_presence)
        robot_presence = int(robot_presence)
        if robot.carrying_package is None:
            holding_package = False
        else:
            holding_package = True
        holding_package = int(holding_package)

        return (max_index, package_presence, robot_presence, holding_package)

    def choose_action(self, state):

        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}

        if self.training and random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}

        old_value = self.q_table[state][action]
        next_max = max(self.q_table.get(next_state, {a: 0 for a in self.actions}).values())
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = np.clip(new_value, -100, 100)



class Robot(Agent):
    def __init__(self, model, q_table_file="q_table_avg.json", q_learning=None):
        super().__init__(model)

        self.carrying_package = None
        self.target_location = None
        self.q_learning = q_learning
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.package_delivered = 0
        self.previous_pos = self.pos

        if self.q_learning != None:

            self.q_learning = QLearning(q_learning=self.model.q_learning,
                                        q_table_file=q_table_file)
        else:
            self.q_learning = QLearning()
            #self.q_learning.load_q_table(model.q_table_file)

        #print(self.q_learning.epsilon)
        self.rewards = []
        self.last_package_distance = float('inf')
        self.last_robot_distance = float('inf')

        self.picked_package = False
        self.use_learning = self.model.learning
        self.shared_reward = 0


    def get_best_step(self, possible_steps, pheromones, action=None):
        valid_steps = []
        for step in possible_steps:
            cell_contents = self.model.grid.get_cell_list_contents([step])
            is_free = not any(isinstance(obj, (Obstacle, Package)) for obj in cell_contents)
            if is_free:
                valid_steps.append(step)
        possible_steps = valid_steps
        threshold = self.model.pheromone_treshold
        filtered_pheromones = []
        for ph in pheromones:
            robot_conc = ph.robot_pheromone if ph.robot_pheromone >= threshold else 0
            package_conc = ph.package_pheromone if ph.package_pheromone >= threshold else 0
            filtered_pheromones.append(Pheromone(robot_pheromone=robot_conc, package_pheromone=package_conc))
        pheromones = filtered_pheromones
        if self.carrying_package:
            min_distance = float('inf')
            for step in possible_steps:
                distance = abs(step[0] - self.target_location[0]) + abs(step[1] - self.target_location[1])
                if distance < min_distance:
                    min_distance = distance
                    best_steps = [step]
                elif distance == min_distance:
                    best_steps.append(step)
            return best_steps
        if action == 0: #mi avvicino al pacco
            pheromone_concentrations = [ph.package_pheromone for ph in pheromones]
            max_pheromone = max(pheromone_concentrations)
            return [step for step, conc in zip(possible_steps, pheromone_concentrations) if
                    conc == max_pheromone]
        elif action == 1: #mi avvicino al robot
            pheromone_concentrations = [ph.robot_pheromone for ph in pheromones]
            max_pheromone = max(pheromone_concentrations)
            return [step for step, conc in zip(possible_steps, pheromone_concentrations) if
                    conc == max_pheromone]
        elif action == 3:  #random movemente
            return [self.random.choice(possible_steps)]


    def calculate_reward(self):

        base_reward = 0
        #print("shared reward: ", self.shared_reward)
        base_reward += self.shared_reward
        self.shared_reward = 0

        if self.previous_pos == self.pos:
            #print("Penalità per essere rimasto fermo")
            base_reward -= 1

        if self.picked_package:
            #print("ho raccolto un package allo step:", self.model.steps)

            base_reward += 10.0


            self.last_package_distance = self.model.get_closest_package_distance(self.pos)
            #self.last_goal_distance = self.model.get_distance(self.pos, self.target_location)
            self.picked_package = False
            #print("reward per aver raccolto un pacco", base_reward)
            return base_reward


        current_dist = self.model.get_closest_package_distance(self.pos)
        #print("Last distance: ", self.last_package_distance)
        #print("Current distance: ", current_dist)
        if hasattr(self, 'last_package_distance') and self.model.steps > 0:
            dist_change = self.last_package_distance - current_dist

            distance_reward = dist_change  # * 2
            #base_reward += distance_reward
            #print("Reward per avvicinamento package: ", distance_reward)

        self.last_package_distance = current_dist

        current_dist = self.model.get_closest_robot_distance(self.pos)
        # print("Last distance: ", self.last_package_distance)
        # print("Current distance: ", current_dist)
        if hasattr(self, 'last_robot_distance') and self.model.steps > 0:
            dist_change = self.last_robot_distance - current_dist

            distance_reward = dist_change  # * 2
            #base_reward += distance_reward
            #print("Reward per avvicinamento robot: ", distance_reward)

        self.last_robot_distance = current_dist


        #print("Reward per pass: ", base_reward)
        return base_reward



    def get_package(self):

        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False, radius=1)

        for step in possible_steps:
            cell_contents = self.model.grid.get_cell_list_contents([step])
            for obj in cell_contents:
                if isinstance(obj, Package) and not obj.collected and obj.check_robot_nearby() >= obj.weight:
                    #print("Pacco raccolto con peso: ", obj.weight)

                    #devo controllare i robot adiacenti al pacco e darli reward
                    neighborhood = self.model.grid.get_neighborhood(
                        obj.pos, moore=True, include_center=False
                    )
                    for cell in neighborhood:
                        cell_contents = self.model.grid.get_cell_list_contents([cell])
                        for r in cell_contents:
                            if isinstance(r, Robot) and r is not self:
                                r.shared_reward += 10 #reward per aver aiutato


                    self.carrying_package = obj
                    obj.collected = True
                    self.model.grid.remove_agent(obj)

                    self.target_location = obj.destination
                    self.release_package()

                    self.model.grid.remove_agent(obj)
                    return True
                    #obj.collected = True
                    #obj.delivered = True
                    #self.model.grid.remove_agent(obj)
                    #return True
        return False

    def release_package(self):


        self.model.grid.place_agent(self.carrying_package, tuple(self.pos))
        self.carrying_package.delivered = True
        self.carrying_package = None
        self.target_location = None
        self.package_delivered += 1


    def step(self):
        #il rilascio e il raccoglimento di un package non conta come mossa dello step, possono muoversi dopo
        if(self.model.steps == 1):
            self.last_package_distance = self.model.get_closest_package_distance(self.pos)
            self.last_robot_distance = self.model.get_closest_robot_distance(self.pos)
            #self.last_goal_distance = self.model.get_distance(self.pos, self.target_location)
            #print("Aggiornate le distanze")


        if self.pos == self.target_location:
            #print("package rilasciato")
            self.release_package()
            self.last_package_distance = self.model.get_closest_package_distance(self.pos)
            return




        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False, radius=1)
        


        pheromones = [
            Pheromone(
                robot_pheromone=self.model.robot_pheromone_layer.data[x, y],
                package_pheromone=self.model.package_pheromone_layer.data[x, y]
            ) for (x, y) in possible_steps
        ] if not self.model.render_pheromone else [
            next((obj.pheromone for obj in self.model.grid.get_cell_list_contents(step) if isinstance(obj, Pheromones)),
                 Pheromone())
            for step in possible_steps
        ]
        if self.carrying_package:
            #print("ho in mano un package")
            valid_steps = []
            for step in possible_steps:
                cell_contents = self.model.grid.get_cell_list_contents([step])
                is_free = not any(isinstance(obj, (Obstacle)) for obj in cell_contents)
                if is_free:
                    valid_steps.append(step)
            min_distance = float('inf')

            for step in valid_steps:
                distance = abs(step[0] - self.target_location[0]) + abs(step[1] - self.target_location[1])

                if distance < min_distance:
                    min_distance = distance
                    best_steps = [step]
                elif distance == min_distance:
                    best_steps.append(step)
            if best_steps:
                self.model.grid.move_agent(self, self.random.choice(best_steps))
            return
        #se ho un package mi muovo verso la destinazione e salto lo step di reward

        #print("Mi muovo usando q-learning")
        if self.use_learning:

            state = self.q_learning.get_state(self, pheromones)
            action = self.q_learning.choose_action(state)
            self.action_counts[action] += 1
        else:
            action = -1

        if self.carrying_package:
            best_steps = self.get_best_step(possible_steps, pheromones)
            if best_steps:
                self.model.grid.move_agent(self, self.random.choice(best_steps))

        elif action == 2:
            self.update_pheromone()
        else:
            best_steps = self.get_best_step(possible_steps, pheromones, action)
            if best_steps:
                self.model.grid.move_agent(self, self.random.choice(best_steps))

        if self.carrying_package is None:
            self.picked_package = self.get_package()


        if self.use_learning:

            reward = self.calculate_reward()

            self.rewards.append(reward)

            next_possible_steps = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False
            )
            next_pheromones = [
                Pheromone(
                    robot_pheromone=self.model.robot_pheromone_layer.data[x, y],
                    package_pheromone=self.model.package_pheromone_layer.data[x, y]
                ) for (x, y) in next_possible_steps
            ] if not self.model.render_pheromone else [
                next((obj.pheromone for obj in self.model.grid.get_cell_list_contents(step) if
                      isinstance(obj, Pheromones)),
                     Pheromone())
                for step in next_possible_steps
            ]

            next_package_presence = any(
                any(isinstance(obj, Package) for obj in self.model.grid.get_cell_list_contents(step)) for step in
                next_possible_steps)

            next_state = self.q_learning.get_state(self, next_pheromones, next_package_presence)
            self.q_learning.learn(state, action, reward, next_state)

        #print("Previous pos: ", self.previous_pos)
        self.previous_pos = self.pos
        #print("Aggiornato previous_pos: ", self.previous_pos)


    def update_pheromone(self):
        if not self.carrying_package:
            self.model.robot_pheromone_layer.set_cell(self.pos, self.model.robot_pheromone_layer.data[self.pos]
                                                    + self.model.pheromone_added)

            if self.model.render_pheromone:
                x, y = self.pos
                cell_contents = self.model.grid.get_cell_list_contents((x, y))
                for obj in cell_contents:
                    if isinstance(obj, Pheromones):
                        obj.pheromone.robot_pheromone += self.model.pheromone_added

class Package(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.collected = False
        self.delivered = False
        self.destination = self.assign_random_destination()
        self.weight = self.random.randint(1, self.model.max_weight)

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
        self.update_pheromone()

    def update_pheromone(self):
        if not self.collected:
            self.model.package_pheromone_layer.set_cell(self.pos, self.model.package_pheromone_layer.data[self.pos]
                                                    + self.model.pheromone_added)

            if self.model.render_pheromone:
                x, y = self.pos
                cell_contents = self.model.grid.get_cell_list_contents((x, y))
                for obj in cell_contents:
                    if isinstance(obj, Pheromones):
                        obj.pheromone.package_pheromone += self.model.pheromone_added
    def check_robot_nearby(self): #funzione che ritorna quanti robot sono nel neighborhood
        neighborhood = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False
            )
        count_robots = 0
        for cell in neighborhood:
            cell_contents = self.model.grid.get_cell_list_contents([cell])
            for obj in cell_contents:
                if isinstance(obj, Robot):
                    count_robots += 1
        #print("Ho robot vicini pari a: ", count_robots)
        return count_robots
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

