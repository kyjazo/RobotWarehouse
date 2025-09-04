import solara
from mesa.visualization import SolaraViz, make_space_component, Slider
from model import WarehouseModel
from agents import Robot, Package, Obstacle, Pheromones, QLearning

treshold = 0.001

def agent_portrayal(agent):
    portrayal = {
        "size": 120,
    }

    if isinstance(agent, Robot):
        portrayal["color"] = "darkred"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2

    elif isinstance(agent, Package):
        if(agent.delivered):
            portrayal["color"] = "blue"
            portrayal["marker"] = "o"
            portrayal["zorder"] = 2
        else:
            portrayal["color"] = "green"
            portrayal["marker"] = "o"
            portrayal["zorder"] = 2


    elif isinstance(agent, Obstacle):
        portrayal["color"] = "black"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2

    elif isinstance(agent, Pheromones):
        if agent.pheromone.robot_pheromone == 0 and agent.pheromone.package_pheromone == 0:
            portrayal["color"] = "white"
            portrayal["marker"] = "s"
            portrayal["size"] = 75
        else:
            max_pheromone = max(agent.pheromone.robot_pheromone, agent.pheromone.package_pheromone)
            red_intensity = (agent.pheromone.robot_pheromone / max_pheromone) if max_pheromone != 0 else 0
            green_intensity = (agent.pheromone.package_pheromone / max_pheromone) if max_pheromone != 0 else 0


            blend_to_white = 0.5


            red_val = int(red_intensity * 255)
            green_val = int(green_intensity * 255)
            blue_val = 0


            red_val = int(red_val + blend_to_white * (255 - red_val))
            green_val = int(green_val + blend_to_white * (255 - green_val))
            blue_val = int(blue_val + blend_to_white * (255 - blue_val))

            portrayal["color"] = f"#{red_val:02x}{green_val:02x}{blue_val:02x}"
            portrayal["marker"] = "s"
            portrayal["size"] = 75

    return portrayal





q_learning_params = {
        "actions": [0, 1, 2, 3],
        "alpha": 0.01,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 0.9985,
        "min_epsilon": 0.01
    }

q = QLearning(**q_learning_params, q_table_file="q_table_avg.json")

model_params = {

    "render_pheromone": {
        "type": "Select",
        "value": False,
        "values": [True, False],
        "label": "Render Pheromone?",
    },
    "learning": {
        "type": "Select",
        "value": True,
        "values": [True, False],
        "label": "learning?",
    },
    "testing": {
        "type": "Select",
        "value": True,
        "values": [True, False],
        "label": "testing?",
    },


    "height": Slider("Height", 45, 5, 100, 5, dtype=int),
    "width": Slider("Width", 45, 5, 100, 5, dtype=int),
    "num_robot": Slider("Number of robots", 10, 1, 10, 1, dtype=int),
    "num_package": Slider("Number of packages", 20, 1, 20, 1, dtype=int),
    "num_obstacle": Slider("Number of obstacles", 128, 1, 200, 1, dtype=int),
    "pheromone_evaporation": Slider("Pheromone Evaporation", 0.1, 0, 1, 0.01, dtype=float),
    "pheromone_added": Slider("Pheromone Released", 1, 0, 10, 0.1, dtype=float),
    "diffusion_rate": Slider("Diffusion Rate", 0.5, 0.01, 1, 0.1, dtype=float),
    "max_weight": Slider("Max Weight", 3, 2, 10, 1, dtype=float),
    "q_learning": q

}



SpaceGraph = make_space_component(
    agent_portrayal=agent_portrayal,

)
myWarehouse = WarehouseModel()
viz = SolaraViz(
    model=myWarehouse,
    components=[SpaceGraph],
    model_params=model_params,
    name="WolfSheepModel"
)


@solara.component
def Page():
    return viz

