import solara
from mesa.visualization import SolaraViz, make_space_component, Slider
from model import WarehouseModel
from agents import Robot, Package, Obstacle, Pheromones


def agent_portrayal(agent):
    portrayal = {
        "size": 120,
    }

    if isinstance(agent, Robot):
        portrayal["color"] = "darkred"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2

    elif isinstance(agent, Package):
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


           red_hex = int(red_intensity * 255)
           green_hex = int(green_intensity * 255)
           portrayal["color"] = f"#{red_hex:02x}{green_hex:02x}00"
           portrayal["marker"] = "s"
           portrayal["size"] = 75

    return portrayal







model_params = {

    "render_pheromone": {
        "type": "Select",
        "value": False,
        "values": [True, False],
        "label": "Render Pheromone?",
    },

    "height": Slider("Height", 15, 5, 100, 5, dtype=int),
    "width": Slider("Width", 15, 5, 100, 5, dtype=int),
    "num_robot": Slider("Number of robots", 2, 1, 5, 1, dtype=int),
    "num_package": Slider("Number of packages", 3, 1, 20, 1, dtype=int),
    "num_obstacle": Slider("Number of obstacles", 4, 1, 20, 1, dtype=int),
    "pheromone_evaporation": Slider("Pheromone Evaporation", 0.1, 0, 1, 0.01, dtype=float),
    "pheromone_added": Slider("Pheromone Released", 1, 0, 5, 0.1, dtype=float),
    "diffusion_rate": Slider("Diffusion Rate", 0.5, 0.01, 1, 0.1, dtype=float),

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

