import gc
from collections import defaultdict

import mesa
import pandas as pd
import matplotlib.pyplot as plt
from model import WarehouseModel
from agents import QLearning, Robot
import os
import json
from datetime import datetime
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def save_q_table_to_results(q_table_file, output_dir):
    if os.path.exists(q_table_file):
        dest_path = os.path.join(output_dir, os.path.basename(q_table_file))
        shutil.copy2(q_table_file, dest_path)
        print(f"📋 Q-table saved in: {dest_path}")
    else:
        print("⚠️ Q-table file not found, no copy made")

def get_next_test_folder(testing=False, learning=False, prefix="test"):
    global save

    if testing and learning:
        base_dir = "test_results"
    elif testing and not learning:
        base_dir = "test_results_NL"
    else:
        base_dir = "results"

    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(base_dir)
        if isinstance(d, str) and os.path.isdir(os.path.join(base_dir, d)) and d.startswith(str(prefix))
    ]

    numbers = [int(d[len(prefix):]) for d in existing if d[len(prefix):].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1

    new_folder = os.path.join(base_dir, f"{prefix}{next_number}")
    if save:
        os.makedirs(new_folder)

    return new_folder, os.path.abspath(new_folder)

def save_simulation_metadata(params, learning_params=None, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "simulation_params": params
    }

    if learning_params:
        metadata["q_learning_params"] = learning_params

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"simulation_params_{timestamp}.json")
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Parameters saved in {filepath}")

def plot_reward(df, output_dir="./results", window_size=100):
    if "Reward" not in df.columns:
        print("⚠️ 'Reward' column not found.")
        return

    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-v0_8')

    reward_stats = df.groupby('iteration')['Reward'].agg(['mean', 'std']).reset_index()
    reward_stats['mean_rolling'] = reward_stats['mean'].rolling(window=window_size).mean()

    plt.plot(reward_stats['iteration'], reward_stats['mean_rolling'],
             color='darkblue', linewidth=2.5, label='Average Reward')

    if 'run_id' in df.columns and len(df['run_id'].unique()) > 1:
        reward_stats['std_rolling'] = reward_stats['std'].rolling(window=window_size).mean()
        plt.fill_between(reward_stats['iteration'],
                         reward_stats['mean_rolling'] - reward_stats['std_rolling'],
                         reward_stats['mean_rolling'] + reward_stats['std_rolling'],
                         color='blue', alpha=0.2)

    plt.title(f'Average Reward (rolling mean {window_size})', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save:
        filename = f"reward_rolling{window_size}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

def plot_simulation_steps(df, output_dir="./results", window_size=100):
    if save:
        os.makedirs(output_dir, exist_ok=True)
    if "Step" not in df.columns:
        print("'Steps' column not available in data.")
        return

    window_size = window_size

    if 'run_id' in df.columns:
        steps_data = df.groupby(["run_id", "iteration"])["Step"].max().reset_index()
        steps_stats = steps_data.groupby("iteration")["Step"].agg(["mean", "std"]).reset_index()
    else:
        steps_stats = df.groupby("iteration")["Step"].max().reset_index()
        steps_stats["std"] = 0

    steps_stats['mean_rolling'] = steps_stats['mean'].rolling(window=window_size).mean()
    steps_stats['std_rolling'] = steps_stats['std'].rolling(window=window_size).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(steps_stats["iteration"], steps_stats["mean_rolling"],
             color="green", linewidth=2.5,
             label=f"Number of Steps (rolling mean {window_size})")
    plt.fill_between(steps_stats["iteration"],
                     steps_stats["mean_rolling"] - steps_stats["std_rolling"],
                     steps_stats["mean_rolling"] + steps_stats["std_rolling"],
                     color='green', alpha=0.2)
    plt.title("Duration of Episodes in Steps", fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Steps", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout(pad=3)

    filepath = os.path.join(output_dir, f"steps_plot_rolling{window_size}.png")
    if save:
        plt.savefig(filepath)
        print(f"📈 Steps plot saved in: {filepath}")
    plt.show()

def plot_all_actions_in_one(df, output_dir="./results", window_size=100):
    plt.figure(figsize=(20, 10))
    plt.style.use('seaborn-v0_8')

    actions = ["Action_0", "Action_1", "Action_2", "Action_3"]
    action_labels = ["Follow package pheromone", "Follow robot pheromone", "Deposit pheromone", "Random movement"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    window_size = window_size

    if 'run_id' in df.columns:
        grouped = df.groupby(["run_id", "iteration"])[actions].sum().reset_index()
        mean_grouped = grouped.groupby("iteration")[actions].mean().reset_index()
        std_grouped = grouped.groupby("iteration")[actions].std().reset_index()
    else:
        mean_grouped = df.groupby("iteration")[actions].sum().reset_index()
        std_grouped = mean_grouped.copy()
        for action in actions:
            std_grouped[action] = 0

    for action, label, color in zip(actions, action_labels, colors):
        mean_grouped[f'{action}_rolling'] = mean_grouped[action].rolling(window=window_size).mean()
        std_grouped[f'{action}_rolling'] = std_grouped[action].rolling(window=window_size).mean()

        plt.plot(mean_grouped["iteration"], mean_grouped[f'{action}_rolling'],
                 label=label, linewidth=3, color=color, alpha=0.8)

        plt.fill_between(mean_grouped["iteration"],
                         mean_grouped[f'{action}_rolling'] - std_grouped[f'{action}_rolling'],
                         mean_grouped[f'{action}_rolling'] + std_grouped[f'{action}_rolling'],
                         color=color, alpha=0.2)

    plt.title(f"Actions Distribution (rolling mean {window_size})", fontsize=18, pad=20)
    plt.xlabel("Episode", fontsize=16, labelpad=15)
    plt.ylabel("Action Count", fontsize=16, labelpad=15)
    plt.legend(title="Actions", title_fontsize=14, fontsize=12,
               frameon=True, shadow=True, facecolor='white', edgecolor='gray',
               bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir, f"all_actions_usage_rolling{window_size}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()


def plot_packages_delivered(df, output_dir="./results", window_size=100):
    if "Package_delivered" not in df.columns:
        print("⚠️ 'Package_delivered' column not found.")
        return

    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-v0_8')

    if 'run_id' in df.columns:
        # Per simulazioni multiple
        delivered_stats = df.groupby(["run_id", "iteration"])["Package_delivered"].sum().reset_index()
        stats = delivered_stats.groupby("iteration")["Package_delivered"].agg(["mean", "std"]).reset_index()
    else:
        # Per simulazioni singole
        stats = df.groupby("iteration")["Package_delivered"].sum().reset_index()
        stats["std"] = 0

    stats['mean_rolling'] = stats['mean'].rolling(window=window_size).mean()
    stats['std_rolling'] = stats['std'].rolling(window=window_size).mean()

    plt.plot(stats['iteration'], stats['mean_rolling'],
             color='green', linewidth=2.5, label='Packages Delivered')

    if 'std_rolling' in stats.columns:
        plt.fill_between(stats['iteration'],
                         stats['mean_rolling'] - stats['std_rolling'],
                         stats['mean_rolling'] + stats['std_rolling'],
                         color='green', alpha=0.2)

    plt.title(f'Packages Delivered (rolling mean {window_size})', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Number of Packages', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save:
        filename = f"packages_delivered_rolling{window_size}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()


def run_single_simulation(run_id, base_params, q_learning_params, num_episodes=5000):
    try:
        params = base_params.copy()

        if not base_params['testing']:
            run_output_dir = os.path.join("./tmp_runs", f"run_{run_id}")
            os.makedirs(run_output_dir, exist_ok=True)
            q_table_file = os.path.join(run_output_dir, "q_table_anchor.json")
            params["q_table_file"] = q_table_file
            q = QLearning(**q_learning_params, q_table_file=q_table_file)
        else:
            q_table_file = params['q_table_file']
            q = QLearning(**q_learning_params, q_table_file=q_table_file)

        all_results = []

        for iteration in range(num_episodes):
            model = WarehouseModel(**{k: v for k, v in params.items() if k != "q_learning_params"},
                                   q_learning=q)

            while model.running:
                model.step()

            agent_data = model.datacollector.get_agenttype_vars_dataframe(Robot).reset_index()

            agent_data["iteration"] = iteration
            agent_data["run_id"] = run_id

            all_results.extend(agent_data.to_dict('records'))


            model.grid = None
            model.remove_all_agents()
            model.datacollector = None

            del model
            gc.collect()

            print("Iteration:", iteration)

        return all_results, run_output_dir

    except Exception as e:
        import traceback
        print(f"⚠️ Error in simulation {run_id}: {traceback.format_exc()}")
        return [], None
def merge_q_tables(run_dirs, output_dir, n_robots):
    q_tables_out_dir = os.path.join(output_dir, "q_tables")
    os.makedirs(q_tables_out_dir, exist_ok=True)

    for idx in range(n_robots):
        combined_q = defaultdict(lambda: defaultdict(float))
        counts = defaultdict(lambda: defaultdict(int))

        for run_dir in run_dirs:
            q_file = os.path.join(run_dir, "q_tables", f"q_table_{idx}.json")
            if not os.path.exists(q_file):
                continue
            with open(q_file, "r") as f:
                q_table = json.load(f)

            for state, actions in q_table.items():
                for action, value in actions.items():
                    combined_q[state][action] += float(value)
                    counts[state][action] += 1

        avg_q = {}
        for state, actions in combined_q.items():
            avg_q[state] = {}
            for action, total in actions.items():
                avg_q[state][action] = total / counts[state][action]

        out_file = os.path.join(q_tables_out_dir, f"q_table_{idx}.json")
        with open(out_file, "w") as f:
            json.dump(avg_q, f)


def clean_up_q_tables(run_dirs):
    """
    Rimuove le cartelle temporanee delle run dopo aver unito le Q-tables
    """
    for run_dir in run_dirs:
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
            print(f"🗑️ Removed temporary run directory: {run_dir}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    num_parallel_runs = 3  # Numero di run parallele

    base_params = {
        "width": 45,
        "height": 45,
        "num_robot": 10,  
        "num_package": 20,
        "learning": True,
        "max_steps": 200,
        "diffusion_rate": 0.5,
        "pheromone_evaporation": 0.1,
        "testing": False,
        "render_pheromone": False,
        "max_weight": 3,
        "q_table_file": None,  # Sarà impostato automaticamente
    }

    q_learning_params = {
        "actions": [0, 1, 2, 3],  # Azioni specifiche per i robot
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 0.9985,
        "min_epsilon": 0.01
    }

    all_results = []
    run_dirs = []  # Per tenere traccia delle cartelle temporanee
    save = True

    try:
        with ProcessPoolExecutor(max_workers=num_parallel_runs) as executor:
            futures = [executor.submit(run_single_simulation, i, base_params, q_learning_params)
                       for i in range(num_parallel_runs)]

            for i, future in enumerate(as_completed(futures)):
                try:
                    sim_result, run_dir = future.result()

                    for r in sim_result:
                        r["run_id"] = i
                    all_results.extend(sim_result)

                    if base_params['learning'] and not base_params['testing']:
                        run_dirs.append(run_dir)

                except Exception as e:
                    import traceback

                    print(f"⚠️ Error in simulation {i}: {traceback.format_exc()}")

    except KeyboardInterrupt:
        print("⛔ Interrupted by user.")

    if all_results:
        df = pd.DataFrame(all_results)
        df = df.dropna(subset=['Reward'])  # O 'Package_delivered' a seconda dei tuoi dati

        output_dir, abs_output_dir = get_next_test_folder(
            base_params['testing'],
            base_params['learning']
        )

        if base_params['learning'] and not base_params['testing']:
            # Unisci le Q-tables dalle diverse run
            merge_q_tables(run_dirs, abs_output_dir, n_robots=base_params["num_robot"])
            # Pulisci le cartelle temporanee
            clean_up_q_tables(run_dirs)

        if save:
            save_simulation_metadata(base_params, q_learning_params, output_dir=output_dir)

        # Plot dei risultati
        window_size = 100
        plot_reward(df, output_dir=output_dir, window_size=window_size)
        plot_packages_delivered(df, output_dir=output_dir, window_size=window_size)
        plot_simulation_steps(df, output_dir=output_dir, window_size=window_size)
        plot_all_actions_in_one(df, output_dir=output_dir, window_size=window_size)