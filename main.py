import gc
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
            q_table_file = f"q_table_run_{run_id}.json"
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

        return all_results, q_table_file

    except Exception as e:
        import traceback
        print(f"⚠️ Error in simulation {run_id}: {traceback.format_exc()}")
        return [], None
def merge_q_tables(q_table_files, output_file="q_table_avg.json"):
    combined_q_table = {}
    valid_files = []

    for file in q_table_files:
        if not os.path.exists(file):
            print(f"⚠️ Skipping missing Q-table: {file}")
            continue

        with open(file, "r") as f:
            q_table = json.load(f)
        valid_files.append(file)

        for state, actions in q_table.items():
            state = eval(state)
            if state not in combined_q_table:
                combined_q_table[state] = {int(a): float(v) for a, v in actions.items()}
            else:
                for a, v in actions.items():
                    a = int(a)
                    if a in combined_q_table[state]:
                        combined_q_table[state][a] += float(v)
                    else:
                        combined_q_table[state][a] = float(v)

    if not valid_files:
        print("❌ No valid Q-table files found. Cannot merge.")
        return

    for state in combined_q_table:
        for action in combined_q_table[state]:
            combined_q_table[state][action] /= len(valid_files)

    with open(output_file, "w") as f:
        json.dump({str(k): v for k, v in combined_q_table.items()}, f, indent=2)
    print(f"✅ Average Q-table saved in {output_file}")


def clean_up_q_tables(q_table_files, keep_file="q_table_avg.json"):
    for file in q_table_files:
        try:
            if os.path.exists(file) and file != keep_file:
                os.remove(file)
                print(f"🗑️ Q-table deleted: {file}")
        except Exception as e:
            print(f"⚠️ Error deleting {file}: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    num_parallel_runs = 3

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
        "q_table_file": "",
    }

    q_learning_params = {
        "actions": [0, 1, 2, 3],
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 0.9985,
        "min_epsilon": 0.01
    }

    all_results = []
    q_tables_paths = []
    partial_results_file = "partial_results.csv"
    save = True

    if os.path.exists(partial_results_file):
        os.remove(partial_results_file)

    try:
        with ProcessPoolExecutor(max_workers=num_parallel_runs) as executor:
            futures = [executor.submit(run_single_simulation, i, base_params, q_learning_params)
                       for i in range(num_parallel_runs)]

            for i, future in enumerate(as_completed(futures)):
                try:
                    sim_result, q_table_file = future.result()

                    for r in sim_result:
                        r["run_id"] = i
                    all_results.extend(sim_result)

                    pd.DataFrame(sim_result).to_csv(partial_results_file, mode='a', index=False,
                                                    header=not os.path.exists(partial_results_file))

                    if base_params['learning'] and not base_params['testing']:
                        q_tables_paths.append(q_table_file)

                except Exception as e:
                    import traceback
                    print(f"⚠️ Error in simulation {i}: {traceback.format_exc()}")

    except KeyboardInterrupt:
        print("⛔ Interrupted by user.")

    if all_results:
        if base_params['learning'] and not base_params['testing']:
            merge_q_tables(q_tables_paths, output_file="q_table_avg.json")
            clean_up_q_tables(q_tables_paths, keep_file="q_table_avg.json")

        df = pd.DataFrame(all_results)
        print(all_results)
        df = df.dropna(subset=['Reward'])


        print(all_results)


        output_dir, abs_output_dir = get_next_test_folder(base_params['testing'], base_params['learning'])

        if save:
            save_simulation_metadata(base_params, q_learning_params, output_dir=output_dir)
            save_q_table_to_results("q_table_avg.json", abs_output_dir)

        window_size = 100
        plot_reward(df, output_dir=output_dir, window_size=window_size)
        plot_packages_delivered(df, output_dir=output_dir, window_size=window_size)
        plot_simulation_steps(df, output_dir=output_dir, window_size=window_size)
        plot_all_actions_in_one(df, output_dir=output_dir, window_size=window_size)