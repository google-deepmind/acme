import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def gather_evaluator_csv_files_from_base_dir(base_dir=None, selected_acme_ids=None):
    """
    Here the base_dir is assumed to be ~/acme, and under the base_dir are the dirs
    named with `acme_id`, under which has `logs/evaluator/logs.csv`.

    Args:
        base_dir (_type_): acme results dir, if None use the default acme setting
        selected_acme_ids (_type_): if None, use all acme_ids, otherwise use the selected ones
        
    Returns:
        a dict of {acme_id: csv_path}
    """
    if base_dir is None:
        base_dir = os.path.expanduser('~/acme')

    # find all subdir and get the acme_id of all
    id_to_csv = {}
    for acme_id in os.listdir(base_dir):
        if selected_acme_ids is not None and acme_id not in selected_acme_ids:
            continue
        exp_dir = os.path.join(base_dir, acme_id)
        if not os.path.isdir(exp_dir):
            continue
        csv_path = os.path.join(exp_dir, 'logs', 'evaluator', 'logs.csv')
        id_to_csv[acme_id] = csv_path

    return id_to_csv


def plot_all_learning_curves(id_to_csv):
    """Given a dict of {acme_id: csv_path}, plot all learning curves on one figure

    Args:
        id_to_csv: a dict of {acme_id: csv_path}
    """
    for acme_id, csv_path in id_to_csv.items():
        print(f"Plotting {acme_id}")
        df = pd.read_csv(csv_path)
        steps = df['actor_steps']
        frames = 4 * steps
        returns = df['episode_return']
        
        # sparsify data for plotting
        every_n = 5
        frames = frames[frames.index % every_n == 0]
        returns = returns[returns.index % every_n == 0]
        
        plt.plot(frames, returns, label=acme_id)
    plt.legend()
    plt.show()
    save_path = os.path.expanduser('~/acme/learning_curves.png')
    print(f"Saving to {save_path}")
    plt.savefig(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--acme_id', type=str, default=None, help='if None, use all acme_ids, otherwise use the selected ones')
    parser.add_argument('--base_dir', type=str, default=None, help='acme results dir, if None use the default acme setting')
    args = parser.parse_args()

    if args.acme_id is None:
        selected_acme_ids = None
    else:
        selected_acme_ids = args.acme_id.split(',')

    id_to_csv = gather_evaluator_csv_files_from_base_dir(base_dir=args.base_dir, selected_acme_ids=selected_acme_ids)
    plot_all_learning_curves(id_to_csv)

