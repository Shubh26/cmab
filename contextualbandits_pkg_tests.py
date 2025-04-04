"""
This file shows applications of this package: https://github.com/david-cortes/contextualbandits.
These are being tested in parallel to VW.
"""
import json, logging, itertools
import user_model
import pandas as pd, numpy as np, re, os, math
import seaborn as sns; sns.set()
from collections import Counter, defaultdict, deque
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from contextualbandits.online import BootstrappedUCB, BootstrappedTS, LogisticUCB, \
            SeparateClassifiers, EpsilonGreedy, AdaptiveGreedy, ExploreFirst, \
            ActiveExplorer, SoftmaxExplorer
from copy import deepcopy
import matplotlib.pyplot as plt
from pylab import rcParams
from datetime import datetime

logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(funcName)s() — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

priors = {'BootstrappedUCB': 'ucb',
          'BootstrappedTS': 'ts',
          'SeparateClassifiers': 'default',
          'EpsilonGreedy':  'default',
          'LogisticUCB':  'default',
          'AdaptiveGreedy':  'default',
          'ExploreFirst': None,
          'ActiveExplorer': 'default',
          'SoftmaxExplorer': 'default'
          }


# these are params in addition to the common set of params required
# TODO: there other decay types for AdaptiveGreedy - see ex. notebook
specialized_bandit_params = {
  'BootstrappedUCB': {'percentile': 80},
  'BootstrappedTS': dict(),
  'SeparateClassifiers': dict(),
  'EpsilonGreedy': dict(),
  'LogisticUCB':  {'percentile': 70},
  'AdaptiveGreedy': {'decay_type': 'percentile', 'decay': 0.9997},
  'ExploreFirst': {'explore_rounds': 1500},
  'ActiveExplorer': dict(),
  'SoftmaxExplorer': dict()
}


def get_prior(num_advts, bandit_type=None):
    """
    Return prior based on bandit type. The priors are from the sample notebook.
    :param num_advts:
    :param bandit_type:
    :return:
    """
    if bandit_type is None:
        beta_prior = None  # some cases dont require beta prior, so this becomes a pass-through
    elif bandit_type.lower() == 'default':
        beta_prior = (
            (3. / num_advts, 4), 2)  # until there are at least 2 observations of each class, will use this prior
    elif bandit_type.lower() == "ucb":
        beta_prior = ((5. / num_advts, 4), 2)  # UCB gives higher numbers, thus the higher positive prior
    elif bandit_type.lower() == "ts":
        beta_prior = ((2. / np.log2(num_advts), 4), 2)
    else:
        logging.warning(f"Can't understand bandit type {bandit_type}, returning default.")
        beta_prior = (
            (3. / num_advts, 4), 2)
    return beta_prior


def simulate_rounds(model, rewards, actions_hist, X_global, y_global, batch_st, batch_end):
    np.random.seed(batch_st)

    ## choosing actions for this batch
    actions_this_batch = model.predict(X_global[batch_st:batch_end, :]).astype('uint8')

    # keeping track of the sum of rewards received
    rewards.append(y_global[np.arange(batch_st, batch_end), actions_this_batch].sum())

    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, actions_this_batch)

    # now refitting the algorithms after observing these new rewards
    np.random.seed(batch_st)
    model.fit(X_global[:batch_end, :], new_actions_hist, y_global[np.arange(batch_end), new_actions_hist],
              warm_start=True)

    return new_actions_hist


def demo_batch(model_class, num_advts, num_impr, num_segments, num_features=10, batch_size=50, op_dir=None,
               measure_window_size=None, n_jobs=None):
    """
    The batch mode implies the model is retrained after a certain # of impressions, given by batch_size.
    :param model_class:
    :param num_advts:
    :param num_impr:
    :param num_segments:
    :param num_features:
    :param batch_size:
    :return:
    """
    beta_prior = get_prior(num_advts, priors[model_class.__name__])
    additional_params = specialized_bandit_params[model_class.__name__]
    base_algorithm = LogisticRegression(solver='saga', warm_start=True, n_jobs=n_jobs)
    cmab = model_class(deepcopy(base_algorithm), nchoices=num_advts,
                                       beta_prior=beta_prior, **additional_params)
    logging.info(f"Created cmab of type {model_class.__name__}.")
    um = user_model.UserModel(num_features=num_features, num_segments=num_segments, num_advts=num_advts)

    num_batches = int(math.floor(1.0 * num_impr/batch_size))
    # this takes care of handling unequal batches
    batches = np.array_split(np.arange(num_impr), num_batches)
    logging.info(f"total impressions: {num_impr}, # batches = {num_batches} with batch_size={batch_size}.")
    logging.info(f"Will begin serving impressions.")

    # these have the info of users served to so far, and clicks/non-clicks seen
    X_all, actions_all, rew_all = [], [], []
    for batch_idx, batch in enumerate(batches):
        curr_batch_size = len(batch)
        logging.info(f"At batch {batch_idx+1} of {num_batches}. Batch size={curr_batch_size}.")
        X_curr, actions_curr, rew_curr = [], [], []

        if batch_idx == 0:
            # random actions in first batch
            actions_curr = np.random.randint(low=0, high=num_advts, size=curr_batch_size)
            for impr_idx, _ in enumerate(batch):
                u = um.get_current_user() # since first batch is random, we ignore the user vector in prediction
                X_curr.append(u.reshape(1, -1))
                clicked = um.serve_advt_to_current_user(actions_curr[impr_idx])
                rew_curr.append(1 if clicked else 0)  # this library only deals with 0-1 rewards
            X_all = np.vstack(X_curr)
            actions_all = np.array(actions_curr, dtype=int)
            rew_all = np.array(rew_curr, dtype=int)
        else:
            for impr_idx, _ in enumerate(batch):
                u = um.get_current_user()
                best_action = int(cmab.predict(u)[0])
                actions_curr.append(best_action)
                X_curr.append(u.reshape(1, -1))
                clicked = um.serve_advt_to_current_user(best_action)
                rew_curr.append(1 if clicked else 0)
            X_all = np.vstack((X_all, np.vstack(X_curr)))
            actions_all = np.append(actions_all, actions_curr)
            rew_all = np.append(rew_all, rew_curr)

        # train the model with all history here - this would be used for serving the next batch of users
        logging.info(f"Training model, shape of X_all: {np.shape(X_all)}, actions: {np.shape(actions_all)}, "
              f"rewards: {np.shape(rew_all)}")
        cmab.fit(X=X_all, a=actions_all, r=rew_all, warm_start=True)

    um.print_stats()
    if op_dir:
        if not os.path.exists(op_dir) or not os.path.isdir(op_dir):
            logging.info(f"Output directory {op_dir} doesn't exist, creating it.")
            os.makedirs(op_dir)

        event_records_file = f"{op_dir}/event_records.csv"
        um.event_record.to_csv(event_records_file)

        ctr_matrix_file =  f"{op_dir}/ctr_matrix.csv"
        np.savetxt(ctr_matrix_file, um.ctr_matrix)

        results_file = f"{op_dir}/results.json"
        results = dict()
        results['ctr'] = ctr_matrix_file
        results['event_record'] = event_records_file
        results['model_class'] = model_class.__name__
        results['num_impr'] = num_impr
        results['batch_size'] = batch_size
        results['measure_window_size'] = measure_window_size

        seg_acc_snapshots, ad_probs = summarize_trend(op_dir, um.event_record, um.ctr_matrix, window_size=measure_window_size,
                        title_info=model_class.__name__)
        # results['seg_acc_snapshots'] = seg_acc_snapshots

        with open(results_file, 'w') as f_res:
            f_res.write(json.dumps(results, indent=4))


def summarize_trend(op_dir, event_record=None, ctr_matrix=None, window_size=50, title_info=None):
    """
    Creates plots
    :param op_dir:
    :param event_record:
    :param window_size:
    :return:
    """
    logging.info("Summarizing trends, this might take a while.")
    if event_record is None or ctr_matrix is None:
        logging.info(f'Will use {op_dir} as the results dir.')
        results_file = f"{op_dir}/results.json"
        with open(results_file) as f:
            h = json.loads(f.read())
            ctr_matrix = np.loadtxt(h['ctr'])
            event_record = pd.read_csv(h['event_record'])
    elif not os.path.exists(op_dir) or not os.path.isdir(op_dir):
        logging.info(f"Output directory {op_dir} doesn't exist, creating it.")
        os.makedirs(op_dir)
    else:
        pass

    seg_acc_snapshots, ad_probs = calculate_snapshot_metrics(event_record=event_record, ctr_matrix=ctr_matrix,
                               window_size=window_size, show_every=100)
    unique_segs = sorted(set(event_record['segment_ID']))
    best_advt_ID = np.argmax(ctr_matrix, axis=1)
    # unique_advts = sorted(set(event_record['advt_ID']))

    if title_info:
        title_info = f"({title_info})"
    else:
        title_info = ""

    # create an additional column thats descriptive wrt advts, we will group based on this in seaborn
    labels = []
    for r_idx, r in ad_probs.iterrows():
        labels.append(f"{int(r['advt_ID'])} ({ctr_matrix[int(r['segment_ID']), int(r['advt_ID'])]:.2f})")
    ad_probs['advt (ctr)'] = labels

    for seg_ID in unique_segs:
        seg_ad_probs = ad_probs[ad_probs['segment_ID'] == seg_ID]
        seg_ad_probs = seg_ad_probs.sort_values(by=['impr_idx', 'advt_ID'])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.lineplot(x='impr_idx', y='prob_impr', hue='advt (ctr)', data=seg_ad_probs)
        ax.set_title(f"seg_ID={seg_ID}, best act. ad={best_advt_ID[seg_ID]}, smooth={window_size}, {title_info}")
                         #f"acc={seg_acc[-1]:.2f} {title_info}")
        plt.savefig(f"{op_dir}/seg_{seg_ID}_summary.png")

    logging.info(f"Snapshot summary created, size={len(seg_acc_snapshots)}.")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 1)
    sns.lineplot(x='impr_idx', y='seg_acc', data=seg_acc_snapshots)
    plt.savefig(f"{op_dir}/seg_acc_overall.png")
    plt.clf()
    return seg_acc_snapshots, ad_probs


def calculate_snapshot_metrics(event_record, ctr_matrix, window_size=None, show_every=100):
    """
    NOTE: deprecated in favor of a similar function in user_model
    :param df:
    :return:
    """
    unique_segs = sorted(set(event_record['segment_ID']))
    best_advt_ID = np.argmax(ctr_matrix, axis=1)
    best_ctr_per_seg = np.max(ctr_matrix, axis=1)
    min_ctr_per_seg = np.min(ctr_matrix, axis=1)
    error_lower_bound = np.zeros(len(unique_segs))
    error_upper_bound = best_ctr_per_seg - min_ctr_per_seg
    error_range_per_seg = error_upper_bound - error_lower_bound
    ad_probs = pd.DataFrame(columns=["impr_idx", "prob_impr", "advt_ID", "segment_ID"])
    ad_counts = defaultdict(lambda: deque(maxlen=window_size))
    seg_acc_snapshots = pd.DataFrame(columns=['impr_idx', 'seg_acc'])
    unique_advt_IDs = set(range(np.shape(ctr_matrix)[1]))

    for r_idx, r in event_record.iterrows():
        if r_idx % show_every == 0:
            logging.info(f"Processed {r_idx} event records.")
        ad_counts[r['segment_ID']].append(r['advt_ID'])
        # find per segment acc
        seg_acc_all = []
        for seg_ID, dq in ad_counts.items():
            num_impr_window = len(dq)
            c = Counter(dq)
            # pad with ads not shown - give them 0 counts
            for missing_ads in unique_advt_IDs - set(c.keys()):
                c[missing_ads] = 0
            # find the best ad empirical
            # temp_ad_probs = dict([(k, 1.0*v/num_impr_window) for k, v in c.items()])
            temp_ad_probs = [(r_idx, seg_ID, k, 1.0 * v / num_impr_window) for k, v in c.items()]
            ad_probs = ad_probs.append(
                pd.DataFrame(temp_ad_probs, columns=["impr_idx", "segment_ID", "advt_ID", "prob_impr"]))
            # for k, v in temp_ad_probs.items():
            #     ad_probs = ad_probs.append({"impr_idx": r_idx, "prob_impr": v, "advt_ID": k, "segment_ID": seg_ID},
            #                                ignore_index=True)
            best_ad_empr = max(c.items(), key=lambda t: t[1])[0]
            ctr_best_ad_empr = ctr_matrix[seg_ID][best_ad_empr]
            # seg_acc = 1-abs(ctr_best_ad_empr - max(ctr_matrix[seg_ID, :]))
            seg_acc = 1 - abs(ctr_best_ad_empr - best_ctr_per_seg[seg_ID])/ error_range_per_seg[seg_ID]
            seg_acc_all.append(seg_acc)

        seg_acc_snapshots = seg_acc_snapshots.append({'impr_idx': r_idx, 'seg_acc': np.nanmean(seg_acc_all)},
                                                     ignore_index=True)
    seg_acc_snapshots = seg_acc_snapshots.astype({"impr_idx": int})
    ad_probs = ad_probs.astype({"impr_idx": int, "advt_ID": int, "segment_ID": int})
    return seg_acc_snapshots, ad_probs


def collate_sim_data(mab_op_locs_dict, op_dir, max_events=None, window_size=None):
    indv_dfs = []
    for cmab, dirlist in mab_op_locs_dict.items():
        logging.info(f"Processing outputs for cmab={cmab}.")
        for dir_idx, op_sim_dir in enumerate(dirlist):
            logging.info(f"Processing sim idx={dir_idx}, dir={op_sim_dir}.")
            results_file = f"{op_sim_dir}/results.json"
            with open(results_file) as f:
                h = json.loads(f.read())
                ctr_matrix = np.loadtxt(h['ctr'])
                event_record = pd.read_csv(h['event_record'])
                event_record = event_record.astype({'segment_ID': int, 'advt_ID': int})
                if max_events:
                    event_record = event_record.iloc[:max_events]
                logging.info(f"{len(event_record)} events in this sim.")
            # get the metrics for this file
            seg_acc_snapshots, ad_probs = user_model.calculate_snapshot_metrics(event_record=event_record, ctr_matrix=ctr_matrix,
                                                                     window_size=window_size, show_every=100)
            sns.lineplot(x='impr_idx', y='prob', hue='advt_ID', data=ad_probs[ad_probs['seg_ID']==0])
            plt.show()

            seg_acc_snapshots['cmab'] = [cmab] * len(seg_acc_snapshots)
            seg_acc_snapshots['sim_idx'] = [dir_idx] * len(seg_acc_snapshots)
            indv_dfs.append(seg_acc_snapshots)
    logging.info(f"Created {len(indv_dfs)} dataframes.")
    concat_seg_acc = pd.concat(indv_dfs)

    if not os.path.exists(op_dir) or not os.path.isdir(op_dir):
        logging.info(f"Output directory {op_dir} doesn't exist, creating it.")
        os.makedirs(op_dir)
    concat_seg_acc.to_csv(f"{op_dir}/concat_sim_seg_scores_smooth_{window_size}.csv")
    sns.lineplot(x="impr_idx", y="seg_acc", hue="cmab", data=concat_seg_acc)
    plt.title(f"smooth={window_size}")
    plt.savefig(f"{op_dir}/concat_seg_smooth_{window_size}.png", bbox_inches='tight')

    return concat_seg_acc

if __name__ == "__main__":
    # model_class = BootstrappedUCB
    model_class = AdaptiveGreedy
    num_sims = 2
    op_dir = f'cmab_generated/{model_class.__name__}_small'
    # demo_batch(model_class=model_class, num_advts=4, num_impr=100, num_segments=4, num_features=10, batch_size=50,
    #            op_dir=op_dir, measure_window_size=50, n_jobs=-1)
    # model_class = AdaptiveGreedy
    # for i in range(num_sims):
    #     op_dir = f'cmab_generated/{model_class.__name__}_sims_small/sim_{i+1}'
    #     demo_batch(model_class=model_class, num_advts=4, num_impr=100, num_segments=4, num_features=10, batch_size=50,
    #            op_dir=op_dir, measure_window_size=100, n_jobs=-1)
    # summarize_trend(op_dir=op_dir, window_size=100, title_info=f"{model_class.__name__}")
    # h = {'BootstrappedUCB': ['cmab_generated/BootstrappedUCB_sims_small/sim_2',
    #                          'cmab_generated/BootstrappedUCB_sims_small/sim_1'],
    #      'AdaptiveGreedy': ['cmab_generated/AdaptiveGreedy_sims_small/sim_2',
    #                          'cmab_generated/AdaptiveGreedy_sims_small/sim_1']
    #      }
    # collate_sim_data(h, r'cmab_generated/collated_sims_small', 100)

    all_sim_ops = defaultdict(list)
    for cmab_name, sim_idx in itertools.product(['BootstrappedUCB', 'AdaptiveGreedy'], range(num_sims)):
        op_dir = f'cmab_generated/{cmab_name}_sims/sim_{sim_idx+1}'
        all_sim_ops[cmab_name].append(op_dir)
    # del all_sim_ops['BootstrappedUCB']
    print(json.dumps(all_sim_ops, indent=4))

    collate_sim_data(all_sim_ops, r'cmab_generated/collated_sims', max_events=None, window_size=500)