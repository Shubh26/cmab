import numpy as np, re, sys, os, pandas as pd
#np.random.seed(23)
import json, itertools

import user_model
from collections import Counter, defaultdict
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
from vowpalwabbit import pyvw
import seaborn as sns; sns.set()
import math
import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )


def vw_cmab_predict(vw, context, adf=False, num_actions=None): #, actions): #TODO
    '''
    Pass context to vw to get an action.
    :param vw:
    :param context:
    :return: index of chosen action (in range [0, num_actions-1]) and prob of choosing that action
    '''
    # Convert context vector to vw format
    if adf:
        vw_example = convert_to_vw_adf_format(example=context, is_vw_format_nolabel=False,
                                              num_actions=num_actions, cb_label=None)
    else:
        vw_example = convert_to_vw_format(example=context, is_vw_format_nolabel=False, cb_label=None)
    # print(vw_example)
    pmf = np.array(vw.predict(vw_example))
    # print(sum(pmf), pmf)
    vw.finish_example(vw_example)
    chosen_action_index, prob = sample_action_from_pmf(pmf)

    # chosen_action_index = pmf.argmax()
    # prob = pmf[chosen_action_index]
    return chosen_action_index, prob, vw, vw_example


def vw_cmab_learn(vw, context, clicked, chosen_action_id, chosen_action_prob, is_vw_format_nolabel,
                  adf=False, num_actions=None): #TODO
    '''
    Update vw model with new example
    :param vw:
    :param context:
    :return: updated vw cmab model
    '''

    # # Get cost of the action we chose
    # # VW tries to minimize loss/cost, therefore we will pass cost as -reward
    cost = - int(clicked == True)

    # Convert context vector to vw format
    #  Values for action ids in VW are in range [1, num_actions], so need to +1 below
    if adf: #TODO
        vw_format = convert_to_vw_adf_format(example=context, is_vw_format_nolabel=is_vw_format_nolabel,
                                             num_actions=num_actions,
                                             cb_label=(chosen_action_id, cost, chosen_action_prob))
    else:
        vw_format = convert_to_vw_format(example=context, is_vw_format_nolabel=is_vw_format_nolabel,
                                         cb_label=(chosen_action_id, cost, chosen_action_prob))

    # Learn
    # print(vw_format)
    # vw_format = vw.parse(vw_format, pyvw.vw.lContextualBandit)
    vw.learn(vw_format)
    vw.finish_example(vw_format)
    return vw


def sample_action_from_pmf(pmf):
    total = sum(pmf)
    scale = 1 / total
    pmf = np.array([x * scale for x in pmf])
    chosen_action_index = np.random.choice(np.arange(len(pmf)), p=pmf)
    prob = pmf[chosen_action_index]
    return chosen_action_index, prob


def convert_to_vw_format(example, is_vw_format_nolabel, cb_label=None):
    '''
    This function to convert examples to VW required format:
    action:cost:probability | features
    Note: action is a positive integer in [1, num_actions]
    Ref: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Logged-Contextual-Bandit-Example
         https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format#contextual-Bandit

    :param example:
    :param is_vw_format_nolabel:
    :param cb_label: tuple of form (chosen_action_id, cost, prob).
                    Note chosen_action_id here is in range [0, num_actions-1]
                    while *action* in the vw format above is a positive integer in range [1, num_actions]
    :return:
    '''

    if is_vw_format_nolabel:
        vw_example = example
        if cb_label is not None:
            chosen_action_id, cost, prob = cb_label
            vw_example = f"{str(chosen_action_id+1)}:{str(cost)}:{str(prob)} | " + vw_example
    else:
        vw_example = ''
        if cb_label is not None:
            chosen_action_id, cost, prob = cb_label
            vw_example += f"{str(chosen_action_id+1)}:{str(cost)}:{str(prob)} "
        vw_example += "| "

        for feature_id in range(len(example)):
            vw_example += f"uf{feature_id}:{example[feature_id]} "
    return vw_example


def convert_to_vw_adf_format(example, is_vw_format_nolabel, actions, cb_label=None): #TODO
    '''

    :param example:
    :param is_vw_format_nolabel:
    :param actions:
    :param cb_label:
    :return:
    '''
    if cb_label is not None:
        chosen_action_id, cost, prob = cb_label

    if is_vw_format_nolabel:
        if cb_label is None:
            return example
        else:
            # chosen_action, cost, prob = cb_label
            splitted_example = example.split('\n')
            chosen_action_with_label = "0:{}:{} ".format(cost, prob) + splitted_example[chosen_action_id + 1]
            splitted_example[chosen_action_id + 1] = chosen_action_with_label
            return '\n'.join(splitted_example)
    else:
        vw_example = ''
        vw_example += 'shared |Uf '
        for feature_id in range(len(example)):
            vw_example += f"Uf{feature_id}:{example[feature_id]} "
        vw_example += '\n'

        for action_id in range(len(actions)):
            if cb_label is not None and action_id == chosen_action_id:
                vw_example += "0:{}:{} ".format(cost, prob)
            vw_example += "|A action={} \n".format(str(actions[action_id]))

        # Strip the last newline
        # print('====\n', vw_example[:-1])
        return vw_example[:-1]


def calculate_ctr(click_sequence):
    num_clicks = 0
    ctr_df = pd.DataFrame(columns=['# impression', 'ctr'])
    for t, click in enumerate(click_sequence):
        num_clicks += int(click == True)
        ctr_df = ctr_df.append({'# impression': t+1, 'ctr': num_clicks/(t+1)}, ignore_index=True)
    return ctr_df


def run_simulation(vw_algo, num_impressions, num_advts, num_features, num_segments,
                   randomized_size, randomized_periodicity,
                   measure_window_size=None, op_dir=None, is_summarized=False,
                   adf=True, do_learn=True):
    '''
    One simulation with a given cmab algo and num_impressions.
    :param vw_algo: str presenting all flags needed to run an VW CMAB algo. Examples are:
            f"--cb_explore {num_advts} --epsilon 0.2" for epsilon-greedy algo with epsilon=0.2
    :param num_impressions: integer nb of impressions/iterations to run.
    :param num_advts: integer nb of advts
    :param num_features
    :param num_segments
    :param randomized_size
    :param randomized_periodicity
    :param measure_window_size
    :param op_dir
    :param is_summarized
    :param adf: bool, whether to use Action Dependent Features which allows the learner to use features from actions/advts.
    :param do_learn: bool, whether to update the model with newly available examples.
    :return:
    '''
    um = user_model.UserModel(num_features=num_features, num_segments=num_segments, num_advts=num_advts)
    actual_best_ads = {seg_id: np.argmax(ctrs) for seg_id, ctrs in zip(range(num_segments), um.ctr_matrix)}

    # Initialize vw model
    vw = pyvw.vw(vw_algo)

    # Calculate number of times performing randomized experiments, one every randomized_periodicity users,
    # for a total number of randomized_size*num_impressions
    num_rand_experiments = int(num_impressions/randomized_periodicity)
    num_impr_per_experiment = int(randomized_size*num_impressions/num_rand_experiments)

    for i in range(num_impressions):
        # get the current user's feature vec
        user_vec = um.get_current_user()[0]

        # the cmab suggests which advt to serve
        if i % randomized_periodicity < num_impr_per_experiment and i/randomized_periodicity < num_rand_experiments:
            advt_id_to_serve = np.random.choice(range(num_advts))
            prob = 1. / num_advts
            vw_example = user_vec
            is_vw_format_nolabel = False
        else:
            advt_id_to_serve, prob, vw, vw_example = vw_cmab_predict(vw=vw, context=user_vec, adf=adf, num_actions=num_advts)
            is_vw_format_nolabel = True

        # serve this advt
        clicked = um.serve_advt_to_current_user(advt_id_to_serve)

        # learn from what happened
        if do_learn:
            vw = vw_cmab_learn(vw, context=vw_example, clicked=clicked,
                               chosen_action_id=advt_id_to_serve, chosen_action_prob=prob,
                               is_vw_format_nolabel=is_vw_format_nolabel,
                               adf=adf, num_actions=num_advts)

    # del vw model when finished
    del vw

    um.print_stats()
    if op_dir:
        if not os.path.exists(op_dir) or not os.path.isdir(op_dir):
            logging.info(f"Output directory {op_dir} doesn't exist, creating it.")
            os.makedirs(op_dir)

        event_records_file = f"{op_dir}/event_records.csv"
        um.event_record.head()
        um.event_record.to_csv(event_records_file)

        ctr_matrix_file =  f"{op_dir}/ctr_matrix.csv"
        np.savetxt(ctr_matrix_file, um.ctr_matrix)

        results_file = f"{op_dir}/results.json"
        results = dict()
        results['ctr'] = ctr_matrix_file
        results['event_record'] = event_records_file
        results['model_class'] = vw_algo
        results['num_impr'] = num_impressions
        # results['batch_size'] = batch_size
        results['measure_window_size'] = measure_window_size

        if is_summarized:
            seg_acc_snapshots, ad_probs = summarize_trend(op_dir, um.event_record, um.ctr_matrix, window_size=measure_window_size,
                            title_info=vw_algo + f', rand. size={randomized_size}')
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
    seg_acc_snapshots, ad_probs = user_model.calculate_snapshot_metrics(event_record=event_record, ctr_matrix=ctr_matrix,
                               window_size=window_size, show_every=5000)
    unique_segs = sorted(set(event_record['segment_ID']))
    best_advt_ID = np.argmax(ctr_matrix, axis=1)
    # unique_advts = sorted(set(event_record['advt_ID']))

    if title_info:
        title_info = f"\n({title_info})"
    else:
        title_info = ""

    # create an additional column thats descriptive wrt advts, we will group based on this in seaborn
    labels = []

    for r_idx, r in ad_probs.iterrows():
        labels.append(f"{int(r['advt_ID'])} ({ctr_matrix[int(r['seg_ID']), int(r['advt_ID'])]:.2f})")
    ad_probs['advt (ctr)'] = labels

    for seg_ID in unique_segs:
        seg_ad_probs = ad_probs[ad_probs['seg_ID'] == seg_ID]
        seg_ad_probs = seg_ad_probs.sort_values(by=['impr_idx', 'advt_ID'])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.lineplot(x='impr_idx', y='prob', hue='advt (ctr)', data=seg_ad_probs)
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
                                                                     window_size=window_size, show_every=5000)
            # sns.lineplot(x='impr_idx', y='prob', hue='advt_ID', data=ad_probs[ad_probs['seg_ID'] == 0])
            # plt.show()

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
    num_impressions = 10000
    randomized_size = 0.1  # fraction of impressions to be used for randomized experiment
    randomized_periodicity = 100  # number of impressions between the start of 2 consecutive randomized experiments

    num_advts = 4
    num_features = 10
    num_segments = 4
    num_simulations = 5
    window_size = 500

    algo_params = [('cover', 4), ('bag', 4), ('epsilon', 0.1)]
    # algo_params = [('bag', 4)]
    # run without Action Dependent Features
    adf = False
    # vw_algo = f"--cb_explore {num_advts} --quiet  --{algo} {algo_arg}" #--bag 10"#--epsilon 0.2"  # --cover 1"

    # # run with Action Dependent Features -- CURRENTLY does not work properly
    # adf = True
    # vw_algo = f"--cb_explore_adf --softmax --lambda 10"  #  --softmax --lambda 10"
    # # vw_algo = f"--cb_explore_adf --cb_type mtr --interactions UA --squarecb --gamma_scale 1000" #--epsilon 0.1"  # --softmax --lambda 10"

    # setup randomization
    rand_seed = 22

    op_dir_base = f"vw_cmab_generated"
    for (algo, algo_arg) in algo_params:
        vw_algo = f"--cb_explore {num_advts} --quiet  --{algo} {algo_arg}"
        np.random.seed(rand_seed)
        print(vw_algo)
        logging.info(f"CMAB algo: {vw_algo}, randomized group size: {randomized_size}, with random seed: {rand_seed}")
        for i in range(num_simulations):
            op_dir = os.path.join(op_dir_base, f'{algo}{algo_arg}_sims/sim_{i+1}')
            run_simulation(vw_algo=vw_algo, num_impressions=num_impressions, num_advts=num_advts,
                           num_features=num_features, num_segments=num_segments,
                           randomized_size=randomized_size, randomized_periodicity=randomized_periodicity,
                           measure_window_size=window_size, op_dir=op_dir, is_summarized=True,
                           adf=adf, do_learn=True)

    all_sim_ops = defaultdict(list)
    for (algo, algo_arg), sim_idx in itertools.product(algo_params, range(num_simulations)):
        op_dir = os.path.join(op_dir_base, f'{algo}{algo_arg}_sims/sim_{sim_idx+1}')
        cmab_name = f"{algo}{algo_arg}"
        all_sim_ops[cmab_name].append(op_dir)

    collate_sim_data(all_sim_ops, os.path.join(op_dir_base,'collated_sims'), max_events=None, window_size=window_size)
