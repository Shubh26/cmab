import numpy as np, re, sys, os, pandas as pd
from collections import Counter
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import logging
from datetime import datetime
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(funcName)s() — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )


class UserModel:
    def __init__(self, num_features=2, num_segments=3, num_advts=3):
        """
        The segments are essentially a Voronoi tesellation in the space. The segmentation is implicit: in the sense that
        it is determined on the fly what segment a user belongs too. To get an idea of what this looks like, see the
        method visualize()
        :param num_features: features that define a user
        :param num_segments: number of distinct user segments we want; each segment has it own CTR per sdvt
        :param num_advts: number of advertisements we want to serve
        """
        assert num_features >= 2, f"Need at least 2 features."
        self.num_features = num_features
        self.num_segments = num_segments
        self.num_advts = num_advts
        self.segments_served_to, self.clicks_recorded = [], []
        self.event_record = pd.DataFrame(columns=['segment_ID', 'advt_ID', 'click'])
        self.current_user = None
        temp = []
        for _ in range(num_features):
            temp.append(np.random.sample(num_segments).reshape(-1, 1))
        self.centroid_matrix = np.hstack(temp)
        # get ctrs per segment, note CTRs only need to be [0, 1], they don't need to add up to 1
        self.ctr_matrix = np.random.sample(size=(self.num_segments, self.num_advts))
        logging.info(f"Created centroid matrix with shape {np.shape(self.centroid_matrix)}")
        logging.info(f"Created CTR matrix with shape {np.shape(self.ctr_matrix)}, for {self.num_segments} segments "
                     f"and {self.num_advts} advts.")

    def generate_users(self, num_users=10):
        """
        Generates user feature vectors.
        :param num_users:
        :return:
        """
        temp = []
        for _ in range(self.num_features):
            temp.append(np.random.sample(num_users).reshape(-1, 1))
        user_vecs = np.hstack(temp)
        # logging.info(f"Generated {len(user_vecs)} users. Shape of user vec: {np.shape(user_vecs)}")
        return user_vecs

    def get_user_segment(self, user_vec):
        """
        Works for only one user_vec. Gets the segment ID of the user vec, which is the idx of the
        closest segment centroid.
        :param user_vec:
        :return:
        """
        d = pairwise_distances(self.centroid_matrix, np.array(user_vec).reshape(1, -1), metric='euclidean')
        segment = np.argmin(d)
        return segment

    def visualize(self, num_users=10):
        """
        This is a 2D plot; when num_features >2, only the first 2-dimensions are used.
        This is to visually explore how segment assignments happen. It is recommended to use this with num_features==2,
        with higher # of features, you'd see a projected plot which can be unintuitive.
        :param num_users:
        :return:
        """
        temp = []
        for _ in range(self.num_features):
            temp.append(np.random.sample(num_users).reshape(-1, 1))
        user_vecs = np.hstack(temp)
        logging.info(f"Generated {len(user_vecs)} user vectors.")

        allocated_segments = [self.get_user_segment(u) for u in user_vecs]
        projected_centroids = self.centroid_matrix[:, :2]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(user_vecs[:, 0], user_vecs[:, 1], c='r')
        for user_vec, seg in zip(user_vecs[:, :2], allocated_segments):
            ax.plot([user_vec[0], projected_centroids[seg][0]], [user_vec[1], projected_centroids[seg][1]], c='r',
                    linestyle="--", linewidth=0.7)
        ax.scatter(projected_centroids[:, 0], projected_centroids[:, 1], c='k', s=50)
        # plt.show()
        plt.savefig(r'um_viz.png')
    def get_current_user(self):
        self.current_user = self.generate_users(1)
        return self.current_user

    def serve_advt_to_current_user(self, advt_id):
        if self.current_user is None:
            logging.error(f"There is no current user. Did you call get_current_user() before this method?")
            return
        assert 0 <= advt_id < self.num_advts, f"advt_id can only be an integer in [0, {self.num_advts - 1}]"
        user_vec = self.current_user
        segment = self.get_user_segment(user_vec)
        actual_ctr = self.ctr_matrix[segment, advt_id]
        clicked = False
        if np.random.random() <= actual_ctr:
            clicked = True
        else:
            clicked = False

        self.event_record = self.event_record.append({'segment_ID': segment, 'advt_ID': advt_id, 'click': clicked},
                                                     ignore_index=True)
        self.current_user = None
        return clicked

    def print_stats(self):
        print(f"# segments={self.num_segments}, # advts={self.num_advts}, # features={self.num_features}.")
        df = pd.DataFrame(columns=['segment_ID', 'best_advt_ID'])
        for seg_ID, ctrs in zip(range(self.num_segments), self.ctr_matrix):
            df = df.append({'segment_ID': seg_ID, 'best_advt_ID': np.argmax(ctrs)}, ignore_index=True)

        # if we find recorded events lets add that data too
        if len(self.event_record) > 0:
            highest_impressions = []
            for seg_ID in range(self.num_segments):
                impr = self.event_record[self.event_record['segment_ID'] == seg_ID]
                if len(impr) > 0:
                    empirically_best = max(Counter(impr['advt_ID']).items(), key=lambda t: t[1])[0]
                    highest_impressions.append(empirically_best)
                else:
                    highest_impressions.append(np.NAN)

            df['highest_impressions'] = highest_impressions
            segments_optimized = df.dropna().query('highest_impressions == best_advt_ID')
            seg_acc_offer = 100.0 * len(segments_optimized) / len(df.dropna())
            print(f"Segment Accuracy across segments offered: {seg_acc_offer:.2f}%")
            seg_acc_all = 100.0 * len(segments_optimized) / self.num_segments
            print(f"Segment Accuracy across all segments: {seg_acc_all:.2f}%")
        print(df)


def random_cmab_algo(user_vec, num_advts):
    """
    This is a purely random algo that picks an advt id given num_advts
    :param user_vec: some user vec, but this is ignored. This is here to indicate this is a required input for an
        actual cmab.
    :param num_advts:
    :return:
    """
    return np.random.choice(range(num_advts))


def calculate_snapshot_metrics(event_record, ctr_matrix, window_size=None, show_every=100):
    """
    A reasonably fast function to compute windowed metrics like overall segment accuracy and per segment accuracy over
    a window of impressions.
    :param event_record:
    :param ctr_matrix:
    :param window_size: an integer window size of past impressions. None implies all history would be used.
    :param show_every: an integer that decides a status log should be printed after a specified number of records
        are processed
    :return:
        seg_acc_snapshots_df: dataframe with columns for impression index and aggregate segment accuracy for that
            impression index. Colnames are 'impr_idx', 'seg_acc' respectively.
        per_seg_ad_acc_df: dataframe with columns for impression idx, segment ID, advt ID and probability of offering
            the advt within that segment at that impr. idx.

            **NOTE**: although at a particular impression, not all segments are served, this contains entries for all
            segments; for such 'unserved' segments the probabilities are repeated from the last time
            the segment was served. If you want to retain only the segment IDs served at an impression idx, consider
            performing a join with the event record.
    """
    logging.info("experimental code")
    start_time = datetime.now()
    logging.info(f"Will process {event_record.shape[0]} event records.")
    seg_dist = Counter(event_record['segment_ID'])
    unique_segs = sorted(seg_dist.keys())
    unique_advt_IDs = set(range(np.shape(ctr_matrix)[1]))

    best_ctr_per_seg = np.max(ctr_matrix, axis=1)
    min_ctr_per_seg = np.min(ctr_matrix, axis=1)
    # the lowest error one can have is zero - the correct best ad is found
    error_lower_bound = np.zeros(len(unique_segs))
    # the greatest error is the difference of the ctr of the best and worst advts,
    # occurs when the cmab empirically selects the worst advt as the top advt
    error_upper_bound = best_ctr_per_seg - min_ctr_per_seg
    error_range_per_seg = error_upper_bound - error_lower_bound

    # these store the statistics within the active window
    ad_counts_arr = np.zeros((len(unique_segs), len(unique_advt_IDs)))

    # seg ad snapshots - this structure can get quite big, since this has
    # num_impressions x num_segments x num_advts entries
    per_seg_ad_acc = np.zeros((event_record.shape[0] * len(unique_segs), len(unique_advt_IDs)))

    # this is the running seg acc for each segment - we should have a value for each impression per segment
    running_seg_acc_arr = np.zeros(len(unique_segs))
    seg_acc_snapshots_arr = np.zeros((event_record.shape[0], 2))
    seg_acc_snapshots_arr[:, 0] = np.arange(event_record.shape[0])
    # get the segment and advts per impression from the dataframe, we just need those for these metrics
    list_records = list(zip(event_record['segment_ID'], event_record['advt_ID']))

    win_start = 0  # this defines the starting index of the active window for metric calculation
    for r_idx, (curr_seg_ID, curr_advt_ID) in enumerate(list_records):

        # Stores IDs of segs where info needs to change either because the current impression is relevant,
        # or the window start index needs to be incremented. Needs to be a set since the seg ID may be affected twice.
        segs_affected = set()

        if r_idx % show_every == 0:
            logging.info(f"Processed {r_idx} event records.")

        # add the advt to the right seg info
        # seg_ad_info[curr_seg_ID]['ad_counts'][curr_advt_ID] += 1
        ad_counts_arr[curr_seg_ID, curr_advt_ID] += 1
        segs_affected.add(curr_seg_ID)
        # seg_ad_info[curr_seg_ID]['last_filled_idx'] += 1
        # seg_ad_info[curr_seg_ID]['ads']['last_filled_idx'] = curr_advt_ID

        # does something need to be deleted because we are at the window boundary?
        # check (1) if window size is provided (2) if win_start is beyond window size
        if window_size and window_size <= (r_idx - win_start):
            # note which ad for what segment needs to be ignored to honor the window
            to_discard = list_records[win_start]
            win_start += 1
            # seg_ad_info[to_discard[0]]['ad_counts'][to_discard[1]] -= 1
            ad_counts_arr[to_discard[0], to_discard[1]] -= 1
            segs_affected.add(to_discard[0])

        # update stats for only segments where counts changed
        segs_affected = list(segs_affected)
        seg_acc_all = []
        tot_imprs_in_affected = np.sum(ad_counts_arr[segs_affected, :], axis=1)

        arr_row_start_idx = r_idx * len(unique_segs)
        arr_row_end_idx = arr_row_start_idx + len(unique_segs) - 1
        # first create a copy of the stats in the last impression on the assumption that not much might've changed
        if r_idx > 0:  # nothing to copy at first iteration
            prev_row_start_idx = (r_idx-1) * len(unique_segs)
            prev_row_end_idx = prev_row_start_idx + len(unique_segs) - 1
            copy_last_impr = per_seg_ad_acc[prev_row_start_idx: prev_row_end_idx + 1, :]
        else:
            copy_last_impr = np.zeros((len(unique_segs), len(unique_advt_IDs)))
        # now, only change what is required
        copy_last_impr[segs_affected, :] = ad_counts_arr[segs_affected, :] / tot_imprs_in_affected.reshape(-1, 1)
        # attach this to the original matrix
        per_seg_ad_acc[arr_row_start_idx: arr_row_end_idx + 1, :] = copy_last_impr

        # "empr" is for empirical
        best_ad_empr_in_affected = np.argmax(ad_counts_arr[segs_affected, :], axis=1)
        best_ad_empr_ctr_in_affected = ctr_matrix[segs_affected, best_ad_empr_in_affected]

        # get the current segment errors, scale them by the error range, and subtract from 1 to find accuracy
        # again: just modify these numbers for affected segments
        running_seg_acc_arr[segs_affected] = 1 - abs(best_ad_empr_ctr_in_affected - best_ctr_per_seg[segs_affected]) / \
                                             error_range_per_seg[segs_affected]
        seg_acc_snapshots_arr[r_idx, 1] = np.nanmean(running_seg_acc_arr)

    # dump data into dataframes for convenience, and return
    seg_acc_snapshots_df = pd.DataFrame(seg_acc_snapshots_arr, columns=['impr_idx', 'seg_acc'])
    seg_acc_snapshots_df = seg_acc_snapshots_df.astype({"impr_idx": int})
    logging.info(f"Shape of overall seg acc dataframe: {seg_acc_snapshots_df.shape}.")

    # create the per_seg_ad_acc_df step-by-step ... create the impr_idx, seg_ID cols first, and then the actual data
    indexing_cols = {'impr_idx': np.array([[i]*len(unique_segs) for i in range(event_record.shape[0])]).flatten(),
                     'seg_ID': list(unique_segs)*event_record.shape[0]}
    advt_prob_cols = dict([(i, per_seg_ad_acc[:, i])for i in range(per_seg_ad_acc.shape[1])])
    indexing_cols.update(advt_prob_cols)
    per_seg_ad_acc_df = pd.DataFrame(indexing_cols)
    logging.info(f"Shape of per seg acc dataframe (pre-melt): {per_seg_ad_acc_df.shape}.")
    # we probably want a long form for ease of plotting etc
    per_seg_ad_acc_df = pd.melt(per_seg_ad_acc_df, id_vars=['impr_idx', 'seg_ID'], value_vars=unique_advt_IDs)
    per_seg_ad_acc_df = per_seg_ad_acc_df.rename({'value': 'prob', 'variable': 'advt_ID'}, axis='columns')
    per_seg_ad_acc_df.sort_values(by=['impr_idx', 'seg_ID', 'advt_ID'], ignore_index=True, inplace=True)
    logging.info(f"Shape of per seg acc dataframe (post-melt): {per_seg_ad_acc_df.shape}.")

    end_time = datetime.now()
    logging.info(f"Finished computing snapshot metrics. Took {(end_time-start_time).total_seconds()}s "
                 f"for {event_record.shape[0]} events.")

    return seg_acc_snapshots_df, per_seg_ad_acc_df



def demo_user_model():
    num_advts = 5
    um = UserModel(num_features=2, num_segments=10, num_advts=num_advts)
    um.visualize(100)
    num_impressions = 20
    for i in range(num_impressions):
        # get the current user's feature vec
        user_vec = um.get_current_user()

        # the cmab suggests what advt to serve
        advt_id_to_serve = random_cmab_algo(user_vec, num_advts)

        # serve this advt
        um.serve_advt_to_current_user(advt_id_to_serve)

    print(um.event_record)
    um.print_stats()

if __name__ == "__main__":
    demo_user_model()