import json
import os
import pdb
from tqdm import tqdm
import torch
import pickle
import pandas as pd
from libcity.data.dataset.bertlm_dataset import BERTSubDataset
from libcity.data.dataset.contrastive_split_dataset import collate_unsuperv_contrastive_split
from libcity.data.dataset.bertlm_contrastive_dataset import collate_unsuperv_mask, ContrastiveLMDataset
from libcity.data.dataset.bertlm_dataset import noise_mask
import numpy as np
from collections import defaultdict


def jaccard_similarity(seq1, seq2):
    """Calculate Jaccard similarity between two sequences."""
    set1 = set(seq1)
    set2 = set(seq2)
    if set1.isdisjoint(set2) or (not set1 and not set2):  # Both are empty sets
        return 0.0
    intersection = len(set1 & set2)
    if intersection == 0:
        return 0.0
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


def time_similarity(query_time, traj_time):
    """Calculate similarity based on week_id and minute id."""
    dist = np.linalg.norm(query_time - traj_time)
    sim = 1 / (1 + dist)  # Convert distance to similarity
    return sim


def calculate_similarities(args):
    i, query_grid_seq, query_path_seq, query_time, traj = args
    grid_sim = jaccard_similarity(query_grid_seq, traj[:, 1].astype(int))
    path_sim = jaccard_similarity(query_path_seq, traj[:, 0].astype(int))
    time_sim = time_similarity(query_time, traj[0, [4, 3]].astype(float))
    total_sim = ((5 * grid_sim) + (5 * path_sim) + (1 * time_sim)) / (5 + 5 + 1)
    return i, total_sim

class ContrastiveSplitLMDataset(ContrastiveLMDataset):
    def __init__(self, config):
        super().__init__(config)
        self.argu1 = config.get('out_data_argument1', 'trim')
        self.argu2 = config.get('out_data_argument2', 'time')
        self.data_argument1 = self.config.get("data_argument1", [])
        self.data_argument2 = self.config.get("data_argument2", [])
        self.collate_fn = collate_unsuperv_contrastive_split_lm

    def _gen_dataset(self):
        train_dataset = TrajectoryProcessingDatasetSplitLM(
            data_name=self.dataset, data_type='train', vocab=self.vocab,
            seq_len=self.seq_len, add_cls=self.add_cls, merge=self.merge, min_freq=self.min_freq,
            max_train_size=self.max_train_size,
            data_argument1=self.data_argument1,
            data_argument2=self.data_argument2,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode,
            distribution=self.distribution, avg_mask_len=self.avg_mask_len,
            argu1=self.argu1, argu2=self.argu2)
        eval_dataset = TrajectoryProcessingDatasetSplitLM(
            data_name=self.dataset, data_type='eval', vocab=self.vocab,
            seq_len=self.seq_len, add_cls=self.add_cls, merge=self.merge, min_freq=self.min_freq,
            max_train_size=None,
            data_argument1=self.data_argument1,
            data_argument2=self.data_argument2,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode,
            distribution=self.distribution, avg_mask_len=self.avg_mask_len,
            argu1=self.argu1, argu2=self.argu2)
        test_dataset = TrajectoryProcessingDatasetSplitLM(
            data_name=self.dataset, data_type='test', vocab=self.vocab,
            seq_len=self.seq_len, add_cls=self.add_cls, merge=self.merge, min_freq=self.min_freq,
            max_train_size=None,
            data_argument1=self.data_argument1,
            data_argument2=self.data_argument2,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode,
            distribution=self.distribution, avg_mask_len=self.avg_mask_len,
            argu1=self.argu1, argu2=self.argu2)
        return train_dataset, eval_dataset, test_dataset


class TrajectoryProcessingDatasetSplitLM(BERTSubDataset):

    def __init__(self, data_name, data_type, vocab, seq_len=512, add_cls=True,
                 merge=True, min_freq=1, max_train_size=None,
                 data_argument1=None, data_argument2=None,
                 masking_ratio=0.2, masking_mode='together',
                 distribution='random', avg_mask_len=3, argu1=None, argu2=None):
        self.data_type = data_type
        if argu1 is not None:
            self.data_path1 = 'raw_data/{}/{}_{}_enhancedby{}.csv'.format(
                data_name, data_name, data_type, argu1)
            self.cache_path1 = 'raw_data/{}/cache_{}_{}_{}_{}_{}_enhancedby{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq, argu1)
        else:
            self.data_path1 = 'raw_data/{}/{}_{}.csv'.format(
                data_name, data_name, data_type)
            self.cache_path1 = 'raw_data/{}/cache_{}_{}_{}_{}_{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq)
        if argu2 is not None:
            self.data_path2 = 'raw_data/{}/{}_{}_enhancedby{}.csv'.format(
                data_name, data_name, data_type, argu2)
            self.cache_path2 = 'raw_data/{}/cache_{}_{}_{}_{}_{}_enhancedby{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq, argu2)
        else:
            self.data_path2 = 'raw_data/{}/{}_{}.csv'.format(
                data_name, data_name, data_type)
            self.cache_path2 = 'raw_data/{}/cache_{}_{}_{}_{}_{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq)

        self.temporal_mat_path1 = self.cache_path1[:-4] + '_temporal_mat.pkl'
        self.temporal_mat_path2 = self.cache_path2[:-4] + '_temporal_mat.pkl'

        super().__init__(data_name, data_type, vocab, seq_len, add_cls, merge,
                         min_freq, max_train_size, masking_ratio, masking_mode, distribution, avg_mask_len)
        self._logger.info('Init TrajectoryProcessingDatasetSplitLM!')
        self.data_argument1 = data_argument1
        self.data_argument2 = data_argument2
        self._load_data_split()
        # if os.path.exists('raw_data/porto/similarity')
        # if self.data_type != 'train':
        #     self.pre_calculate_similarity_traj()

    def _load_data_split(self):
        if os.path.exists(self.cache_path1) and os.path.exists(self.temporal_mat_path1) \
                and os.path.exists(self.cache_path2) and os.path.exists(self.temporal_mat_path2):
            self.traj_list1 = pickle.load(open(self.cache_path1, 'rb'))
            self.temporal_mat_list1 = pickle.load(open(self.temporal_mat_path1, 'rb'))
            self.traj_list2 = pickle.load(open(self.cache_path2, 'rb'))
            self.temporal_mat_list2 = pickle.load(open(self.temporal_mat_path2, 'rb'))
            self._logger.info('Load dataset from {}, {}'.format(self.cache_path1, self.cache_path2))
        else:
            origin_data_df1 = pd.read_csv(self.data_path1, sep=';')
            origin_data_df2 = pd.read_csv(self.data_path2, sep=';')
            assert origin_data_df1.shape == origin_data_df2.shape
            self.traj_list1, self.temporal_mat_list1 = self.data_processing(
                origin_data_df1, self.data_path1, cache_path=self.cache_path1, tmat_path=self.temporal_mat_path1)
            self.traj_list2, self.temporal_mat_list2 = self.data_processing(
                origin_data_df2, self.data_path2, cache_path=self.cache_path2, tmat_path=self.temporal_mat_path2)
        self.simi_indexes_1, self.simi_indexes_2 = [], []
        for ind in range(len(self.traj_list1)):
            simi_score = json.load(open(f'raw_data/porto/{self.data_type}_similarity/simi_list_1_{ind}.json'))
            simi_index = [i[0] for i in simi_score[1: ]]
            self.simi_indexes_1.append(simi_index)
        for ind in range(len(self.traj_list2)):
            simi_score = json.load(open(f'raw_data/porto/{self.data_type}_similarity/simi_list_2_{ind}.json'))
            simi_index = [i[0] for i in simi_score[1: ]]
            self.simi_indexes_2.append(simi_index)

    def __len__(self):
        assert len(self.traj_list1) == len(self.traj_list2) == len(self.traj_list)
        return len(self.traj_list)

    def __getitem__(self, ind):
        traj_ind, mask, temporal_mat = super().__getitem__(ind)
        traj_ind1 = self.traj_list1[ind]  # (seq_length, feat_dim)
        traj_ind2 = self.traj_list2[ind]  # (seq_length, feat_dim)
        simi_ind1 = self.simi_indexes_1[ind]
        simi_ind2 = self.simi_indexes_2[ind]
        simi_trajs1 = [self.traj_list1[i] for i in simi_ind1]
        simi_trajs2 = [self.traj_list2[i] for i in simi_ind2]
        temporal_mat1 = self.temporal_mat_list1[ind]  # (seq_length, seq_length)
        temporal_mat2 = self.temporal_mat_list2[ind]  # (seq_length, seq_length)
        simi_temporal_mat1 = [self.temporal_mat_list1[i] for i in simi_ind1]
        simi_temporal_mat2 = [self.temporal_mat_list2[i] for i in simi_ind2]
        mask1 = None
        mask2 = None
        if 'mask' in self.data_argument1:
            mask1 = noise_mask(traj_ind1, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                               self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array
        if 'mask' in self.data_argument2:
            mask2 = noise_mask(traj_ind2, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                               self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array



        return traj_ind, mask, temporal_mat, \
               torch.LongTensor(traj_ind1), torch.LongTensor(traj_ind2), \
               torch.LongTensor(temporal_mat1), torch.LongTensor(temporal_mat2), \
               [torch.LongTensor(i) for i in simi_trajs1], [torch.LongTensor(i) for i in simi_trajs2], \
               [torch.LongTensor(i) for i in simi_temporal_mat1], [torch.LongTensor(i) for i in simi_temporal_mat2], \
               torch.LongTensor(mask1) if mask1 is not None else None, \
               torch.LongTensor(mask2) if mask2 is not None else None

    def pre_calculate_similarity_traj(self):
        traj_list1 = self.traj_list1
        traj_list2 = self.traj_list2
        simi_index_1, simi_index_2 = [], []

        grid_idx_1, path_idx_1 = {}, {}
        for idx, traj in enumerate(traj_list1):
            for grid_id in set(traj[1:, 0].tolist()):
                if grid_id not in grid_idx_1:
                    grid_idx_1[grid_id] = [idx]
                else:
                    grid_idx_1[grid_id].append(idx)
            for path_id in set(traj[1:, -1].tolist()):
                if path_id not in path_idx_1:
                    path_idx_1[path_id] = [idx]
                else:
                    path_idx_1[path_id].append(idx)
                    

        grid_idx_2, path_idx_2 = {}, {}
        for idx, traj in enumerate(traj_list2):
            for grid_id in set(traj[1:, 0].tolist()):
                if grid_id not in grid_idx_2:
                    grid_idx_2[grid_id] = [idx]
                else:
                    grid_idx_2[grid_id].append(idx)
            for path_id in set(traj[1:, -1].tolist()):
                if path_id not in path_idx_2:
                    path_idx_2[path_id] = [idx]
                else:
                    path_idx_2[path_id].append(idx)

        from multiprocessing import Pool, cpu_count

        args_1 = [(traj, traj_list1, grid_idx_1, path_idx_1, i, 1, self.data_type) for i, traj in enumerate(traj_list1)]
        args_2 = [(traj, traj_list2, grid_idx_2, path_idx_2, i, 2, self.data_type) for i, traj in enumerate(traj_list2)]
        pool = Pool(20)
        similarities_1 = pool.starmap(extract_similarity_traj, args_1)
        similarities_2 = pool.starmap(extract_similarity_traj, args_2)
        pool.close()
        if self.data_type == 'test':
            exit()

        # for i, traj1 in enumerate(tqdm(traj_list1)):
        #     simi_list = self.extract_similarity_traj(traj1, traj_list1, grid_idx_1, path_idx_1, i)
        #     simi_index_1.append([i[0] for i in simi_list])
        #
        # for i, traj2 in enumerate(tqdm(traj_list2)):
        #     simi_list = self.extract_similarity_traj(traj2, traj_list2, grid_idx_2, path_idx_2, i)
        #     simi_index_2.append([i[0] for i in simi_list])

        # pdb.set_trace()


def extract_similarity_traj(query_traj, traj_list, grid_idx, path_idx, ind, t, tp, w1=5, w2=5, w3=1, k=10):
    query_loc_list = query_traj[:, 0][0: ].tolist()
    query_path_list = query_traj[:, -1][0: ].tolist()
    query_week_id = query_traj[0][2]
    query_day_id = query_traj[0][3]
    simi_list = []
    k += 1
    potential_loc_index, potential_path_index = set(), set()
    for loc in query_loc_list:
        if loc in grid_idx:
            potential_loc_index.update(grid_idx[loc])
        else:
            continue

    for path in query_path_list:
        if path in path_idx:
            potential_path_index.update(path_idx[path])
        else:
            continue


    for i, traj in enumerate(traj_list):
        if i not in potential_loc_index or i not in potential_path_index:
            continue
        loc_list = traj[:, 0][0: ]
        path_list = traj[:, -1][0: ]
        week_id = traj[0][2]
        day_id = traj[0][3]
        jacaard_path = jaccard_similarity(query_path_list, path_list)
        if jacaard_path == 0:
            continue
        jacaard_loc = jaccard_similarity(query_loc_list, loc_list)
        time_similarity = 0.
        if week_id == query_week_id:
            time_similarity += 0.5
            # (1439 + 1440 - 1) % 1440
        day_distance = min(abs(day_id - query_day_id), 1440 - abs(day_id - query_day_id))
        day_similarity = 1 - day_distance / 1440
        time_similarity += day_similarity
        similarity = w1 * jacaard_loc + w2 * jacaard_path + w3 * time_similarity
        simi_list.append([i, similarity])
    simi_list.sort(key=lambda x: x[1], reverse=True)
    simi_list = simi_list[:k]
    if len(simi_list) < k:
        max_simi = simi_list[0]
        for i in range(k - len(simi_list)):
            simi_list.insert(0, max_simi)
    json.dump(simi_list, open(f'raw_data/porto/{tp}_similarity/simi_list_{t}_{ind}.json', 'w'))
    print(ind)
    return simi_list



def collate_unsuperv_contrastive_split_lm(data, max_len=None, vocab=None, add_cls=True):
    features, masks, temporal_mat, features1, features2, temporal_mat1, temporal_mat2, simi_1, simi_2, simi_tempral_mat1, simi_temporal_mat2, mask1, mask2 = zip(*data)
    data_for_mask = list(zip(features, masks, temporal_mat))
    dara_for_contra = list(zip(features1, features2, temporal_mat1, temporal_mat2, simi_1, simi_2, simi_tempral_mat1, simi_temporal_mat2, mask1, mask2))

    X1, X2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2, simi_X1, simi_X2, simi_batch_temporal_mat1, simi_batch_temporal_mat2, simi_masks_1, simi_masks_2 \
        = collate_unsuperv_contrastive_split(data=dara_for_contra, max_len=max_len, vocab=vocab, add_cls=add_cls)

    masked_x, targets, target_masks, padding_masks, batch_temporal_mat = collate_unsuperv_mask(
        data=data_for_mask, max_len=max_len, vocab=vocab, add_cls=add_cls)
    return X1, X2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2, simi_X1, simi_X2, simi_batch_temporal_mat1, simi_batch_temporal_mat2, simi_masks_1, simi_masks_2, \
           masked_x, targets, target_masks, padding_masks, batch_temporal_mat
