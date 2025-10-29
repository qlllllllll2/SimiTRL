import json
import numpy as np
from logging import getLogger
from libcity.data.dataset import BaseDataset, TrajectoryProcessingDataset, padding_mask
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import torch
import datetime
import pickle
import pandas as pd
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

class ClusterDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.cluster_data_path = config.get('cluster_data_path', None)
        self._load_geo_latlon()
        self.collate_fn = collate_unsuperv_down

    def _load_geo_latlon(self):
        if self.dataset in ['bj', 'geolife']:
            self.geo_file = pd.read_csv('raw_data/bj_roadmap_edge/bj_roadmap_edge.geo')
        if self.dataset in ['porto']:
            self.geo_file = pd.read_csv('raw_data/porto_roadmap_edge/porto_roadmap_edge.geo')
        assert self.geo_file['type'][0] == 'LineString'
        self.geoid2latlon = {}
        for i in range(self.geo_file.shape[0]):
            geo_id = int(self.geo_file.iloc[i]['geo_id'])
            coordinates = eval(self.geo_file.iloc[i]['coordinates'])
            self.geoid2latlon[geo_id] = coordinates
        self._logger.info("Loaded Geo2LatLon, num_nodes=" + str(len(self.geoid2latlon)))

    def _gen_dataset(self):
        test_dataset = DownStreamSubDataset(data_name=self.dataset,
                                            data_path=self.cluster_data_path,
                                            vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                            max_train_size=None,
                                            geo2latlon=self.geoid2latlon)
        return [None], [None], test_dataset

    def _gen_dataloader(self, train_dataset, eval_dataset, test_dataset):
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=False,
                                     collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                          vocab=self.vocab, add_cls=self.add_cls))
        return [None], [None], test_dataloader


class DownStreamSubDataset(TrajectoryProcessingDataset):
    def __init__(self, data_name, data_type, data_path, vocab, seq_len=512,
                 add_cls=True, max_train_size=None, geo2latlon=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.add_cls = add_cls
        self.max_train_size = max_train_size
        self.geo2latlon = geo2latlon
        self._logger = getLogger()
        base_path = 'raw_data/{}/'.format(data_name)
        self.data_type = data_type
        self.data_path = base_path + data_path + '.csv'
        self.cache_path = base_path + data_path + '_add_id.pkl'
        self.cache_path_wkt = base_path + data_path + '_add_id.json'
        self.temporal_mat_path = self.cache_path[:-4] + '_temporal_mat.pkl'
        self._load_data()
        self.simi_indexes = []
        for ind in range(len(self.traj_list)):
            simi_score = json.load(open(f'raw_data/porto/{self.data_type}_similarity/simi_list_0_{ind}.json'))
            simi_index = [i[0] for i in simi_score[1:]]
            self.simi_indexes.append(simi_index)
        # print(self.data_type)
        # os.makedirs(f'raw_data/porto/{self.data_type}_similarity', exist_ok=True)
        # self.pre_calculate_similarity_traj()
    def _load_data(self):
        if os.path.exists(self.cache_path) and os.path.exists(self.temporal_mat_path) \
                and os.path.exists(self.cache_path_wkt):
            self.traj_list = pickle.load(open(self.cache_path, 'rb'))
            self.traj_wkt = json.load(open(self.cache_path_wkt, 'r'))
            self.temporal_mat_list = pickle.load(open(self.temporal_mat_path, 'rb'))
            self._logger.info('Load dataset from {}'.format(self.cache_path))
        else:
            origin_data_df = pd.read_csv(self.data_path, sep=';')
            self.traj_list, self.temporal_mat_list, self.traj_wkt = self.data_processing(origin_data_df)
        if self.max_train_size is not None:
            self.traj_list = self.traj_list[:self.max_train_size]
            self.temporal_mat_list = self.temporal_mat_list[:self.max_train_size]


    def data_processing(self, origin_data):
        self._logger.info('Processing dataset in DownStreamSubDataset!')
        sub_data = origin_data[['id', 'path', 'tlist', 'usr_id', 'traj_id', 'vflag']]
        traj_list = []
        traj_wkt = {}
        temporal_mat_list = []
        for i in tqdm(range(sub_data.shape[0]), desc=self.data_path):
            traj = sub_data.iloc[i]
            loc_list = eval(traj['path'])
            tim_list = eval(traj['tlist'])
            usr_id = traj['usr_id']
            vflag = int(traj['vflag'])
            # assert vflag == 0 or vflag == 1
            id_ = int(traj['id'])

            new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in loc_list]
            new_tim_list = [datetime.datetime.utcfromtimestamp(tim) for tim in tim_list]
            minutes = [new_tim.hour * 60 + new_tim.minute + 1 for new_tim in new_tim_list]
            weeks = [new_tim.weekday() + 1 for new_tim in new_tim_list]
            usr_list = [self.vocab.usr2index.get(usr_id, self.vocab.unk_index)] * len(new_loc_list)
            vflag_list = [vflag] * len(new_loc_list)
            id_list = [id_] * len(new_loc_list)

            # cal wkt str
            wkt_str = 'LINESTRING('
            for j in range(len(loc_list)):
                rid = loc_list[j]
                coordinates = self.geo2latlon[rid]  # [(lat1, lon1), (lat2, lon2), ...]
                for coor in coordinates:
                    wkt_str += (str(coor[0]) + ' ' + str(coor[1]) + ',')
            if wkt_str[-1] == ',':
                wkt_str = wkt_str[:-1]
            wkt_str += ')'
            traj_wkt[id_] = wkt_str

            if self.add_cls:
                new_loc_list = [self.vocab.sos_index] + new_loc_list
                minutes = [self.vocab.pad_index] + minutes
                weeks = [self.vocab.pad_index] + weeks
                usr_list = [usr_list[0]] + usr_list
                tim_list = [tim_list[0]] + tim_list
                vflag_list = [vflag_list[0]] + vflag_list
                id_list = [id_list[0]] + id_list
            temporal_mat = self._cal_mat(tim_list)  # (seq_len, seq_len)
            temporal_mat_list.append(temporal_mat)
            traj_fea = np.array([new_loc_list, tim_list, minutes, weeks,
                                 usr_list, vflag_list, id_list]).transpose((1, 0))  # (seq_length, feat_dim)
            traj_list.append(traj_fea)
        pickle.dump(traj_list, open(self.cache_path, 'wb'))
        json.dump(traj_wkt, open(self.cache_path_wkt, 'w'))
        pickle.dump(temporal_mat_list, open(self.temporal_mat_path, 'wb'))
        return traj_list, temporal_mat_list, traj_wkt  # [loc, tim, mins, weeks, usr, vflag, id]

    def pre_calculate_similarity_traj(self):
        traj_list1 = self.traj_list
        simi_index_1 = []
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

        from multiprocessing import Pool, cpu_count

        args_1 = [(traj, traj_list1, grid_idx_1, path_idx_1, i, 0, self.data_type) for i, traj in
                  enumerate(traj_list1)]
        pool = Pool(20)
        similarities_1 = pool.starmap(extract_similarity_traj, args_1)
        pool.close()
        if self.data_type == 'test':
            exit()

def extract_similarity_traj(query_traj, traj_list, grid_idx, path_idx, ind, t, tp, w1=5, w2=5, w3=1, k=10):
    query_loc_list = query_traj[:, 0][0:].tolist()
    query_path_list = query_traj[:, -1][0:].tolist()
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
        loc_list = traj[:, 0][0:]
        path_list = traj[:, -1][0:]
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

def collate_unsuperv_down(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, temporal_mat, simi, simi_temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)
    simi_X = torch.zeros(batch_size, len(simi[0]), max_len, features[0].shape[-1], dtype=torch.long)
    simi_batch_temporal_mat = torch.zeros(batch_size, len(simi[0]), max_len, max_len,
                                          dtype=torch.long)
    lengths_simi = [[s.shape[0] for s in t] for t in simi]


    for i in range(batch_size):
        for j in range(10):
            end = min(lengths_simi[i][j], max_len)
            simi_X[i, j, :end, :] = simi[i][j][:end, :]
            simi_batch_temporal_mat[i, j, :end, :end] = simi_temporal_mat[i][j][:end, :end]
    lengths_simi = [[s.shape[0] for s in t] for t in simi]
    lengths_simi_ = torch.tensor(lengths_simi, dtype=torch.int16)
    simi_masks = (
        torch.arange(0, max_len, device=lengths_simi_.device).type_as(lengths_simi_).repeat(batch_size, 1).unsqueeze(
            1).repeat(1, len(simi[0]), 1).lt(
            lengths_simi_.unsqueeze(2)))

    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)
    return X.long(), padding_masks, batch_temporal_mat.long(), simi_X.long(), simi_batch_temporal_mat.long(), simi_masks.long()

