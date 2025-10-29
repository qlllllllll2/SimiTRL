import os
import json
import pdb

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import datetime
import pickle
import pandas as pd
import scipy.sparse as sp
from logging import getLogger
from torch.utils.data import DataLoader
from libcity.data.dataset import AbstractDataset, WordVocab
from tqdm import trange, tqdm

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

class BaseDataset(AbstractDataset):
    def __init__(self, config):
        self.logger = getLogger()
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.max_train_size = self.config.get('max_train_size', None)
        self.batch_size = self.config.get('batch_size', 64)
        self.num_workers = self.config.get('num_workers', 0)
        self.vocab_path = self.config.get('vocab_path', None)
        self.baseline_bert = self.config.get('baseline_bert', False)
        self.baseline_tf = self.config.get('baseline_tf', False)
        self.min_freq = self.config.get('min_freq', 1)
        self.merge = self.config.get('merge', True)
        if self.vocab_path is None:
            self.vocab_path = 'raw_data/vocab_{}_True_{}.pkl'.format(self.dataset, self.min_freq)
            if self.merge:
                self.vocab_path = 'raw_data/vocab_{}_True_{}_merge.pkl'.format(self.dataset, self.min_freq)
        if self.baseline_bert:
            self.vocab_path = self.vocab_path[:-4]
            self.vocab_path += '_eos.pkl'
        self.seq_len = self.config.get('seq_len', 512)
        self.add_cls = self.config.get('add_cls', True)
        self.usr_num = 0
        self.vocab_size = 0
        self.vocab = None
        self._load_vocab()
        self.collate_fn = None
        self._logger = getLogger()

        self.roadnetwork = self.config.get('roadnetwork', 'bj_road_edge')
        self.data_path = './raw_data/' + self.roadnetwork + '/'
        self.geo_file = self.config.get('geo_file', self.roadnetwork)
        self.rel_file = self.config.get('rel_file', self.roadnetwork)
        self.bidir_adj_mx = self.config.get('bidir_adj_mx', False)
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.rel')

        self.append_degree2gcn = self.config.get('append_degree2gcn', True)
        self.add_gat = self.config.get('add_gat', True)
        self.gat_K = self.config.get('gat_K', 1)
        self.load_trans_prob = self.config.get('load_trans_prob', True)
        self.normal_feature = self.config.get('normal_feature', False)
        self.device = self.config.get('device', torch.device('cpu'))

        self.geo_file = self._load_geo()
        self.rel_file = self._load_rel()

        if self.add_gat:
            self.node_features, self.node_fea_dim = self._load_geo_feature(self.geo_file)
            self.edge_index, self.loc_trans_prob = self._load_k_neighbors_and_trans_prob()
            self.adj_mx_encoded = None
        else:
            self.node_features, self.node_fea_dim = None, 0
            self.adj_mx_encoded = None
            self.edge_index = None
            self.loc_trans_prob = None

    def _load_vocab(self):
        self.logger.info("Loading Vocab from {}".format(self.vocab_path))
        self.vocab = WordVocab.load_vocab(self.vocab_path)
        self.usr_num = self.vocab.user_num
        self.vocab_size = self.vocab.vocab_size
        self.logger.info('vocab_path={}, usr_num={}, vocab_size={}'.format(
            self.vocab_path, self.usr_num, self.vocab_size))

    def _load_geo(self):
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        for index, geo_id in enumerate(self.geo_ids):
            self.geo_to_ind[geo_id] = index
            self.ind_to_geo[index] = geo_id
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_ids)))
        return geofile

    def _load_rel(self):
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')[['origin_id', 'destination_id']]
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        for row in relfile.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
            if self.bidir_adj_mx:
                self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1

        self._logger.info("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx.shape) +
                          ', edges=' + str(self.adj_mx.sum()))
        return relfile


    def _load_geo_feature(self, road_info):
        node_fea_path = self.data_path + '{}_node_features.npy'.format(self.roadnetwork)
        if self.append_degree2gcn:
            node_fea_path = node_fea_path[:-4] + '_degree.npy'
        if os.path.exists(node_fea_path):
            node_features = np.load(node_fea_path)
        else:
            useful = ['highway', 'lanes', 'length', 'maxspeed']
            if self.append_degree2gcn:
                useful += ['outdegree', 'indegree']
            node_features = road_info[useful]
            norm_dict = {
                'length': 2,
            }
            for k, v in norm_dict.items():
                d = node_features[k]
                min_ = d.min()
                max_ = d.max()
                dnew = (d - min_) / (max_ - min_)
                node_features = node_features.drop(k, 1)
                node_features.insert(v, k, dnew)
            onehot_list = ['lanes', 'maxspeed', 'highway']
            if self.append_degree2gcn:
                onehot_list += ['outdegree', 'indegree']
            for col in onehot_list:
                dum_col = pd.get_dummies(node_features[col], col)
                node_features = node_features.drop(col, axis=1)
                node_features = pd.concat([node_features, dum_col], axis=1)
            node_features = node_features.values
            np.save(node_fea_path, node_features)

        self._logger.info('node_features: ' + str(node_features.shape))  # (N, fea_dim)
        node_fea_vec = np.zeros((self.vocab.vocab_size, node_features.shape[1]))
        for ind in range(len(node_features)):
            geo_id = self.ind_to_geo[ind]
            encoded_geo_id = self.vocab.loc2index[geo_id]
            node_fea_vec[encoded_geo_id] = node_features[ind]
        if self.normal_feature:
            self._logger.info('node_features by a/row_sum(a)')  # (vocab_size, fea_dim)
            row_sum = np.clip(node_fea_vec.sum(1), a_min=1, a_max=None)
            for i in range(len(node_fea_vec)):
                node_fea_vec[i, :] = node_fea_vec[i, :] / row_sum[i]
        node_fea_pe = torch.from_numpy(node_fea_vec).float().to(self.device)  # (vocab_size, fea_dim)
        self._logger.info('node_features_encoded: ' + str(node_fea_pe.shape))  # (vocab_size, fea_dim)
        return node_fea_pe, node_fea_pe.shape[1]

    def _load_k_neighbors_and_trans_prob(self):
        """
        Args:

        Returns:
            (vocab_size, pretrain_dim)
        """
        source_nodes_ids, target_nodes_ids = [], []
        seen_edges = set()
        geoid2neighbors = json.load(open(self.data_path + self.roadnetwork + '_neighbors_{}.json'.format(self.gat_K)))
        if self.load_trans_prob:
            loc_trans_prob = []
            link2prob = json.load(open(self.data_path + self.roadnetwork + '_trans_prob_{}.json'.format(self.gat_K)))
        for k, v in geoid2neighbors.items():
            src_node = self.vocab.loc2index[int(k)]
            for tgt in v:
                trg_node = self.vocab.loc2index[int(tgt)]
                if (src_node, trg_node) not in seen_edges:
                    source_nodes_ids.append(src_node)
                    target_nodes_ids.append(trg_node)
                    seen_edges.add((src_node, trg_node))
                    if self.load_trans_prob:
                        loc_trans_prob.append(link2prob[str(k) + '_' + str(tgt)])
        # add_self_edge
        for i in range(self.vocab.vocab_size):
            if (i, i) not in seen_edges:
                source_nodes_ids.append(i)
                target_nodes_ids.append(i)
                seen_edges.add((i, i))
                if self.load_trans_prob:
                    loc_trans_prob.append(link2prob.get(str(i) + '_' + str(i), 0.0))
        # shape = (2, E), where E is the number of edges in the graph
        edge_index = torch.from_numpy(np.row_stack((source_nodes_ids, target_nodes_ids))).long().to(self.device)
        self.logger.info('edge_index: ' + str(edge_index.shape))  # (vocab_size, pretrain_dim)
        if self.load_trans_prob:
            loc_trans_prob = torch.from_numpy(np.array(loc_trans_prob)).unsqueeze(1).float().to(self.device)  # (E, 1)
            self._logger.info('Trajectory loc-transfer prob shape={}'.format(loc_trans_prob.shape))
        else:
            loc_trans_prob = None
        return edge_index, loc_trans_prob

    def _gen_dataset(self):
        train_dataset = TrajectoryProcessingDataset(data_name=self.dataset,
                                                    data_type='train', vocab=self.vocab,
                                                    seq_len=self.seq_len, add_cls=self.add_cls,
                                                    merge=self.merge, min_freq=self.min_freq,
                                                    max_train_size=self.max_train_size)
        eval_dataset = TrajectoryProcessingDataset(data_name=self.dataset,
                                                   data_type='eval', vocab=self.vocab,
                                                   seq_len=self.seq_len, add_cls=self.add_cls,
                                                   merge=self.merge, min_freq=self.min_freq,
                                                   max_train_size=None)
        test_dataset = TrajectoryProcessingDataset(data_name=self.dataset,
                                                   data_type='test', vocab=self.vocab,
                                                   seq_len=self.seq_len, add_cls=self.add_cls,
                                                   merge=self.merge, min_freq=self.min_freq,
                                                   max_train_size=None)
        return train_dataset, eval_dataset, test_dataset

    def _gen_dataloader(self, train_dataset, eval_dataset, test_dataset):
        assert self.collate_fn is not None
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers, shuffle=True,
                                      collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                           vocab=self.vocab, add_cls=self.add_cls))
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=True,
                                     collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                          vocab=self.vocab, add_cls=self.add_cls))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=False,
                                     collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                          vocab=self.vocab, add_cls=self.add_cls))
        return train_dataloader, eval_dataloader, test_dataloader

    def get_data(self):
        self.logger.info("Loading Dataset!")
        train_dataset, eval_dataset, test_dataset = self._gen_dataset()
        self.logger.info('Size of dataset: ' + str(len(train_dataset)) +
                         '/' + str(len(eval_dataset)) + '/' + str(len(test_dataset)))

        self.logger.info("Creating Dataloader!")
        return self._gen_dataloader(train_dataset, eval_dataset, test_dataset)

    def get_data_feature(self):
        data_feature = {'usr_num': self.usr_num, 'vocab_size': self.vocab_size, 'vocab': self.vocab,
                        "adj_mx": self.adj_mx, "num_nodes": self.num_nodes,
                        "geo_file": self.geo_file, "rel_file": self.rel_file,
                        "geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
                        "node_features": self.node_features, "node_fea_dim": self.node_fea_dim,
                        "adj_mx_encoded": self.adj_mx_encoded, "edge_index": self.edge_index,
                        "loc_trans_prob": self.loc_trans_prob}
        return data_feature


class TrajectoryProcessingDataset(Dataset):

    def __init__(self, data_name, data_type, vocab, seq_len=512,
                 add_cls=True, merge=True, min_freq=1, max_train_size=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.add_cls = add_cls
        self.max_train_size = max_train_size
        self._logger = getLogger()
        self.data_type = data_type

        self.data_path = 'raw_data/{}/{}_{}.csv'.format(data_name, data_name, data_type)
        self.cache_path = 'raw_data/{}/cache_{}_{}_{}_{}_{}.pkl'.format(
            data_name, data_name, data_type, add_cls, merge, min_freq)
        self.temporal_mat_path = self.cache_path[:-4] + '_temporal_mat.pkl'
        self._load_data()
        self.simi_indexes = []
        for ind in trange(len(self.traj_list)):
            simi_score = json.load(open(f'raw_data/porto/{self.data_type}_similarity/simi_list_0_{ind}.json'))
            simi_index = [i[0] for i in simi_score[1:]]
            self.simi_indexes.append(simi_index)
        # self.pre_calculate_similarity_traj()

    def _load_data(self):
        if os.path.exists(self.cache_path) and os.path.exists(self.temporal_mat_path):
            self.traj_list = pickle.load(open(self.cache_path, 'rb'))
            self.temporal_mat_list = pickle.load(open(self.temporal_mat_path, 'rb'))
            self._logger.info('Load dataset from {}'.format(self.cache_path))
        else:
            origin_data_df = pd.read_csv(self.data_path, sep=';')
            self.traj_list, self.temporal_mat_list = self.data_processing(origin_data_df)
        if self.max_train_size is not None:
            self.traj_list = self.traj_list[:self.max_train_size]
            self.temporal_mat_list = self.temporal_mat_list[:self.max_train_size]

    def _cal_mat(self, tim_list):
        # calculate the temporal relation matrix
        seq_len = len(tim_list)
        mat = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                off = abs(tim_list[i] - tim_list[j])
                mat[i][j] = off
        return mat  # (seq_len, seq_len)

    def data_processing(self, origin_data, desc=None, cache_path=None, tmat_path=None):
        self._logger.info('Processing dataset in TrajectoryProcessingDataset!')
        sub_data = origin_data[['path', 'tlist', 'usr_id', 'traj_id', 'vflag']]
        traj_list = []
        temporal_mat_list = []
        for i in tqdm(range(sub_data.shape[0]), desc=desc):
            traj = sub_data.iloc[i]
            loc_list = eval(traj['path'])
            tim_list = eval(traj['tlist'])
            usr_id = traj['usr_id']
            new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in loc_list]
            new_tim_list = [datetime.datetime.utcfromtimestamp(tim) for tim in tim_list]
            minutes = [new_tim.hour * 60 + new_tim.minute + 1 for new_tim in new_tim_list]
            weeks = [new_tim.weekday() + 1 for new_tim in new_tim_list]
            usr_list = [self.vocab.usr2index.get(usr_id, self.vocab.unk_index)] * len(new_loc_list)
            if self.add_cls:
                new_loc_list = [self.vocab.sos_index] + new_loc_list
                minutes = [self.vocab.pad_index] + minutes
                weeks = [self.vocab.pad_index] + weeks
                usr_list = [usr_list[0]] + usr_list
                tim_list = [tim_list[0]] + tim_list
            temporal_mat = self._cal_mat(tim_list)
            temporal_mat_list.append(temporal_mat)
            loc_list.insert(0, 10000000)
            traj_fea = np.array([new_loc_list, tim_list, minutes, weeks, usr_list, loc_list]).transpose((1, 0))
            traj_list.append(traj_fea)
        if cache_path is None:
            cache_path = self.cache_path
        if tmat_path is None:
            tmat_path = self.temporal_mat_path
        pickle.dump(traj_list, open(cache_path, 'wb'))
        pickle.dump(temporal_mat_list, open(tmat_path, 'wb'))
        return traj_list, temporal_mat_list

    def __len__(self):
        return len(self.traj_list)

    def __getitem__(self, ind):
        traj_ind = self.traj_list[ind]  # (seq_length, feat_dim)
        temporal_mat = self.temporal_mat_list[ind]  # (seq_length, seq_length)
        simi_ind = self.simi_indexes[ind]
        simi_trajs = [self.traj_list[i] for i in simi_ind]
        simi_temporal_mat = [self.temporal_mat_list[i] for i in simi_ind]
        return torch.LongTensor(traj_ind), torch.LongTensor(temporal_mat), [torch.LongTensor(i) for i in simi_trajs],\
               [torch.LongTensor(i) for i in simi_temporal_mat]

    def pre_calculate_similarity_traj(self):
        traj_list1 = self.traj_list
        simi_index_1= []
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

        args_1 = [(traj, traj_list1, grid_idx_1, path_idx_1, i, 0, self.data_type) for i, traj in enumerate(traj_list1)]
        pool = Pool(20)
        similarities_1 = pool.starmap(extract_similarity_traj, args_1)
        pool.close()
        if self.data_type == 'test':
            exit()


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
def padding_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
