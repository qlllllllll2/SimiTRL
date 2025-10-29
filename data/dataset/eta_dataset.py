import torch
from libcity.data.dataset import BaseDataset, padding_mask


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


class ETADataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.collate_fn = collate_superv_eta

    def __getitem__(self, ind):
        traj_ind, temporal_mat = super().__getitem__(ind)
        simi_ind = self.simi_indexes[ind]
        simi_trajs = [self.traj_list[i] for i in simi_ind]
        simi_temporal_mat = [self.temporal_mat_list[i] for i in simi_ind]

        return torch.LongTensor(traj_ind), torch.LongTensor(temporal_mat), [torch.LongTensor(i) for i in simi_trajs],\
               [torch.LongTensor(i) for i in simi_temporal_mat]

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

def collate_superv_eta(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, temporal_mat, simi, simi_temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)  # (batch_size, padded_length, padded_length)
    simi_X = torch.zeros(batch_size, len(simi[0]), max_len, features[0].shape[-1], dtype=torch.long)
    simi_batch_temporal_mat = torch.zeros(batch_size, len(simi[0]), max_len, max_len,
                                          dtype=torch.long)
    lengths_simi = [[s.shape[0] for s in t] for t in simi]
    for i in range(batch_size):
        for j in range(10):
            end = min(lengths_simi[i][j] - 1, max_len - 1)
            simi_X[i, j, :end, :] = simi[i][j][:end, :]
            simi_batch_temporal_mat[i, j, :end, :end] = simi_temporal_mat[i][j][:end, :end]
            if add_cls:
                simi_X[i, j, 2:end, 2:4] = vocab.pad_index
            else:
                simi_X[i, j, 1:end, 2:4] = vocab.pad_index
    lengths_simi = [[s.shape[0] - 1 for s in t] for t in simi]
    lengths_simi_ = torch.tensor(lengths_simi, dtype=torch.int16)
    simi_masks = (
        torch.arange(0, max_len, device=lengths_simi_.device).type_as(lengths_simi_).repeat(batch_size, 1).unsqueeze(1).repeat(1,len(simi[0]),1).lt(
            lengths_simi_.unsqueeze(2)))


    labels = []
    update_length = []
    for i in range(batch_size):
        if lengths[i] <= max_len:
            end = lengths[i] - 1
        else:
            end = max_len - 1
        update_length.append(end)
        if add_cls:
            start_time = features[i][1][1]
        else:
            start_time = features[i][0][1]
        labels.append(float((features[i][end][1] - start_time) / 60))
        X[i, :end, :] = features[i][:end, :]
        if add_cls:
            X[i, 2:end, 2:4] = vocab.pad_index
        else:
            X[i, 1:end, 2:4] = vocab.pad_index

    targets = torch.FloatTensor(labels).unsqueeze(-1)  # (batch_size, 1)

    padding_masks = padding_mask(torch.tensor(update_length, dtype=torch.int16), max_len=max_len)

    return X.long(), targets.float(), padding_masks, batch_temporal_mat.long(), simi_X.long(), simi_batch_temporal_mat.long(), simi_masks.long()  # batch_temporal_mat全0
