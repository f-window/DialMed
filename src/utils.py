"""
@Time: 2022/9/3 17:52
@Author: hezf
@desc:
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import logging
import random
import math
import os
from scipy.special import comb
from typing import Optional
import json
from tqdm import tqdm


def get_metrics(pred, gold):
    p = precision_score(gold, pred, average='micro')
    r = recall_score(gold, pred, average='micro')
    f1 = f1_score(gold, pred, average='micro')
    j = jaccard_score(gold, pred, average='micro')
    return p, r, f1, j


def logits_to_multi_hot(data: torch.Tensor, threshold=0.5):
    """
    :return:
    """
    if data.is_cuda:
        data = data.cpu().detach().numpy()
    else:
        data = data.numpy()
    result = []
    for i in range(data.shape[0]):
        result.append([1 if v >= threshold else 0 for v in list(data[i])])
    return np.array(result)


def setup_seed(seed):
    """
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class MyLogger(object):
    def __init__(self, log_file, debug='1'):
        self.log_file = log_file
        path = os.path.split(self.log_file)[0]
        if not os.path.exists(path):
            os.mkdir(path)
        self.ch = logging.StreamHandler()
        self.formatter = logging.Formatter("%(asctime)s - %(message)s")
        self.fh = logging.FileHandler(log_file, mode='w')
        self.logger = logging.getLogger()
        self.debug = debug
        self.init()

    def init(self):
        self.logger.setLevel(logging.INFO)
        self.fh.setLevel(logging.INFO)
        self.fh.setFormatter(self.formatter)
        self.ch.setLevel(logging.INFO)
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)
        self.logger.addHandler(self.fh)

    def log(self, message):
        if self.debug == '1':
            self.logger.info(message)


def load_model(model_path, map_location=None):
    print('Loading model...')
    model = torch.load(model_path, map_location=map_location)
    return model


def get_sub_graph(graph, ids, result: set, k=5):
    """
    Randomly sample on KG
    :param ids:
    :param graph: [[adj_id1, adj_id2,...],... [], [], []]
    :param result: set
    :param k:
    :return:
    """
    for idx in ids:
        result.add(idx)
        if len(graph[idx]) > k:
            adj = set(random.sample(graph[idx], k=k))
        else:
            adj = set(graph[idx])
        adj = adj - result
        get_sub_graph(graph, adj, result, k)


def get_sub_graph_with_relation(graph, ids, result: set, depth, k=5, max_depth=3):
    """
    Randomly sample on KG (considering relations)
    :param graph: {entity_id: {rel_id: [], ...}}
    :param ids:
    :param result:
    :param k:
    :param depth
    :param max_depth
    :return:
    """
    if depth > max_depth:
        return None
    for idx in ids:
        result.add(idx)
        relations = graph[idx].keys()
        if len(relations) > 0:
            adj = set()
            fine_k = math.ceil(1.0 * k / len(relations))
            if len(graph[idx]) > k:
                relations = random.sample(relations, k)
                fine_k = 1
            for relation in relations:
                if k > len(graph[idx][relation]):
                    a_ids = graph[idx][relation]
                else:
                    a_ids = random.sample(graph[idx][relation], k=fine_k)
                for a_id in a_ids:
                    adj.add(a_id)
            adj = adj - result
            get_sub_graph_with_relation(graph, adj, result, depth + 1, k)


def ddi_rate(multi_hot: Optional[torch.Tensor], ddi_matrix):
    """
    calculate ddi rate for one sample
    :param multi_hot:
    :param ddi_matrix:
    :return: ddi rate; ddi count; Comb
    """
    if isinstance(multi_hot, np.ndarray):
        multi_hot = torch.tensor(multi_hot)
    if multi_hot.requires_grad:
        multi_hot = multi_hot.data.cpu()
    N = (multi_hot == 1).sum().item()
    if N >= 2:
        c = comb(N, 2)
        if len(multi_hot.shape) == 1:
            multi_hot = multi_hot.unsqueeze(1)
        current_dd = multi_hot * multi_hot.T
        current_ddi = (torch.mul(current_dd, ddi_matrix).sum() / 2).item()
        return current_ddi / c, current_ddi, c
    else:
        return None, None, None


def batch_ddi_rate(multi_hot: torch.Tensor, ddi_matrix):
    """
    calculate ddi rate for one batch
    :param multi_hot:
    :param ddi_matrix:
    :return: ddi rate; ddi count; Comb
    """
    rate = []
    for i in range(len(multi_hot)):
        line = multi_hot[i, :]
        res = ddi_rate(line, ddi_matrix)
        if res[0] is not None:
            rate.append(res[0])
    if len(rate) > 0:
        return np.mean(rate)
    else:
        return 0


def get_ddi_matrix_from_file(ddi_file, label_file):
    """
    load the ddi matrix from ddi file
    :param ddi_file:
    :param label_file:
    :return:
    """
    with open(ddi_file, 'r', encoding='utf-8') as f:
        ddi_data = json.load(f)
    with open(label_file, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)
    matrix = torch.zeros(len(label_dict), len(label_dict))
    for m, dd_list in ddi_data.items():
        if m in label_dict:
            m_id = label_dict[m]
            for dd in dd_list:
                if dd in label_dict:
                    dd_id = label_dict[dd]
                    matrix[m_id, dd_id] = 1
    for i in range(len(label_dict)):
        matrix[i, i] = 0
        for j in range(len(label_dict)):
            if i != j:
                if matrix[i, j] != 0:
                    matrix[j, i] = 1
    num_mat = matrix.numpy()
    return matrix


def calculate_ground_truth_DDI(dataset_path):
    """
    :param dataset_path:
    :return:
    """
    from src.data import LabelData
    label_data = LabelData(label_path=os.path.join(dataset_path, 'label.json'))
    ddi_matrix = get_ddi_matrix_from_file('/appendix/ddi.json', os.path.join(dataset_path, 'label.json'))
    all_labels = []
    for line in tqdm(open(os.path.join(dataset_path, 'test.txt'))):
        instance = json.loads(line)
        label = [0] * len(label_data.id_to_label)
        for lb in instance['label']:
            label[label_data.label_to_id[lb]] = 1
        all_labels.append(label)
    print(batch_ddi_rate(torch.tensor(all_labels), ddi_matrix))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1.0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, logits, labels, efficiency=None):
        """
        logits: batch_size * n_class
        labels: batch_size
        """
        batch_size, n_class = logits.shape[0], logits.shape[1]
        p = logits
        pt = (labels - (1 - p)) * (2*labels-1)
        alpha_t = (labels - (1 - self.alpha)) * (2*labels-1)
        loss = - alpha_t * ((1 - pt)**self.gamma) * torch.log(pt)
        if self.size_average:
            return loss.sum()/batch_size
        else:
            return loss.sum()


def DDI_adaptor(multi_hots, data_path='/data2/hezhenfeng/drug_recommendation/data/dialogue/数据集v3.9/全部数据'):
    """
    :param multi_hots: (B, C)
    :param data_path: a directory containing label.json
    :return:
    """
    if isinstance(multi_hots, (np.ndarray, list)):
        multi_hots = torch.tensor(multi_hots)
    ddi_matrix = get_ddi_matrix_from_file('/data2/hezhenfeng/drug_recommendation/data/medication/ddi.json',
                                          os.path.join(data_path, 'label.json'))
    return batch_ddi_rate(multi_hots, ddi_matrix)
