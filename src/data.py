"""
@Time: 2022/9/3 17:52
@Author: hezf
@desc: 
"""
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List
import torch
import json
from copy import deepcopy
import pickle
import numpy as np
from src.utils import get_sub_graph_with_relation


class PretrainedDataset(Dataset):
    def __init__(self, data_path, tokenizer, kg_info, label_data=None, dialogue_percent=1):
        super(PretrainedDataset, self).__init__()
        self.data_path = data_path
        self.label_to_id = dict()
        self.id_to_label = dict()
        self.disease_to_id = dict()
        self.tokenizer = tokenizer
        self.kg_info = kg_info
        self.label_data = label_data
        # [0.2, 0.4, 0.6, 0.8]
        self.dialogue_percent = dialogue_percent
        self.MAX_LENGTH = 512
        self.raw_data = []
        self.data = []
        self.init_data()

    def dialogue_adj(self, dialogue: List[str]):
        """
        Get the adjacent matrix of Dialogue Graph
        :param dialogue:
        :return:
        """
        adj = [[1 if i == j else 0 for j in range(len(dialogue))] for i in range(len(dialogue))]
        for i, utt in enumerate(dialogue):
            current_person = utt[:2]
            other = False
            for j in range(i + 1, len(dialogue)):
                if other and dialogue[j][:2] == current_person:
                    break
                elif not other and dialogue[j][:2] == current_person:
                    adj[i][j] = adj[j][i] = 1
                else:
                    other = True
                    adj[i][j] = adj[j][i] = 1
        return torch.tensor(adj)

    def init_data(self):
        def to_multi_hot(max_len, id_list):
            """
            :param max_len:
            :param id_list:
            :return:
            """
            result = [0 for _ in range(max_len)]
            for idx in id_list:
                result[idx] = 1
            return result

        print('Loading data...')
        with open(self.data_path, 'r', encoding='utf-8') as f:
            temp = f.readlines()
            self.raw_data = temp
        # label processing
        diseases = set()
        for dialog_block in temp:
            dialog_dict = json.loads(dialog_block.strip())
            if self.label_data is None:
                for lab in dialog_dict['label']:
                    if lab not in self.label_to_id:
                        temp_id = len(self.label_to_id)
                        self.label_to_id[lab] = temp_id
                        self.id_to_label[temp_id] = lab
            if dialog_dict['disease'][0] == '其他疾病或无确诊' or dialog_dict['disease'][0] not in self.kg_info.entity2id:
                dialog_dict['disease'][0] = '其他疾病或无确诊'
            diseases.add(self.kg_info.entity2id[dialog_dict['disease'][0]])
        if self.label_data is not None:
            self.label_to_id = self.label_data.label_to_id
            self.id_to_label = self.label_data.id_to_label
        # dialogue processing
        diseases = list(diseases)
        for line_id, dialog_block in enumerate(tqdm(temp)):
            if line_id >= 1000:
                break
            dialog_dict = json.loads(dialog_block.strip())
            dialog_str = ''
            speaker_idx = []
            dialog = dialog_dict['dialog']
            # For the experiment of task feasibility analysis
            if self.dialogue_percent < 1 and len(dialog) < 5:
                continue
            remain_turn = int(len(dialog) * self.dialogue_percent)
            dialog = dialog[:remain_turn]
            for i, u in enumerate(dialog):
                speaker_idx.append(1 if u[:2] == '患者' else 0)
                dialog_str += ('[CLS]{}[SEP]'.format(u))
            if dialog_str == '':
                continue
            label_ids = [self.label_to_id[lab] for lab in dialog_dict['label']]
            tokens = self.tokenizer(dialog_str, add_special_tokens=False, max_length=self.MAX_LENGTH,
                                    padding='max_length', return_tensors='pt', truncation=True)
            if tokens.data['input_ids'][0][-1].item() != self.tokenizer.sep_token_id and tokens.data['input_ids'][0][-1].item() != self.tokenizer.pad_token_id:
                tokens.data['input_ids'][0][-1] = self.tokenizer.sep_token_id
            cls_ids = [i for i, item in enumerate(tokens.data['input_ids'][0]) if item.item() == self.tokenizer.cls_token_id]
            utter_block = [i for i, item in enumerate(tokens.data['input_ids'][0]) if item.item() == self.tokenizer.sep_token_id]
            # speaker type ids
            token_type_flag = False
            speaker_type_ids = []
            if token_type_flag:
                temp_cls_ids = np.roll(cls_ids, -1)
                for i, idx in enumerate(temp_cls_ids):
                    length = idx if i < len(temp_cls_ids)-1 else self.MAX_LENGTH
                    # patients
                    if speaker_idx[i] == 1:
                        speaker_type_ids.extend([1] * (length-len(speaker_type_ids)))
                    else:
                        speaker_type_ids.extend([0] * (length-len(speaker_type_ids)))
            else:
                speaker_type_ids = [0] * self.MAX_LENGTH
            tokens.data['token_type_ids'] = torch.tensor(speaker_type_ids, dtype=torch.long).unsqueeze(0)

            disease = dialog_dict['disease'][0] if dialog_dict['disease'][0] not in self.kg_info.entity2id else '其他疾病或无确诊'
            node_id = set()
            get_sub_graph_with_relation(self.kg_info.graph, diseases, node_id, 1, k=5)
            node_id = sorted(node_id)
            disease_id = node_id.index(self.kg_info.entity2id[disease])
            node_id = torch.tensor(node_id, dtype=torch.long)
            dialog_adj = self.dialogue_adj(dialog)[:len(cls_ids), :len(cls_ids)]
            self.data.append({'dialog': tokens, 'label': to_multi_hot(len(self.label_to_id), label_ids), 'utter_block': utter_block,
                              'adj': dialog_adj, 'speaker_idx': torch.tensor(speaker_idx[:len(utter_block)]), 'node_id': node_id,
                              'kg_adj': self.kg_info.graph_adj[node_id][:, node_id], 'disease_idx': disease_id, 'cls_ids': cls_ids
                              })

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def pretrained_collate(batch_data):
        input_ids, token_type_ids, attention_mask, labels = [], [], [], []
        utter_blocks, adjs = [], []
        cls_ids, node_ids, kg_adjs = [], [], []
        disease_idx = []
        for instance in deepcopy(batch_data):
            input_ids.append(instance['dialog']['input_ids'][0].squeeze(0))
            token_type_ids.append(instance['dialog']['token_type_ids'][0].squeeze(0))
            attention_mask.append(instance['dialog']['attention_mask'][0].squeeze(0))
            labels.append(instance['label'])
            utter_blocks.append(instance['utter_block'])
            adjs.append(instance['adj'])
            cls_ids.append(torch.tensor(instance['cls_ids'], dtype=torch.long))
            node_ids.append(torch.tensor(instance['node_id'], dtype=torch.long))
            kg_adjs.append(instance['kg_adj'])
            disease_idx.append(instance['disease_idx'])
        return (torch.stack(input_ids), torch.stack(token_type_ids),
                torch.stack(attention_mask), torch.tensor(labels), adjs, cls_ids,
                node_ids, kg_adjs, torch.tensor(disease_idx, dtype=torch.long))


class KGInfo(object):
    def __init__(self, kg_file):
        super(KGInfo, self).__init__()
        self.kg_file = kg_file
        self.graph = None
        self.graph_adj = None
        self.entity2id = None
        self.id2entity = None
        self.relation2id = None
        self.id2relation = None
        self._init()

    def _init(self):
        with open(self.kg_file, 'rb') as f:
            kg_info = pickle.load(f)
        self.graph = kg_info['graph']
        self.sub_graph = kg_info['sub_graph']
        self.graph_adj = kg_info['graph_adj']
        self.entity2id = kg_info['entity2id']
        self.id2entity = kg_info['id2entity']
        self.relation2id = kg_info['relation2id']
        self.id2relation = kg_info['id2relation']


class LabelData(object):
    def __init__(self, label_path):
        super(LabelData, self).__init__()
        self.label_path = label_path
        self.label_to_id = {}
        self.id_to_label = {}
        self.label_list = []
        self.init_data()

    def init_data(self):
        print('Loading label data...')
        with open(self.label_path, 'r', encoding='utf-8') as f:
            self.label_to_id = json.load(f)
        for label, idx in self.label_to_id.items():
            self.id_to_label[idx] = label
            self.label_list.append(label)
