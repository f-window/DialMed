"""
Statistic
"""
import os
import json
import re
import random
from tqdm import tqdm


def statistic_result_medication(data_path):
    """
    
    :param data_path:
    :return:
    """
    pattern = ['啥.*药', '什么.*药', '哪.*药', '那.*药', '哪些.*药', '怎么用.*药', '怎样用.*药', '[吃涂擦].*药吗', '什么.*偏方',
               '有没有.*药', '推荐.*药', '建议.*药', '怎.*药', '推荐.*什么', '用药.*问题', '如何用.*药']
    ps = [re.compile(p) for p in pattern]

    def judge_utterance_positive(utterances):
        for p in ps:
            if len(re.findall(p, utterances)) != 0:
                return True
        return False

    all_str = set()
    all_count = result_count = 0
    for file_name in os.listdir(data_path):
        if file_name.find('.txt') != -1:
            print('Start processing{}...'.format(file_name))
            with open(os.path.join(data_path, file_name), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                all_str.update(set(lines))
    all_data = list(all_str)
    if len(all_str) > 200000:
        all_data = random.sample(all_data, k=100000)
    all_count, result_count = len(all_data), 0
    for line in tqdm(all_data):
        instance = json.loads(line)
        for u in instance['dialog']:
            if u[:2] == '患者' and judge_utterance_positive(u):
                result_count += 1
                break

    print('All count: {}, medication dialogue count: {}'.format(all_count, result_count))
    print('Propation: ', result_count/all_count)

