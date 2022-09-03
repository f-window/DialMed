"""
@Author: hezf
@Time: 2022/9/3 17:52
@Author: hezf
@desc:
"""
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


def get_dataloader(kg_info, label_data, tokenizer):
    train_dataset = PretrainedDataset(os.path.join(args.data_path, 'train.txt'), tokenizer,
                                      kg_info, label_data, dialogue_percent=args.dialogue_percent)
    dev_dataset = PretrainedDataset(os.path.join(args.data_path, 'dev.txt'), tokenizer,
                                    kg_info, label_data, dialogue_percent=args.dialogue_percent)
    test_dataset = PretrainedDataset(os.path.join(args.data_path, 'test.txt'), tokenizer,
                                     kg_info, label_data, dialogue_percent=args.dialogue_percent)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=PretrainedDataset.pretrained_collate,
                                  pin_memory=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=PretrainedDataset.pretrained_collate,
                                pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=PretrainedDataset.pretrained_collate,
                                 pin_memory=True)
    return train_dataloader, dev_dataloader, test_dataloader


def main():
    setup_seed(args.seed)
    model_path = '/huggingface-model/chinese-roberta-wwm-ext'
    # data
    kg_info = KGInfo(os.path.join(root_path, 'data/kg/kg_info.pkl'))
    kg_embedding = pickle.load(open(os.path.join(root_path, 'data/kg/transr_embedding_500.pkl'), 'rb'))
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ddi_matrix = get_ddi_matrix_from_file('{}/appendix/ddi.json'.format(root_path), os.path.join(args.data_path, 'label.json')) if args.ddi else None
    label_data = LabelData(label_path=os.path.join(args.data_path, 'label.json'))
    train_dataloader, dev_dataloader, test_dataloader = get_dataloader(kg_info, label_data, tokenizer)
    # model and setup
    pretrained_model = AutoModel.from_pretrained(model_path, config=config)
    device = torch.device('cuda')
    model = DDN(model=pretrained_model, n_label=len(label_data.label_to_id), hidden_size=config.hidden_size,
                output_size=500, entity_embedding=kg_embedding['entity_embedding'], relation_embedding=kg_embedding['relation_embedding'],
                entity_nums=len(kg_info.entity2id), relation_nums=len(kg_info.id2relation), graph_adj=kg_info.graph_adj.to(device),
                dim=500, config=config, kg_info=kg_info, relation_types=None if args.relation_types == -1 else args.relation_types,
                relation_dim=args.relation_dim).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gama)
    loss_func = torch.nn.BCELoss().to(device)
    max_dev_jaccard, best_dev_epoch, best_dev_model = 0, 0, None
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    logger = MyLogger(os.path.join(os.path.join(root_path, 'data/log/'), args.log_name+time_str+'.log'))
    logger.log('python ' + ' '.join(sys.argv))
    logger.log(str(model))
    # Train and eval
    for epoch in range(args.epoch):
        logger.log('\n\nepoch: {}'.format(epoch))
        logger.log('lr: {}'.format(optimizer.param_groups[0]['lr']))
        train_loss, train_jaccard, dev_loss, dev_jaccard, test_loss, test_jaccard = [], [], [], [], [], []
        train_p, train_r, train_f1, = [], [], []
        model.train()
        for input_ids, token_type_ids, attention_mask, labels, adjs, cls_ids, node_ids, kg_adjs, disease_ids in tqdm(train_dataloader):
            input_ids, token_type_ids = input_ids.to(device), token_type_ids.to(device)
            attention_mask, labels = attention_mask.to(device), labels.to(device)
            disease_ids = disease_ids.to(device)
            outputs = model(input_ids, token_type_ids, attention_mask, adjs, cls_ids, node_ids, kg_adjs, disease_ids)
            result = logits_to_multi_hot(outputs)
            optimizer.zero_grad()
            loss = loss_func(outputs, labels.float())

            batch_labels = labels.data.cpu().numpy()
            p, r, f1, j = get_metrics(result, batch_labels)
            train_p.append(p), train_r.append(r), train_f1.append(f1), train_jaccard.append(j)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()
        logger.log('train_loss: {}'.format(sum(train_loss) / len(train_loss)))
        logger.log('train_prescription: {}'.format(sum(train_p) / len(train_p)))
        logger.log('train_recall: {}'.format(sum(train_r) / len(train_r)))
        logger.log('train_f1: {}'.format(sum(train_f1) / len(train_f1)))
        logger.log('train_jaccard_score: {}'.format(sum(train_jaccard) / len(train_jaccard)))
        if dev_dataloader:
            with torch.no_grad():
                model.eval()
                dev_results, dev_labels = [], []
                for input_ids, token_type_ids, attention_mask, labels, adjs, cls_ids, node_ids, kg_adjs, disease_ids in tqdm(dev_dataloader):
                    dev_labels.append(labels.numpy())
                    input_ids, token_type_ids = input_ids.to(device), token_type_ids.to(device)
                    attention_mask, labels = attention_mask.to(device), labels.to(device)
                    disease_ids = disease_ids.to(device)
                    outputs = model(input_ids, token_type_ids, attention_mask, adjs, cls_ids, node_ids, kg_adjs, disease_ids)

                    dev_results.append(logits_to_multi_hot(outputs))
                    dev_loss.append(loss_func(outputs, labels.float()).item())
                dev_p, dev_r, dev_f1, dev_j = get_metrics(np.concatenate(dev_results), np.concatenate(dev_labels))
                logger.log('dev_loss: {}'.format(np.nanmean(dev_loss)))
                logger.log('dev_prescription: {}'.format(dev_p))
                logger.log('dev_recall: {}'.format(dev_r))
                logger.log('dev_f1: {}'.format(dev_f1))
                logger.log('dev_jaccard_score: {}'.format(dev_j))
                if max_dev_jaccard < dev_j:
                    max_dev_jaccard = dev_j
                    best_dev_epoch = epoch
                    logger.log('Saving model...')
                    best_state_dict = copy.deepcopy(model.state_dict())
                logger.log('best jaccard: {}, best epoch: {}'.format(max_dev_jaccard, best_dev_epoch))
    # Test
    if test_dataloader:
        with torch.no_grad():
            model.load_state_dict(best_state_dict)
            model.eval()
            test_results, test_labels, test_loss = [], [], []
            for input_ids, token_type_ids, attention_mask, labels, adjs, cls_ids, node_ids, kg_adjs, disease_ids in tqdm(test_dataloader):
                test_labels.append(labels.numpy())
                input_ids, token_type_ids = input_ids.to(device), token_type_ids.to(device)
                attention_mask, labels = attention_mask.to(device), labels.to(device)
                disease_ids = disease_ids.to(device)
                outputs = model(input_ids, token_type_ids, attention_mask, adjs, cls_ids, node_ids, kg_adjs, disease_ids)
                test_results.append(logits_to_multi_hot(outputs))
                test_loss.append(loss_func(outputs, labels.float()).item())
            test_results = np.concatenate(test_results)
            test_labels = np.concatenate(test_labels)
            test_p, test_r, test_f1, test_j = get_metrics(test_results, test_labels)

            logger.log('test_loss: {}'.format(np.nanmean(test_loss)))
            logger.log('test_prescription: {}'.format(test_p))
            logger.log('test_recall: {}'.format(test_r))
            logger.log('test_f1: {}'.format(test_f1))
            logger.log('test_jaccard_score: {}'.format(test_j))
            if ddi_matrix is not None:
                logger.log('test_ddi: {}'.format(batch_ddi_rate(test_results, ddi_matrix)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", '-cu', type=str, default=0, help='GPU id')
    parser.add_argument('--seed', '-s', type=int, default=20, help='Random seed')
    parser.add_argument('--lr', '-l', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--epoch', '-e', type=int, default=50, help='Epochs of training')
    parser.add_argument('--log_name', '-lg', type=str, default='log-', help='Log file name')
    parser.add_argument('--data_path', '-p', type=str, required=True, help='A directory containing [(train)|(dev)|test].txt and label.json')
    parser.add_argument('--dialogue_percent', '-dp', type=float, default=1, choices=[0.2, 0.4, 0.6, 0.8, 1], help='A hyper-parameter for the experiment of task feasibility analysis')
    parser.add_argument('--relation_types', type=int, default=-1, help='Relation types in Dialogue Graph')
    parser.add_argument('--relation_dim', type=int, default=768/2, help='Relation dim in Dialogue Graph')
    parser.add_argument('--ddi', action='store_true', help='The flag of calculate ddi')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.cuda)
    root_path = '/DialMed'

    import time
    import numpy as np
    import copy
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoConfig, AutoModel
    from tqdm import tqdm
    import copy
    import pickle
    import warnings
    from src.model import DDN
    from src.data import KGInfo, PretrainedDataset, LabelData
    from src.utils import *
    warnings.filterwarnings("ignore")
    milestones = [20, 30, 40]
    gama = 0.5

    main()
