import os
import argparse
import time
from collections import defaultdict
import random
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from tensorboardX import SummaryWriter
from util import get_logger
from data_io import get_data_tensor, pregress, get_conll_data, pregress_char
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from model import Bilstmcrf
from tqdm import tqdm
import torch.optim as optim
from metrics import compute_f1
from span_util import compute_f1_no_crf_BIEOS, compute_f1_crf, compute_f1_crf_BIEOS
import codecs
import json
from sklearn.model_selection import StratifiedKFold, KFold
from functools import reduce

import warnings
warnings.filterwarnings("ignore")

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# relax-f1
def compute_instance_f1(y_true, y_pred):
    metrics = {}
    TP, FP, FN = defaultdict(int), defaultdict(int), defaultdict(int)
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            if y_true[i][j] not in ['[SEP]', '[CLS]','[PAD]','O']:
                if y_true[i][j][2:] == y_pred[i][j][2:]:
                    TP[y_true[i][j][2:]] += 1
                else:
                    FN[y_true[i][j][2:]] += 1
        for k in range(len(y_pred[i])):
            if y_pred[i][k] not in ['[SEP]', '[CLS]','[PAD]','O']:
                if y_pred[i][k] != y_true[i][k]:
                    FP[y_pred[i][k][2:]] += 1

    all_labels = set(TP.keys()) | set(FP.keys()) | set(FN.keys())

    macro_f1 = 0
    for label in all_labels:
        precision, recall, f1 = _compute_instance_f1(TP[label], FP[label], FN[label])
        metrics["precision-%s" % label] = precision
        metrics["recall-%s" % label] = recall
        metrics["f1-measure-%s" % label] = f1
        macro_f1 += f1
    precision, recall, f1 = _compute_instance_f1(sum(TP.values()), sum(FP.values()), sum(FN.values()))
    metrics["precision-overall"] = precision
    metrics["recall-overall"] = recall
    metrics["f1-measure-overall"] = f1
    metrics['micro-f1'] = f1
    metrics['macro-f1'] = macro_f1 / len(all_labels)

    return metrics


def _compute_instance_f1(TP, FP, FN):
    precision = float(TP) / float(TP + FP) if TP + FP > 0 else 0
    recall = float(TP) / float(TP + FN) if TP + FN > 0 else 0
    f1 = 2. * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0
    return precision, recall, f1

def evaluate_instance(y_true, y_pred):
    metric = compute_instance_f1(y_true, y_pred)
    return metric
    
def evaluate_crf(y_true, y_pred, tag):
    if tag == 'BIO':
        gold_sentences = [compute_f1_crf(i) for i in y_true]
        pred_sentences = [compute_f1_crf(i) for i in y_pred]
    elif tag == 'BIEOS':
        gold_sentences = [compute_f1_crf_BIEOS(i) for i in y_true]
        pred_sentences = [compute_f1_crf_BIEOS(i) for i in y_pred]
    metric = compute_f1(gold_sentences, pred_sentences)
    return metric



def evaluate(data, model, label_map, tag,use_crf):
    print("Evaluating on test set...")
    test_iterator = tqdm(data, desc="test_interation")

    y_pred = []
    y_true = []
    test_loss = 0.

    for step, test_batch in enumerate(test_iterator):
        model.eval()
        _test_batch = tuple(t.to(device) for t in test_batch)
        input_ids, input_mask, label_ids, char_test = _test_batch
        loss, logits = model.calculate_loss(input_ids, input_mask, label_ids, char_test)
        
        if use_crf == False:
            logits = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)

        logits = logits.detach().cpu().numpy()

        test_loss += loss.item()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.cpu().data.numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []

            for j, m in enumerate(label):
                if input_mask[i][j] != 0:
                    temp_1.append(label_map[m])
                    temp_2.append(label_map[logits[i][j]])
                    if j == label.size -1:
                        assert (len(temp_1) == len(temp_2))
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                else:
                    assert (len(temp_1) == len(temp_2))
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break

        test_iterator.set_postfix(test_loss=loss.item())
        
    metric_instance = evaluate_instance(y_true, y_pred)
    
    metric = evaluate_crf(y_true, y_pred, tag)
    metric['test_loss'] = test_loss / len(data)
    return metric, metric_instance


def train(model, train_dataloader, test_dataloader, num_train_epochs, device, lr, tb_writer, logging_steps,
          train_logger, label_map, tag,args):
    # param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = optim.Adam(model.parameters(), lr=lr)

    decay_rate = 0.05
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 / (1 + decay_rate * epoch))
    test_result = []
    test_result_instance = []
    bestscore = 0
    best_epoch = 0
    bestscore_instance = 0
    best_epoch_instance = 0
#     save_model_list = [0, 0, 0, 0, 0]
    tr_loss, logging_loss = 0.0, 0.0
    global_step = 0
    tq = tqdm(range(num_train_epochs), desc="Epoch")
    for epoch in tq:
        # scheduler.step()
        avg_loss = 0.
        model.train()
        model.zero_grad()

        if epoch > 0:
            scheduler.step()

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            _batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids, char_input = _batch
            loss, _ = model.calculate_loss(input_ids, input_mask, label_ids, char_input)
            # loss, _ = model(input_ids, input_mask, label_ids)
            tr_loss += loss.item()
            avg_loss += loss.item() / len(train_dataloader)

            loss.backward()
            optimizer.step()
            model.zero_grad()
            global_step += 1

            if logging_steps > 0 and global_step % logging_steps == 0:
#                 print(scheduler.get_lr())
                tb_writer.add_scalar('lr', scheduler.get_lr(), global_step)
                tb_writer.add_scalar('train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            epoch_iterator.set_postfix(train_loss=loss.item())

        tq.set_postfix(avg_loss=avg_loss)

        print('%d epoch，global_step: %d ,loss: %.2f' % (epoch, global_step, tr_loss / global_step))
        
        metric,metric_instance = evaluate(test_dataloader, model, label_map, tag,args.use_crf)
        
        print(metric)
        metric_instance['epoch'] = epoch
        metric['epoch'] = epoch
        print(metric['test_loss'], epoch)
        tb_writer.add_scalar('test_loss', metric['test_loss'], epoch)

        test_result.append(metric)
        test_result_instance.append(metric_instance)
        
        if metric['micro-f1'] > bestscore:
            bestscore=metric['micro-f1']
            best_epoch = epoch

        
        if metric_instance['micro-f1'] > bestscore_instance:
            bestscore_instance=metric_instance['micro-f1']
            best_epoch_instance = epoch
            print('best model epoch is: %d' %epoch)
            model_name = args.model_save_dir + "best.model"
            torch.save(model.state_dict(), model_name)
            
#             print('best model epoch is: %d' %epoch)
#             model_name = args.model_save_dir + "best.model"
#             torch.save(model.state_dict(), model_name)

        # for socre in range(len(save_model_list)):
        #     if metric['micro-f1'] > save_model_list[socre]:
        #         print(socre, metric['micro-f1'])
        #         save_model_list[socre] = metric['micro-f1']
        #         model_name = args.model_save_dir + str(socre) + ".model"
        #         torch.save(model.state_dict(), model_name)
        #         break

    test_result.append({'best_test_f1': bestscore,
                        'best_test_epoch': best_epoch})
    test_result_instance.append({'best_test_f1': bestscore_instance,
                        'best_test_epoch': best_epoch_instance})
    tb_writer.close()
    return test_result,test_result_instance

def save_config(config, path, verbose=True):
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config

if __name__ == "__main__":
    print(os.getcwd())
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", default=True, type=str2bool, help="Whether to run training.")
    parser.add_argument("--do_test", default=True, type=str2bool, help="Whether to run test on the test set.")
    parser.add_argument('--save_best_model', type=str2bool, default=True, help='Whether to save best model.')

    parser.add_argument('--model_save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
    parser.add_argument('--data_path', default='', type=str, help='数据路径')
    parser.add_argument('--pred_embed_path', default='', type=str, help='预训练词向量路径(glove、word2vec),空则随机初始化词向量')
    parser.add_argument('--tensorboard_dir', default='./saved_models/runs/', type=str)

    parser.add_argument("--use_bieos", default=False, type=str2bool, help="True:BIEOS False:BIO")
    parser.add_argument('--data_type', default='con;;', help='数据类型 -conll - cyber')
    parser.add_argument('--optimizer', default='Adam', type=str)

    parser.add_argument('--max_seq_length', default=128, type=int, help='Sequence max_length.')
    parser.add_argument('--max_word_len', default=20, type=int)
    parser.add_argument('--do_lower_case', default=False, type=str2bool, help='False 区分大小写，true 全为小写')
    parser.add_argument('--freeze', default=True, type=str2bool, help='是否冻结词向量')
    parser.add_argument('--number_normalized', default=False, type=str2bool, help='数字正则化')
    parser.add_argument('--use_crf', default=True, type=str2bool, help='是否使用crf')
    parser.add_argument('--rnn_type', default='LSTM', type=str, help='LSTM/GRU')
    parser.add_argument('--gpu', default=torch.cuda.is_available(), type=str2bool)
    parser.add_argument('--use_pre', default=True, type=str2bool, help='是否使用预训练的词向量')

    parser.add_argument("--learning_rate", default=0.015, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size.')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logging_steps', default=2, type=int)
    parser.add_argument('--word_emb_dim', default=300, type=int)
    parser.add_argument('--char_hidden_dim', default=50, type=int)
    parser.add_argument('--use_char', default=True, type=str2bool, help='是否使用char向量')
    parser.add_argument('--use_highway', default=True, type=str2bool)
    parser.add_argument('--char_emb_dim', default=30, type=int)
    parser.add_argument('--rnn_hidden_dim', default=100, type=int, help='rnn的隐状态的大小')
    parser.add_argument('--num_layers', default=1, type=int, help='rnn中的层数')
    parser.add_argument('--lr_decay', default=0.05, type=float)
    parser.add_argument('--min_count', default=1, type=int)
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--dropoutlstm', default=0.5, type=float,help='lstm后的dropout')
    parser.add_argument('--use_number_norm', default=False, type=str2bool)

    args = parser.parse_args()

    opt = vars(args)  # dict

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    print(args)
    start_time = time.time()
    train_logger = get_logger(args.model_save_dir + '/train_log.log')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def seed_everything(SEED):
        random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True


    def data_ana(data):
        label_alphabet = set()
        entity_num = {}
        for line, label in data:
            for j in range(len(label) - 1):
                label_alphabet.add(label[j])
                if (label[j][0] == 'B' or label[j][0] == 'I') and (label[j + 1] == 'O' or label[j + 1][0] == 'B'):
                    entity_num[label[j][2:]] = entity_num.get(label[j][2:], 0) + 1
                elif j == len(label) - 1 and label[j + 1] == 'I':
                    entity_num[label[j + 1][2:]] = entity_num.get(label[j + 1][2:], 0) + 1

        return label_alphabet, entity_num


    if args.use_bieos == True:
        tag = 'BIEOS'
    else:
        tag = 'BIO'

    if args.data_type == 'cyber':
        data_ans, word2idx, idx2word, label2index, index2label, pretrain_word_embedding, char2idx,vocab = get_data_tensor(
            args.data_path,
            args.min_count,
            args.use_number_norm,
            args.pred_embed_path)

        args.vocab_size = len(vocab)

        seed_everything(args.seed)
        skf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        for fold_, (train_idx, test_idx) in enumerate(skf.split(data_ans)):
            train_data = np.array(data_ans)[train_idx]
            test_data = np.array(data_ans)[test_idx]

            # tr_la, tr_en = data_ana(list(train_data))
            # test_la, test_en = data_ana(list(test_data))
            # print(test_en, len(test_en))
            #
            # print(reduce(lambda x, y: x + y, test_en.values()))
            # print(reduce(lambda x, y: x + y, tr_en.values()))
            #
            # print(reduce(lambda x, y: x + y, test_en.values()) + reduce(lambda x, y: x + y, tr_en.values()))
            # print(tr_en)

            # train_data, train_mask, train_label = pregress(train_data, word2idx, label2index,
            #                                                max_seq_lenth=args.max_seq_length)

            train_data, train_mask, train_label, char_data = pregress_char(train_data, word2idx, label2index,
                                                                           args.max_seq_length, char2idx,
                                                                           args.max_word_len)

            train_data = torch.tensor([f for f in train_data], dtype=torch.long)
            train_mask = torch.tensor([f for f in train_mask], dtype=torch.long)
            train_label = torch.tensor([f for f in train_label], dtype=torch.long)

            bat = len(char_data)
            char_data = np.array(char_data).reshape(bat, args.max_seq_length, args.max_word_len)
            char_data = torch.tensor(char_data, dtype=torch.long)

            train_dataset = TensorDataset(train_data, train_mask, train_label, char_data)

            test_data, test_mask, test_label, char_test = pregress_char(test_data, word2idx, label2index,
                                                                        args.max_seq_length, char2idx,
                                                                        args.max_word_len)

            test_data = torch.tensor([f for f in test_data], dtype=torch.long)
            test_mask = torch.tensor([f for f in test_mask], dtype=torch.long)
            test_label = torch.tensor([f for f in test_label], dtype=torch.long)

            tab = len(char_test)
            char_test = np.array(char_test).reshape(tab, args.max_seq_length, args.max_word_len)
            char_test = torch.tensor(char_test, dtype=torch.long)

            test_dataset = TensorDataset(test_data, test_mask, test_label, char_test)

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

            args.char_alphabet_size = len(char2idx)
            model = Bilstmcrf(args, pretrain_word_embedding, len(label2index))
            model = model.to(device)

            tb_writer = SummaryWriter(args.tensorboard_dir + str(fold_))

            test_result,test_result_instance = train(model, train_dataloader, test_dataloader, args.num_train_epochs, device,
                                args.learning_rate,
                                tb_writer, args.logging_steps, train_logger, index2label, tag)

            with codecs.open(args.model_save_dir + '/test_result_fold_%d.txt' % fold_ , 'w', encoding='utf-8') as f:
                json.dump(test_result, f, indent=4, ensure_ascii=False)
            with codecs.open(args.model_save_dir + '/test_result_instance_fold_%d.txt' % fold_ , 'w', encoding='utf-8') as f:
                json.dump(test_result_instance, f, indent=4, ensure_ascii=False)
                
    elif args.data_type == 'conll':

        train_data, test_data, dev_data, pretrain_word_embedding, vocab, word2idx, idx2word, label2index, index2label,char2idx = get_conll_data(args.data_path,args.min_count,args.use_number_norm,args.pred_embed_path)

        train_data, train_mask, train_label,char_data = pregress_char(train_data, word2idx, label2index,args.max_seq_length,char2idx,args.max_word_len)
        train_data = torch.tensor([f for f in train_data], dtype=torch.long)
        train_mask = torch.tensor([f for f in train_mask], dtype=torch.long)
        train_label = torch.tensor([f for f in train_label], dtype=torch.long)
        bat = len(char_data)
        char_data = np.array(char_data).reshape(bat, args.max_seq_length, args.max_word_len)
        char_data = torch.tensor(char_data, dtype=torch.long)
        train_dataset = TensorDataset(train_data, train_mask, train_label, char_data)

        test_data, test_mask, test_label, char_test = pregress_char(test_data, word2idx, label2index,
                                                                    args.max_seq_length, char2idx,
                                                                    args.max_word_len)
        test_data = torch.tensor([f for f in test_data], dtype=torch.long)
        test_mask = torch.tensor([f for f in test_mask], dtype=torch.long)
        test_label = torch.tensor([f for f in test_label], dtype=torch.long)
        tab = len(char_test)
        char_test = np.array(char_test).reshape(tab, args.max_seq_length, args.max_word_len)
        char_test = torch.tensor(char_test, dtype=torch.long)

        test_dataset = TensorDataset(test_data, test_mask, test_label, char_test)
        
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        args.char_alphabet_size = len(char2idx)
        model = Bilstmcrf(args, pretrain_word_embedding, len(label2index))
        model = model.to(device)

        tb_writer = SummaryWriter(args.tensorboard_dir)

        test_result,test_result_instance = train(model, train_dataloader, test_dataloader, args.num_train_epochs, device,
                            args.learning_rate, tb_writer, args.logging_steps, train_logger, index2label,tag,args)

        with codecs.open(args.model_save_dir + '/test_result.txt','w', encoding='utf-8') as f:
            json.dump(test_result, f, indent=4, ensure_ascii=False)
            
#         with codecs.open(args.model_save_dir + '/test_result_instance.txt','w', encoding='utf-8') as f:
#             json.dump(test_result_instance, f, indent=4, ensure_ascii=False)

    opt = vars(args)  # dict
    # save config
    save_config(opt, args.model_save_dir + '/args_config.json', verbose=True)