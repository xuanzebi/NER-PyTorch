import json
import codecs
from sklearn.model_selection import train_test_split
from util import rep_text, replace_text, clean_text, build_pretrain_embedding,normalize_word
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def build_vocab(data, min_count,use_number_norm=False):
    unk = '</UNK>'
    pad = '</PAD>'
    label2index = {}
    vocab = {}
    label2index[pad] = 0
    index = 1

    word2Idx = {}

    for i, line in enumerate(data):
        text = line[0]
        label = line[1]
        for te, la in zip(text, label):
            if use_number_norm:
                te = normalize_word(te) 
            if te in vocab:
                vocab[te] += 1
            else:
                vocab[te] = 1

            if la not in label2index:
                label2index[la] = index
                index += 1

    index2label = {j: i for i, j in label2index.items()}

    word2Idx[pad] = len(word2Idx)
    word2Idx[unk] = len(word2Idx)

    vocab = {i: j for i, j in vocab.items() if j >= min_count}

    for idx in vocab:
        if idx not in word2Idx:
            word2Idx[idx] = len(word2Idx)
    idx2word = {j: i for i, j in word2Idx.items()}

    return vocab, word2Idx, idx2word, label2index, index2label


def build_char_vocab(data):
    char2idx = {}
    char2idx['</pad>'] = 0
    word_len = []
    for i, line in enumerate(data):
        text = line[0]
        for te in text:
            word_len.append(len(te))
            for t in te:
                if t not in char2idx:
                    char2idx[t] = len(char2idx)
    return char2idx


def get_data(data_path, min_count, use_number_norm, embed_path):
    data_ans = json.load(open(data_path, encoding='utf-8'))
    data = replace_text(clean_text(data_ans), use_number_norm)
    new_data = rep_text(data)
    new_data = [(line[0], line[2]) for line in new_data]  # 2 是BIO 1 是 BIEOS
    char2idx = build_char_vocab(new_data)
    vocab, word2idx, idx2word, label2index, index2label = build_vocab(new_data, min_count)
    pretrain_word_embedding, unk_words = build_pretrain_embedding(embedding_path=embed_path, word_index=word2idx)
    return new_data, pretrain_word_embedding, vocab, word2idx, idx2word, label2index, index2label, char2idx


def get_conll_data(data_path, min_count, use_number_norm, embed_path):
    train_data = json.load(open(data_path + 'train_data.json', encoding='utf-8'))
    test_data = json.load(open(data_path + 'test_data.json', encoding='utf-8'))
    dev_data = json.load(open(data_path + 'dev_data.json', encoding='utf-8'))
    train_data = [(line['text'], line['bioes']) for line in train_data]  # 2 是BIO 1 是 BIEOS
    test_data = [(line['text'], line['bioes']) for line in test_data]  # 2 是BIO 1 是 BIEOS
    dev_data = [(line['text'], line['bioes']) for line in dev_data]  # 2 是BIO 1 是 BIEOS
    new_data = []
    new_data.extend(train_data)
    new_data.extend(test_data)
    new_data.extend(dev_data)
    vocab, word2idx, idx2word, label2index, index2label = build_vocab(new_data, min_count,use_number_norm)
    char2idx = build_char_vocab(new_data)
    pretrain_word_embedding, unk_words = build_pretrain_embedding(embedding_path=embed_path, word_index=word2idx)
    return train_data, test_data, dev_data, pretrain_word_embedding, vocab, word2idx, idx2word, label2index, index2label,char2idx


def pregress(data, word2idx, label2idx, max_seq_lenth):
    INPUT_ID = []
    INPUT_MASK = []
    LABEL_ID = []
    for text, label in data:
        input_mask = []
        input_id = []
        label_id = []
        for te, la in zip(text, label):
            if te in word2idx:
                input_id.append(word2idx[te])
            else:
                input_id.append(word2idx['</UNK>'])
            label_id.append(label2idx[la])
            input_mask.append(1)

        if len(input_id) > max_seq_lenth:
            input_id = input_id[:max_seq_lenth]
            label_id = label_id[:max_seq_lenth]
            input_mask = input_mask[:max_seq_lenth]

        while len(input_id) < max_seq_lenth:
            input_id.append(0)
            label_id.append(0)
            input_mask.append(0)

        assert len(input_id) == len(label_id) == len(input_mask) == max_seq_lenth
        INPUT_ID.append(input_id)
        LABEL_ID.append(label_id)
        INPUT_MASK.append(input_mask)

    return INPUT_ID, INPUT_MASK, LABEL_ID


def pregress_char(data, word2idx, label2idx, max_seq_lenth, char2idx, max_word_lenth):
    INPUT_ID = []
    INPUT_MASK = []
    LABEL_ID = []
    CHAR_ID = []
    for text, label in data:
        input_mask = []
        input_id = []
        label_id = []
        char_id = []
        for te, la in zip(text, label):
            charid = []
            for t in te:
                charid.append(char2idx[t])

            if len(charid) > max_word_lenth:
                charid = charid[:max_word_lenth]

            while len(charid) < max_word_lenth:
                charid.append(0)

            char_id.append(charid)

            if te in word2idx:
                input_id.append(word2idx[te])
            else:
                input_id.append(word2idx['</UNK>'])

            label_id.append(label2idx[la])
            input_mask.append(1)

        if len(input_id) > max_seq_lenth:
            input_id = input_id[:max_seq_lenth]
            label_id = label_id[:max_seq_lenth]
            input_mask = input_mask[:max_seq_lenth]
            char_id = char_id[:max_seq_lenth]

        while len(input_id) < max_seq_lenth:
            input_id.append(0)
            label_id.append(0)
            input_mask.append(0)
            char_id.append([0] * max_word_lenth)

        assert len(input_id) == len(label_id) == len(input_mask) == max_seq_lenth
        INPUT_ID.append(input_id)
        LABEL_ID.append(label_id)
        INPUT_MASK.append(input_mask)
        CHAR_ID.append(char_id)
    return INPUT_ID, INPUT_MASK, LABEL_ID, CHAR_ID


def get_data_tensor(data_path, min_count, use_number_norm, embed_path):
    new_data, pretrain_word_embedding, vocab, word2idx, idx2word, label2index, index2label, char2idx = get_data(
        data_path,
        min_count,
        use_number_norm,
        embed_path)

    # train_data, test_data = train_test_split(new_data, test_size=0.2, random_state=seed)
    # train_data, train_mask, train_label = pregress(train_data, word2idx, label2index, max_seq_lenth=max_seq_lenth)
    # train_data = torch.tensor([f for f in train_data], dtype=torch.long)
    # train_mask = torch.tensor([f for f in train_mask], dtype=torch.long)
    # train_label = torch.tensor([f for f in train_label], dtype=torch.long)
    # train_dataset = TensorDataset(train_data, train_mask, train_label)
    #
    # test_data, test_mask, test_label = pregress(test_data, word2idx, label2index, max_seq_lenth=max_seq_lenth)
    # test_data = torch.tensor([f for f in test_data], dtype=torch.long)
    # test_mask = torch.tensor([f for f in test_mask], dtype=torch.long)
    # test_label = torch.tensor([f for f in test_label], dtype=torch.long)
    # test_dataset = TensorDataset(test_data, test_mask, test_label)

    return new_data, word2idx, idx2word, label2index, index2label, pretrain_word_embedding, char2idx,vocab


# TODO 分桶
class SequenceBucketCollator():
    def __init__(self, choose_length, sequence_index, length_index,
                 label_index=None):
        self.choose_length = choose_length
        self.sequence_index = sequence_index
        self.length_index = length_index
        self.label_index = label_index

    def __call__(self, batch):
        batch = [torch.stack(x) for x in list(zip(*batch))]

        sequences = batch[self.sequence_index]
        lengths = batch[self.length_index]

        length = self.choose_length(lengths).long()
        mask = torch.arange(start=128, end=0, step=-1) < length

        print(sequences.shape)
        padded_sequences = sequences[:, mask]

        batch[self.sequence_index] = padded_sequences

        if self.label_index is not None:
            return [x for i, x in enumerate(batch) if i != self.label_index], batch[
                self.label_index]

        return batch

# mask = torch.gt(torch.unsqueeze(t, 2), 0).type(torch.cuda.FloatTensor)
# train_data, test_data = train_test_split(new_data, test_size=0.2, random_state=1234)
# data_path = 'D:/projects/cybersecurity_ner/dataset/data_punct_bieos_bio.json'
# data_ans = json.load(open(data_path, encoding='utf-8'))
# data = replace_text(clean_text(data_ans), False)
# new_data = rep_text(data)
# new_data = [(line[0], line[2]) for line in new_data]
# char2idx, word_len = build_char_vocab(new_data)
# print(len(char2idx))
# print(char2idx)
#
# word_len = np.array(word_len)
# print(word_len.max(),word_len.min(),word_len.mean())
# print(np.percentile(word_len,99))
