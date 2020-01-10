import logging
import numpy as np


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename, mode='a', encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


# 数据清洗
def clean_text(data):
    data_clean = []
    for line in data:
        to = line['text']
        la1 = line['bioes']
        la2 = line['bio']
        token = []
        label1 = []
        label2 = []
        for i, tok in enumerate(to):
            punct = '"”‘:“([)]_'
            for pu in punct:
                tok = tok.replace(pu, '')
            if len(tok) > 1:
                if tok[-1] == "'":
                    tok = tok[:-1]
                if tok[0] == "'":
                    tok = tok[1:]
                if tok[0] == "@":
                    tok = tok[1:]

            if len(tok) == 1:
                token.append(tok)
                label1.append(la1[i])
                label2.append(la2[i])
            if len(tok) > 1:
                if tok[-1] in '!.;?,':
                    token.append(tok[:-1])
                    label1.append(la1[i])
                    label2.append(la2[i])
                    token.append(tok[-1])
                    label1.append('O')
                    label2.append('O')
                else:
                    token.append(tok)
                    label1.append(la1[i])
                    label2.append(la2[i])
        assert len(token) == len(label1) == len(label2)
        data_clean.append((token, label1, label2))

    # print(len(data_clean))
    return data_clean


def rep_text(data):
    punct = 'and earlier versions'
    rep_label = 'Other_Technical_Terms'
    k = 0
    new_data = []
    for line in data:
        text = line[0]
        bieos_label = line[1]
        label = line[2]
        punct_list = punct.split(' ')
        i = 0
        while i < len(text) - len(punct_list) + 1:
            if text[i] == punct_list[0] and text[i + 1] == punct_list[1] and text[i + 2] == punct_list[2]:
                if 'O' in label[i:i + 3]:
                    # print(text[i:i + 3])
                    label[i] = 'B-NER_Modifier'
                    label[i + 1] = 'I-NER_Modifier'
                    label[i + 2] = 'I-NER_Modifier'
                    bieos_label[i] = 'B-NER_Modifier'
                    bieos_label[i + 1] = 'I-NER_Modifier'
                    bieos_label[i + 2] = 'E-NER_Modifier'
                    k += 1
                i = i + 2
            i += 1
        for i, te in enumerate(text):
            if label[i][2:] == rep_label:
                label[i] = 'O'
                bieos_label[i] = 'O'
        new_data.append((text, bieos_label, label))
    return new_data


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def replace_text(data_clean, use_number_norm):
    data_final = []

    for token, label1, label2 in data_clean:
        _token = []
        _label = []
        _label2 = []
        for i, to in enumerate(token):
            # TODO: IP 邮箱 数字 缩写（’）  CVE  (p.m.)  带/的路径

            if to in ['Email-Stealing', 'RTF-formatted', 'Mac-compatible']:
                to = to.split('-')[0]

            if use_number_norm:
                to = normalize_word(to)  # 数字的字符全部替换为0

            _token.append(to)
            _label.append(label1[i])
            _label2.append(label2[i])
        data_final.append((_token, _label, _label2))
    return data_final


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_pretrain_embedding(embedding_path, word_index):
    embedding_matrix = np.zeros((len(word_index), 300))
    scale = np.sqrt(3.0 / 300)
    for index in range(1, len(word_index)):
        embedding_matrix[index, :] = np.random.uniform(-scale, scale, [1, 300])
    
    if embedding_path == None or embedding_path == '':
        print('================NONE================')
        return embedding_matrix, 0
    else:
        print('===============GLOVE===================')
        embedding_index = load_embeddings(embedding_path)
        unknown_words = []
        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                try:
                    embedding_matrix[i] = embedding_index[word.lower()]
                except KeyError:
                    try:
                        embedding_matrix[i] = embedding_index[word.title()]
                    except KeyError:
                        unknown_words.append(word)
        return embedding_matrix, unknown_words
