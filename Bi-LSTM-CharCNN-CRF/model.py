import torch
import torch.nn as nn
from crf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, input_dim, num_layers=1):
        super(Highway, self).__init__()

        self._layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        for layer in self._layers:
            # Bias the highway layer to just carry its input forward.
            # Set the bias on B(x) to be positive, then g will be biased to be high
            # The bias on B(x) is the second half of the bias vector in each linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        current_inputs = inputs
        for layer in self._layers:
            linear_part = current_inputs
            projected_inputs = layer(current_inputs)

            nonlinear_part, gate = projected_inputs.chunk(2, dim=-1)
            nonlinear_part = torch.relu(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_inputs = gate * linear_part + (1 - gate) * nonlinear_part
        return current_inputs


class CharCNN(nn.Module):
    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.hidden_dim = args.char_hidden_dim
        self.char_drop = nn.Dropout(args.dropout)
        self.char_embeddings = nn.Embedding(args.char_alphabet_size, args.char_emb_dim)
        self.char_embeddings.weight.data.copy_(
            torch.from_numpy(self.random_embedding(args.char_alphabet_size, args.char_emb_dim)))

        self.char_cnn = nn.Conv1d(args.char_emb_dim, self.hidden_dim, kernel_size=3, padding=1)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.zeros([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(0, vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, input):
        """

        :param input:
        :return: batch_size * len * char_emb_dim
        """
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds).transpose(2, 1).contiguous()
        return char_cnn_out

    def get_last_hiddens(self, input):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, -1)
        return char_cnn_out


class WordRep(nn.Module):
    """
    词向量：glove/字向量/elmo/bert/flair
    """

    def __init__(self, args, pretrain_word_embedding):
        super(WordRep, self).__init__()
        self.word_emb_dim = args.word_emb_dim
        self.char_emb_dim = args.char_emb_dim
        self.use_char = args.use_char
        self.use_pre = args.use_pre
        self.freeze = args.freeze
        self.drop = nn.Dropout(args.dropout)

        if self.use_pre:
            if self.freeze:
                self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrain_word_embedding),
                                                                   freeze=True).float()
            else:
                self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrain_word_embedding),
                                                                   freeze=False)
        else:
            self.word_embedding = nn.Embedding(args.vocab_size, 300)
        if self.use_char:
            self.char_feature = CharCNN(args)

    def forward(self, word_input, char_inputs=None):
        batch_size = word_input.size(0)
        sent_len = word_input.size(1)
        word_embs = self.word_embedding(word_input)
        if self.use_char:
            word_len = char_inputs.size(2)
            char_inputs = char_inputs.view(-1, word_len)
            char_features = self.char_feature.get_last_hiddens(char_inputs)
            char_features = char_features.view(batch_size, sent_len, -1)
            word_embs = torch.cat([word_embs, char_features], 2)
            # print(word_embs.shape)
        word_represent = self.drop(word_embs)
        return word_represent


class Bilstmcrf(nn.Module):
    """
    bilstm-crf模型
    """

    def __init__(self, args, pretrain_word_embedding, label_size):
        super(Bilstmcrf, self).__init__()
        self.use_crf = args.use_crf
        self.use_char = args.use_char
        self.gpu = args.gpu
        self.use_char = args.use_char
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.rnn_type = args.rnn_type
        self.max_seq_length = args.max_seq_length
        self.use_highway = args.use_highway
        self.dropoutlstm = nn.Dropout(args.dropoutlstm)
        self.wordrep = WordRep(args, pretrain_word_embedding)

        if self.use_char:
            self.lstm = nn.LSTM(350, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                                bidirectional=True)
            self.gru = nn.GRU(350, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                              bidirectional=True)
        else:
            self.lstm = nn.LSTM(300, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                                bidirectional=True)
            self.gru = nn.GRU(300, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                              bidirectional=True)

        self.label_size = label_size
        if self.use_crf:
            self.crf = CRF(self.label_size, self.gpu)
            self.label_size += 2
        if self.use_highway:
            self.highway = Highway(args.rnn_hidden_dim * 2, 1)

        self.hidden2tag = nn.Linear(args.rnn_hidden_dim * 2, self.label_size)

    # pack_padded  pad_packed_sequence
    def forward(self, word_input, input_mask, labels, char_input=None):
        # word_input input_mask   FloatTensor
        if self.use_char:
            word_input = self.wordrep(word_input, char_input)
        else:
            word_input = self.wordrep(word_input)

        input_mask.requires_grad = False
        word_input = word_input * (input_mask.unsqueeze(-1).float())
        batch_size = word_input.size(0)

        total_length = word_input.size(1)
        ttt = input_mask.ge(1)
        word_seq_lengths = [int(torch.sum(i).cpu().numpy()) for i in ttt]

        if self.rnn_type == 'LSTM':
            packed_words = pack_padded_sequence(word_input, word_seq_lengths, True, enforce_sorted=False)
            lstm_out, hidden = self.lstm(packed_words)
            output, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=total_length)
        elif self.rnn_type == 'GRU':
            packed_words = pack_padded_sequence(word_input, word_seq_lengths, True, enforce_sorted=False)
            lstm_out, hidden = self.gru(packed_words)
            output, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=total_length)

        if self.use_highway:
            output = self.highway(output)

        output = self.dropoutlstm(output)
        output = self.hidden2tag(output)
        maskk = input_mask.ge(1)

        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(output, maskk, labels)
            scores, tag_seq = self.crf._viterbi_decode(output, input_mask)
            return total_loss / batch_size, tag_seq
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            active_loss = input_mask.view(-1) == 1
            active_logits = output.view(-1, self.label_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss, output

    def calculate_loss(self, word_input, input_mask, labels, char_input=None):
        # word_input input_mask   FloatTensor
        if self.use_char:
            word_input = self.wordrep(word_input, char_input)
        else:
            word_input = self.wordrep(word_input)
        
#         print(word_input.shape)
        input_mask.requires_grad = False
        word_input = word_input * (input_mask.unsqueeze(-1).float())

        batch_size = word_input.size(0)
        if self.rnn_type == 'LSTM':
            output, _ = self.lstm(word_input)
        elif self.rnn_type == 'GRU':
            output, _ = self.gru(word_input)

        if self.use_highway:
            output = self.highway(output)

        output = self.dropoutlstm(output)
        output = self.hidden2tag(output)
        maskk = input_mask.ge(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(output, maskk, labels)
            scores, tag_seq = self.crf._viterbi_decode(output, input_mask)
            return total_loss / batch_size, tag_seq
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)

            active_loss = input_mask.view(-1) == 1
            active_logits = output.view(-1, self.label_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

            return loss, output
