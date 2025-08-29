"""BERT and RNN model for sentence pair classification.

Author: Tsinghuaboy (tsinghua9boy@sina.com)

Used for SMP-CAIL2020-Argmine.
"""
import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
import torch.nn.functional as F
# from transformers.modeling_bert import BertModel
from transformers import BertModel, BertConfig
from transformers import ElectraConfig,ElectraModel,ElectraTokenizer
import math

def attention(q, k, v, d_k, mask=None, dropout=None):  # Scaled Dot-Product Attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        # self.k_linear(k).shape = (64, 10, 100)
        # (bs, -1, self.h, self.d_k) = (64, -1, 8, 12)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class BertForClassification(nn.Module):
    """BERT with simple linear model."""
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.bert_model_path: pretrained model path or model type
                    e.g. 'bert-base-chinese'
                config.hidden_size: The same as BERT model, usually 768
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        model_config =  BertConfig.from_pretrained(config.bert_model_path)
        self.bert = BertModel.from_pretrained(config.bert_model_path, config=model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size, config.num_classes)
        self.bn = nn.BatchNorm1d(config.num_classes)
        self.num_classes = config.num_classes

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward inputs and get logits.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.shape[0]
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # encoder_hidden_states=False
        )
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        # bert_output[1]: (batch_size, hidden_size)
        pooled_output = bert_output[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output).view(batch_size, self.num_classes)
        # logits = self.bn(logits)
        logits = nn.functional.softmax(logits, dim=-1)
        # logits: (batch_size, num_classes)
        return logits

class BertXForClassification(nn.Module):
    """BERT with simple linear model."""
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.bert_model_path: pretrained model path or model type
                    e.g. 'bert-base-chinese'
                config.hidden_size: The same as BERT model, usually 768
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        # data(b, 512, 768) -> conv(b, 511,767) -> bn -> mp(b, 4, 6)
        self.conv_module = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(128, 128), stride=(128, 128), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 255,255) -> bn -> mp(b, 4, 4)
        self.conv_module2 = nn.Sequential(
            nn.Conv2d(1,1, kernel_size=(2,3), stride=(2,3),padding=(0,0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(64, 64), stride=(64, 64),padding=(1,1))
        )
        # data(b, 512, 768) -> conv(b, 169, 192) -> bn -> mp(b, 5, 6)
        self.conv_module3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 4), stride=(3, 4), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(32, 32), stride=(32, 32), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 127, 127) -> bn -> mp(b, 4, 4)
        self.conv_module4 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(4, 6), stride=(4, 6), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(32, 32), stride=(32, 32), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 101, 108) -> bn -> mp(b, 6, 6)
        self.conv_module5 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(5, 7), stride=(5, 7), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(16, 16), stride=(16, 16), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 84, 84) -> bn -> mp(b, 5, 5)
        self.conv_module6 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(6, 9), stride=(6, 9), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(16, 16), stride=(16, 16), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 72, 75) -> bn -> mp(b, 9, 9)
        self.conv_module7 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(7, 10), stride=(7, 10), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8), padding=(1, 1))
        )

        #cnn feature map has a total number of 228 dimensions.
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size + 228 , config.num_classes)
        self.bn = nn.BatchNorm1d(config.num_classes)
        self.num_classes = config.num_classes

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward inputs and get logits.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.shape[0]
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # encoder_hidden_states=False
        )
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        encoded_output = bert_output[0]
        # encoded_output[0]: (batch_size, 1, sequence_length, hidden_size)
        encoded_output = encoded_output.view(batch_size, 1, encoded_output.shape[1], -1)
        cnn_feats = []
        cnn_feats.append(self.conv_module(encoded_output))
        cnn_feats.append(self.conv_module2(encoded_output))
        cnn_feats.append(self.conv_module3(encoded_output))
        cnn_feats.append(self.conv_module4(encoded_output))
        cnn_feats.append(self.conv_module5(encoded_output))
        cnn_feats.append(self.conv_module6(encoded_output))
        cnn_feats.append(self.conv_module7(encoded_output))

        for index in range(len(cnn_feats)):
            cnn_feats[index] = cnn_feats[index].reshape((batch_size, -1))
        con_cnn_feats = torch.cat(cnn_feats, dim=1)

        # bert_output[1]: (batch_size, hidden_size)
        pooled_output = bert_output[1]
        # 228 + 768 ->
        pooled_output = torch.cat([con_cnn_feats, pooled_output], dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output).view(batch_size, self.num_classes)
        logits = self.bn(logits)
        logits = nn.functional.softmax(logits, dim=-1)
        # logits: (batch_size, num_classes)
        return logits

class BertYForClassification(nn.Module):
    """BERT with simple linear model."""
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.bert_model_path: pretrained model path or model type
                    e.g. 'bert-base-chinese'
                config.hidden_size: The same as BERT model, usually 768
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        model_config = BertConfig.from_pretrained(config.bert_model_path)
        self.bert = BertModel.from_pretrained(config.bert_model_path, config=model_config)
        for param in self.bert.parameters():
            param.requires_grad = True

        # data(b, 512, 768) -> conv(b, 511,767) -> bn -> mp(b, 4, 6)
        self.conv_module = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(128, 128), stride=(128, 128), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 255,255) -> bn -> mp(b, 4, 4)
        self.conv_module2 = nn.Sequential(
            nn.Conv2d(1,1, kernel_size=(2,3), stride=(2,3),padding=(0,0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(64, 64), stride=(64, 64),padding=(1,1))
        )
        # data(b, 512, 768) -> conv(b, 169, 192) -> bn -> mp(b, 5, 6)
        self.conv_module3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 4), stride=(3, 4), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(32, 32), stride=(32, 32), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 127, 127) -> bn -> mp(b, 4, 4)
        self.conv_module4 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(4, 6), stride=(4, 6), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(32, 32), stride=(32, 32), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 101, 108) -> bn -> mp(b, 6, 6)
        self.conv_module5 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(5, 7), stride=(5, 7), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(16, 16), stride=(16, 16), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 84, 84) -> bn -> mp(b, 5, 5)
        self.conv_module6 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(6, 9), stride=(6, 9), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(16, 16), stride=(16, 16), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 72, 75) -> bn -> mp(b, 4, 4)
        self.conv_module7 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(7, 10), stride=(7, 10), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(16, 16), stride=(8, 8), padding=(1, 1))
        )
        self.conv_module8 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(8, 12), stride=(8, 12), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 255,255) -> bn -> mp(b, 4, 4)
        self.conv_module9 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(9, 13), stride=(9, 13), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 169, 192) -> bn -> mp(b, 5, 6)
        self.conv_moduleA = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(10, 15), stride=(10, 15), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 127, 127) -> bn -> mp(b, 4, 4)
        self.conv_moduleB = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(11, 16), stride=(11, 16), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 101, 108) -> bn -> mp(b, 6, 6)
        self.conv_moduleC = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(12, 18), stride=(12, 18), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 84, 84) -> bn -> mp(b, 5, 5)
        self.conv_moduleD = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(13, 19), stride=(13, 19), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        )
        # data(b, 512, 768) -> conv(b, 72, 75) -> bn -> mp(b, 4, 4)
        self.conv_moduleE = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(14, 21), stride=(14, 21), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        )

        #cnn feature map has a total number of 228 dimensions.
        self.att = MultiHeadAttention(d_model=config.hidden_size, heads=8)
        self.att2 = MultiHeadAttention(d_model=config.hidden_size, heads=8)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size + 1865, config.num_classes)  #689
        # self.bn = nn.BatchNorm1d(config.num_classes)
        self.num_classes = config.num_classes

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward inputs and get logits.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.shape[0]
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # encoder_hidden_states=False
        )
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        encoded_output = bert_output[0]
        # encoded_output[0]: (batch_size, 1, sequence_length, hidden_size)
        encoded_output = encoded_output.view(batch_size, 1, encoded_output.shape[1], -1)
        cnn_feats = []
        cnn_feats.append(self.conv_module(encoded_output))
        cnn_feats.append(self.conv_module2(encoded_output))
        cnn_feats.append(self.conv_module3(encoded_output))
        cnn_feats.append(self.conv_module4(encoded_output))
        cnn_feats.append(self.conv_module5(encoded_output))
        cnn_feats.append(self.conv_module6(encoded_output))
        cnn_feats.append(self.conv_module7(encoded_output))
        cnn_feats.append(self.conv_module8(encoded_output))
        cnn_feats.append(self.conv_module9(encoded_output))
        cnn_feats.append(self.conv_moduleA(encoded_output))
        cnn_feats.append(self.conv_moduleB(encoded_output))
        cnn_feats.append(self.conv_moduleC(encoded_output))
        cnn_feats.append(self.conv_moduleD(encoded_output))
        cnn_feats.append(self.conv_moduleE(encoded_output))
        for index in range(len(cnn_feats)):
            cnn_feats[index] = cnn_feats[index].reshape((batch_size, -1))
        con_cnn_feats = torch.cat(cnn_feats, dim=1)
        
        # bert_output[1]: (batch_size, hidden_size)
        pooled_output = bert_output[1]
        pooled_output = self.dropout(pooled_output)
        # pooled_output = self.att(pooled_output.unsqueeze(0), pooled_output.unsqueeze(0), pooled_output.unsqueeze(0)).squeeze(0)
        att_output = self.att(pooled_output.unsqueeze(0), pooled_output.unsqueeze(0), pooled_output.unsqueeze(0)).squeeze(0)
        pooled_output_1 = pooled_output + att_output
        pooled_output_1 = self.dropout(pooled_output_1)
        att_output2 = self.att2(pooled_output_1.unsqueeze(0), pooled_output_1.unsqueeze(0), pooled_output_1.unsqueeze(0)).squeeze(0)
        pooled_output_2 = pooled_output_1 + att_output2
        # pooled_output = bert_output[0][:,0,:]
        # # 228 + 768 ->  
        pooled_output_2 = torch.cat([con_cnn_feats, pooled_output_2], dim=1)
        pooled_output_2 = self.dropout(pooled_output_2)
        logits = self.linear(pooled_output_2).view(batch_size,self.num_classes)
        # logits = self.bn(logits)
        logits = nn.functional.softmax(logits, dim=-1)
        # logits: (batch_size, num_classes)
        return logits

class RnnForSentencePairClassification(nn.Module):
    """Unidirectional GRU model for sentences pair classification.
    2 sentences use the same encoder and concat to a linear model.
    """
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.vocab_size: vocab size
                config.hidden_size: RNN hidden size and embedding dim
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.rnn = nn.GRU(
            config.hidden_size, hidden_size=config.hidden_size,
            bidirectional=False, batch_first=True)
        self.linear = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_classes = config.num_classes

    def forward(self, s1_ids, s2_ids, s1_lengths, s2_lengths):
        """Forward inputs and get logits.

        Args:
            s1_ids: (batch_size, max_seq_len)
            s2_ids: (batch_size, max_seq_len)
            s1_lengths: (batch_size)
            s2_lengths: (batch_size)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = s1_ids.shape[0]
        # ids: (batch_size, max_seq_len)
        s1_embed = self.embedding(s1_ids)
        s2_embed = self.embedding(s2_ids)
        # embed: (batch_size, max_seq_len, hidden_size)
        s1_packed: PackedSequence = pack_padded_sequence(
            s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)
        s2_packed: PackedSequence = pack_padded_sequence(
            s2_embed, s2_lengths, batch_first=True, enforce_sorted=False)
        # packed: (sum(lengths), hidden_size)
        self.rnn.flatten_parameters()
        _, s1_hidden = self.rnn(s1_packed)
        _, s2_hidden = self.rnn(s2_packed)
        s1_hidden = s1_hidden.view(batch_size, -1)
        s2_hidden = s2_hidden.view(batch_size, -1)
        hidden = torch.cat([s1_hidden, s2_hidden], dim=-1)
        hidden = self.linear(hidden).view(-1, self.num_classes)
        hidden = self.dropout(hidden)
        logits = nn.functional.softmax(hidden, dim=-1)
        # logits: (batch_size, num_classes)
        return logits
