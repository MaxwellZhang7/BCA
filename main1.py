import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel
import jieba
sentence  = '上的飞机飞过酒店客房'


print(jieba.lcut(sentence))