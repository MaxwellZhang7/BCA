"""Training file for SMP-CAIL2020-Argmine.

Author: Tsinghuaboy tsinghua9boy@sina.com

Usage:
    python -m torch.distributed.launch train.py \
        --config_file 'config/bert_config.json'
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch train.py \
        --config_file 'config/rnn_config.json'
"""
#batch size:24->14 and max length:512-256 have negtive impact on acc, after 10 epochs drop to (train_acc: 0.490404, train_f1: 0.489924, valid_acc: 0.375000, valid_f1: 0.361930,)

#for bert 2 epochs
#train_acc: 0.873826, train_f1: 0.873886, valid_acc: 0.775000, valid_f1: 0.701429
#rnn 2 epochs
#train_acc: 0.552470, train_f1: 0.552839, valid_acc: 0.425000, valid_f1: 0.407884
#单attention效果目前最佳,18,无resnet结构
from typing import Dict
import argparse #解析命令行参数
import json
import os
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule)

from data import Data
from evaluate import evaluate, evaluatex, calculate_accuracy_f1, get_labels_from_file
from model import BertForClassification, RnnForSentencePairClassification, BertYForClassification
from utils import get_csv_logger, get_path
from vocab import build_vocab
from transformers import ElectraConfig,ElectraModel,ElectraTokenizer  
import numpy as np

torch.cuda.set_device(0)
MODEL_MAP = {
    'bert_origin':BertYForClassification,
    'bert': BertYForClassification,
    'rnn': RnnForSentencePairClassification
}

class MultiFocalLoss(nn.Module):
    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        # logit = F.softmax(input, dim=1)
        logit = input
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class Trainer:
    """Trainer for SMP-CAIL2020-Argmine.


    """
    def __init__(self,
                 model, data_loader: Dict[str, DataLoader], device, config):
        """Initialize trainer with model, data, device, and config.
        Initialize optimizer, scheduler, criterion.

        Args:
            model: model to be evaluated
            data_loader: dict of torch.utils.data.DataLoader
            device: torch.device('cuda') or torch.device('cpu')
            config:
                config.experiment_name: experiment name
                config.model_type: 'bert' or 'rnn'
                config.lr: learning rate for optimizer
                config.num_epoch: epoch number
                config.num_warmup_steps: warm-up steps number
                config.gradient_accumulation_steps: gradient accumulation steps
                config.max_grad_norm: max gradient norm

        """
        self.model = model
        self.device = device
        self.config = config
        self.data_loader = data_loader
        self.config.num_training_steps = config.num_epoch * (
            len(data_loader['train']) // config.batch_size)
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        # self.criterion = nn.CrossEntropyLoss()  
        self.criterion = MultiFocalLoss(num_class=2, smooth=0.2)

    def _get_optimizer(self):
        """Get optimizer for different models.

        Returns:
            optimizer
        """
        if self.config.model_type == 'bert_origin':
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_parameters = [
                {'params': [p for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.001},
                {'params': [p for n, p in self.model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}]
            optimizer = AdamW(
                optimizer_parameters,
                lr=self.config.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-8,
                correct_bias=False)
        else:  # rnn
            optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def _get_scheduler(self):
        """Get scheduler for different models.

        Returns:
            scheduler
        """
        if self.config.model_type == 'bert_origin':
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=self.config.num_training_steps)
        else:  # rnn
            scheduler = get_constant_schedule(self.optimizer)
        return scheduler

    def _evaluate_for_train_valid(self):
        """Evaluate model on train and valid set and get acc and f1 score.

        Returns:
            train_acc, train_f1, valid_acc, valid_f1
        """
        train_predictions = evaluatex(
            model=self.model, data_loader=self.data_loader['valid_train'],
            device=self.device)
        valid_predictions = evaluatex(
            model=self.model, data_loader=self.data_loader['valid_valid'],
            device=self.device)
        train_answers = get_labels_from_file(self.config.train_file_path)
        valid_answers = get_labels_from_file(self.config.valid_file_path)
        train_acc, train_f1 = calculate_accuracy_f1(
            train_answers, train_predictions)
        valid_acc, valid_f1 = calculate_accuracy_f1(
            valid_answers, valid_predictions)
        #with open("dkasjdklas.txt","a") as f:
        #    f.write(train_f1," ",valid_f1,"\n")
        return train_acc, train_f1, valid_acc, valid_f1

    def _epoch_evaluate_update_description_log(
            self, tqdm_obj, logger, epoch):
        """Evaluate model and update logs for epoch.

        Args:
            tqdm_obj: tqdm/trange object with description to be updated
            logger: logging.logger
            epoch: int

        Return:
            train_acc, train_f1, valid_acc, valid_f1
        """
        # Evaluate model for train and valid set
        results = self._evaluate_for_train_valid()
        train_acc, train_f1, valid_acc, valid_f1 = results
        # Update tqdm description for command line
        tqdm_obj.set_description(
            'Epoch: {:d}, train_acc: {:.6f}, train_f1: {:.6f}, '
            'valid_acc: {:.6f}, valid_f1: {:.6f}, '.format(
                epoch, train_acc, train_f1, valid_acc, valid_f1))
        # Logging
        logger.info(','.join([str(epoch)] + [str(s) for s in results]))
        return train_acc, train_f1, valid_acc, valid_f1

    def save_model(self, filename):
        """Save model to file.

        Args:
            filename: file name
        """
        torch.save(self.model.state_dict(), filename)

    def train(self):
        """Train model on train set and evaluate on train and valid set.

        Returns:
            state dict of the best model with highest valid f1 score
        """
        epoch_logger = get_csv_logger(
            os.path.join(self.config.log_path,
                         self.config.experiment_name + '-epoch.csv'),
            title='epoch,train_acc,train_f1,valid_acc,valid_f1')
        step_logger = get_csv_logger(
            os.path.join(self.config.log_path,
                         self.config.experiment_name + '-step.csv'),
            title='step,loss')
        trange_obj = trange(self.config.num_epoch, desc='Epoch', ncols=120)
        # self._epoch_evaluate_update_description_log(
        #     tqdm_obj=trange_obj, logger=epoch_logger, epoch=0)
        best_model_state_dict, best_train_f1, global_step = None, 0, 0
        for epoch, _ in enumerate(trange_obj):
            self.model.train()
            tqdm_obj = tqdm(self.data_loader['train'], ncols=80)
            for step, batch in enumerate(tqdm_obj):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    'input_ids':batch[0],
                    'attention_mask':batch[1],
                    'token_type_ids':batch[2],
                }
                logits = self.model(**inputs)  # the last one is label
                loss = self.criterion(logits, batch[-1])
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps
                # self.optimizer.zero_grad()
                loss.backward()  #更新模型的梯度，包括weight和bias

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm)
                    #after 梯度累加的基本思想在于，在优化器更新参数前，也就是执行 optimizer.step() 前，进行多次反向传播，是的梯度累计值自动保存在 parameter.grad 中，最后使用累加的梯度进行参数更新。
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    tqdm_obj.set_description('loss: {:.6f}'.format(loss.item()))
                    step_logger.info(str(global_step) + ',' + str(loss.item()))

                if epoch >= 7 and step % (self.config.gradient_accumulation_steps * 200) == 0:
                        results = self._epoch_evaluate_update_description_log(
                            tqdm_obj=trange_obj, logger=epoch_logger, epoch=epoch + 1)
                        self.save_model(os.path.join(
                            self.config.model_path, self.config.experiment_name,
                            self.config.model_type + '-' + str(epoch + 1) + '.bin'))

                        if results[-1] > best_train_f1:
                            best_model_state_dict = deepcopy(self.model.state_dict())
                            best_train_f1 = results[-1]
        return best_model_state_dict


def main(config_file='config/bert_config-l.json'):
    """Main method for training.

    Args:
        config_file: in config dir
    """
    # 0. Load config and mkdir
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    # get_path(os.path.join(config.model_path, config.experiment_name))
    # get_path(config.log_path)
    # if config.model_type == 'rnn':  # build vocab for rnn
    #     build_vocab(file_in=config.all_train_file_path,
    #                 file_out=os.path.join(config.model_path, 'vocab.txt'))
    # 1. Load data
    data = Data(vocab_file=os.path.join(config.bert_model_path, 'vocab.txt'),
                max_seq_len=config.max_seq_len,
                model_type=config.model_type, config=config)
    datasets = data.load_train_and_valid_files(
        train_file=config.train_file_path,
        valid_file=config.valid_file_path)
    train_set, valid_set_train, valid_set_valid = datasets
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # device = torch.device('cpu')
        # torch.distributed.init_process_group(backend="nccl")
        # sampler_train = DistributedSampler(train_set)
        sampler_train = RandomSampler(train_set)
    else:
        device = torch.device('cpu')
        sampler_train = RandomSampler(train_set)
    data_loader = {
        'train': DataLoader(
            train_set, shuffle=False, batch_size=config.batch_size),
        'valid_train': DataLoader(
            valid_set_train, batch_size=config.batch_size, shuffle=False),
        'valid_valid': DataLoader(
            valid_set_valid, batch_size=config.batch_size, shuffle=False)}
    # 2. Build model
    model = MODEL_MAP[config.model_type](config)
    model.to(device)
    if torch.cuda.is_available():
        model = model
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model, find_unused_parameters=True)
    # 3. Train
    trainer = Trainer(model=model, data_loader=data_loader,
                      device=device, config=config)
    best_model_state_dict = trainer.train()
    # 4. Save model
    torch.save(best_model_state_dict,
               os.path.join(config.model_path, 'model.bin'))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', default='config/bert_config.json',help='model config file')
    parser.add_argument('--local_rank', default=0,help='used for distributed parallel')
    args = parser.parse_args()
    config_file = 'config/bert_config.json'
    main(config_file)
