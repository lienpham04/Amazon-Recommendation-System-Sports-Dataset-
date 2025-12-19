# coding: utf-8
# @email: enoche.chow@gmail.com

r"""
################################
"""

import datetime
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt
import os

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model, mg=False):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.valid_results_dict = dict() 
        self.optimizer = self._build_optimizer()

        #fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']        # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.mg = mg
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.beta = config['beta']
        
        cur = datetime.datetime.now()
        cur = cur.strftime('%b-%d-%Y-%H-%M-%S')
        self.save_path = os.path.join('/Users/lienpham/Documents/NEU KÌ 7/Computer Vision/Final/MMRec/images', cur)
        os.makedirs(self.save_path, exist_ok=True)

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            second_inter = interaction.clone()
            losses = loss_func(interaction)
            
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)
            
            if self.mg and batch_idx % self.beta == 0:
                first_loss = self.alpha1 * loss
                first_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                losses = loss_func(second_inter)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                else:
                    loss = losses
                    
                if self._check_nan(loss):
                    self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                    return loss, torch.tensor(0.0)
                second_loss = -1 * self.alpha2 * loss
                second_loss.backward()
            else:
                loss.backward()
                
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
            # for test
            #if batch_idx == 0:
            #    break
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            #raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            #for param_group in self.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.valid_results_dict[epoch_idx] = valid_result
                
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                # test
                _, test_result = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                if update_flag:
                    # Save the best model at this epoch 
                    cur = datetime.datetime.now()
                    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')
                    # torch.save(self.model.state_dict(), f'{self.save_path}/best_model.pth')
                    t_final,v_final,t_feat,v_feat = self.model.print_embed()
                    embeddings_dict = {
                        't_final': t_final,
                        'v_final': v_final,
                        't_feat': t_feat,
                        'v_feat': v_feat
                    }
                    # torch.save(embeddings_dict, f'{self.save_path}/embeddings_best_epoch.pth')
        
                    
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        
        # ============================================================
        # FIXED: Generate all plots after training is complete
        # ============================================================
        if verbose:
            self.logger.info("=" * 60)
            self.logger.info("Generating plots...")
        
        try:
            # Plot individual metrics
            self.plot_train_loss(show=False, save=True, markersize=3)
            self.plot_recall(show=False, save=True, markersize=3)
            self.plot_ndcg(show=False, save=True, markersize=3)
            self.plot_precision(show=False, save=True, markersize=3)
            self.plot_map(show=False, save=True, markersize=3)
            
            # Plot all metrics in one dashboard
            self.plot_all(show=False, save=True)
            
            if verbose:
                self.logger.info(f"All plots saved to: {self.save_path}")
                self.logger.info("=" * 60)
        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
        
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, figsize = (15,12), save = True, rotation = 0, markersize = 1):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.figure(figsize=figsize)
        plt.plot(epochs, values, label = "Train loss", marker='o', markersize=markersize)
        step = max(1, len(epochs) // 20)  # hiện khoảng 50 nhãn
        plt.xticks(epochs[::step], rotation=rotation)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        if save:
            plt.savefig(f'{self.save_path}/train_loss.png')
        if show:
            plt.show()
        plt.close()
            
    # --- HELPER METHOD ---
    def _plot_metric_curve(self, metric_name, k_list, show=True, figsize=(15, 12), save=True, rotation=0, markersize = 1):
        """
        Helper function to plot specific metrics at multiple k values over epochs.
        """
        if not self.valid_results_dict:
            self.logger.warning(f"No validation results found to plot {metric_name}.")
            return

        epochs = list(self.valid_results_dict.keys())
        epochs.sort()
        
        plt.figure(figsize=figsize)
        
        for k in k_list:
            values = []
            target_key_lower = f'{metric_name.lower()}@{k}'
            
            for epoch in epochs:
                res = self.valid_results_dict[epoch]
                # Case-insensitive key lookup
                val = 0.0
                for key, v in res.items():
                    if key.lower() == target_key_lower:
                        val = v
                        break
                values.append(val)
            
            plt.plot(epochs, values, label=f'{metric_name}@{k}', marker='o', markersize=markersize)
        
        step = max(1, len(epochs) // 20)
        plt.xticks(epochs[::step], rotation=rotation)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Metrics over Epochs')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if save:
            plt.savefig(f'{self.save_path}/{metric_name.lower()}_metrics.png')
        if show:
            plt.show()
        plt.close()

    # --- PLOT METHODS ---

    def plot_recall(self, show=True, figsize=(15, 12), save=True, rotation=0, markersize=1):
        r"""Plot Recall@K for K in [5, 10, 20, 50]"""
        self._plot_metric_curve('Recall', [5, 10, 20, 50], show, figsize, save, rotation, markersize)

    def plot_ndcg(self, show=True, figsize=(15, 12), save=True, rotation=0, markersize=1):
        r"""Plot NDCG@K for K in [5, 10, 20, 50]"""
        self._plot_metric_curve('NDCG', [5, 10, 20, 50], show, figsize, save, rotation, markersize)

    def plot_precision(self, show=True, figsize=(15, 12), save=True, rotation=0, markersize=1):
        r"""Plot Precision@K for K in [5, 10, 20, 50]"""
        self._plot_metric_curve('Precision', [5, 10, 20, 50], show, figsize, save, rotation, markersize)

    def plot_map(self, show=True, figsize=(15, 12), save=True, rotation=0, markersize=1):
        r"""Plot MAP@K for K in [5, 10, 20, 50]"""
        self._plot_metric_curve('MAP', [5, 10, 20, 50], show, figsize, save, rotation, markersize)
        
    def plot_all(self, show=True, figsize=(20, 24), save=True):
        r"""
        Vẽ tất cả 5 metric (Train Loss, Recall, NDCG, Precision, MAP) 
        trên cùng 1 Figure lớn (Dashboard) với các subplots riêng biệt.
        """
        # Kiểm tra dữ liệu train
        if not self.train_loss_dict:
            self.logger.warning("No training data to plot.")
            return

        # Chuẩn bị dữ liệu Loss
        loss_epochs = list(self.train_loss_dict.keys())
        loss_epochs.sort()
        loss_values = [float(self.train_loss_dict[epoch]) for epoch in loss_epochs]
        
        # Chuẩn bị dữ liệu Valid
        if not self.valid_results_dict:
            self.logger.warning("No validation data to plot metrics.")
            valid_epochs = []
        else:
            valid_epochs = list(self.valid_results_dict.keys())
            valid_epochs.sort()
        
        # Tạo lưới 3 hàng, 2 cột (Kích thước lớn)
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # Tiêu đề chung cho cả ảnh lớn
        fig.suptitle(f'Training Dashboard - {self.config["model"]}', fontsize=24, fontweight='bold')
        
        # ---------------------------------------------------------
        # 1. Vẽ TRAIN LOSS tại vị trí (0, 0)
        # ---------------------------------------------------------
        ax_loss = axes[0, 0]
        ax_loss.plot(loss_epochs, loss_values, color='tab:red', linewidth=2, label='Train Loss')
        ax_loss.set_title('Training Loss', fontsize=16, fontweight='bold')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, linestyle='--', alpha=0.6)
        ax_loss.legend(loc='upper right')

        # ---------------------------------------------------------
        # 2. Vẽ các Metric đánh giá (Recall, NDCG, Precision, MAP)
        # ---------------------------------------------------------
        
        # Cấu hình: (Tên Metric, Vị trí Axes trên lưới)
        metrics_config = [
            ('Recall', axes[0, 1]),    # Hàng 0, Cột 1
            ('NDCG', axes[1, 0]),      # Hàng 1, Cột 0
            ('Precision', axes[1, 1]), # Hàng 1, Cột 1
            ('MAP', axes[2, 0])        # Hàng 2, Cột 0
        ]
        
        k_list = [5, 10, 20, 50] # Các mốc Top-K cần vẽ

        for metric_name, ax in metrics_config:
            if not valid_epochs:
                ax.text(0.5, 0.5, 'No Validation Data', ha='center', va='center', fontsize=12)
                continue
            
            # Duyệt qua từng K (5, 10, 20, 50)
            for k in k_list:
                values = []
                # Tạo key cần tìm, ví dụ: 'recall@5'
                target_key_lower = f'{metric_name.lower()}@{k}'
                
                for epoch in valid_epochs:
                    res = self.valid_results_dict[epoch]
                    val = 0.0
                    # Tìm key trong dictionary (không phân biệt hoa thường)
                    for key, v in res.items():
                        if key.lower() == target_key_lower:
                            val = v
                            break
                    values.append(val)
                
                # Vẽ đường cho K hiện tại
                ax.plot(valid_epochs, values, label=f'{metric_name}@{k}', marker='o', markersize=4)
            
            # Trang trí cho subplot hiện tại
            ax.set_title(f'{metric_name} Metrics', fontsize=16, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        # ---------------------------------------------------------
        # 3. Xử lý ô cuối cùng (Vị trí 2, 1) - Ẩn đi vì không dùng
        # ---------------------------------------------------------
        axes[2, 1].axis('off')
        
        # Căn chỉnh khoảng cách để không bị đè chữ
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
        
        # Lưu và Hiển thị
        if save:
            save_file = os.path.join(self.save_path, 'all_metrics_dashboard.png')
            plt.savefig(save_file)
            self.logger.info(f"Dashboard saved to {save_file}")
            
        if show:
            plt.show()
            
        plt.close()