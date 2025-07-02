from net_modules.hier_transforms import VSTransformer,LabTransformer,CXRCNN, TimeTransformer, MMFusionTransformer, PredTimeTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import  ARDSDataset
import pytorch_warmup as warmup
from torch.utils.tensorboard import SummaryWriter
import warnings
import random
from utils.misc import AverageMeter
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from thop import profile
from fvcore.nn import FlopCountAnalysis

warnings.filterwarnings('ignore')

class ModelConfig:
    def __init__(self, args):
        self.d_model = args.latent_dim
        self.base_encoder_layers = 2
        self.timer_encoder_layers = 4 if not args.use_mft else 2
        self.pos_type = args.pos_type
        self.output_dim = 2 if args.w_int > 0 else 1
        self.add_learnable_encoding = args.pos_type == 'learn_naive'
        self.device = 'cuda'
        
    def get_base_encoder_kwargs(self):
        return {
            'd_model': self.d_model,
            'num_encoder_layers': self.base_encoder_layers,
            'pos_type': self.pos_type
        }
    
    def get_cxr_encoder_kwargs(self):
        return {
            'd_model': self.d_model,
            'pos_type': self.pos_type
        }
    
    def get_timer_kwargs(self):
        return {
            'd_model': self.d_model,
            'num_encoder_layers': self.timer_encoder_layers,
            'use_learnable_pos': self.add_learnable_encoding
        }

class Coach:
    def __init__(self, args, is_train=True):
        self.args = args
        self.complete_m = [mod for mod in ['cxr', 'lab', 'vs'] 
                      if getattr(args, f'use_{mod}', False)]
        # if args.w_int > 0:
        #     output_dim = 2
        # else:
        #     output_dim = 1
        if is_train:
            self.train_dataset, self.val_dataset = self.configure_datasets()
        # add_learnable_encoding = False
        # if args.pos_type=='learn_naive':
        #     add_learnable_encoding = True

        # self.labencoder = LabTransformer(d_model=args.latent_dim,num_encoder_layers=2, pos_type=args.pos_type).cuda()
        # self.vsencoder = VSTransformer(d_model=args.latent_dim,num_encoder_layers=2, pos_type=args.pos_type).cuda()
        # self.cxrencoder = CXRCNN(d_model=args.latent_dim, model_type=args.cnn_name, pos_type=args.pos_type).cuda()
        # if args.use_mft:
        #     self.labtimer = TimeTransformer(d_model=args.latent_dim,num_encoder_layers=2, use_learnable_pos=add_learnable_encoding).cuda()
        #     self.vstimer = TimeTransformer(d_model=args.latent_dim,num_encoder_layers=2, use_learnable_pos=add_learnable_encoding).cuda()
        #     self.cxrtimer = TimeTransformer(d_model=args.latent_dim,num_encoder_layers=2, use_learnable_pos=add_learnable_encoding).cuda()
        #     self.fusion_model = MMFusionTransformer(d_model=args.latent_dim,num_encoder_layers=2, output_dim=output_dim).cuda()
        # else:
        #     self.timer = TimeTransformer(d_model=args.latent_dim,num_encoder_layers=4, output_dim=output_dim, use_learnable_pos=add_learnable_encoding).cuda()
        # if args.use_ptt:
        #     self.pred_fusion_model = PredTimeTransformer(d_model=args.latent_dim,num_encoder_layers=2, output_dim=output_dim).cuda()
        self._create_models()
        print('model initialized !!')
        self.is_test_compare_mod = False

    def _create_models(self):
        """统一的模型创建方法"""
        config = ModelConfig(self.args)
        
        # 基础编码器
        self.encoders = nn.ModuleDict({
            'lab': LabTransformer(**config.get_base_encoder_kwargs()),
            'vs': VSTransformer(**config.get_base_encoder_kwargs()),
            'cxr': CXRCNN(**config.get_cxr_encoder_kwargs(), model_type=self.args.cnn_name)
        }).to(config.device)
        
        # 时间模型
        if self.args.use_mft:
            self.timers = nn.ModuleDict({
                'lab': TimeTransformer(**config.get_timer_kwargs()),
                'vs': TimeTransformer(**config.get_timer_kwargs()),
                'cxr': TimeTransformer(**config.get_timer_kwargs())
            }).to(config.device)
            
            self.fusion_model = MMFusionTransformer(
                d_model=config.d_model,
                num_encoder_layers=config.base_encoder_layers,
                output_dim=config.output_dim
            ).to(config.device)
        else:
            self.timer = TimeTransformer(
                **config.get_timer_kwargs(),
                output_dim=config.output_dim
            ).to(config.device)
        
        # 可选模型
        if self.args.use_ptt:
            self.pred_fusion_model = PredTimeTransformer(
                d_model=config.d_model,
                num_encoder_layers=config.base_encoder_layers,
                output_dim=config.output_dim
            ).to(config.device)
        
    def balanced_sampler(self, dataset, batch_size):
        pos_indices = dataset.pos_indices
        neg_indices = dataset.neg_indices
        while True:
            pos_samples = random.sample(pos_indices, batch_size // 2)
            neg_samples = random.sample(neg_indices, batch_size // 2)
            indices = pos_samples + neg_samples
            random.shuffle(indices)
            for idx in indices:
                yield idx
    def random_sampler(self, dataset):
        while True:
            indices = [i for i, label in enumerate(dataset.all_labels)]
            random.shuffle(indices)
            for idx in indices:
                yield idx

    def _get_all_models(self):
        """获取所有模型的字典"""
        models = {}
        
        # 基础编码器
        for name, model in self.encoders.items():
            models[f'{name}encoder'] = model
        
        # 时间模型
        if self.args.use_mft:
            for name, model in self.timers.items():
                models[f'{name}timer'] = model
            models['fusion_model'] = self.fusion_model
        else:
            models['timer'] = self.timer
        
        # 可选模型
        if self.args.use_ptt:
            models['pred_fusion_model'] = self.pred_fusion_model
        
        return models

    def save_checkpoint(self, is_best=False):
        """简化的保存检查点方法"""
        models = self._get_all_models()
        save_dict = {name: model.state_dict() for name, model in models.items()}
        save_dict.update({
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch
        })
        
        save_path = f'{self.args.save_dir}/{"best" if is_best else self.current_epoch}.pth'
        torch.save(save_dict, save_path)
        
        if is_best:
            with open(f'{self.args.save_dir}/best.txt', 'a') as f:
                f.write(f'best epoch: {self.current_epoch}, auroc: {self.auroc_mean}, mae: {self.pred_error_mean}\n')
        
        print(f"Model saved at {'best' if is_best else f'epoch {self.current_epoch}'}")

    def load_checkpoint(self, is_train=True):
        """简化的加载检查点方法"""
        ckpt = torch.load(self.args.ckpt)
        models = self._get_all_models()
        
        for name, model in models.items():
            if name in ckpt:
                model.load_state_dict(ckpt[name])
        
        if is_train:
            self.start_epoch = ckpt['epoch'] + 1
            self.optimizer.load_state_dict(ckpt['optimizer'])

    def configure_optimizers(self):
        """简化的优化器配置方法"""
        models = self._get_all_models()
        param_list = [{'params': model.parameters()} for model in models.values()]
        
        self.optimizer = optim.AdamW(
            param_list, 
            lr=self.args.lr,
            betas=(0.9, 0.999), 
            weight_decay=0.0001
        )
        
        num_train_data = len(self.train_dataset)
        num_steps = num_train_data * self.args.total_epoch
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_steps * self.args.num_interval
        )
        self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)
    
    def _init_train(self):
        """初始化训练相关组件"""
        self.configure_optimizers()
        self.writer = SummaryWriter(log_dir=self.args.save_dir)
        self.start_epoch = 1
        if self.args.ckpt is not None:
            self.load_checkpoint(is_train=True)
        self.criterion = nn.BCEWithLogitsLoss()
        if self.args.w_int > 0:
            self.mseloss = nn.MSELoss()

        self.best_f1 = -10.0
        self.loss_keys = ['cls']
        if self.args.w_int > 0:
            self.loss_keys.append('int')
        self.loss_log_dict = {k: {} for k in self.loss_keys}

    def _forward(self,fi):
        # fidx == 0 indicates missing
        mm_data = []  # [Bx3xD, ...]
        mm_mask = []  # [B,B,..]
        if not self.args.use_fagg:
            agg = False
        else:
            agg = True


        for mod in self.inputs_dict.keys():
            if self.args.use_ptt:
                bool_index = (self.inputs_dict[mod]['fidx'] == fi) # >=
            else:
                bool_index = (self.inputs_dict[mod]['fidx'] >= fi-1)
            if not bool_index.any().item(): # no valid data at pred point fi
                mm_mask.append(True)
                continue

            mm_mask.append(False)
            relative_timestamps = self.inputs_dict[mod]['relative_timestamps'][bool_index]

            if mod == 'vs':
                input_cont = self.inputs_dict[mod]['data_cont'][bool_index]
                input_cat = self.inputs_dict[mod]['data_cat'][bool_index]
                mask_cont = self.inputs_dict[mod]['mask_cont'][bool_index]
                mask_cat = self.inputs_dict[mod]['mask_cat'][bool_index]

                # Calculate FLOPs for vsencoder
                if self.args.cal_flops:
                    with torch.no_grad():
                        flops = FlopCountAnalysis(self.encoders['vs'],
                                                  (input_cont, input_cat, mask_cont, mask_cat, agg, relative_timestamps))
                        self.total_flops += flops.total()

                f = self.encoders['vs'](input_cont, input_cat, mask_cont,
                                   mask_cat, agg=agg, time_stamps=relative_timestamps)  # T,num_v (B=T) mask: T,num_v --> T,D
            elif mod == 'lab':
                mask = self.inputs_dict[mod]['mask'][bool_index]
                if self.args.cal_flops:
                    with torch.no_grad():
                        flops = FlopCountAnalysis(self.encoders['lab'],
                                                  (self.inputs_dict[mod]['data'][bool_index], mask, agg, relative_timestamps))
                        self.total_flops += flops.total()

                f = self.encoders['lab'](self.inputs_dict[mod]['data'][bool_index], mask, agg=agg, time_stamps=relative_timestamps)  # T,num_v (B=T) mask: T,num_v --> T,D
            elif mod == 'cxr':
                if self.args.cal_flops:
                    flops, _ = profile(self.encoders['cxr'],
                                       inputs=(self.inputs_dict[mod]['data'][bool_index],relative_timestamps),
                                       verbose=False)
                    self.total_flops += flops

                f = self.encoders['cxr'](self.inputs_dict[mod]['data'][bool_index], time_stamps=relative_timestamps)
            else:
                raise NotImplementedError
            if not self.args.use_fagg:
                f = f.reshape(-1, f.shape[-1])
            if self.args.cal_flops:
                with torch.no_grad():
                    flops = FlopCountAnalysis(self.timers[mod],
                                              (f.unsqueeze(0),),
                                              verbose=False)
                    self.total_flops += flops.total()

            ft = self.timers[mod](f.unsqueeze(0))  # B,padded_T,D - > B, D
            mm_data.append(ft)
        mm_mask = torch.tensor(mm_mask, dtype=torch.bool).cuda()
        if self.is_test_compare_mod and torch.any(mm_mask, dim=0):
            return None, False

        all_missing_mask = torch.all(mm_mask, dim=0) # True = missing
        if all_missing_mask:
            return None, False
        # fusion
        mm_data = torch.stack(mm_data, dim=0).cuda()
        if self.args.use_ptt:
            if self.args.cal_flops:
                with torch.no_grad():
                    flops = FlopCountAnalysis(self.fusion_model,
                                              (mm_data,),
                                              verbose=False)
                    self.total_flops += flops.total()
            fi_feature = self.fusion_model(mm_data)
            return fi_feature, ~all_missing_mask
        else:
            if self.args.cal_flops:
                with torch.no_grad():
                    flops = FlopCountAnalysis(self.timer,
                                              (mm_data,),
                                              verbose=False)
                    self.total_flops += flops.total()
            logits = self.fusion_model(mm_data, output_logits=True)[0]
            return logits, ~all_missing_mask

    def getPositionEncoding(self, specified_ks, d=128, n=10000):
            '''
            seq_len should be the relative time in this interval
            '''
            seq_len = len(specified_ks)
            P = torch.zeros((seq_len, d)).cuda()
            for j, k in enumerate(specified_ks):
                for i in range(int(d/2)):
                    denominator = torch.pow(torch.tensor(n).cuda(), 2*i/d)
                    P[j, 2*i] = torch.sin(torch.tensor(k).cuda()/denominator)
                    P[j, 2*i+1] = torch.cos(torch.tensor(k).cuda()/denominator)
            return P
    def getPositionEncoding_naive(self, seq_len, d=128, n=10000):
            '''
            seq_len should be the relative time in this interval
            '''

            P = torch.zeros((seq_len, d)).cuda()
            for k in range(seq_len):
                for i in range(int(d/2)):
                    denominator = torch.pow(torch.tensor(n).cuda(), 2*i/d)
                    P[k, 2*i] = torch.sin(torch.tensor(k).cuda()/denominator)
                    P[k, 2*i+1] = torch.cos(torch.tensor(k).cuda()/denominator)
            return P

    def _forward_no_mft(self,fi):
        # fidx == 0 indicates missing
        mm_data = []  # [Bx3xD, ...]
        mm_mask = []  # [B,B,..]
        if not self.args.use_fagg:
            agg = False
        else:
            agg = True
        for mod in self.inputs_dict.keys():
            if self.args.use_ptt:
                bool_index = (self.inputs_dict[mod]['fidx'] == fi) # >=
            else:
                bool_index = (self.inputs_dict[mod]['fidx'] >= fi-1)
            if not bool_index.any().item(): # no valid data at pred point fi
                mm_mask.append(True)
                # print(f'missing mod: {mod}')
                continue
            relative_timestamps = self.inputs_dict[mod]['relative_timestamps'][bool_index]

            mm_mask.append(False)
            if mod == 'vs':
                input_cont = self.inputs_dict[mod]['data_cont'][bool_index]
                input_cat = self.inputs_dict[mod]['data_cat'][bool_index]
                mask_cont = self.inputs_dict[mod]['mask_cont'][bool_index]
                mask_cat = self.inputs_dict[mod]['mask_cat'][bool_index]
                if self.args.cal_flops:
                    with torch.no_grad():
                        flops = FlopCountAnalysis(self.encoders['vs'],
                                                  (input_cont, input_cat, mask_cont, mask_cat, agg, relative_timestamps))
                        self.total_flops += flops.total()
                f = self.encoders['vs'](input_cont, input_cat, mask_cont,
                                   mask_cat,agg=agg, time_stamps=relative_timestamps)  # T,num_v (B=T) mask: T,num_v --> T,D
            elif mod == 'lab':
                mask = self.inputs_dict[mod]['mask'][bool_index]
                if self.args.cal_flops:

                    with torch.no_grad():
                        flops = FlopCountAnalysis(self.encoders['lab'],
                                                  (self.inputs_dict[mod]['data'][bool_index], mask, agg, relative_timestamps))
                    # print('lab flops:',flops.total())
                    self.total_flops += flops.total()
                f = self.encoders['lab'](self.inputs_dict[mod]['data'][bool_index], mask, agg=agg, time_stamps=relative_timestamps)
            elif mod == 'cxr':
                if self.args.cal_flops:
                    flops, _ = profile(self.encoders['cxr'],
                                       inputs=(self.inputs_dict[mod]['data'][bool_index],relative_timestamps),
                                       verbose=False)
                    # print(f'cxr flops: {flops}')
                    self.total_flops += flops
                f = self.encoders['cxr'](self.inputs_dict[mod]['data'][bool_index], time_stamps=relative_timestamps)
            else:
                raise NotImplementedError
            if not self.args.use_fagg:
                f = f.reshape(-1, f.shape[-1])

            '''
            for f (T,D), add modality encoding (1,D), and time encoding (T,D)
            '''

            mm_data.append(f) # B,T,D

        mm_mask = torch.tensor(mm_mask, dtype=torch.bool).cuda()
        all_missing_mask = torch.all(mm_mask, dim=0) # True = missing
        if all_missing_mask:
            return None, ~all_missing_mask
        # fusion
        mm_data = torch.concat(mm_data, dim=0).cuda()
        if self.args.use_ptt:
            if self.args.cal_flops:
                with torch.no_grad():
                    flops = FlopCountAnalysis(self.fusion_model,
                                              (mm_data,),
                                            )
                    # print(f'fusion flops:',flops.total())
                    self.total_flops += flops.total()
            fi_feature = self.timer(mm_data.unsqueeze(0))
            return fi_feature, ~all_missing_mask
        else:
            if self.args.cal_flops:
                with torch.no_grad():

                    flops = FlopCountAnalysis(self.timer,
                                              (mm_data.unsqueeze(0)),
                                              )
                # print(f'timer flops: ',flops.total())
                self.total_flops += flops.total()
            logits = self.timer(mm_data.unsqueeze(0), output_logits=True)[0]
            return logits, ~all_missing_mask

    def train(self):
        self._init_train()

        for epoch in range(self.start_epoch, self.args.total_epoch + 1):
            if self.args.use_bsample:
                sampler = self.balanced_sampler(self.train_dataset, self.args.batch_size)
            else:
                sampler = self.random_sampler(self.train_dataset)

            self.current_epoch = epoch
            for encoder in self.encoders.values():
                encoder.train()
            if self.args.use_mft:
                for timer in self.timers.values():
                    timer.train()
                self.fusion_model.train()
            else:
                self.timer.train()
            if self.args.use_ptt:
                self.pred_fusion_model.train()
            self.optimizer.zero_grad()

            avg_loss_dict = {}
            for lk in self.loss_keys:
                avg_loss_dict[lk] = AverageMeter()

            accumulated_steps = 0
            # 原本是每个batch下每个fidx预测点update一次，现在是每个batch下所有fidx累积才update一次
            # for step in range((len(self.train_dataset) // self.args.batch_size)+1):
            for step in range(50):
                for _ in range(self.args.batch_size):
                    idx = next(sampler)
                    self.inputs_dict, self.labels = self.train_dataset[idx]
                    if len(self.inputs_dict) == 0:
                        # print('empty input for ards patient')
                        # will not cause accumulation for update
                        continue
                    self.labels = self.labels.unsqueeze(0).cuda()
                    for i in self.inputs_dict.keys():
                        for j in self.inputs_dict[i].keys():
                            self.inputs_dict[i][j] = self.inputs_dict[i][j].cuda()

                    max_fidx = max([torch.max(self.inputs_dict[i]['fidx']).item() for i in self.inputs_dict.keys()])

                    # num_train_intervals = min(self.args.num_pred_interval, int(max_fidx))
                    num_train_intervals = int(max_fidx)
                    # for each sample, set buffer for each previous fi
                    previous_fis = []
                    for ifi in range(1, num_train_intervals + 1): # from 1 to 8
                        fi = num_train_intervals + 1 - ifi # from 8 to 1
                        # obtain feature for single fi
                        if self.args.use_ptt:
                            if self.args.use_mft:
                                fifeature, am_mask = self._forward(fi)
                            else:
                                fifeature, am_mask = self._forward_no_mft(fi)  # am_mask: B , False means all mods missing
                            if not am_mask:
                                continue
                            all_features = torch.stack([fifeature] + previous_fis, dim=1)
                            logits = self.pred_fusion_model(all_features)[0]
                            previous_fis.append(fifeature)
                        else:
                            if self.args.use_mft:
                                logits, am_mask = self._forward(fi)
                            else:
                                logits, am_mask = self._forward_no_mft(fi)

                        loss = torch.tensor(0.0).cuda()
                        if self.args.w_int > 0:
                            pred_int = logits[1]
                            logits = logits[0].unsqueeze(0)
                            if self.labels[0] == 1:  # only for ards patient
                                label_int = self._get_label_int(fi)
                            # else:
                            #     label_int = torch.tensor(0.0).cuda()
                                loss_int = self.mseloss(pred_int, label_int)
                                avg_loss_dict['int'].update(loss_int.item())
                                loss += self.args.w_int * loss_int

                        if self.args.w_cls > 0:
                            loss_cls = self.criterion(logits, self.labels)
                            avg_loss_dict['cls'].update(loss_cls.item())
                            loss += self.args.w_cls * loss_cls


                        loss.backward(retain_graph=True)

                    accumulated_steps += 1

                    # 如果达到 batch size，更新梯度
                    if accumulated_steps == self.args.batch_size:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        accumulated_steps = 0
                ####
                if step % 20 == 0:
                    for lk in self.loss_keys:
                        l = avg_loss_dict[lk].avg
                        print(
                        f'train iter: {step}, {lk}_loss: {l}')
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()

            for lk in self.loss_keys:
                mloss = avg_loss_dict[lk].avg
                self.loss_log_dict[lk]['train'] = mloss
                print(f'{lk} loss of epoch {epoch}  is {mloss}')

            self.validate()

            # write loss for both train and val
            for lk in self.loss_keys:
                self.writer.add_scalars(f'Loss_{lk}', self.loss_log_dict[lk], epoch)
        print('training finished')
        self.args.ckpt = f'{self.args.save_dir}/best.pth'

    def validate(self):
        auroc_log_dict = {}
        with torch.no_grad():
            fi_labels = [[] for _ in range(self.args.num_interval)]
            fi_probs = [[] for _ in range(self.args.num_interval)]
            all_labels = []
            all_probs = []
            if self.args.w_int > 0:
                all_labels_int = []
                all_preds_int = []
                # fi_labels_int = [[] for _ in range(self.args.num_interval)]
                # fi_preds_int = [[] for _ in range(self.args.num_interval)]
                level_preds = {i:[] for i in range(1,5)}
                pred_error_log_dict = {}

            avg_loss_dict = {}
            for lk in self.loss_keys:
                avg_loss_dict[lk] = AverageMeter()
            print('validating...')
            for idx in tqdm(range(len(self.val_dataset))):
                self.inputs_dict, self.labels = self.val_dataset[idx]
                self.labels = self.labels.unsqueeze(0).cuda()
                if len(self.inputs_dict) == 0:
                #     print('empty input for ards patient')
                #     # will not cause accumulation for update
                    continue
                for i in self.inputs_dict.keys():
                    for j in self.inputs_dict[i].keys():
                        self.inputs_dict[i][j] = self.inputs_dict[i][j].cuda()
                # for each mod 按照 fidx 1~k 分k组来foward
                max_fidx = int(max([torch.max(self.inputs_dict[i]['fidx']) for i in self.inputs_dict.keys()]))
                previous_fis = []
                for ifi in range(1, max_fidx + 1):  # from 1 to 8
                    fi = max_fidx + 1 - ifi  # from 8 to 1
                    # obtain feature for single fi
                    if self.args.use_ptt:
                        if self.args.use_mft:
                            fifeature, am_mask = self._forward(fi)
                        else:
                            fifeature, am_mask = self._forward_no_mft(fi)
                        if not am_mask:
                            continue
                        all_features = torch.stack([fifeature] + previous_fis, dim=1)
                        logits = self.pred_fusion_model(all_features)[0]
                        previous_fis.append(fifeature)
                    else:
                        if self.args.use_mft:
                            logits, am_mask = self._forward(fi)
                        else:
                            logits, am_mask = self._forward_no_mft(fi)


                    loss = torch.tensor(0.0).cuda()
                    if self.args.w_int > 0:

                        pred_int = logits[1]
                        logits = logits[0].unsqueeze(0)
                        if self.labels[0] == 1:  # only for ards patient
                            label_int = self._get_label_int(fi)
                        # else:
                        #     label_int = torch.tensor(0).cuda()
                            loss_int = self.mseloss(pred_int, label_int)
                            avg_loss_dict['int'].update(loss_int.item())
                            loss += self.args.w_int * loss_int

                            all_labels_int.append(label_int.cpu().numpy())
                            all_preds_int.append(pred_int.cpu().numpy())
                            level_preds[int(label_int.cpu().numpy())].append(pred_int.cpu().numpy())
                            # fi_labels_int[fi - 1].append(label_int.cpu().numpy())
                            # fi_preds_int[fi - 1].append(pred_int.cpu().numpy())

                    if self.args.w_cls > 0:
                        loss_vector = self.criterion(logits, self.labels)
                        masked_loss = loss_vector * am_mask
                        loss_cls = masked_loss.sum() / am_mask.sum()
                        avg_loss_dict['cls'].update(loss_cls.item())
                        loss += self.args.w_cls * loss_cls



                    probs = torch.sigmoid(logits)
                    fi_labels[fi - 1].append(self.labels.cpu().numpy())
                    fi_probs[fi - 1].append(probs.cpu().numpy())
                    all_labels.append(self.labels.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())


            for lk in self.loss_keys:
                mloss = avg_loss_dict[lk].avg
                self.loss_log_dict[lk]['val'] = mloss
            print('val loss:', avg_loss_dict['cls'].avg)

            if self.args.w_int > 0:


                pred_errors =  []
                all_labels_int_ = np.array(all_labels_int)
                all_preds_int_ = np.array(all_preds_int)
                all_pred_error = np.mean(np.abs(all_labels_int_ - all_preds_int_))
                pred_error_log_dict[f'val_all'] = all_pred_error
                print(f"all pred points, pred error: {all_pred_error:.4f}")
                # for fidx in range(self.args.num_interval):
                #     fi_labels_int_ = np.array(fi_labels_int[fidx])
                #     fi_preds_int_ = np.array(fi_preds_int[fidx])
                #     pred_error = np.mean(np.abs(fi_labels_int_ - fi_preds_int_))
                #     pred_errors.append(pred_error)
                #     pred_error_log_dict[f'val_{fidx + 1}'] = pred_error
                #     print(f"pred idx: {fidx + 1}, pred error: {pred_error:.4f}")
                # self.writer.add_scalars('pred_error', pred_error_log_dict, self.current_epoch)

                for level in range(1,5):
                    level_preds_ = np.array(level_preds[level])
                    level_pred_error = np.mean(np.abs(level_preds_ - level))
                    pred_error_log_dict[f'val_{level}'] = level_pred_error
                    print(f"level: {level}, pred error: {level_pred_error:.4f}")
                    pred_errors.append(level_pred_error)
                self.writer.add_scalars('pred_error', pred_error_log_dict, self.current_epoch)


            aurocs = []
            for fidx in range(self.args.num_interval):
                fi_labels_ = np.array(fi_labels[fidx])
                fi_probs_ = np.array(fi_probs[fidx])
                auroc = roc_auc_score(fi_labels_, fi_probs_)
                aurocs.append(auroc)
                print(f"pred idx: {fidx + 1}, AUROC: {auroc:.4f}")
                auroc_log_dict[f'val_{fidx + 1}'] = auroc
            all_labels_ = np.array(all_labels)
            all_probs_ = np.array(all_probs)
            auroc_all = roc_auc_score(all_labels_, all_probs_)
            auroc_log_dict[f'val_all'] = auroc_all
            print(f"all pred points, AUROC: {auroc_all:.4f}")
            self.writer.add_scalars('aurocs', auroc_log_dict, self.current_epoch)


            self.auroc_mean = np.mean(aurocs[:8])
            self.pred_error_mean = np.mean(pred_errors)
            indicator = self.auroc_mean - self.pred_error_mean
            if indicator > self.best_f1:
                self.best_f1 = indicator
                best_epoch = self.current_epoch
                print(f'best indicator:{indicator}, at epoch: {best_epoch}, auroc: {self.auroc_mean}, pred_error: {self.pred_error_mean}')
                self.save_checkpoint(is_best=True)
        if self.current_epoch % self.args.save_interval == 0:
            self.save_checkpoint()



    def test(self, is_compare_mod=False, return_results=False):
        # todo: write other metrics: F1, sensitivity, specificity, accuracy, precision, recall
        if self.args.ckpt is None:
            raise ValueError('no checkpoint to test!!!')
        self.load_checkpoint(is_train=False)
        if is_compare_mod:
            self.is_test_compare_mod = True
        test_dataset = self.configure_test_dataset()

        metric_names = ['auroc', 'mae', 'mse', 'accuracy', 'ppv', 'npv', 'sensitivity', 'specificity', 'mae_d', 'f1']
        repeated_results = {k: [[] for _ in range(self.args.num_interval)] for k in metric_names}
        repeated_results_all = {k: [] for k in metric_names}
        for i in range(self.args.num_test_repeat):
            with torch.no_grad():

                fi_labels = [[] for _ in range(self.args.num_interval)]
                fi_probs = [[] for _ in range(self.args.num_interval)]
                all_labels = []
                all_probs = []
                if self.args.w_int > 0:
                    all_labels_int = []
                    all_preds_int = []
                    # fi_labels_int = [[] for _ in range(self.args.num_interval)]
                    # fi_preds_int = [[] for _ in range(self.args.num_interval)]
                    level_preds = {i: [] for i in range(1, 5)}
                print('testing...')
                for idx in tqdm(range(len(test_dataset))):
                    # one patient
                    self.inputs_dict, self.labels = test_dataset[idx]
                    self.labels = self.labels.unsqueeze(0).cuda()
                    if len(self.inputs_dict) == 0:
                        # print('empty input for ards patient')
                        # will not cause accumulation for update
                        continue
                    for i in self.inputs_dict.keys():
                        for j in self.inputs_dict[i].keys():
                            self.inputs_dict[i][j] = self.inputs_dict[i][j].cuda()
                    # for each mod 按照 fidx 1~k 分k组来foward

                    max_fidx = int(max([torch.max(self.inputs_dict[i]['fidx']) for i in self.inputs_dict.keys()]))
                    previous_fis = []
                    # for each pred point of one patient
                    for ifi in range(1, max_fidx + 1):  # from 1 to 8
                        fi = max_fidx + 1 - ifi  # from 8 to 1
                        # obtain feature for single fi
                        if self.args.use_ptt:
                            if self.args.use_mft:
                                fifeature, am_mask = self._forward(fi)
                            else:
                                fifeature, am_mask = self._forward_no_mft(fi)
                            if not am_mask:
                                continue
                            all_features = torch.stack([fifeature] + previous_fis, dim=1)
                            logits = self.pred_fusion_model(all_features)[0]
                            previous_fis.append(fifeature)
                        else:
                            if self.args.use_mft:
                                logits, am_mask = self._forward(fi)
                            else:
                                logits, am_mask = self._forward_no_mft(fi)

                        if self.args.w_int > 0:
                            pred_int = logits[1]
                            logits = logits[0].unsqueeze(0)
                            # if self.labels[0] == 1:  # only for ards patient
                            # if torch.sigmoid(logits) > 0.5:  # only for positive preds
                            if self.labels[0] == 1:
                                label_int = self._get_label_int(fi, to_tensor=False)

                                # test only calculate for ards patient
                                all_labels_int.append(label_int)
                                all_preds_int.append(pred_int.cpu().numpy())
                                level_preds[int(label_int)].append(pred_int.cpu().numpy())
                            # fi_labels_int[fi - 1].append(label_int)
                            # fi_preds_int[fi - 1].append(pred_int.cpu().numpy())
                        probs = torch.sigmoid(logits)
                        fi_labels[fi - 1].append(self.labels.cpu().numpy())
                        fi_probs[fi - 1].append(probs.cpu().numpy())
                        all_labels.append(self.labels.cpu().numpy())
                        all_probs.append(probs.cpu().numpy())
                #######################################################################################
                # pred error, mae & mse
                if self.args.w_int > 0:
                    all_labels_int_ = np.array(all_labels_int)
                    all_preds_int_ = np.array(all_preds_int)
                    all_mae = np.mean(np.abs(all_labels_int_ - all_preds_int_))
                    all_mse = np.mean((all_labels_int_ - all_preds_int_) ** 2)
                    all_mae_d = np.mean(np.abs(all_labels_int_ - np.round(all_preds_int_)))
                    repeated_results_all['mae'].append(all_mae)
                    repeated_results_all['mse'].append(all_mse)
                    repeated_results_all['mae_d'].append(all_mae_d)
                    print(f"all pred points, mae: {all_mae:.4f}, mse: {all_mse:.4f}, mae_d: {all_mae_d:.4f}")
                    # different_levels = {i: [] for i in range(5)}
                    # for fidx in range(self.args.num_interval):
                    #     fi_labels_int_ = np.array(fi_labels_int[fidx]) * 4.0
                    #     fi_preds_int_ = np.array(fi_preds_int[fidx]) * 4.0
                    #     level = self._get_label_int(fidx + 1, to_tensor=False) * 4
                    #     different_levels[int(level)]+= fi_preds_int_[fi_labels_int_ == level].tolist()
                    #     different_levels[0] += fi_preds_int_[fi_labels_int_ == 0].tolist()

                    # for level, preds in different_levels.items():
                    for level, preds in level_preds.items():
                        # if empty, continue
                        if len(preds) == 0:
                            continue
                        labels_int_ = np.array([level] * len(preds))
                        preds_int_ = np.array(preds)
                        mae = np.mean(np.abs(labels_int_ - preds_int_))
                        mse = np.mean((labels_int_ - preds_int_) ** 2)
                        mae_d = np.mean(np.abs(labels_int_ - np.round(preds_int_)))
                        repeated_results['mae'][level].append(mae)
                        repeated_results['mse'][level].append(mse)
                        repeated_results['mae_d'][level].append(mae_d)
                        print(f"pred level: {level}, mae: {mae:.4f},  mse: {mse:.4f}, mae_d: {mae_d:.4f}")

                # auroc, acc, ppv, npv, sensitivity, specificity
                threshold = 0.7
                for fidx in range(self.args.num_interval):
                    if len(fi_labels[fidx]) == 0:
                        continue
                    fi_labels_ = np.array(fi_labels[fidx])
                    fi_probs_ = np.array(fi_probs[fidx])
                    auroc = roc_auc_score(fi_labels_, fi_probs_)
                    accuracy = np.mean((fi_probs_ > threshold) == fi_labels_)
                    ppv = np.sum((fi_probs_ > threshold) * fi_labels_) / np.sum(fi_probs_ > threshold)
                    npv = np.sum((fi_probs_ <= threshold) * (1 - fi_labels_)) / np.sum(fi_probs_ <= threshold)
                    sensitivity = np.sum((fi_probs_ > threshold) * fi_labels_) / np.sum(fi_labels_)
                    specificity = np.sum((fi_probs_ <= threshold) * (1 - fi_labels_)) / np.sum(1 - fi_labels_)
                    f1 = 2 * ppv * sensitivity / (ppv + sensitivity)
                    repeated_results['auroc'][fidx].append(auroc)
                    repeated_results['accuracy'][fidx].append(accuracy)
                    repeated_results['ppv'][fidx].append(ppv)
                    repeated_results['npv'][fidx].append(npv)
                    repeated_results['sensitivity'][fidx].append(sensitivity)
                    repeated_results['specificity'][fidx].append(specificity)
                    repeated_results['f1'][fidx].append(f1)
                    print(
                        f"pred idx: {fidx + 1}, AUROC: {auroc:.4f}, accuracy: {accuracy:.4f}, ppv: {ppv:.4f}, npv: {npv:.4f}, sensitivity: {sensitivity:.4f}, specificity: {specificity:.4f}, f1: {f1:.4f}")
                all_labels_ = np.array(all_labels)
                all_probs_ = np.array(all_probs)
                auroc_all = roc_auc_score(all_labels_, all_probs_)
                accuracy_all = np.mean((all_probs_ > threshold) == all_labels_)
                ppv_all = np.sum((all_probs_ > threshold) * all_labels_) / np.sum(all_probs_ > threshold)
                npv_all = np.sum((all_probs_ <= threshold) * (1 - all_labels_)) / np.sum(all_probs_ <= threshold)
                sensitivity_all = np.sum((all_probs_ > threshold) * all_labels_) / np.sum(all_labels_)
                specificity_all = np.sum((all_probs_ <= threshold) * (1 - all_labels_)) / np.sum(1 - all_labels_)
                f1_all = 2 * ppv_all * sensitivity_all / (ppv_all + sensitivity_all)
                repeated_results_all['auroc'].append(auroc_all)
                repeated_results_all['accuracy'].append(accuracy_all)
                repeated_results_all['ppv'].append(ppv_all)
                repeated_results_all['npv'].append(npv_all)
                repeated_results_all['sensitivity'].append(sensitivity_all)
                repeated_results_all['specificity'].append(specificity_all)
                repeated_results_all['f1'].append(f1_all)
                print(
                    f"all pred points, AUROC: {auroc_all:.4f}, accuracy: {accuracy_all:.4f}, ppv: {ppv_all:.4f}, npv: {npv_all:.4f}, sensitivity: {sensitivity_all:.4f}, specificity: {specificity_all:.4f}, f1: {f1_all:.4f}")
        if return_results:
            return repeated_results, repeated_results_all
        with open(f'{self.args.save_dir}/test_results.txt', 'a') as f:
            for fidx in range(self.args.num_interval):
                if len(repeated_results['auroc'][fidx]) == 0:
                    continue
                auroc_txt = self.text_mead_std('auroc', repeated_results['auroc'][fidx])
                acc_txt = self.text_mead_std('accuracy', repeated_results['accuracy'][fidx])
                ppv_txt = self.text_mead_std('ppv', repeated_results['ppv'][fidx])
                npv_txt = self.text_mead_std('npv', repeated_results['npv'][fidx])
                sen_txt = self.text_mead_std('sensitivity', repeated_results['sensitivity'][fidx])
                spe_txt = self.text_mead_std('specificity', repeated_results['specificity'][fidx])
                f1_txt = self.text_mead_std('f1', repeated_results['f1'][fidx])
                print(
                    f"at pred idx: {fidx + 1}, {auroc_txt}, {acc_txt}, {ppv_txt}, {npv_txt}, {sen_txt}, {spe_txt}, {f1_txt}\n")
                f.write(
                    f"at pred idx: {fidx + 1}, {auroc_txt}, {acc_txt}, {ppv_txt}, {npv_txt}, {sen_txt}, {spe_txt}, {f1_txt}\n")
            auroc_all_txt = self.text_mead_std('auroc', repeated_results_all['auroc'])
            acc_all_txt = self.text_mead_std('accuracy', repeated_results_all['accuracy'])
            ppv_all_txt = self.text_mead_std('ppv', repeated_results_all['ppv'])
            npv_all_txt = self.text_mead_std('npv', repeated_results_all['npv'])
            sen_all_txt = self.text_mead_std('sensitivity', repeated_results_all['sensitivity'])
            spe_all_txt = self.text_mead_std('specificity', repeated_results_all['specificity'])
            f1_all_txt = self.text_mead_std('f1', repeated_results_all['f1'])
            print(
                f"all pred points, {auroc_all_txt}, {acc_all_txt}, {ppv_all_txt}, {npv_all_txt}, {sen_all_txt}, {spe_all_txt}, {f1_all_txt}\n")
            f.write(
                f"all pred points, {auroc_all_txt}, {acc_all_txt}, {ppv_all_txt}, {npv_all_txt}, {sen_all_txt}, {spe_all_txt}, {f1_all_txt}\n")
            if self.args.w_int > 0:
                for level in range(5):
                    if len(repeated_results['mae'][level]) == 0:
                        continue
                    mae_txt = self.text_mead_std('mae', repeated_results['mae'][level])
                    mse_txt = self.text_mead_std('mse', repeated_results['mse'][level])
                    mae_d_txt = self.text_mead_std('mae_d', repeated_results['mae_d'][level])
                    print(f"at level: {level}, {mae_txt}, {mse_txt}, {mae_d_txt}\n")
                    f.write(f"at level: {level}, {mae_txt}, {mse_txt}, {mae_d_txt}\n")
                mae_all_txt = self.text_mead_std('mae', repeated_results_all['mae'])
                mse_all_txt = self.text_mead_std('mse', repeated_results_all['mse'])
                mae_d_all_txt = self.text_mead_std('mae_d', repeated_results_all['mae_d'])
                print(f"all pred points, {mae_all_txt}, {mse_all_txt}, {mae_d_all_txt}\n")
                f.write(f"all pred points, {mae_all_txt}, {mse_all_txt}, {mae_d_all_txt}\n")
        # save repeated_results_all as csv for further analysis
        df = pd.DataFrame(repeated_results_all)
        df.to_csv(f'{self.args.save_dir}/test_repeated_results_all.csv', index=False)

    def test_cls(self, is_compare_mod=False, return_results=False):
        # todo: write other metrics: F1, sensitivity, specificity, accuracy, precision, recall
        if self.args.ckpt is None:
            raise ValueError('no checkpoint to test!!!')
        self.load_checkpoint(is_train=False)
        if is_compare_mod:
            self.is_test_compare_mod = True
        test_dataset = self.configure_test_dataset()

        metric_names = ['auroc', 'mae', 'mse', 'accuracy', 'ppv', 'npv', 'sensitivity', 'specificity', 'mae_d',
                        'f1']
        repeated_results = {k: [[] for _ in range(self.args.num_interval)] for k in metric_names}
        repeated_results_all = {k: [] for k in metric_names}
        for i in range(self.args.num_test_repeat):
            with torch.no_grad():

                fi_labels = [[] for _ in range(self.args.num_interval)]
                fi_probs = [[] for _ in range(self.args.num_interval)]
                all_labels = []
                all_probs = []
                print('testing...')
                for idx in tqdm(range(len(test_dataset))):
                    # one patient
                    self.inputs_dict, self.labels = test_dataset[idx]
                    self.labels = self.labels.unsqueeze(0).cuda()
                    if len(self.inputs_dict) == 0:
                        # print('empty input for ards patient')
                        # will not cause accumulation for update
                        continue
                    for i in self.inputs_dict.keys():
                        for j in self.inputs_dict[i].keys():
                            self.inputs_dict[i][j] = self.inputs_dict[i][j].cuda()
                    # for each mod 按照 fidx 1~k 分k组来foward

                    max_fidx = int(max([torch.max(self.inputs_dict[i]['fidx']) for i in self.inputs_dict.keys()]))
                    previous_fis = []
                    # for each pred point of one patient
                    for ifi in range(1, max_fidx + 1):  # from 1 to 8
                        fi = max_fidx + 1 - ifi  # from 8 to 1
                        # obtain feature for single fi
                        if self.args.use_ptt:
                            if self.args.use_mft:
                                fifeature, am_mask = self._forward(fi)
                            else:
                                fifeature, am_mask = self._forward_no_mft(fi)
                            if not am_mask:
                                continue
                            all_features = torch.stack([fifeature] + previous_fis, dim=1)
                            logits = self.pred_fusion_model(all_features)[0]
                            previous_fis.append(fifeature)
                        else:
                            if self.args.use_mft:
                                logits, am_mask = self._forward(fi)
                            else:
                                logits, am_mask = self._forward_no_mft(fi)

                        if self.args.w_int > 0:
                            logits = logits[0].unsqueeze(0)
                        probs = torch.sigmoid(logits)
                        fi_labels[fi - 1].append(self.labels.cpu().numpy())
                        fi_probs[fi - 1].append(probs.cpu().numpy())
                        all_labels.append(self.labels.cpu().numpy())
                        all_probs.append(probs.cpu().numpy())

                # auroc, acc, ppv, npv, sensitivity, specificity
                threshold = 0.7 # todo
                for fidx in range(self.args.num_interval):
                    if len(fi_labels[fidx]) == 0:
                        continue
                    fi_labels_ = np.array(fi_labels[fidx])
                    fi_probs_ = np.array(fi_probs[fidx])
                    auroc = roc_auc_score(fi_labels_, fi_probs_)
                    fpr, tpr, thresholds = roc_curve(fi_labels_, fi_probs_)
                    self.draw_roc_curve(fpr, tpr, auroc, fidx, thresholds)
                    accuracy = np.mean((fi_probs_ > threshold) == fi_labels_)

                    sensitivity = np.sum((fi_probs_ > threshold) * fi_labels_) / np.sum(fi_labels_)
                    specificity = np.sum((fi_probs_ <= threshold) * (1 - fi_labels_)) / np.sum(1 - fi_labels_)
                    prior_pos = np.sum(fi_labels_) / len(fi_labels_)
                    prior_neg = 1 - prior_pos
                    # ppv = np.sum((fi_probs_ > threshold) * fi_labels_) / np.sum(fi_probs_ > threshold)
                    # npv = np.sum((fi_probs_ <= threshold) * (1 - fi_labels_)) / np.sum(fi_probs_ <= threshold)
                    print('yes it is..')
                    ppv = sensitivity * prior_pos / (sensitivity * prior_pos + (1 - specificity) * prior_neg)
                    npv = specificity * prior_neg / ((1 - sensitivity) * prior_pos + specificity * prior_neg)

                    f1 = 2 * ppv * sensitivity / (ppv + sensitivity)
                    repeated_results['auroc'][fidx].append(auroc)
                    repeated_results['accuracy'][fidx].append(accuracy)
                    repeated_results['ppv'][fidx].append(ppv)
                    repeated_results['npv'][fidx].append(npv)
                    repeated_results['sensitivity'][fidx].append(sensitivity)
                    repeated_results['specificity'][fidx].append(specificity)
                    repeated_results['f1'][fidx].append(f1)
                    print(
                        f"pred idx: {fidx + 1}, AUROC: {auroc:.4f}, accuracy: {accuracy:.4f}, ppv: {ppv:.4f}, npv: {npv:.4f}, sensitivity: {sensitivity:.4f}, specificity: {specificity:.4f}, f1: {f1:.4f}")
                all_labels_ = np.array(all_labels)
                all_probs_ = np.array(all_probs)
                auroc_all = roc_auc_score(all_labels_, all_probs_)
                accuracy_all = np.mean((all_probs_ > threshold) == all_labels_)
                ppv_all = np.sum((all_probs_ > threshold) * all_labels_) / np.sum(all_probs_ > threshold)
                npv_all = np.sum((all_probs_ <= threshold) * (1 - all_labels_)) / np.sum(all_probs_ <= threshold)
                sensitivity_all = np.sum((all_probs_ > threshold) * all_labels_) / np.sum(all_labels_)
                specificity_all = np.sum((all_probs_ <= threshold) * (1 - all_labels_)) / np.sum(1 - all_labels_)
                f1_all = 2 * ppv_all * sensitivity_all / (ppv_all + sensitivity_all)
                repeated_results_all['auroc'].append(auroc_all)
                repeated_results_all['accuracy'].append(accuracy_all)
                repeated_results_all['ppv'].append(ppv_all)
                repeated_results_all['npv'].append(npv_all)
                repeated_results_all['sensitivity'].append(sensitivity_all)
                repeated_results_all['specificity'].append(specificity_all)
                repeated_results_all['f1'].append(f1_all)
                print(
                    f"all pred points, AUROC: {auroc_all:.4f}, accuracy: {accuracy_all:.4f}, ppv: {ppv_all:.4f}, npv: {npv_all:.4f}, sensitivity: {sensitivity_all:.4f}, specificity: {specificity_all:.4f}, f1: {f1_all:.4f}")
        if return_results:
            return repeated_results, repeated_results_all
        with open(f'{self.args.save_dir}/test_results.txt', 'a') as f:
            for fidx in range(self.args.num_interval):
                if len(repeated_results['auroc'][fidx]) == 0:
                    continue
                auroc_txt = self.text_mead_std('auroc', repeated_results['auroc'][fidx])
                acc_txt = self.text_mead_std('accuracy', repeated_results['accuracy'][fidx])
                ppv_txt = self.text_mead_std('ppv', repeated_results['ppv'][fidx])
                npv_txt = self.text_mead_std('npv', repeated_results['npv'][fidx])
                sen_txt = self.text_mead_std('sensitivity', repeated_results['sensitivity'][fidx])
                spe_txt = self.text_mead_std('specificity', repeated_results['specificity'][fidx])
                f1_txt = self.text_mead_std('f1', repeated_results['f1'][fidx])
                print(
                    f"at pred idx: {fidx + 1}, {auroc_txt}, {acc_txt}, {ppv_txt}, {npv_txt}, {sen_txt}, {spe_txt}, {f1_txt}\n")
                f.write(
                    f"at pred idx: {fidx + 1}, {auroc_txt}, {acc_txt}, {ppv_txt}, {npv_txt}, {sen_txt}, {spe_txt}, {f1_txt}\n")
            auroc_all_txt = self.text_mead_std('auroc', repeated_results_all['auroc'])
            acc_all_txt = self.text_mead_std('accuracy', repeated_results_all['accuracy'])
            ppv_all_txt = self.text_mead_std('ppv', repeated_results_all['ppv'])
            npv_all_txt = self.text_mead_std('npv', repeated_results_all['npv'])
            sen_all_txt = self.text_mead_std('sensitivity', repeated_results_all['sensitivity'])
            spe_all_txt = self.text_mead_std('specificity', repeated_results_all['specificity'])
            f1_all_txt = self.text_mead_std('f1', repeated_results_all['f1'])
            print(
                f"all pred points, {auroc_all_txt}, {acc_all_txt}, {ppv_all_txt}, {npv_all_txt}, {sen_all_txt}, {spe_all_txt}, {f1_all_txt}\n")
            f.write(
                f"all pred points, {auroc_all_txt}, {acc_all_txt}, {ppv_all_txt}, {npv_all_txt}, {sen_all_txt}, {spe_all_txt}, {f1_all_txt}\n")

        # save repeated_results_all as csv for further analysis
        df = pd.DataFrame(repeated_results_all)
        df.to_csv(f'{self.args.save_dir}/test_repeated_results_all.csv', index=False)

    def test_box_plot(self, is_compare_mod=False, return_results=False):
        if self.args.ckpt is None:
            raise ValueError('no checkpoint to test!!!')
        self.load_checkpoint(is_train=False)
        if is_compare_mod:
            self.is_test_compare_mod = True
        test_dataset = self.configure_test_dataset()
        with torch.no_grad():

            fi_probs = {f'{6*i}h': [] for i in range(self.args.num_interval,0,-1)}
            fi_emergency = {f'{6*i}h': [] for i in range(self.args.num_interval, 0, -1)}

            print('testing...')
            for idx in tqdm(range(len(test_dataset))):
                # one patient
                self.inputs_dict, self.labels = test_dataset[idx]
                self.labels = self.labels.unsqueeze(0).cuda()
                if self.labels[0] == 0:
                    continue
                if len(self.inputs_dict) == 0:
                    # print('empty input for ards patient')
                    # will not cause accumulation for update
                    continue
                for i in self.inputs_dict.keys():
                    for j in self.inputs_dict[i].keys():
                        self.inputs_dict[i][j] = self.inputs_dict[i][j].cuda()

                max_fidx = int(max([torch.max(self.inputs_dict[i]['fidx']) for i in self.inputs_dict.keys()]))
                previous_fis = []
                # for each pred point of one patient
                for ifi in range(1, max_fidx + 1):  # from 1 to 8
                    fi = max_fidx + 1 - ifi  # from 8 to 1
                    # obtain feature for single fi
                    if self.args.use_ptt:
                        if self.args.use_mft:
                            fifeature, am_mask = self._forward(fi)
                        else:
                            fifeature, am_mask = self._forward_no_mft(fi)
                        if not am_mask:
                            continue
                        all_features = torch.stack([fifeature] + previous_fis, dim=1)
                        logits = self.pred_fusion_model(all_features)[0]
                        previous_fis.append(fifeature)
                    else:
                        if self.args.use_mft:
                            logits, am_mask = self._forward(fi)
                        else:
                            logits, am_mask = self._forward_no_mft(fi)

                    if self.args.w_int > 0:
                        emergency_level = logits[1]
                        fi_emergency[f'{6*fi}h'].append(emergency_level.cpu().numpy())
                        logits = logits[0].unsqueeze(0)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]
                    fi_probs[f'{6*fi}h'].append(probs)

            pvalues = [sorted(v)[10:] for v in fi_probs.values()]
            plabels = list(fi_probs.keys())
            fig, ax = plt.subplots(figsize=(15, 4)) # todo
            box =  ax.boxplot(pvalues,
                        patch_artist=True,
                        # showmeans=True,
                        # meanline=True,
                        showcaps=True,
                        showfliers=False,
                        labels=plabels,
                        widths=0.4)
            # 修改4: 调整坐标轴样式
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            # 修改5: 设置坐标轴字体大小
            ax.tick_params(axis='both', labelsize=14)
            # x轴title
            ax.set_xlabel('Time to ARDS Onset', fontsize=14, weight='semibold')
            ax.set_ylabel('Predicted Risk Score', fontsize=14, weight='semibold')

            # 颜色映射部分
            raw_values = [np.mean(v) for v in fi_emergency.values()]
            vmin, vmax = np.min(raw_values), np.max(raw_values)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.Reds
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

            for patch, value in zip(box['boxes'], raw_values):
                patch.set_facecolor(cmap(norm(value)))
                patch.set_edgecolor('black')

            # 修改6: 调整colorbar
            cbar = fig.colorbar(sm, ax=ax, pad=0.01, aspect=40)  # 调整长宽比
            cbar.set_label('Average Emergency Level',
                           fontsize=14,  # 修改字体大小
                           weight='bold')
            cbar.ax.tick_params(labelsize=14)  # colorbar刻度字体

            # 保存结果
            fig.tight_layout()
            plt.savefig(f'{self.args.save_dir}/box_plot.png', dpi=300, bbox_inches='tight')
            plt.show()

    def test_flops(self, is_compare_mod=False, return_results=False):
        # todo: write other metrics: F1, sensitivity, specificity, accuracy, precision, recall
        if self.args.ckpt is None:
            raise ValueError('no checkpoint to test!!!')
        self.load_checkpoint(is_train=False)
        if is_compare_mod:
            self.is_test_compare_mod = True
        test_dataset = self.configure_test_dataset()
        # Initialize FLOPs tracking
        self.total_flops = 0.0
        patient_count = 0

        with torch.no_grad():
            print('testing flops...')
            cc =  0
            for idx in tqdm(range(len(test_dataset))):
                cc+=1
                if cc>100:
                    break
                # one patient
                self.inputs_dict, self.labels = test_dataset[idx]
                self.labels = self.labels.unsqueeze(0).cuda()
                if len(self.inputs_dict) == 0:
                    continue
                for i in self.inputs_dict.keys():
                    for j in self.inputs_dict[i].keys():
                        self.inputs_dict[i][j] = self.inputs_dict[i][j].cuda()
                # for each mod 按照 fidx 1~k 分k组来foward
                # Reset FLOPs counter for current patient
                patient_flops_start = self.total_flops
                max_fidx = int(max([torch.max(self.inputs_dict[i]['fidx']) for i in self.inputs_dict.keys()]))
                previous_fis = []
                # for each pred point of one patient

                for ifi in range(1, max_fidx + 1):  # from 1 to 8
                    fi = max_fidx + 1 - ifi  # from 8 to 1
                    # obtain feature for single fi
                    if self.args.use_ptt:
                        if self.args.use_mft:
                            fifeature, am_mask = self._forward(fi)
                        else:
                            fifeature, am_mask = self._forward_no_mft(fi)
                        if not am_mask:
                            continue
                        all_features = torch.stack([fifeature] + previous_fis, dim=1)
                        if self.args.test_only:
                            with torch.no_grad():
                                flops = FlopCountAnalysis(self.pred_fusion_model, (all_features,))
                                self.total_flops += flops.total()
                            print(f'FLOPs for pred_fusion_model: {flops.total():.2f}')
                        logits = self.pred_fusion_model(all_features)[0]
                        previous_fis.append(fifeature)
                    else:
                        if self.args.use_mft:
                            logits, am_mask = self._forward(fi)
                        else:
                            logits, am_mask = self._forward_no_mft(fi)
                patient_count += 1
                # if patient_count > 0:  # Prevent division by zero
                #     avg_flops = self.total_flops / patient_count
                #     print(f'Current average FLOPs: {avg_flops:.2f}')
            # Final average calculation
            final_avg_flops = self.total_flops / patient_count
            print(f'\nFinal Average FLOPs per patient: {final_avg_flops:.2f}')
            return final_avg_flops

    def test_draw_roc(self, is_compare_mod=False, return_results=False):
        if self.args.ckpt is None:
            raise ValueError('no checkpoint to test!!!')
        self.load_checkpoint(is_train=False)
        test_dataset = self.configure_test_dataset()
        all_y_true = {i:[[] for _ in range(10)] for i in range(1,9)}
        all_y_score = {i:[[] for _ in range(10)] for i in range(1,9)}
        with torch.no_grad():
            for exp_idx in range(10):
                print('testing...')
                for idx in tqdm(range(len(test_dataset))):
                    # one patient
                    self.inputs_dict, self.labels = test_dataset[idx]
                    self.labels = self.labels.unsqueeze(0).cuda()
                    if len(self.inputs_dict) == 0:
                        continue
                    for i in self.inputs_dict.keys():
                        for j in self.inputs_dict[i].keys():
                            self.inputs_dict[i][j] = self.inputs_dict[i][j].cuda()
                    # for each mod 按照 fidx 1~k 分k组来foward
                    # max_fidx = int(max([torch.max(self.inputs_dict[i]['fidx']) for i in self.inputs_dict.keys()]))
                    max_fidx = 8
                    previous_fis = []
                    # for each pred point of one patient
                    for ifi in range(1, max_fidx + 1):  # from 1 to 8
                        fi = max_fidx + 1 - ifi  # from 8 to 1
                        # obtain feature for single fi
                        if self.args.use_ptt:
                            if self.args.use_mft:
                                fifeature, am_mask = self._forward(fi)
                            else:
                                fifeature, am_mask = self._forward_no_mft(fi)
                            if not am_mask:
                                continue
                            all_features = torch.stack([fifeature] + previous_fis, dim=1)
                            logits = self.pred_fusion_model(all_features)[0]
                            previous_fis.append(fifeature)
                        else:
                            if self.args.use_mft:
                                logits, am_mask = self._forward(fi)
                            else:
                                logits, am_mask = self._forward_no_mft(fi)
                        if self.args.w_int > 0:
                            logits = logits[0].unsqueeze(0)
                        probs = torch.sigmoid(logits)
                        all_y_true[fi][exp_idx].append(self.labels.cpu().numpy())
                        all_y_score[fi][exp_idx].append(probs.cpu().numpy())
        # cal metrics auroc, acc, sen, spe for <6h, <24h,<48h
        # < 6h
        y_true = np.concatenate([np.array(i) for i in all_y_true[1]], axis=0)
        y_score = np.concatenate([np.array(i) for i in all_y_score[1]], axis=0)
        results_6h = self.cal_metrics(y_true, y_score)
        # <24h
        y_true = np.concatenate([np.concatenate([np.array(i) for i in all_y_true[fidx]], axis=0) for fidx in range(1,5)], axis=0)
        y_score = np.concatenate([np.concatenate([np.array(i) for i in all_y_score[fidx]], axis=0) for fidx in range(1,5)], axis=0)
        results_24h = self.cal_metrics(y_true, y_score)
        # <48h
        y_true = np.concatenate(
            [np.concatenate([np.array(i) for i in all_y_true[fidx]], axis=0) for fidx in range(1, 9)], axis=0)
        y_score = np.concatenate(
            [np.concatenate([np.array(i) for i in all_y_score[fidx]], axis=0) for fidx in range(1, 9)], axis=0)
        results_48h = self.cal_metrics(y_true, y_score)
        print(results_6h,'\n',results_24h,'\n',results_48h)
        # draw roc curves
        for fidx in [1, 2, 3, 4]:
            print('fidx:', fidx)
            self.draw_roc_curve(all_y_true[fidx], all_y_score[fidx], fidx)

    def test_drawRR(self, is_compare_mod=False, return_results=False):
        if self.args.ckpt is None:
            raise ValueError('no checkpoint to test!!!')
        self.load_checkpoint(is_train=False)
        test_dataset = self.configure_test_dataset()
        all_patients = [] #{'label':0,'probs':[],'at':[]}]
        with torch.no_grad():
            print('testing...')

            for idx in tqdm(range(len(test_dataset))):
                # one patient
                self.inputs_dict, self.labels = test_dataset[idx]
                self.labels = self.labels.unsqueeze(0).cuda()
                if len(self.inputs_dict) == 0:
                    continue
                for i in self.inputs_dict.keys():
                    for j in self.inputs_dict[i].keys():
                        self.inputs_dict[i][j] = self.inputs_dict[i][j].cuda()
                # for each mod 按照 fidx 1~k 分k组来foward
                # max_fidx = int(max([torch.max(self.inputs_dict[i]['fidx']) for i in self.inputs_dict.keys()]))
                max_fidx = 8
                previous_fis = []
                all_patients.append({'label':self.labels.cpu().numpy()[0],'probs':[],'at':[]})
                # for each pred point of one patient
                for ifi in range(1, max_fidx + 1):  # from 1 to 8
                    fi = max_fidx + 1 - ifi  # from 8 to 1
                    # obtain feature for single fi
                    if self.args.use_ptt:
                        if self.args.use_mft:
                            fifeature, am_mask = self._forward(fi)
                        else:
                            fifeature, am_mask = self._forward_no_mft(fi)
                        if not am_mask:
                            continue
                        all_features = torch.stack([fifeature] + previous_fis, dim=1)
                        logits = self.pred_fusion_model(all_features)[0]
                        previous_fis.append(fifeature)
                    else:
                        if self.args.use_mft:
                            logits, am_mask = self._forward(fi)
                        else:
                            logits, am_mask = self._forward_no_mft(fi)
                    if self.args.w_int > 0:
                        logits = logits[0].unsqueeze(0)
                    probs = torch.sigmoid(logits)
                    all_patients[-1]['probs'].append(probs.cpu().numpy()[0])
                    all_patients[-1]['at'].append(fi)
        # save all patients as pkl for further analysis
        with open(f'{self.args.save_dir}/all_patients.pkl', 'wb') as f:
            pickle.dump(all_patients, f)





    def cal_metrics(self, y_true, y_score):
        auroc = roc_auc_score(y_true, y_score)
        acc = np.mean((y_score > 0.5) == y_true)
        sen = np.sum((y_score > 0.5) * y_true) / np.sum(y_true)
        spe = np.sum((y_score <= 0.5) * (1 - y_true)) / np.sum(1 - y_true)
        return {'auroc': auroc, 'acc': acc, 'sen': sen, 'spe': spe}

    def draw_roc_curve(self, all_y_true, all_y_score, fidx):
        # 公共FPR网格
        fpr_grid = np.linspace(0, 1, 100)
        tpr_interpolated = []

        # 处理每条ROC曲线
        for y_true, y_score in zip(all_y_true, all_y_score):
            fpr, tpr, _ = roc_curve(y_true, y_score)

            # 插值到公共FPR网格
            tpr_interp = np.interp(fpr_grid, fpr, tpr)
            tpr_interpolated.append(tpr_interp)

        tpr_interpolated = np.array(tpr_interpolated)

        # 计算统计量
        mean_tpr = np.mean(tpr_interpolated, axis=0)
        std_tpr = np.std(tpr_interpolated, axis=0)

        # 可视化
        plt.figure(figsize=(6, 5))

        # 绘制所有实验曲线
        for i in range(10):
            plt.plot(fpr_grid, tpr_interpolated[i], color='grey', alpha=0.3, lw=1)

        # 绘制平均曲线
        plt.plot(fpr_grid, mean_tpr, color='palevioletred', lw=2,
                 label=f'Mean ROC (AUC = {np.mean(mean_tpr):.2f}±{np.mean(std_tpr):.2f})')
        # set fontsize of label

        # 绘制标准差范围
        plt.fill_between(fpr_grid,
                         mean_tpr - std_tpr,
                         mean_tpr + std_tpr,
                         color='palevioletred', alpha=0.2,
                         label='±1 Std. Dev.')

        # 对角线参考线
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        # set font size
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        if fidx == 1:
            plt.title('(c) Prediction time to ARDS Onset < 6h',fontsize=14, fontweight='semibold')
        elif fidx == 2:
            plt.title('(d) Prediction time to ARDS Onset 6~12h',fontsize=14, fontweight='semibold')
        elif fidx ==3:
            plt.title('(e) Prediction time to ARDS Onset 12~18h',fontsize=14, fontweight='semibold')
        else:
            plt.title('(f) Prediction time to ARDS Onset 18~24h',fontsize=14, fontweight='semibold')
        plt.legend(loc="lower right", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def text_mead_std(self,metric_name, results):
        mean = np.mean(results)
        std = np.std(results)
        return f"{metric_name}: {mean:.4f}+-{std:.4f}"
    # todo
    def configure_datasets(self):

        train_dataset = ARDSDataset(src_pth=self.args.data_src_dir, phase=self.args.train_data_name,num_interval=self.args.num_interval, complete_types=self.complete_m, length_interval=self.args.interval_length, use_single_cxr=self.args.use_single_cxr)
        val_dataset = ARDSDataset(src_pth=self.args.data_src_dir, phase=self.args.val_data_name,num_interval=self.args.num_interval, complete_types=self.complete_m, length_interval=self.args.interval_length, use_single_cxr=self.args.use_single_cxr)
        return train_dataset, val_dataset
    # todo
    def configure_test_dataset(self):
        test_dataset = ARDSDataset(src_pth=self.args.data_src_dir, phase=self.args.test_data_name,num_interval=self.args.num_interval, complete_types=['cxr','vs','lab'], length_interval=self.args.interval_length, use_single_cxr=self.args.use_single_cxr)
        return test_dataset

    def _get_label_int(self, fi, to_tensor=True):
        if fi < 3:  # 0-12 hrs  level 4
            label_int = 4.0
        elif fi < 5:  # 12-24 hrs  level 3
            label_int = 3.0
        elif fi < 9:  # 24-48 hrs   level 2
            label_int = 2.0
        else:  # >48 hrs        level 1
            label_int = 1.0

        if to_tensor:
            return torch.tensor(label_int).cuda()
        else:
            return label_int



    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def cal_params(self, model):
        # parameter size of each module:
        # original 3d cnn: 63471171
        # latents = 4096 *n
        # Print parameter informationx
        # num_cnn = 0
        # num_tab = 0
        # num_lat = 0
        # for n, p in Encoder.named_parameters():
        #     print(n)
        #     if n.startswith('cnn_encoder'):
        #         num_cnn += p.numel()
        #     if n.startswith('tab_encoder'):
        #         num_tab += p.numel()
        #     if n.startswith('latents'):
        #         num_lat += p.numel()
        # print(f'param cnns: {num_cnn}, param tab: {num_tab}, param latents: {num_lat}, param MA: ',
        #       total_params - num_lat - num_tab - num_cnn)
        return sum(p.numel() for p in model.parameters())

    def pad_collate_fn(self,batch):
        # Separate the data and labels
        data_dicts, labels = zip(*batch)
        b = len(data_dicts)
        out_dict = {}
        for mod in self.complete_m:
            out_dict[mod] = {}
            for k in data_dicts[0][mod].keys():
                seqs = []
                for i in range(b):
                    seqs.append(data_dicts[i][mod][k])
                if k == 'fidx':
                    padded_sequences, mask = self.pad_fidx(seqs)
                elif mod == 'cxr':
                    padded_sequences, mask = self.pad_cxr(seqs)
                else:
                    padded_sequences, mask = self.pad(seqs)
                out_dict[mod][k] = padded_sequences
                out_dict[mod][f'{k}_mask'] = ~mask
        return out_dict, torch.tensor(labels)  # Return padded data, mask, and labels



    def pad(self, list_data):
        # Get the maximum sequence length in the batch
        max_len = max([seq.shape[0] for seq in list_data])

        # Pad the sequences to the same length (padding with zeros)
        padded_sequences = torch.zeros(len(list_data), max_len,
                                       list_data[0].shape[-1])  # (batch_size, max_len, feature_dim)
        for i, seq in enumerate(list_data):

            padded_sequences[i, :seq.shape[0], :] = seq


        # Create a mask indicating where the padding is (1 for valid, 0 for padding)
        mask = torch.ones((len(list_data), max_len), dtype=torch.bool)
        for i, seq in enumerate(list_data):
            mask[i, :seq.shape[0]] = False  # 1 where data exists, 0 where padding
        return padded_sequences, mask


    def pad_fidx(self, list_data):
        max_len = max([seq.shape[0] for seq in list_data])
        # Pad the sequences to the same length (padding with zeros)
        padded_sequences = torch.zeros(len(list_data), max_len)  # (batch_size, max_len)
        for i, seq in enumerate(list_data):
            padded_sequences[i, :seq.shape[0]] = seq
        # Create a mask indicating where the padding is (1 for valid, 0 for padding)
        mask = torch.ones((len(list_data), max_len), dtype=torch.bool)
        for i, seq in enumerate(list_data):
            mask[i, :seq.shape[0]] = False  # 1 where data exists, 0 where padding
        return padded_sequences, mask
    def pad_cxr(self, list_data):
        # Get the maximum sequence length in the batch
        max_len = max([seq.shape[0] for seq in list_data])

        # Pad the sequences to the same length (padding with zeros)
        padded_sequences = torch.zeros(len(list_data), max_len,3,224,224)
                                         # (batch_size, max_len, feature_dim)
        for i, seq in enumerate(list_data):
            padded_sequences[i, :seq.shape[0], :,:,:] = seq

        # Create a mask indicating where the padding is (1 for valid, 0 for padding)
        mask = torch.ones((len(list_data), max_len), dtype=torch.bool)
        for i, seq in enumerate(list_data):
            mask[i, :seq.shape[0]] = False  # 1 where data exists, 0 where padding
        return padded_sequences, mask




























