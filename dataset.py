import os.path
import random

import torchio as tio
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler
from torchvision import transforms
import pandas as pd
import SimpleITK as sitk
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import glob
import json
from PIL import Image

random.seed(42)
from utils.misc import cat_mask

# Define transformations (you can customize these based on your requirements)

with open('cat_dict_map.json','r') as f:
    VS_CAT_MAP = json.load(f)
with open('mu_std_dict_lab.json','r') as f:
    lab_mu_std = json.load(f)
with open('mu_std_dict_vs.json','r') as f:
    vs_mu_std = json.load(f)

for cat in VS_CAT_MAP.keys():
    for k in VS_CAT_MAP[cat].keys():
        VS_CAT_MAP[cat][k] = int(VS_CAT_MAP[cat][k]) + 1
    VS_CAT_MAP[cat].update({'miss':0})

def get_transform(phase='train'):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    trans = []
    trans.append(transforms.Resize(256))
    if phase == 'train':
        trans.append(transforms.RandomHorizontalFlip())
        trans.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
    trans.append(transforms.CenterCrop(224))
    trans.append(transforms.ToTensor())
    trans.append(normalize)
    return transforms.Compose(trans)


class ARDSDataset(Dataset):
    def __init__(self, src_pth='', phase='train', num_interval=12,
                 complete_types=['cxr', 'lab', 'vs'], length_interval='6 hours', use_single_cxr=False):
        ### new
        # journal_src_pth = '/home/yd/ards-miccai/ards-med-journal/data_splits'
        ###
        # patient_df = pd.read_csv(f'{journal_src_pth}/{phase}_data.csv')

        self.patient_df = pd.read_csv(f'{src_pth}/{phase}.csv')
        self.src_pth = src_pth


        self.transform = get_transform(phase)
        self.num_interval = num_interval
        self.complete_types = complete_types
        self.length_interval = length_interval
        self.all_labels = torch.tensor(self.patient_df['label'].values, dtype=torch.float)
        # self.all_labels = torch.tensor(self.patient_df['classification'].values, dtype=torch.float)
        self.pos_indices = [i for i, label in enumerate(self.all_labels) if label == 1]
        self.neg_indices = [i for i, label in enumerate(self.all_labels) if label == 0]
        self.use_single_cxr = use_single_cxr
    def __len__(self):

        # return len(self.stay_pths)
        return len(self.all_labels)

    def get_labdata(self, pths):
        # todo: normalize lab values
        dd = []
        ddm = []
        if len(pths) == 0:
            raise ValueError
        for pth in pths:
            pth = os.path.join(self.src_pth, pth)
            df = pd.read_csv(pth)
            # remove Unamed 0
            df = df.drop('Unnamed: 0', axis=1)
            df = df.iloc[0]
            #
            # # todo
            # if df.isna().sum() > 20:
            #     f_mask.append(False)
            #     continue
            ddm.append(torch.tensor(df.isna().values.astype('bool')))

            df = df.astype('float')
            for t in lab_mu_std.keys():
                df[t] = np.clip((df[t] - lab_mu_std[t]['mu']) / lab_mu_std[t]['std'], -2, 2)
                # df[t] = (df[t] - lab_mu_std[t]['mu'])/lab_mu_std[t]['std']
            df = df.fillna(1e-6)
            dd.append(torch.tensor(df.values.astype('float')))
        # # todo
        # if len(dd) == 0:
        #     return torch.zeros(1, 45), torch.zeros(1, 45)

        tab_feature = torch.stack(dd, dim=0)
        mask = torch.stack(ddm, dim=0)

        return tab_feature.type(torch.float32), mask.type(torch.float32)

    def get_vsdata(self, pths):
        cont = []
        cat = []
        contm = []
        catm = []
        if len(pths) == 0:
            raise ValueError
        for pth in pths:
            pth = os.path.join(self.src_pth, pth)
            vs_df = pd.read_csv(pth).drop('Unnamed: 0', axis=1)
            # 1. split to cat and cont, map nan to 'miss'(-1 in CAT_MAP), fill0 for cont
            vs_cat = vs_df[VS_CAT_MAP.keys()].iloc[0]
            vs_cat = vs_cat.astype('string')
            catm.append(torch.tensor(vs_cat.isna().values.astype('bool')))
            vs_cat = vs_cat.fillna('miss')
            vs_cont = vs_df.drop(VS_CAT_MAP.keys(), axis=1).iloc[0].astype('float')
            for t in vs_mu_std.keys():
                vs_cont[t] = (vs_cont[t] - vs_mu_std[t]['mu']) / vs_mu_std[t]['std']

            contm.append(torch.tensor(vs_cont.isna().values.astype('bool')))
            vs_cont = vs_cont.fillna(0)
            # 2. map cat values
            for c in VS_CAT_MAP.keys():
                vs_cat[c] = str(VS_CAT_MAP[c][vs_cat[c]])
            cont.append(torch.tensor(vs_cont.values.astype('float')))
            cat.append(torch.tensor(vs_cat.values.astype('long')))
        cont_feature = torch.stack(cont, dim=0)
        cat_feature = torch.stack(cat, dim=0)
        cont_mask = torch.stack(contm, dim=0)
        cat_mask = torch.stack(catm, dim=0)
        return cont_feature.type(torch.float32), cat_feature.type(torch.float32), cont_mask.type(torch.float32), cat_mask.type(torch.float32)

    def get_imgdata(self, pths):
        dd = []
        if len(pths) == 0:
            return torch.zeros(1, 3, 224, 224)
        for pth in pths:
            # img_pths = glob.glob(f'{pth}/*.jpg')
            # for img_pth in img_pths:
            img = Image.open(pth).convert('RGB')
            dd.append(self.transform(img))
        return torch.stack(dd, dim=0)

    def __getitem__(self, idx):
        ##### new
        # patient_id, stay_id, label, _, severity, onset_time = self.patient_df.iloc[idx].values
        # spth = f'{self.src_pth}/patient_stay_data/{patient_id}/{stay_id}'
        #####
        patient_id, stay_id, label = self.patient_df.iloc[idx].values
        spth = f'{self.src_pth}/patient_stay_data/{patient_id}/{stay_id}'
        stay_df = pd.read_csv(f'{spth}/stay.csv')
        data = {i: {} for i in self.complete_types}
        time_axis = pd.read_csv(f'{spth}/time_axis.csv')
        time_axis['time'] = pd.to_datetime(time_axis['time'])
        time_axis = time_axis.sort_values('time')
        start_time = time_axis['time'].iloc[0]
        label = self.all_labels[idx]
        # todo
        self.im_meta = pd.read_csv(f'{spth}/cxr_meta.csv')


        # if stay_df['ards'][0] == 1:
        if label == 1:
            # label = 1
            # print('time ac before:', time_axis.shape[0])
            ards_onset = pd.to_datetime(stay_df['ards_time'][0]) + pd.Timedelta('6 hour')
            end_point = random.choice(
                pd.date_range(ards_onset-pd.Timedelta(self.length_interval), ards_onset, freq='1h'))
            time_axis = time_axis[time_axis['time'] < end_point]
            # print('time ac after:', time_axis.shape[0])
        else:
            # label = 0
            total_length_considered = self.num_interval * pd.Timedelta(self.length_interval)
            if time_axis['time'].iloc[-1] - start_time > total_length_considered:
                # choose a random time as end_point between start_time+total_length_considered and time_axis['time'].iloc[-1]
                end_point = random.choice(
                    pd.date_range(start_time + total_length_considered, time_axis['time'].iloc[-1], freq='2h'))
                time_axis = time_axis[time_axis['time'] < end_point]
            else:
                end_point = time_axis['time'].iloc[-1]
        # get last row 'time' value
        # get fragment idx [5554444,33,2222,1111]
        # for type, sub_time_axis in time_axis.groupby('type'):
        for type in self.complete_types:

            sub_time_axis = time_axis[time_axis['type'] == type] # time axis for each mod
            ## todo
            if self.use_single_cxr and type == 'cxr':
                # only consider the last AP view cxr
                # rank time_axis by StudyDateTime from latest to earliest
                sub_time_axis = sub_time_axis.sort_values('time', ascending=False).reset_index(drop=True)
                flag_any = False
                for i, row in sub_time_axis.iterrows():
                    name = row['value'].split('/')[-1].split('.')[0]
                    immeta = self.im_meta[self.im_meta.values[:,0]==name]
                    try:
                        if 'AP' in immeta['PerformedProcedureStepDescription'].values[0] or 'PA' in immeta['PerformedProcedureStepDescription'].values[0]:
                            sub_time_axis = sub_time_axis.iloc[i:i+1]
                            flag_any = True
                            break
                    except:
                        continue
                if not flag_any:
                    continue


            if sub_time_axis.shape[0] == 0:  # T
                # mod not exist
                continue
                # data[type]['fidx'] = torch.zeros(1)
                # sub_list = [] # to retrieve data (detail data paths)
            else:
                last_time = end_point
                # last_time = time_axis['time'].iloc[-1]
                # get time_series data
                sub_pths = sub_time_axis['value'].copy()
                idx_value = 1
                fidx_tmp = torch.zeros(sub_time_axis.shape[0])
                abs_timestamps = sub_time_axis['time'].copy()
                relative_timestamps  = torch.zeros(sub_time_axis.shape[0])
                #
                previous_bound = sub_time_axis.shape[0]
                while last_time - pd.Timedelta(self.length_interval) > start_time:
                    last_time = last_time - pd.Timedelta(self.length_interval)
                    sub_time_axis = sub_time_axis[sub_time_axis['time'] < last_time]
                    current_bound = sub_time_axis.shape[0]
                    fidx_tmp[current_bound:previous_bound] = idx_value
                    relative_timestamps_current = (abs_timestamps[current_bound:previous_bound]
                                     - pd.Timestamp(last_time)).dt.total_seconds()/pd.Timedelta(self.length_interval).total_seconds()

                    relative_timestamps[current_bound:previous_bound] = torch.tensor(relative_timestamps_current.values)
                    previous_bound = current_bound
                    idx_value += 1
                    if idx_value > self.num_interval:
                        break
                if sub_time_axis.shape[0] > 0 and idx_value <= self.num_interval:
                    fidx_tmp[:sub_time_axis.shape[0]] = idx_value
                # don't consider those time outside considered interval
                bool_fidx_not0 = ~(fidx_tmp == 0).numpy()
                sub_list = sub_pths[bool_fidx_not0].tolist()
                relative_timestamps = relative_timestamps[bool_fidx_not0]

                if any(bool_fidx_not0):
                    fidx_tmp = fidx_tmp[bool_fidx_not0]
                else:
                    # data[type]['fidx'] = torch.zeros(1)
                    continue
            data[type]['fidx'] = fidx_tmp
            data[type]['relative_timestamps'] = relative_timestamps * 10.0
            if type == 'cxr':
                data[type]['data'] = self.get_imgdata(sub_list)
            elif type == 'lab':
                data_, mask_= self.get_labdata(pths=sub_list)
                data[type]['data'], data[type]['mask'] = data_, mask_

            else:
                data[type]['data_cont'], data[type]['data_cat'], data[type]['mask_cont'], data[type][
                    'mask_cat'] = self.get_vsdata(pths=sub_list)
        data = {key: value for key, value in data.items() if value}
        return data, label




class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.positive_indices = torch.where(labels == 1)[0]
        self.negative_indices = torch.where(labels == 0)[0]
        self.half_batch = batch_size // 2

    def __iter__(self):
        # Shuffle positive and negative indices
        pos_indices = self.positive_indices[torch.randperm(len(self.positive_indices))]
        neg_indices = self.negative_indices[torch.randperm(len(self.negative_indices))]

        # Ensure we don't go out of bounds
        min_len = min(len(pos_indices), len(neg_indices))
        pos_indices = pos_indices[:min_len]
        neg_indices = neg_indices[:min_len]

        # Generate balanced batches
        balanced_batches = []
        for i in range(0, min_len, self.half_batch):
            batch_pos = pos_indices[i:i + self.half_batch]
            batch_neg = neg_indices[i:i + self.half_batch]
            balanced_batches.extend(batch_pos.tolist() + batch_neg.tolist())

        return iter(balanced_batches)

    def __len__(self):
        return len(self.positive_indices) // self.half_batch

if __name__ == '__main__':
    ds = ARDSDataset(src_pth='', phase='val',use_single_cxr=True)

    idx = 1

    for _ in range(3):
        d, l = ds[idx]
        print(l)

        print('null')
