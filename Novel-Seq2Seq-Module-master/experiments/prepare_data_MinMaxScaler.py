# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.DoubleTensor')

class DataPrepare:
    def __init__(self,
                 datapath='prepare_data/data',
                 datafile='AT',
                 input_steps=12,
                 pred_horizion=6,
                 split_ratio=(0.75, 0.15, 0.10)):
        """
        نسخه‌ی تصحیح‌شده برای استفاده از فایل AT_data_features.csv
        بدون تغییر در ساختار main.
        """

        # مسیر فایل ویژگی‌ها
        feature_file = os.path.join(datapath, 'AT_data_features.csv')

        # اگر فایل ویژگی‌ها وجود داشت از آن استفاده کن، در غیر این صورت همان مسیر قبلی را ادامه بده
        if os.path.isfile(feature_file):
            df = pd.read_csv(feature_file)
        else:
            csv_path = os.path.join(datapath, datafile + "_data.csv")
            df = pd.read_csv(csv_path, header=0, index_col=0)

        # اگر ستون time وجود دارد، حذفش کن چون ویژگی عددی نیست
        if 'time' in df.columns or 'Time' in df.columns:
            time_col = 'time' if 'time' in df.columns else 'Time'
            df.drop(columns=[time_col], inplace=True)

        # فقط ستون‌های عددی باقی بمانند
        df = df.select_dtypes(include=[np.number])

        self.data = df
        self.datafile = datafile
        self.pred_horizion = pred_horizion
        self.input_steps = input_steps
        self.features = self.data.shape[1]

        # اسکیلر برای load (ستون هدف)
        #تغییر MINMAX به STANDARDSCALER
        #self.scaler_load = preprocessing.MinMaxScaler(feature_range=(0, 1))

        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

        
        # نسبت تقسیم
        self.split_ratio = list(split_ratio)

    def __repr__(self):
        return f"The dataset of {self.datafile} with input length {self.input_steps}"

    def _data_to_supervised(self):
        """تبدیل سری زمانی نرمال‌شده به داده نظارتی (X,y)"""
        column_data = []
        column_name = []

        # ورودی‌ها (lagها)
        for i in range(self.input_steps, 0, -1):
            column_data.append(self.data.shift(periods=i, axis=0))
            for col in self.data.columns:
                column_name.append(f"{col}(t-{i})")

        # خروجی‌ها (افق پیش‌بینی)
        for i in range(0, self.pred_horizion):
            column_data.append(self.data[self.data.columns[0]].shift(periods=-i, axis=0))
            column_name.append(f"{self.data.columns[0]}(t+{i+1})")

        reframed_data = pd.concat(column_data, axis=1)
        reframed_data.columns = column_name
        reframed_data.dropna(how='any', axis=0, inplace=True)
        return reframed_data

    def _standardizedata(self):
        """نرمال‌سازی تمام ستون‌های عددی در بازه [0,1]"""
        for col in self.data.columns:
            #scaler_tmp = preprocessing.MinMaxScaler(feature_range=(0, 1))
            scaler_tmp = StandardScaler()

            self.data[col] = scaler_tmp.fit_transform(self.data[[col]])

    def un_standardize(self, x):
        """بازگردانی خروجی به مقیاس اصلی بار"""
        #return self.scaler_load.inverse_transform(x)
        return self.scaler_y.inverse_transform(x)
    

    def _split_data(self, reframed_data):
        """تقسیم داده‌ها به train / val / test"""
        inputs = reframed_data.values[:, :-self.pred_horizion]
        targets = reframed_data.values[:, -self.pred_horizion:]

        num_examples = inputs.shape[0]
        train_size = int(num_examples * self.split_ratio[0])
        valid_size = int(num_examples * self.split_ratio[1])
        test_size = int(num_examples * self.split_ratio[2])

        def reshape_data(ip, op):
            ip = ip.reshape(ip.shape[0], self.input_steps, self.features)
            return ip, op

        train_ip, train_op = reshape_data(inputs[:train_size, :], targets[:train_size, :])
        valid_ip, valid_op = reshape_data(inputs[train_size:train_size+valid_size, :],
                                          targets[train_size:train_size+valid_size, :])
        test_ip, test_op = reshape_data(inputs[train_size+valid_size:train_size+valid_size+test_size, :],
                                        targets[train_size+valid_size:train_size+valid_size+test_size, :])

        return train_ip, train_op, valid_ip, valid_op, test_ip, test_op

    #def prepare_data(self):
        """Pipeline نهایی برای آماده‌سازی داده"""
        #self._standardizedata()
        #reframed_data = self._data_to_supervised()
        #return self._split_data(reframed_data)

    def prepare_data(self):
    """Pipeline نهایی بدون leakage، با اسکیلینگ بعد از split"""
    # 1) بدون اسکیلینگ اولیه، فقط supervised بساز
        reframed_data = self._data_to_supervised()

    # 2) split
        train_ip, train_op, valid_ip, valid_op, test_ip, test_op = self._split_data(reframed_data)

    # 3) اسکیلینگ X: فقط روی train fit، بقیه transform
        Ntr, L, d = train_ip.shape
        self.scaler_x.fit(train_ip.reshape(Ntr, -1))
        train_ip = self.scaler_x.transform(train_ip.reshape(Ntr, -1)).reshape(Ntr, L, d)

        Nva = valid_ip.shape[0]
        valid_ip = self.scaler_x.transform(valid_ip.reshape(Nva, -1)).reshape(Nva, L, d)

        Nte = test_ip.shape[0]
        test_ip  = self.scaler_x.transform(test_ip.reshape(Nte, -1)).reshape(Nte, L, d)

    # 4) اسکیلینگ y: فقط روی train fit، بقیه transform
        self.scaler_y.fit(train_op.reshape(-1, 1))
        train_op = self.scaler_y.transform(train_op.reshape(-1, 1)).reshape(train_op.shape)
        valid_op = self.scaler_y.transform(valid_op.reshape(-1, 1)).reshape(valid_op.shape)
        test_op  = self.scaler_y.transform(test_op.reshape(-1, 1)).reshape(test_op.shape)

    return train_ip, train_op, valid_ip, valid_op, test_ip, test_op







class Datasets(torch.utils.data.Dataset):
    def __init__(self, ip, op):
        super(Datasets, self).__init__()
        self.input = ip
        self.output = op
        self.len = ip.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.input[idx]), torch.tensor(self.output[idx])

    def __len__(self):
        return self.len

if __name__ == '__main__':
    Data = DataPrepare(datafile='AT', input_steps=12, pred_horizion=6)
    temp = Data.prepare_data()
    train_dataset = Datasets(temp[0], temp[1])
    for x, y in train_dataset:
        print(x.shape, y.shape)
        break
