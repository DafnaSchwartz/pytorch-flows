import numpy as np
import datasets
import datasets.util
import ipdb
import os

non_gait_is_normal = True
eval_pd = False
eval_hc = False

class PD_WRIST:
    class Data:
        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):

        trn, val, tst = load_data_normalised()

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

def load_data_split():
    non_walking_data = []
    pd_walking_data = []
    hc_walking_data = []
    pd_indication = []
    
    data_dir = '/home/dafnas1/datasets/pd_wrist/data_split_to_windows'
    #data = np.load('/home/dafnas1/my_repo/gait_anomaly_detection/Seq2Seq-gait-analysis/healthy_wrist_walking_data.npy')[:,2:5]
    for file in os.listdir(data_dir):
        window_data = np.load(os.path.join(data_dir,file))['arr_0']
        window_data = window_data - np.mean(window_data, axis=1, keepdims=True)
        window_data_power = np.sqrt(np.sum(np.power(window_data,2),axis=2))
        if 'non' in file:
            non_walking_data.append(window_data_power[::10,:])
            if 'PD' in file:
                pd_indication.append(np.ones_like(window_data_power[::10,0]))
            if 'HC' in file:
                pd_indication.append(np.zeros_like(window_data_power[::10,0]))
        else:
            if 'PD' in file:
                pd_walking_data.append(window_data_power)
            if 'HC' in file:
                hc_walking_data.append(window_data_power)
    non_walking_data_arr = np.concatenate(non_walking_data)
    walking_data_arr = np.concatenate(pd_walking_data + hc_walking_data)
    pd_indication = np.concatenate(pd_indication)
    seed = 42
    rng = np.random.RandomState(seed)
    rng.shuffle(non_walking_data_arr)
    rng = np.random.RandomState(seed)
    rng.shuffle(pd_indication)
    # use when gait is anomaly and non-gait is normal
    
    if non_gait_is_normal:
        N_validate = int(0.1 * non_walking_data_arr.shape[0])
        data_validate = non_walking_data_arr[-N_validate:]
        data_train = non_walking_data_arr[0:-N_validate]
        data_test = walking_data_arr
        #data_test = np.concatenate(walking_data_arr)
        pd_indication_validate = pd_indication[-N_validate:]
        if eval_pd:
            data_validate = data_validate[pd_indication_validate==1,:]
            data_test = np.concatenate(pd_walking_data)
        if eval_hc:
            data_validate = data_validate[pd_indication_validate==0,:]
            data_test = np.concatenate(hc_walking_data)
    else:
        # use when gait is normal and non-gait is anomaly
        rng.shuffle(walking_data_arr)
        N_validate = int(0.1 * walking_data_arr.shape[0])
        data_validate = walking_data_arr[-N_validate:]
        data_train = walking_data_arr[0:-N_validate]
        data_test = non_walking_data_arr[:int(0.1 * non_walking_data_arr.shape[0])]

    return data_train, data_validate, data_test

def load_data_normalised():
    filtered = False
    if filtered:
        if non_gait_is_normal:
            data_train = np.load('non_walking_trn_filter.npy')
            data_validate = np.load('non_walking_val_filter.npy')
            data_test = np.load('walking_test.npy')
        else:
            walking_data = np.load('walking_test.npy')
            N_validate = int(0.1 * walking_data.shape[0])
            data_validate = walking_data[-N_validate:]
            data_train = walking_data[0:-N_validate]
            data_test = np.load('non_walking_val_filter.npy')
        return data_train, data_validate, data_test
    data_train, data_validate, data_test = load_data_split()
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test