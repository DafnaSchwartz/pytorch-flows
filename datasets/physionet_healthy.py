import numpy as np
import datasets
import datasets.util
import ipdb
import os

non_gait_is_normal = False

class PHYSIONET:
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
    
    data_dir = '/home/dafnas1/my_repo/gait_anomaly_detection/Seq2Seq-gait-analysis/gait_detection_pd'
    #data = np.load('/home/dafnas1/my_repo/gait_anomaly_detection/Seq2Seq-gait-analysis/healthy_wrist_walking_data.npy')[:,2:5]
    non_walking_data_arr = np.load(os.path.join(data_dir,'phisionet_non_walking_windows.npy'))
    non_walking_data_arr = non_walking_data_arr - np.mean(non_walking_data_arr, axis=1, keepdims=True)
    non_walking_data_arr = np.sqrt(np.sum(np.power(non_walking_data_arr,2),axis=2))
    
    walking_data_arr = np.load(os.path.join(data_dir,'phisionet_walking_windows.npy'))
    walking_data_arr = walking_data_arr - np.mean(walking_data_arr, axis=1, keepdims=True)
    walking_data_arr = np.sqrt(np.sum(np.power(walking_data_arr,2),axis=2))
    ipdb.set_trace()
    rng = np.random.RandomState(42)
    rng.shuffle(non_walking_data_arr)
    

    # use when gait is anomaly and non-gait is normal
    
    if non_gait_is_normal:
        N_validate = int(0.1 * non_walking_data_arr.shape[0])
        data_validate = non_walking_data_arr[-N_validate:]
        data_train = non_walking_data_arr[0:-N_validate]
        data_test = walking_data_arr
        #data_test = np.concatenate(walking_data_arr)
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