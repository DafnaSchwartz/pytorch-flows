import numpy as np
import datasets
import datasets.util
import ipdb

class GAIT:
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
    #data = np.load('/home/dafnas1/my_repo/gait_anomaly_detection/Seq2Seq-gait-analysis/healthy_wrist_walking_data.npy')[:,2:5]
    data = np.load('/home/dafnas1/my_repo/gait_anomaly_detection/Seq2Seq-gait-analysis/windows_2sec_healthy_wirst_walking_data.npy')[:,:,0]
    data_hd_for_test = np.load('/home/dafnas1/my_repo/gait_anomaly_detection/Seq2Seq-gait-analysis/2sec_win_walking_HD_data_list.npy')[:,:,0]
    #N_test = int(0.1 * data.shape[0])
    #data_test = data[-N_test:]
    data = data[:,::2] - np.reshape(np.mean(data, axis=1), [-1,1])
    data_test = data_hd_for_test
    data_test = data_test[:,::2] - np.reshape(np.mean(data_test, axis=1), [-1,1])
    #data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test

def load_data_normalised():
    data_train, data_validate, data_test = load_data_split()
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test