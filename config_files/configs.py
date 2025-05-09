"""
作者：yueyue
日期：2023年10月30日
"""

class Config(object):
    def __init__(self):
        # model configs
        self.domain_num_classes = 23
        self.label_num_classes = 3
        # training configs
        self.num_epoch = 50
        self.batch_size = 128
        self.save_ckp = True
        self.preprocessing_method_acf = 1

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.weight_decay = 0
        self.k_fold_num = 6
        self.n_average_times = 20

        # data parameters
        self.pig_infoset = [1, 2, 3, 5, 6, 7, 17, 18, 19, 20, 21, 25, 26, 28, 30, 31, 32, 36, 37, 40, 41, 44, 45]  # all pigs
        self.pig_train_set = [1, 2, 3, 5, 6, 7, 17, 18, 19, 20, 21, 25, 26, 28, 30, 31, 32, 36, 37, 40, 41, 44,45]  # train and valid pigs
        self.pig_dic = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5, 17: 6, 18: 7, 19: 8, 20: 9, 21: 10, 25: 11, 26: 12, 28: 13,
                   30: 14, 31: 15, 32: 16, 36: 17, 37: 18, 40: 19, 41: 20, 44: 21, 45: 22}

        # self.CPC_DA = CPC_DA_Configs

class Config2(object):
    def __init__(self):
        # model configs
        self.domain_num_classes = 23
        self.label_num_classes = 3
        # training configs
        self.num_epoch = 50
        self.batch_size = 128
        self.save_ckp = True
        self.preprocessing_method_acf = 0

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.weight_decay = 0
        self.k_fold_num = 6
        self.n_average_times = 20

        # data parameters
        self.pig_infoset = [1, 2, 3, 5, 6, 7, 17, 18, 19, 20, 21, 25, 26, 28, 30, 31, 32, 36, 37, 40, 41, 44, 45]  # all pigs
        self.pig_train_set = [1, 2, 3, 5, 6, 7, 17, 18, 19, 20, 21, 25, 26, 28, 30, 31, 32, 36, 37, 40, 41, 44,45]  # train and valid pigs
        self.pig_dic = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5, 17: 6, 18: 7, 19: 8, 20: 9, 21: 10, 25: 11, 26: 12, 28: 13,
                   30: 14, 31: 15, 32: 16, 36: 17, 37: 18, 40: 19, 41: 20, 44: 21, 45: 22}
