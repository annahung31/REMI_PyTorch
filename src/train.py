from model import XLinear
import numpy as np
import pickle
import os
import datetime
import json
from collections import OrderedDict
import torch
import yaml
import ipdb

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'



def main():

    modelConfig, trainConfig = get_configs()
    # load dictionary
    event2word, word2event = pickle.load(open(os.path.join(trainConfig['ROOT'],'dictionary.pkl'), 'rb'))

    # load train data
    if not modelConfig['adaptive_len']:
        training_data = np.load(os.path.join(trainConfig['ROOT'],'train_data.npy'))
    else:
        training_data = np.load(os.path.join(trainConfig['ROOT'],'train_data_all.npy'))
    

    

    device = torch.device("cuda:{}".format(trainConfig['gpuID']) if not trainConfig["no_cuda"] and torch.cuda.is_available() else "cpu")
    print('Device to train:', device)
    
    resume = trainConfig['resume_training_model']

    # declare model
    model = XLinear(
            modelConfig,
            device,
            event2word=event2word, 
            word2event=word2event, 
            is_training=True)
    # train
    model.train(training_data,
                trainConfig,
                device,
                resume)
                




def get_configs():
    cfg = yaml.full_load(open("config.yml", 'r')) 

    modelConfig = cfg['MODEL']
    trainConfig = cfg['TRAIN']


    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_Dir = os.path.join(trainConfig['output_dir'],cur_date)
    if not os.path.exists(experiment_Dir):
        os.mkdir(experiment_Dir) 
    print('Experiment: ', experiment_Dir)
    trainConfig.update({'experiment_Dir': experiment_Dir})


    with open(os.path.join(experiment_Dir, 'config.yml'), 'w') as f:
        doc = yaml.dump(cfg, f)

    print('='*5, 'Model configs', '='*5)
    print(json.dumps(modelConfig, indent=1, sort_keys=True))
    print('='*2, 'Training configs', '='*5)
    print(json.dumps(trainConfig, indent=1, sort_keys=True))
    return modelConfig, trainConfig


if __name__ == '__main__':
    main()


