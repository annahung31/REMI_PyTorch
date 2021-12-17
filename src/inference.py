from model import XLinear
import pickle
import random
import os
import time
import torch
import random
import yaml
import json

import utils


def main():
    cfg = yaml.full_load(open("config.yml", 'r')) 
    inferenceConfig = cfg['INFERENCE']

    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = inferenceConfig['gpuID']

    print('='*2, 'Inferenc configs', '='*5)
    print(json.dumps(inferenceConfig, indent=1, sort_keys=True))

    # checkpoint information
    CHECKPOINT_FOLDER = inferenceConfig['experiment_dir']
    midi_folder = os.path.join(CHECKPOINT_FOLDER, 'midi')
    checkpoint_type = inferenceConfig['checkpoint_type']
    if checkpoint_type == 'best_train':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'model_best.pth.tar')
        output_prefix = 'best_train_'
    elif checkpoint_type == 'best_val':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'model_best_val.pth.tar')
        output_prefix = 'best_val_'
    elif checkpoint_type == 'epoch_idx':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'ep_{}.pth.tar'.format(str(inferenceConfig['model_epoch'])))
        output_prefix = str(inferenceConfig['model_epoch'])+ '_'


    pretrainCfg = yaml.full_load(open(os.path.join(CHECKPOINT_FOLDER,"config.yml"), 'r')) 
    modelConfig = pretrainCfg['MODEL']

    # create result folder
    if not os.path.exists(midi_folder):
        os.mkdir(midi_folder)
    # load dictionary
    event2word, word2event = pickle.load(open(inferenceConfig['dictionary_path'], 'rb'))
    # declare model
    device = torch.device("cuda" if not inferenceConfig["no_cuda"] and torch.cuda.is_available() else "cpu")
    print('Device to generate:', device)

    # declare model
    model = XLinear(
            modelConfig,
            device,
            event2word=event2word, 
            word2event=word2event, 
            is_training=True)

    song_info = {}
    song_info['song_name'] = []
    song_info['song_time'] = []
    song_info['num_word'] = []
    song_info['ave_genTime_per_word'] = []
    # inference
    for i in range(inferenceConfig['num_sample']):
        # hash name
        name = '{:x}'.format(random.getrandbits(128))
        # inference
        song_total_time,  num_word, ave_genTime_per_word= model.inference(
            model_path = model_path,
            n_bar=48,
            strategies=['temperature', 'nucleus'],
            params={'t': 1.2, 'p': 0.9},
            bpm=120,
            output_path='{}/{}.mid'.format(midi_folder, output_prefix + name))


        song_info['song_name'].append(output_prefix + name)
        song_info['song_time'].append(song_total_time)
        song_info['num_word'].append(num_word)
        song_info['ave_genTime_per_word'].append(ave_genTime_per_word)

    utils.record_song_time(song_info)



if __name__ == '__main__':
    main()