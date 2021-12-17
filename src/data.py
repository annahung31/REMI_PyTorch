import os
from collections import Counter
from glob import glob
import numpy as np
import pickle
import utils
from tqdm import tqdm
import string
import random
def extract_event(file_path):
    note_items, ticks_per_beat = utils.read_midi(file_path)
    note_items = utils.quantize_items(note_items, ticks_per_beat/4)
    groups = utils.group_items(note_items, ticks_per_beat*4)
    events = utils.item2event(groups, ticks_per_beat)
    return events


def create_folders(ROOT):
    folder_list = ['event', 'word']
    for fo in folder_list:
        if not os.path.exists(os.path.join(ROOT, fo)):
            os.mkdir(os.path.join(ROOT, fo))


if __name__ == '__main__':
    ROOT = 'dataset/775_subset'
    create_folders(ROOT)
    # list file
    files = sorted(glob(os.path.join(ROOT,'midi','*.midi')))
    print(len(files))

    print('='*5, 'extract events', '='*5)
    # extract events
    for i in tqdm(range(len(files))):
        events = extract_event(files[i])
        folder = files[i].split('/')[-2]
        #name = files[i].split('/')[-1][:5] +'_'+ files[i].split('/')[-1][-10:-4] + '_'+''.join(random.sample(string.ascii_letters+string.digits, 10))
        name = files[i].split('/')[-1][:-5]
        path = os.path.join(ROOT,'event','{}...{}.pkl'.format(folder, name))
        pickle.dump(events, open(path, 'wb'))
        
    # count for dictionary
    print('='*5, 'count for dictionary', '='*5)
    event_files = glob(os.path.join(ROOT,'event','*.pkl'))
    print(len(event_files))
    data = []
    for file in tqdm(event_files):
        for event in pickle.load(open(file, 'rb')):
            data.append('{}_{}'.format(event.name, event.value))
    counts = Counter(data)
    event2word = {key: i for i, key in enumerate(counts.keys())}
    word2event = {i: key for i, key in enumerate(counts.keys())}
    path = os.path.join(ROOT,'dictionary.pkl')
    pickle.dump((event2word, word2event), open(path, 'wb'))
    
    # convert to word
    print('='*5, 'convert to word', '='*5)
    event_files = glob(os.path.join(ROOT,'event','*.pkl'))
    event2word, word2event = pickle.load(open(os.path.join(ROOT,'dictionary.pkl'), 'rb'))
    for file in tqdm(event_files):
        events = pickle.load(open(file, 'rb'))
        words = []
        for e in events:
            word = event2word['{}_{}'.format(e.name, e.value)]
            words.append(word)
        name = file.split('/')[-1]
        path = os.path.join(ROOT,'word','{}.npy'.format(name))
        np.save(path, words)

    # create training data
    print('='*5, 'create training data', '='*5)
    WINDOW_SIZE = 512
    GROUP_SIZE = 5
    INTERVAL = GROUP_SIZE * 2
    word_files = sorted(glob(os.path.join(ROOT,'word','*.npy')))
    print(len(word_files))
    segments = []
    for file in tqdm(word_files):
        words = np.load(file)
        pairs = []
        for i in range(0, len(words)-WINDOW_SIZE-1, WINDOW_SIZE):
            x = words[i:i+WINDOW_SIZE]
            y = words[i+1:i+WINDOW_SIZE+1]
            pairs.append([x, y])
        pairs = np.array(pairs)
        # abandon the last
        for i in np.arange(0, len(pairs)-GROUP_SIZE, INTERVAL):
            data = pairs[i:i+GROUP_SIZE]
            if len(data) == GROUP_SIZE:
                segments.append(data)
    training_data = np.array(segments)
    print(training_data.shape)
    np.save(os.path.join(ROOT,'train_data.npy'), segments)
