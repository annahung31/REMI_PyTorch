import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import miditoolkit
from tqdm import tqdm
import shutil
# import modules
import utils
import os
import time
import json
import ipdb
from sklearn.model_selection import train_test_split
from modules import MemTransformerLM
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import pyplot


class XLTransformer(object):
    def __init__(self, modelConfig, device, event2word, word2event, is_training=True):

        self.event2word = event2word
        self.word2event = word2event
        self.modelConfig = modelConfig

        # model settings    
        self.n_layer= modelConfig['n_layer']
        self.d_model = modelConfig['d_model']
        self.seq_len= modelConfig['seq_len']
        self.mem_len =  modelConfig['mem_len']

        self.tgt_len = modelConfig['tgt_len']
        self.ext_len = modelConfig['ext_len']
        self.eval_tgt_len = modelConfig['eval_tgt_len']


        
        self.init = modelConfig['init']
        self.init_range = modelConfig['init_range']
        self.init_std = modelConfig['init_std']
        self.proj_init_std = modelConfig['proj_init_std']


        #mode
        self.is_training = is_training
        self.device = device
        self.tied = modelConfig['tied']
        self.sample_softmax = modelConfig['sample_softmax']      
        self.adaptive_len = modelConfig['adaptive_len']        
        
       


    def init_weight(self, weight):
        if self.init == 'uniform':
            nn.init.uniform_(weight, -self.init_range, self.init_range)
        elif self.init == 'normal':
            nn.init.normal_(weight, 0.0, self.init_std)

    def init_bias(self, bias):
        nn.init.constant_(bias, 0.0)


    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                self.init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('AdaptiveEmbedding') != -1:
            if hasattr(m, 'emb_projs'):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, self.proj_init_std)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                self.init_weight(m.weight)
        elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
            if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
                self.init_weight(m.cluster_weight)
            if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
                self.init_bias(m.cluster_bias)
            if hasattr(m, 'out_projs'):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, self.proj_init_std)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, self.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                self.init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                self.init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                self.init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                self.init_bias(m.r_bias)




    def get_model(self, pretrain_model=None):
        model = MemTransformerLM(self.modelConfig, is_training=self.is_training)

        st_eopch = 0
        if pretrain_model:
            checkpoint = torch.load(pretrain_model)
            print('Pretrained model config:')
            print('epoch: ', checkpoint['epoch'])
            print('best_loss: ', checkpoint['best_loss'])
            print(json.dumps(checkpoint['model_setting'], indent=1, sort_keys=True))
            print(json.dumps(checkpoint['train_setting'], indent=1, sort_keys=True))
            try:
                model.load_state_dict(checkpoint['state_dict'])
                print('{} loaded.'.format(pretrain_model))
                
                
            except:
                print('Loaded weights have different shapes with the model. Please check your model setting.')
                exit()
            st_eopch = checkpoint['epoch']+1

        else:
            model.apply(self.weights_init)
            model.word_emb.apply(self.weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing


        return st_eopch ,model.to(self.device)




    def save_checkpoint(self, state, root, save_freq=10, is_best=False, is_eval=False):
        if is_eval:
            torch.save(state, os.path.join(root,'model_best_val.pth.tar'))
        
        if is_best:
            torch.save(state, os.path.join(root,'model_best.pth.tar'))

        if state['epoch'] % save_freq == 0:
            torch.save(state, os.path.join(root,'ep_{}.pth.tar'.format(state['epoch'])))






    def early_stop(self, val_loss, train_loss, strategy='overfitting', stop_epoch=10):
        '''
        strategy: 'overfitting', 'specific_loss'
        '''
        if strategy == 'overfitting':
            _val_loss = val_loss[1:] + [0]
            delta = [(_val_loss[i] - val_loss[i])>0 for i in range(len(val_loss))][:-1]
            if True in delta:
                raise_num = len(delta)-delta.index(True)
                if raise_num > stop_epoch:
                    return True
                else:
                    return False
            else:
                return False
        
        elif strategy == 'specific_loss':
            return True if train_loss <= 0.25 else False


    def train_loss_record(self, epoch, train_loss,checkpoint_dir, val_loss=None):

        if val_loss:
            df = pd.DataFrame({'epoch': [epoch+1],
                    'train_loss': ['%.3f'%train_loss],
                    'val_loss': ['%.3f'%val_loss]})
            
        else:
            df = pd.DataFrame({'epoch': [epoch+1],
                    'train_loss': ['%.3f'%train_loss]})

        csv_file = os.path.join(checkpoint_dir, 'loss.csv')

        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(os.path.join(checkpoint_dir, 'loss.csv'), mode='a', header=False,  index=False)


        


        


    def get_batch(self, data_ROOT, group_size):
        '''
        words: ROOT of datas
        output: (bsz, group_size, 2, seq_len)
        '''
        WINDOW_SIZE = self.seq_len
        GROUP_SIZE = group_size
        INTERVAL = GROUP_SIZE * 2
        word_files = sorted(glob(os.path.join(data_ROOT,'word','*.npy')))
        
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

        return training_data



    def get_optim(self, model, OPTIM, mom, lr):

        if OPTIM.lower() == 'sgd':
            if self.sample_softmax > 0:
                dense_params, sparse_params = [], []
                for param in model.parameters():
                    if param.size() == model.word_emb.weight.size():
                        sparse_params.append(param)
                    else:
                        dense_params.append(param)
                optimizer_sparse = optim.SGD(sparse_params, lr=lr * 2)
                optimizer = optim.SGD(dense_params, lr=lr, momentum=mom)
            else:
                optimizer = optim.SGD(model.parameters(), lr=lr,
                    momentum=mom)
        elif OPTIM.lower() == 'adam':
            if self.sample_softmax > 0:
                dense_params, sparse_params = [], []
                for param in model.parameters():
                    if param.size() == model.word_emb.weight.size():
                        sparse_params.append(param)
                    else:
                        dense_params.append(param)
                optimizer_sparse = optim.SparseAdam(sparse_params, lr=lr)
                optimizer = optim.Adam(dense_params, lr=lr)
            else:
                optimizer = optim.Adam(model.parameters(), lr=lr)
        elif OPTIM.lower() == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=lr)
        
        return optimizer





    def get_scheduler(self, SCHEDULER, optimizer, 
                             max_step, eta_min, warmup_step, 
                             patience, lr_min, decay_rate):

        #### scheduler
        if SCHEDULER == 'cosine':
            # here we do not set eta_min to lr_min to be backward compatible
            # because in previous versions eta_min is default to 0
            # rather than the default value of lr_min 1e-6
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                max_step, eta_min=eta_min) # should use eta_min arg
            '''
            if sample_softmax > 0:
                scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
                    max_step, eta_min=eta_min) # should use eta_min arg
            '''
        elif SCHEDULER == 'inv_sqrt':
            # originally used for Transformer (in Attention is all you need)
            def lr_lambda(step):
                # return a multiplier instead of a learning rate
                if step == 0 and warmup_step == 0:
                    return 1.
                else:
                    return 1. / (step ** 0.5) if step > warmup_step \
                        else step / (warmup_step ** 1.5)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif SCHEDULER == 'dev_perf':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                factor=decay_rate, patience=patience, min_lr=lr_min)

            '''
            if sample_softmax > 0:
                scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sparse,
                    factor=decay_rate, patience=patience, min_lr=lr_min)
            '''
        elif SCHEDULER == 'constant':
            pass
        
        return scheduler



    def validate(self, val_data, batch_size, model, seed, max_eval_steps):
        torch.manual_seed(seed)
        group_size = 5
        val_loss = []
        num_batches = len(val_data) // batch_size
        model.eval()

        # If the model does not use memory at all, make the ext_len longer.
        # Otherwise, make the mem_len longer and keep the ext_len the same.
        if self.mem_len == 0:
            model.reset_length(self.eval_tgt_len,
                self.ext_len+self.tgt_len-self.eval_tgt_len, self.mem_len)
        else:
            model.reset_length(self.eval_tgt_len,
                self.ext_len, self.mem_len+self.tgt_len-self.eval_tgt_len)


        total_len, total_loss = 0, 0.
        with torch.no_grad():
            
            for i in range(num_batches):
                mems = tuple()
                if max_eval_steps > 0 and i >= max_eval_steps:
                    break
                segments = val_data[batch_size*i:batch_size*(i+1)]
                for j in range(group_size):
                    batch_x = segments[:, j, 0, :]
                    batch_y = segments[:, j, 1, :]
                    
                    batch_x = torch.from_numpy(batch_x).to(self.device)
                    batch_y = torch.from_numpy(batch_y).to(self.device)
                    ret = model(batch_x, batch_y, *mems)
                    loss, mems = ret[0], ret[1:]
                    loss = loss.mean()
                    total_loss += self.seq_len * loss.float().item()
                    total_len += self.seq_len


        # Switch back to the training mode
        model.reset_length(self.tgt_len, self.ext_len, self.mem_len)
        model.train()


        return total_loss / total_len


    def train(self, training_data, trainConfig, device, resume):
        
        checkpoint_dir = trainConfig['experiment_Dir']
        batch_size = trainConfig['batch_size']
        group_size = 5
        data_ROOT = trainConfig['ROOT']
        torch.manual_seed(trainConfig["seed"])

        if self.adaptive_len:
            training_data = self.get_batch(data_ROOT, group_size)
        

        train_data, val_data = train_test_split(training_data, test_size=0.1, random_state=42)
        print('Data to train  :{}, to validate: {}'.format(train_data.shape, val_data.shape))
        #Prepare data
        index = np.arange(len(train_data))
        np.random.shuffle(index)
        train_data = train_data[index]
        num_batches = len(train_data) // batch_size

        #Prepare model
        if resume != 'None':
            st_epoch, model = self.get_model(resume)
            print('Continue to train from {} epoch'.format(st_epoch))
        else:
            st_epoch, model = self.get_model()

        optimizer = self.get_optim(model, trainConfig['optim'], trainConfig['mom'], trainConfig['lr'])
        scheduler = self.get_scheduler(trainConfig['SCHEDULER'], optimizer, 
                                        trainConfig['max_step'], 
                                        trainConfig['eta_min'], trainConfig['warmup_step'], 
                                        trainConfig['patience'], trainConfig['lr_min'], trainConfig['decay_rate'])


        train_step = 0
        best_loss = 1000
        #best_eval_loss = 1000
        #epoch_val_loss = []
        epoch_train_loss = []
        save_freq = trainConfig['save_freq']
        
        print('>>> Start training')
        for epoch in range(st_epoch, trainConfig['num_epochs']):
            
            train_loss = []
            st_time = time.time()
            model.train()

            
            for i in tqdm(range(num_batches)):
                model.zero_grad()
                segments = train_data[batch_size*i:batch_size*(i+1)]
                mems = tuple()

                for j in range(group_size):
                    
                    batch_x = segments[:, j, 0, :]
                    batch_y = segments[:, j, 1, :]
                    
                    batch_x = torch.from_numpy(batch_x).permute(1, 0).contiguous().to(self.device)  # (seq_len, bsz)
                    batch_y = torch.from_numpy(batch_y).permute(1, 0).contiguous().to(self.device)
                    
                    ret = model(batch_x, batch_y, *mems)
                    loss, mems = ret[0], ret[1:]
                    
                    loss = loss.float().mean().type_as(loss)                     
                    train_loss.append(loss.item())

                    loss.backward()
                optimizer.step()


                #setup scheduler, update lr
                train_step += 1
                if scheduler in ['cosine', 'constant', 'dev_perf']:
                    # linear warmup stage
                    if train_step < trainConfig['warmup_step']:
                        curr_lr = trainConfig['lr'] * train_step / trainConfig['warmup_step']
                        optimizer.param_groups[0]['lr'] = curr_lr

                    else:
                        if scheduler == 'cosine':
                            scheduler.step(train_step)

                elif scheduler == 'inv_sqrt':
                    scheduler.step(train_step)


            #val_loss = self.validate(val_data, batch_size, model, trainConfig["seed"], trainConfig['max_eval_steps'])
            curr_train_loss = sum(train_loss) / len(train_loss)


            #epoch_val_loss.append(val_loss)
            epoch_train_loss.append(curr_train_loss)
            # epoch_info = 'Train Loss: {:.5f} , Val Loss: {:.5f}, T: {:.3f}'.format(curr_train_loss, val_loss, time.time()-st_time)
            epoch_info = 'Epoch: {}, Train Loss: {:.5f} ,  T: {:.3f}'.format(epoch+1, curr_train_loss, time.time()-st_time)
            print(epoch_info)

            # self.train_loss_record(epoch, curr_train_loss, checkpoint_dir, val_loss)
            self.train_loss_record(epoch, curr_train_loss, checkpoint_dir)
            
            is_best = (curr_train_loss < best_loss)

            

            self.save_checkpoint({
                    'epoch': epoch + 1,
                    'model_setting': self.modelConfig,
                    'train_setting': trainConfig,
                    'state_dict': model.state_dict(),
                    'best_loss': curr_train_loss,
                    'optimizer' : optimizer.state_dict(),
                                }, 
                    checkpoint_dir, 
                    save_freq,
                    is_best, is_eval=False)
            

            '''
            if val_loss < #:
                self.save_checkpoint({
                        'epoch': epoch + 1,
                        'model_setting': self.modelConfig,
                        'train_setting': trainConfig,
                        'state_dict': model.state_dict(),
                        'best_loss': val_loss,
                        'optimizer' : optimizer.state_dict(),
                    }, checkpoint_dir,
                    is_eval=True)
                # = val_loss 


            
            if trainConfig['EARLY_STOP'] != 'None':
                if self.early_stop(epoch_val_loss, curr_train_loss, strategy=trainConfig['EARLY_STOP']):
                    print('Stop training at epoch', epoch)
                    break
            '''

            if curr_train_loss < 0.01:
                print('Experiment [{}] finished at loss < 0.01.'.format(checkpoint_dir))
                break

        

    def inference(self, model_path, n_bar, strategies, params, bpm, output_path):
        _, model = self.get_model(model_path)
        model.eval()
        
        # initial start
        words = [[]]
        record_time = []
        # add beat
        words[-1].append(self.event2word['Bar_None'])
        # add position
        words[-1].append(self.event2word['Position_1/16'])
        # add random note on
        candidates = [v for k, v in self.event2word.items() if 'Note On' in k]
        
        words[-1].append(np.random.choice(candidates, size=1)[0])
        
        # initialize mem
        mems = tuple()


        song_init_time = time.time()
        # generate
        initial_flag = True
        generate_n_bar = 0
        batch_size = 1
        while generate_n_bar < n_bar:
            
            # prepare input
            if initial_flag:
                temp_x = np.zeros((len(words[0]), batch_size))

                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[z][b] = t
                
                initial_flag = False
            else:
                temp_x = np.zeros((1, batch_size))
                
                for b in range(batch_size):
                    temp_x[0][b] = words[b][-1] ####?####
                    

            temp_x = torch.from_numpy(temp_x).long().to(self.device)     
            st_time = time.time()
            
            _logits, mems = model.generate(temp_x, *mems)

            record_time.append(time.time() - st_time)

            logits = _logits.cpu().squeeze().detach().numpy()


            # temperature or not
            if 'temperature' in strategies:
                probs = self.temperature(logits=logits, temperature=params['t'])
                
            else:
                probs = self.temperature(logits=logits, temperature=1.)
            # sampling
            word = self.nucleus(probs=probs, p=params['p'])    
            words[0].append(word)
            
            # record n_bar
            if word == self.event2word['Bar_None']:
                generate_n_bar += 1


        song_total_time = time.time() - song_init_time
        print('Total words generated: ', len(words[0]))
        print('Average WORD generation time: ', sum(record_time) / len(record_time))
        #np.save(output_path[:-4]+'.npy', np.array(words[0]))
        # write midi
        utils.write_midi(
            words=words[0],
            ticks_per_beat=480,
            bpm=bpm,
            word2event=self.word2event,
            output_path=output_path)    
        

        

        

        #utils.record_time(
        #    times=record_time,
        #    output_path=output_path
        #)
        print('midi saved to ', output_path)
        return song_total_time, len(words[0]), sum(record_time) / len(record_time)




    ########################################
    # search strategy: temperature (re-shape)
    ########################################
    
    def temperature(self, logits, temperature):
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        return probs

    ########################################
    # search strategy: topk (truncate)
    ########################################
    def topk(self, probs, k):
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:k]
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    ########################################
    # search strategy: nucleus (truncate)
    ########################################
    
    def nucleus(self, probs, p):
        probs /= sum(probs)
        sorted_probs = np.sort(probs)[::-1]
        sorted_index = np.argsort(probs)[::-1]
        cusum_sorted_probs = np.cumsum(sorted_probs)
        after_threshold = cusum_sorted_probs > p
        if sum(after_threshold) > 0:
            last_index = np.where(after_threshold)[0][-1]
            candi_index = sorted_index[:last_index]
        else:
            candi_index = sorted_index[:3] # just assign a value
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word