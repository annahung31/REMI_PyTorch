
* Current best pre-trained model:https://drive.google.com/drive/folders/1xwu_geEUC2o-bLrMWm-DYJAOiVsnWbBK?usp=sharing 
* commit 962f6368b894cc8c5650d7e1b9ef26f38d5f37d0


## Train from scratch
1. Get the midi dataset from The original repo.
2. Change the paths information in `src/config.yml`
3. Use `src/data.py` to get the `vocabulary.pkl` and `train_data.npy`
4. Run `src/train.py` to train.