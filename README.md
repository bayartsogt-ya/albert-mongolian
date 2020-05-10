# ALBERT-Mongolian
ALBERT for Mongolian


## Pretrain from Scratch
### Install Required packeges
```
pip install -r requirement.txt
```
### Download data
```bash
python3 datasets/dl_and_preprop_mn_wiki.py         # Mongolian Wikipedia
python3 datasets/dl_and_preprop_mn_news.py         # 700 million words Mongolian news data set
cat mn_corpus/*.txt > all.txt                      # Put them all to one file
```
### Train SentencePiece model
First you need to [install sentencepiece from source](https://github.com/google/sentencepiece#c-from-source)
Then start training (which requires ~30GB memory)
```
train_spm_model.sh
```
### Build tf records for pretraining

### Start Pretraining

## Evaluation

## Reference

## Citation
