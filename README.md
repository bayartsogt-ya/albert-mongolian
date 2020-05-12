# ALBERT-Mongolian
ALBERT for Mongolian

This repo provides pretrained ALBERT model and trained SentencePiece model for Mongolia text. Training data is the Japanese wikipedia corpus from [Wikimedia Downloads](https://dumps.wikimedia.org/mnwiki/20200501/) and Mongolian News corpus.

As config file, the one [official repo provided](https://tfhub.dev/google/albert_base/3) is used.

As stated by official contributor [here](https://github.com/google-research/ALBERT/issues/104#issuecomment-548636183), we used only 512 for *max sequence length*.

Here we plannig to put pretraining loss
![Pretraining Loss](./images/pretraining_loss.png)

## Pretrain from Scratch
### Install Required packages
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
Now you can use `mn_corpus/*.txt` to produce `*tf_record` files. Here the first parameter is path to `*.txt` files and second one for max sequence length.
```bash
source build_pretraining_data.sh  ./mn_corpus  512
```
After the above command produces `*.tf_record` files, you should upload them to Google Cloud Storage (GCS).
```source
gsutil -m cp ./mn_corpus/*.tf_record gs://YOU_BUCKET/folder/
```
### Start Pretraining
```
```


## Evaluation

## Reference

## Citation
