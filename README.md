# ALBERT-Mongolian
ALBERT for Mongolian

## Dev Notes
| No | Started Date | Ended Date |    size    | seq128 (1) | seq512 (2) | tpu name  | output dir | tmux name |
| -- | --           | --         |    --      | --         | --         | --        | --         | --        |
|  1 | May 13, 20   | -          |    base    | 0          | 1M         | tfrc-v3-1 | gs://bucket-97tsogoo-gmail/pretrain/albert/output | 2 |
|  2 | May 14, 20   | -          |    base    | 900k       | 100k       | node-1    | gs://bucket-97tsogoo-gmail/pretrain/albert/pretrain1/output_512 | pretrain-1 |
|  3 | May 14, 20   | -          |    large   | 900k       | 100k       | -         | - | pretrain-3 |

* `(1) -> max sequence length 128`
* `(2) -> max sequence length 512`

This repo provides pretrained ALBERT model and trained SentencePiece model for Mongolia text. Training data is the Japanese wikipedia corpus from [Wikimedia Downloads](https://dumps.wikimedia.org/mnwiki/20200501/) and Mongolian News corpus.

As config file, the one [official repo provided](https://tfhub.dev/google/albert_base/3) is used.

As stated by official contributor [here](https://github.com/google-research/ALBERT/issues/104#issuecomment-548636183), we used only 512 for *max sequence length*.

Here we plannig to put pretraining loss
![Pretraining Loss](./images/pretraining_loss.png)

## Pretrain from Scratch

### Install Required packages
```
git clone --recurse-submodules https://github.com/bayartsogt-ya/albert-mongolian.git
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
```bash
python -m albert.run_pretraining \
    --input_file=... \
    --output_dir=... \
    --init_checkpoint=... \
    --albert_config_file=... \
    --do_train \
    --do_eval \
    --train_batch_size=512 \
    --eval_batch_size=64 \
    --max_seq_length=512 \
    --max_predictions_per_seq=20 \
    --optimizer='lamb' \
    --learning_rate=.00176 \
    --num_train_steps=1000000 \
    --num_warmup_steps=3125 \
    --save_checkpoints_steps=10000
```

## Evaluation
### TODO after pretraining done
- [ ] Loss Analysis 
- [ ] Comparison with Mongolian BERT and mBERT
- [ ] Benchmark цэгцлэх (Eduge classification, MN NER etc)
- [ ] Pre-trained model paper? https://arxiv.org/abs/1912.00690 ...
- [ ] ...

## Reference
1. [ALBERT - official repo](https://github.com/google-research/albert)
2. [WikiExtrator](https://github.com/attardi/wikiextractor)
3. [ALBERT - Japanese](https://github.com/alinear-corp/albert-japanese)
4. You's paper
5. ...

## Citation
```
@misc{albert-mongolian,
  author = {Bayartsogt Yadamsuren},
  title = {ALBERT Pretrained Model on Mongolian Datasets},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bayartsogt-ya/albert-mongolian/}}
}
```
