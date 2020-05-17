# ALBERT-Mongolian
ALBERT for Mongolian

## Dev Notes
**TODO**
- [ ] uncased sentencepiece training
- [ ] uncased tf record build
- [ ] uncased albert-base training (maybe on tpu v2?)

**Training log:**
Tensorboard Link => https://colab.research.google.com/drive/1QVRmC73yImJL1U0VvhcbhVmHoONKZYVO?authuser=1#scrollTo=dwin7ZHipLou

| No | Started Date | Ended Date | model size | batch size | seq128 (1) | seq512 (2) | tpu name  | output dir | tmux name |
| -- | --           | --         |    --      |     --     | --         | --         | --        | --         | --        |
| 1  | May 13, 20   | failed     |  base      | 512        | 0          | 1M         | tfrc-v3-1 | gs://bucket-97tsogoo-gmail/pretrain/albert/output | 2 |
| 2  | May 14, 20   | May 16, 20 |  base      | 512        | 900k       | 100k       | node-1    | gs://bucket-97tsogoo-gmail/pretrain/albert/pretrain1/output_* | pretrain-1 |
| 3  | May 14, 20   | -          |  large     | 512        | 900k       | 100k       | node-2    | gs://bucket-97tsogoo-gmail/pretrain/albert/pretrain3/output_* | pretrain-3 |
| 4  | May 14, 20   | -          |  xlarge    | 128        | 3.6M       | 400k       | node-3    | gs://bucket-97tsogoo-gmail/pretrain/albert/pretrain4/output_* | pretrain4  |
| 5  | May 16, 20   | -          |  base      | 512        | 0          | 1M         | tfrc-v3-1 | gs://bucket-97tsogoo-gmail/pretrain/albert/pretrain5/output_* | pretrain-5 |
| 6  | May 17, 20   | -          |  large     | 512        | 0          | 4M         | node-1    | gs://bucket-97tsogoo-gmail/pretrain/albert/pretrain6/output_* | pretrain-6 |
| 6  |              | -          |  xxlarge   | --         | 900k       | 100k       | -         | - | - |
| 7  |              | -          |  BERT-large| 512        | 0          | 4M         | -         | - | - |

* `(1) -> max sequence length 128`
* `(2) -> max sequence length 512`

This repo provides pretrained ALBERT model and trained SentencePiece model for Mongolia text. Training data is the Mongolian wikipedia corpus from [Wikipedia Downloads](https://dumps.wikimedia.org/mnwiki/20200501/) and Mongolian News corpus.

Here we plannig to put pretraining loss
![Pretraining Loss](./images/pretraining_loss.png)

## Pretrain from Scratch

### Install Required packages
```
git clone --recurse-submodules https://github.com/bayartsogt-ya/albert-mongolian.git
pip install -r requirement.txt
```

### Download data
This section is done by [tugstugi/mongolian-bert#data-preparation](https://github.com/tugstugi/mongolian-bert#data-preparation)
```bash
python3 datasets/dl_and_preprop_mn_wiki.py         # Mongolian Wikipedia
python3 datasets/dl_and_preprop_mn_news.py         # 700 million words Mongolian news data set
cat mn_corpus/*.txt > all.txt                      # Put them all to one file
```

### Train SentencePiece model
First you need to [install sentencepiece from source](https://github.com/google/sentencepiece#c-from-source)
Then start training (which requires ~30GB memory)

if you are training uncased model, you need to lowercase the input data.
```bash
python do_lowercase.py --input ./all.txt --output ./all_lowercased.txt
# train_spm_model.sh [INPUT_FILE_PATH] [SP_MODEL_PATH]
train_spm_model.sh ./all.txt 30k-mn-uncased
```

Otherwise, just run:
```
train_spm_model.sh ./all.txt 30k-mn-cased
```

### Build tf records for pretraining
Now you can use `mn_corpus/*.txt` to produce `*tf_record` files. Here the first parameter is path to `*.txt` files and second one for max sequence length.
```bash
# source build_pretraining_data.sh [BASE_DIR] [MAX_SEQ_LEN] [SP_MODEL_PREFIX]
source build_pretraining_data.sh ./mn_corpus 512 30k-mn-cased
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
    --save_checkpoints_steps=10000 \
    --use_tpu=true \
    --tpu_name=your_tpu_name \
    --tpu_zone=your_tpu_zone \
    --num_tpu_cores=8
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
4. [You's paper](https://arxiv.org/abs/1904.00962)
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

## Thank you
