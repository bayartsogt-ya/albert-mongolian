# Pretrain from Scratch


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
