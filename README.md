# ALBERT-Mongolian
This repo provides pretrained ALBERT model ("A Lite" version of BERT) and SentencePiece model (unsupervised text tokenizer and detokenizer) trained on Mongolian text corpus.

## Pretrained models
[Pretrained BERT model and trained SentencePiece model](https://huggingface.co/bayartsogt/albert-mongolian)

Here is pretraining loss:
![Pretraining Loss](./images/loss.svg)

### Transformers usage
Now you can use Pretrained ALBERT Mongolian in [Transformers](https://github.com/huggingface/transformers)
```python
import torch
from transformers import AlbertTokenizer, AlbertForMaskedLM

tokenizer = AlbertTokenizer.from_pretrained('bayartsogt/albert-mongolian')
model = AlbertForMaskedLM.from_pretrained('bayartsogt/albert-mongolian')
```

### Experiment with ALBERT-Mongolian
There are simple usages of the pretrained model

* Text classification using TPU on Colab: [ALBERT_Mongolian_text_classification.ipynb](https://github.com/bayartsogt-ya/ml-tutorials/blob/master/ALBERT_Mongolian_text_classification.ipynb)

Another simple tutorial of ALBERT-Mongolian is recorded in *AWS-Mongolians e-meetup #3*. 
You can follow up using links below:
* Masked Language Modeling (MLM) on Colab: [ALBERT_Mongolian_MLM.ipynb](https://github.com/bayartsogt-ya/ml-tutorials/blob/master/ALBERT_Mongolian_MLM.ipynb)
* video tutorial:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=m-iVftIlRyU&t=6215s
" target="_parent"><img src="http://img.youtube.com/vi/m-iVftIlRyU/0.jpg" 
alt="AWS-Mongolians e-meetup #3" width="240" height="180" border="10" /></a>

## Results

You can easily *fine-tune* ALBERT-Mongolian using [Official repo of ALBERT](https://github.com/google-research/albert).

The model also evaluated on a simple Mongolian text classification problem with [Eduge dataset](https://github.com/tugstugi/mongolian-nlp/blob/master/datasets/eduge.csv.gz).

While ALBERT-base is compatible in terms of results shown below, it is over 10 times (only 135MB) smaller than BERT-base (1.2GB).

* ALBERT-Mongolian:
```
                          precision    recall  f1-score   support

            байгал орчин       0.85      0.83      0.84       999
               боловсрол       0.80      0.80      0.80       873
                   спорт       0.98      0.98      0.98      2736
               технологи       0.88      0.93      0.91      1102
                 улс төр       0.92      0.85      0.89      2647
              урлаг соёл       0.93      0.94      0.94      1457
                   хууль       0.89      0.87      0.88      1651
             эдийн засаг       0.83      0.88      0.86      2509
              эрүүл мэнд       0.89      0.92      0.90      1159

                accuracy                           0.90     15133
               macro avg       0.89      0.89      0.89     15133
            weighted avg       0.90      0.90      0.90     15133
```

* BERT-Mongolian: from [Mongolian Text Classification](https://github.com/sharavsambuu/mongolian-text-classification)
```
                          precision    recall  f1-score   support

            байгал орчин       0.82      0.84      0.83       999
               боловсрол       0.91      0.70      0.79       873
                   спорт       0.97      0.98      0.97      2736
               технологи       0.91      0.85      0.88      1102
                 улс төр       0.87      0.86      0.86      2647
              урлаг соёл       0.88      0.96      0.92      1457
                   хууль       0.86      0.85      0.86      1651
             эдийн засаг       0.84      0.87      0.85      2509
              эрүүл мэнд       0.90      0.90      0.90      1159

                accuracy                           0.88     15133
               macro avg       0.88      0.87      0.87     15133
            weighted avg       0.88      0.88      0.88     15133
```

## Pretrain from Scratch
You can follow the [PRETRAIN_SCRATCH.md](./PRETRAIN_SCRATCH.md) to reproduce the results.

## Reference
1. [ALBERT - official repo](https://github.com/google-research/albert)
2. [WikiExtrator](https://github.com/attardi/wikiextractor)
3. [Mongolian BERT](https://github.com/tugstugi/mongolian-bert)
4. [ALBERT - Japanese](https://github.com/alinear-corp/albert-japanese)
5. [Mongolian Text Classification](https://github.com/sharavsambuu/mongolian-text-classification)
6. [You's paper](https://arxiv.org/abs/1904.00962)
7. [AWS-Mongolia e-meetup #3](https://www.youtube.com/watch?v=m-iVftIlRyU)

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
