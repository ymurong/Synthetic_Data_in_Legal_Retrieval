# Domain-Adapation-French-Legal-Retrieval

The time-consuming and expensive nature of creating large-scale, high-quality annotated datasets presents a significant bottleneck for various IR and natural language processing (NLP) tasks in the Legal field. To address this bottleneck, we experiment different state-of-art domain adaptation techniques and validate its effectiveness particularly in statutory article retrieval in French. 

The project aims to answer the following research question: 
> To what extent can domain adaptation affect the performance of Retrieval-based Legal Question Answering (RLQA) System?

The planned implementation consists of two main steps: 

- Assess the effectiveness of current domain adaptation techniques on the RLQA system in the French language. We aim to gain insights into why these techniques work or do not work under different settings, taking into account the specific characteristics of French legal text discussed in the previous section.

- Explore potential approaches to enhancing the performance of existing domain adaptation techniques in a legal context. 


# Experimentation Setup

For our RLQA experiments setup, mMARCO, a multilingual version of the famous MS MARCO, will be used as our source domain dataset due to its richness and widespread usage in reading comprehension (RC) and question-answering (QA) tasks. Our target domain will be French legal text from the BSARD dataset, which was presented in the previous section. 

Following different adaptation technique, we would tune our dense retrieval model by using the french mMARCO on CamemBERT, a French RoBERTa model that is pre-trained on 147GB of French web pages filtered from Common Crawl. Afterwards, both adaptation techniques require target domain corpus for further unsupervised tuning in a different way. However, they would not need access to the labeled examples.

Evaluation would be done on the labeled pairs of BSARD dataset so that we would be able to compare our results in order to address our sub research questions of SRQ1, SRQ2 and SRQ3. As presented in Table \ref{table:evaluation}, three standard information retrieval metrics are used to evaluate performance, namely the (macro-averaged) recall@k (R@k), mean average precision@k (MAP@k), and mean reciprocal rank@k (MRR@k). We have selected the same value of k to enable a comparison of our results with those presented in the BSARD paper.

Afterwards, we plan to do a qualitative error analysis to figure out possible reasons of the evaluation results and try to come up with some counter measures that may be helpful to improve the performance of domain adaptation in legal context. We will then evaluate the counter measures the same way as we did previously and explain some of the possible reasons under the hood.

Overall, our expected contribution would be to validate the applicability of domain adaptation in the legal domain through ablation studies and qualitative error analysis. Based on these findings, some further improvements can be proposed to the existing techniques in the hope that this research can bring further insights to the legal retrieval field and offering greater benefits to free professional legal assistance services.


# Project Folder Structure

This project has the following folder structure:




# Project Setup

## Windows
First, you need to install dependencies.
```bash
python3 -m venv venv
source  venv/Scripts/activate
pip install -r requirements.txt
```

> For windows, wheel files here: https://pypi.tuna.tsinghua.edu.cn/simple or https://www.lfd.uci.edu/~gohlke/pythonlibs/

Secondly, some sub dependencies are needed.
```bash
python -m spacy download fr_core_news_md
```

# Experiments

## Lexical Models

In order to reproduce the TF-IDF and BM25 models, run:
```bash
python scripts/baseline/bsard/experiments/run_zeroshot_evaluation.py \
    --retriever {tfidf, bm25} \ 
    --lem true
```

## Train Dense Model
```bash
python scripts/baseline/bsard/experiments/train_biencoder.py
```


## Domain Adaptation
### Unsupervisedly train DAM
The source DAM aims to capture the corpus features in the source domain. Therefore, the REM module trained in this source domain can be generic because it will not be dependent on the source-domain features.
```bash
python scripts/disentangled_retriever/adapt/run_adapt_with_mlm.py \
    --corpus_path "../../data/datasets/fr-msmarco/french_collection.tsv" \
    --output_dir "./output/adapt-mlm/french-marco/train_rem/dam" \
    --model_name_or_path camembert-base \
    --logging_first_step \
    --logging_steps 50 \
    --max_seq_length 100 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 1000 \
    --fp16 \
    --learning_rate 5e-5 \
    --max_steps 100000 \
    --dataloader_drop_last \
    --overwrite_output_dir \
    --dataloader_num_workers 16 \
    --weight_decay 0.01 \
    --lr_scheduler_type "constant_with_warmup" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --optim adamw_torch 
```

### Supervisedly train REM in contrastive way

```bash
output_dir="./output/adapt-mlm/french-marco/train_rem/rem-with-hf-dam/contrast"

python 
    --lora_rank 192 --parallel_reduction_factor 4 --new_adapter_name msmarco \
    --pooling average \
    --similarity_metric ip \
    --qrel_path ./data/datasets/msmarco-passage/qrels.train \
    --query_path ./data/datasets/msmarco-passage/query.train \
    --corpus_path ./data/datasets/msmarco-passage/corpus.tsv \
    --negative ./data/datasets/msmarco-passage/msmarco-hard-negatives.tsv \
    --output_dir $output_dir \
    --model_name_or_path jingtao/DAM-bert_base-mlm-msmarco \
    --logging_steps 100 \
    --max_query_len 24 \
    --max_doc_len 128 \
    --per_device_train_batch_size 32 \
    --inv_temperature 1 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --neg_per_query 3 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --dataloader_drop_last \
    --overwrite_output_dir \
    --dataloader_num_workers 0 \
    --weight_decay 0 \
    --lr_scheduler_type "constant" \
    --save_strategy "epoch" \
    --optim adamw_torch
```





