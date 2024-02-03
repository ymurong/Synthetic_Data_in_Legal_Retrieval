# Domain-Adapation-French-Legal-Retrieval

The time-consuming and expensive nature of creating large-scale, high-quality annotated datasets presents a significant bottleneck for various IR and natural language processing (NLP) tasks in the Legal field. To address this bottleneck, we experiment different state-of-art domain adaptation techniques and validate its effectiveness particularly in statutory article retrieval in French. 

The project aims to answer the following research question: 
> To what extent can domain adaptation affect the performance
of french legal retrieval task ?

>  SRQ1 How well does the current state of art domain adap-65
tation (TSDAE, GPL) perform in the task of french legal
retrieval under a zero-shot setting?

> SRQ2 How well does the current state of art domain adap-
tation (TSDAE, GPL) perform in the task of french legal
retrieval under a supervised setting?

> SRQ3 Why would the current state of art domain adaptation
(TSDAE, GPL) perform as expected or not as expected ?

> SRQ4 How could we improve the performance of current
domain adaptation in the task of french legal retrieval?

The planned implementation consists of two main steps: 

- Assessment of the effectiveness of current domain adaptation
techniques on the task of french legal retrieval.

- Proposition of variants to the current domain adaptation tech-
niques in order to mitigate the limits and improve its performance
in the task of french legal retrieval.


# Methodology

The methodology of the proposed research follows the logic of the previously introduced sub-questions. This includes the reproduction of the baseline results based on the methodology provided by the BSARD paper (4.1), the application and evaluation of the current state-of-the-art domain adaptation techniques, namely TSDAE and GPL (4.2) on the statutory article retrieval task (4.2), analysis of the results obtained and designed experimentation to find better variations (4.3), evaluation methods (4,4).


## Baseline

The lexical baselines selected are TF-IDF and BM25. Both methods would calculate a score for each article in corpus and return the k articles with the highest scores as the top-k most relevant results to the input query. We follow the same formula and parameters proposed by BSARD paper to produce the same results. 

The following TF-IDF formula is used, where the term frequency tf is the number of occurrences of term t in article a, and the document frequency df is the number of articles within the corpus that contain term t, C is the total number of articles in corpus.

```math
w(t,a) = tf(t,a) * \log \frac{|C|}{df(t)}
```

The following BM25 formula is used, where k1 $\in$ R+ and b  $\in$ [0, 1] are constant parameters to be fixed, |a| is the article length, and avgal is the average article length in the collection.

```math
w(t,a) = \frac{tf(t,a) * (k_1 + 1)}{tf(t,a) + k_1 * (1-b+b*\frac{|a|}{avgal})*log\frac{|C|-df(t) +0.5}{df(t) + 0.5}} 
```

The dense baselines selected are word2vec, fastText and camembert. Word2vec and fastText are both context-independent word representations while camembert is context-dependent word embedding. Camembert is a pretrained french language model based on Roberta architecture. \cite{Martin_2020} Particularly for camembert model, we would evaluate under both zeroshot setting and supervised setting to be able to compare their results with domain adaptation techniques.

## Domain Adaptation Setup

Two methods will be used for domain adaptation setup, namely TSDAE and GPL. Both methods would be applied separately to the statutory article retrieval task and would be evaluated under both zeroshot setting and supervised setting.

As for TSDAE, we would first preprocess our 22633 statuary articles from the corpus into distinguished sentences. Then, a training dataset would be built by adding noise to each sentence. By default, TSDAE would delete tokens from each sentence randomly based on a deletion ratio, 0.6 as recommended by the TSDAE paper. Then, through an encoder-decoder architecture, TSDAE model would be trained to predict the original sentences in order to learn the semantics of target domain language. 

As for GPL, we would follow the 4 steps provided by paper. For the query generation step, a french mt5 model fine-tuned on msmarco would be used to generate queries. For the negative mining, we would use a fine-tuned camembert bi-encoder model to find similar passages from the corpus that are not relevant to queries. Then, we would score all pairs by using a fine-tuned cross-encoder model. This step allows us to calculate a margin score given the triplet (generated query, positive, mined negative) as follows:

```math
MarginScore = |sim(Query, Pos) - sim(Query, Neg)|
```

Finally, a bi-encoder would then be trained by using MarginMSELoss. This allows the model to learn knowledge from the cross-encoder. 

```math
MarginMSELoss = MarginScore_{example} - MarginScore_{gold}
```

## Analysis and Experimentation

By conducting a comparative analysis of results obtained across various scenarios, we aim to put forth several investigative approaches to discern the underlying reasons behind these outcomes. The primary objective of this step is to identify limitations in current domain adaptation methods for statutory legal retrieval tasks and propose refinements to enhance its performance.

We give some hypotheses that may impact performance:

1) Pre-processing techniques: In the case of TSDAE, we adhered to the default approach of separating each article into sentences, with each sentence representing an individual example. Exploring the impact of treating multiple sentences as a single example may unveil potential differences. Additionally, the recommended deletion ratio, as suggested in the paper, may vary based on the specific domain, providing room for enhancement.

2) Synthetic training data quality: Given that GPL heavily relies on generated queries, the quality of these queries could significantly influence final performance. A qualitative analysis, examining the relevance between generated queries and their corresponding statutory articles, is necessary. We can also come up with some quantitative metrics to measure the quality of synthetic query answer pairs. 

3) Legal Text Related: Statutory Article can be sometimes very long. Thus, the chunking techniques can also influence the performance of the model. By default, GPL directly truncates overflowed text, which can pose a limitation for legal tasks. 

4) Combination of sparse and dense retrieval: Given that GPL employs a cross-encoder to enhance bi-encoder performance, uncertainties arise regarding the stability of performance when retrieving unseen statutory articles. As mentioned in related work, sparse retrieval can be still effective in lexical matching and can be useful if we employ an ensemble-based fusion to mitigate the limit of cross-encoder, it may become more robust when it encounters unseen corpus.

Depending on the results obtained, we would come up with our variation approach and evaluate them respectively.

## Evaluation

Evaluation would be done on the labeled pairs of BSARD dataset so that we would be able to compare our results in order to address our sub research questions of SRQ1, SRQ2 and SRQ3. As presented in Table \ref{table:evaluation}, three standard information retrieval metrics \cite{schutze2008introduction} are used to evaluate performance, namely the (macro-averaged) recall@k (R@k), mean average precision@k (MAP@k), and mean reciprocal rank@k (MRR@k). We have selected the same value of k to enable a comparison of our results with those presented in the BSARD paper.


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

## Mac
First, you need to install dependencies.
```bash
python3 -m venv venv
source  venv/Scripts/activate
pip install -r requirements.txt
```

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





