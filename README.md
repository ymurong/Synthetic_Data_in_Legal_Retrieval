# Synthetic Data in Legal Retrieval: Improving Performance by Combining LLM-Based Synthesis and Step by Step Error Extrapolation

The goal of the proposed research is to investigate how synthetic questions affect the performance of french statutory article retrieval, a task aims to automatically retrieve law articles relevant to a legal question. Synthetic training data generation, especially through large language models (LLMs), is promising and often used to train dense retrieval model on the target domain. This can significantly reduce costs and time constraints in legal retrieval domain where high-quality labeled data is especially valuable due to the significant time and financial resources required to produce it. However, the distribution of generated data often differs from the distribution of target domain data, leading to semantically mismatched answers and impairing the performance of downstream tasks. Therefore, We propose combining chain-of-thought (CoT) prompting techniques with error extrapolation to account for the specific characteristics of legal question language and the distributional differences between synthetic and human-labeled data. Through comprehensive experiments, we demonstrate that using LLM in question synthesis improves recall performance of statutory article retrieval by 8 points and enhances MAP and MRR by 3 points, showing promising results compared to the traditional doc2query with the T5 model. Additionally, we demonstrate that in a supervised setting, augmenting the training data with our synthetic data—generated using our proposed synthesizing method—can significantly enhance performance. Specifically, it improves Recall@100 by 12 points, MAP@100 by 19 points, and MRR@100 by 17 points.

The project aims to answer the following research question: 

> RQ: How can we generate better synthetic legal questions by leveraging large language models (LLMs) that can enhance the end to end accuracy of statutory article retrieval ?

    * SRQ1 What are the differences between current doc2query synthetic questions and real human questions from a syntactical perspective? 

    * SRQ2 What are the prompting techniques that can enhance the accuracy of statutory article retrieval? 

    * SRQ3 How error extrapolation influence the accuracy of statutory article retrieval ?

The specific contributions of the proposed research are as follows:

(1) We provide optimized french synthetic queries for the statutory article retrieval task that can improve its performance under both semi-supervised setting and supervised setting.

(2) We adapt and evaluate step by step error extrapolation method on information retrieval - particularly on statutory article retrieval task.

(3) We propose a method on legal question synthesis through the combination of LLM prompting and step by step error extrapolation in statutory article information retrieval.


## Project Setup

### Windows
First, you need to install dependencies.
```bash
python3 -m venv venv
source  venv/Scripts/activate
pip install -r requirements.txt
```

> For windows, wheel files here: https://pypi.tuna.tsinghua.edu.cn/simple or https://www.lfd.uci.edu/~gohlke/pythonlibs/

### Mac
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

## EDA
* [Exploratory Data Analysis (Before Experimentation)](scripts/eda/Exploratory_Data_Analysis(Before_Experimentation).ipynb)
* [(WIP) Exploratory Data Analysis (After Experimentation)](scripts/eda/Exploratory_Data_Analysis(After_Experimentation).ipynb)


## Experiments

### Baseline Reproduction

#### TF-IDF, BM25, FastText, Word2Vec
In order to reproduce the TF-IDF, BM25, FastText, Word2Vec results, run:
```bash
python scripts/baseline/bsard/experiments/run_zeroshot_evaluation.py \
    --retriever {tfidf, bm25, word2vec} \ 
    --lem true
```

####  GPL + mT5
In order to reproduce the GPL + mT5 results, we need to use gpl tool.

We first generate pseudo questions where the statutory articles have already been transformed to required format under ./scripts/gpl/data/bsard directory. We use doc2query/msmarco-french-mt5-base-v1 as query generator. For each passage we generate 1 query with a batch size setting to 2.

```python
# GPL Generating queries
import gpl
gpl.toolkit.qgen(
    data_path = "./scripts/gpl/data/bsard",
    output_dir = "/content/drive/MyDrive/UVA/Thesis/training/gpl/bsard/generated",
    generator_name_or_path="doc2query/msmarco-french-mt5-base-v1",
    ques_per_passage=1,
    bsz=2,
    qgen_prefix="qgen",
)
```
Once we generated the pseudo queries, for each generated query and positive passage, we mined the similar but non relevant passages using 2 bi encoder retrievers.
```python
# Mining the similar but non relevant passage 
gpl.toolkit.NegativeMiner(
    generated_path = "/content/drive/MyDrive/UVA/Thesis/training/gpl/bsard/generated",
    prefix="qgen",
    retrievers=["antoinelouis/biencoder-camembert-base-mmarcoFR", "antoinelouis/biencoder-mMiniLMv2-L12-mmarcoFR"],
    retriever_score_functions=["cos_sim", "cos_sim"],
    nneg=50,
    use_train_qrels=False,
).run()
```

Afterwards, we use a fine-tuned cross-encoder model to calculate a margin score for the triplet given the triplet (generated query, positive,
mined negative). Then, a bi-encoder would then be trained by using MarginMSELoss.

```bash
# Train the gpl model
python -m gpl.train \
--path_to_generated_data "/content/drive/MyDrive/UVA/Thesis/training/gpl/bsard2/generated" \
--base_ckpt "camembert-base" \
--gpl_score_function "dot" \
--batch_size_gpl 32 \
--batch_size_generation 1 \
--gpl_steps 140000 \
--new_size -1 \
--queries_per_passage 1 \
--output_dir "/content/drive/MyDrive/UVA/Thesis/training/gpl/bsard2/output" \
--evaluation_data "/content/drive/MyDrive/UVA/Thesis/training/gpl/bsard2" \
--evaluation_output "/content/drive/MyDrive/UVA/Thesis/training/gpl/bsard2/evaluation" \
--generator "doc2query/msmarco-french-mt5-base-v1" \
--retrievers "antoinelouis/biencoder-camembert-base-mmarcoFR" "antoinelouis/biencoder-mMiniLMv2-L12-mmarcoFR" \
--retriever_score_functions "cos_sim" "cos_sim" \
--cross_encoder "antoinelouis/crossencoder-mMiniLMv2-L12-mmarcoFR" \
--qgen_prefix "qgen" \
--use_amp   # Use this for efficient training if the machine supports AMP
```

In the end, we evaluated the testset by applying the trained bi-encoder model.
```python
# Evaluation the model
step = 20000
gpl.toolkit.evaluate(
    data_path = "/content/drive/MyDrive/UVA/Thesis/training/gpl/bsard2",
    output_dir = f"/content/drive/MyDrive/UVA/Thesis/training/gpl/bsard2/evaluation/{step}",
    model_name_or_path = f"/content/drive/MyDrive/UVA/Thesis/training/gpl/bsard2/output/{step}",
    score_function = "dot",
    pooling = "cls",
    split = "test",
    k_values = [1, 3, 5, 10, 20, 100, 200, 500]
)
```

####  Full Supervised
Here, we simply reproduce the results by following the BSARD Paper.

```bash
# Train with human labeled data
!python scripts/baseline/bsard/experiments/train_biencoder.py
# Evaluation
!python scripts/baseline/bsard/experiments/test_biencoder.py --checkpoint_path "your model checkpoint_path"
```

### 1st Stage Results Reproduction

The first stage is to find the best prompting strategy. For each statutory article, we generate one pseudo query. Here, the frac means that we generate 5% of queries for each iteration. 

* [First Stage Experiment Code](scripts/experiments/first_stage.ipynb)


### 2nd Stage Results Reproduction
* [Second Stage Experiment Code](scripts/experiments/second_stage.ipynb)

