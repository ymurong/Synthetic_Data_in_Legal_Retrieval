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

First, you need to install dependencies.
```bash
python3 -m venv venv
source  venv/Scripts/activate
```

Secondly, you need to install manually some dependencies manually via wheel files, according to your OS.
* fasttext
* gensim
* spacy
* wordcloud

> For windows, wheel files here: https://www.lfd.uci.edu/~gohlke/pythonlibs

Thirdly, some sub dependencies are needed.
```bash
python -m spacy download fr_core_news_md
```











