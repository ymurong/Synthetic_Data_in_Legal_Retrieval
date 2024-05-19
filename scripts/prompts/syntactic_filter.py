import argparse
import pandas as pd
import spacy
from zss import Node, simple_distance
from functools import partial
import duckdb
import csv

# Load the French language model
nlp = spacy.load("fr_core_news_md")


# Function to convert a spaCy token to a zss Node recursively
def token_to_zss_node(mode, token):
    if mode == "dep":
        node = Node(token.dep_)
    elif mode == "text":
        node = Node(token.orth_)
    elif mode == "pos":
        node = Node(token.pos_)

    for child in token.children:
        node.addkid(token_to_zss_node(mode=mode, token=child))
    return node


def build_zss_tree(spacy_doc, zss_parser):
    root = Node("ROOT")
    for sent in spacy_doc.sents:
        root.addkid(zss_parser(sent.root))
    return root


# Function to compute tree edit distance between two documents
def compute_tree_edit_distance(parsed_doc1, parsed_doc2, zss_parser):
    root1 = build_zss_tree(spacy_doc=parsed_doc1, zss_parser=zss_parser)
    root2 = build_zss_tree(spacy_doc=parsed_doc2, zss_parser=zss_parser)
    # Compute the tree edit distance using zss
    return simple_distance(root1, root2)


def _remove_stop_words(doc):
    return " ".join([str(t) for t in doc if not t.is_stop])


def semantic_similarity(doc1, doc2):
    return doc1.similarity(doc2)


# zss parser
zss_dependency_parser = partial(token_to_zss_node, 'dep')


def semantic_filter(df_questions, threshold):
    df_questions['semantic_similarity'] = df_questions.apply(
        lambda row: semantic_similarity(nlp(_remove_stop_words(nlp(row['Question']))),
                                        nlp(_remove_stop_words(nlp(row['synthetic_question'])))), axis=1)
    df_questions_filtered = df_questions[df_questions['semantic_similarity'] < threshold]
    return df_questions, df_questions_filtered


def syntactic_filter(df_questions, topk, random=False):
    df_questions['tree_edit_distance'] = df_questions.apply(
        lambda row: compute_tree_edit_distance(nlp(row['Question']), nlp(row['synthetic_question']),
                                               zss_parser=zss_dependency_parser), axis=1)
    df_questions['length_diff'] = df_questions.apply(
        lambda row: len(row['synthetic_question'].split(" ")) - len(row['Question'].split(" ")), axis=1)
    df_questions['tree_edit_distance_norm'] = df_questions['tree_edit_distance'] - df_questions['length_diff']
    df_questions = df_questions.sort_values(by='tree_edit_distance_norm', ascending=True).reset_index(drop=True)
    df_questions.drop_duplicates(subset=['synthetic_question', 'article_ids'], keep='first').reset_index(drop=True)
    if random:
        return df_questions, df_questions.sample(n=topk)
    return df_questions, df_questions[:topk]


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--save_folder', type=str, default='./bsard/gpt-synthesizing/step_by_step')
    argparser.add_argument('--extrapolated_queries', type=str, default='./data/extrapolated_queries.csv')
    argparser.add_argument('--wrong_pairs_to_extrapolate', type=str, default='./data/wrong_pairs_to_extrapolate.csv')
    argparser.add_argument('--syntactic_topk', type=int, default=100)
    argparser.add_argument('--semantic_threshold', type=float, default=0.8)
    argparser.add_argument('--do_random', action="store_true", default=True)
    args = argparser.parse_args()

    syntactic_topk = args.syntactic_topk
    semantic_threshold = args.semantic_threshold
    save_path = f"{args.save_folder}/extrapolated_queries_filtered.csv"
    random = args.do_random

    # merge data
    conn = duckdb.connect(database="questions.db", read_only=False)
    cur = conn.cursor()
    cur.execute(f"CREATE OR REPLACE TABLE extrapolated_questions AS SELECT * FROM \"{args.extrapolated_queries}\";")
    cur.execute(f"CREATE OR REPLACE TABLE wrong_pairs AS SELECT * FROM \"{args.wrong_pairs_to_extrapolate}\";")
    cur.execute("""
        CREATE OR REPLACE TABLE questions AS
        (
        select distinct wrong_pairs.Question, extrapolated_questions.synthetic_question, extrapolated_questions.article_ids
        from extrapolated_questions, wrong_pairs
        where extrapolated_questions.article_ids = wrong_pairs.Article_Id
        )
    """)

    df_questions = cur.execute("""
        select *
        from questions
    """).df()

    df_questions, df_questions_filtered = syntactic_filter(df_questions[:100], topk=syntactic_topk, random=random)
    df_questions, df_questions_filtered = semantic_filter(df_questions_filtered, threshold=semantic_threshold)
    df_questions_filtered[['synthetic_question', 'article_ids']].reset_index(drop=True).to_csv(save_path, header=True,
                                                                                               index=True,
                                                                                               quoting=csv.QUOTE_NONNUMERIC,
                                                                                               quotechar='"')
    print(f"{save_path} generated!")
