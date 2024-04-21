import os
import json
from os.path import abspath, join
import argparse

import torch
import pandas as pd

from utils.data import BSARDataset
from utils.eval_step_by_step import BiEncoderEvaluator
from models.trainable_dense_models import BiEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path",
                        type=str,
                        default=abspath(join(__file__, "../../../../../output/training/Nov03-16-00-33/22")),
                        help="Path of the model directory."
                        )
    args, _ = parser.parse_known_args()

    # 1. Load an already-trained BiEncoder.
    checkpoint_path = args.checkpoint_path  # "output/training/Nov10-14-11-42_bitower-camembert-base-1000-200-20-22/99"
    model = BiEncoder.load(checkpoint_path)

    # 2. Load the test set.
    test_queries_df = pd.read_csv(abspath(join(__file__, "../../../bsard/data/questions_fr_test.csv")))
    documents_df = pd.read_csv(abspath(join(__file__, "../../../bsard/data/articles_fr.csv")))
    test_dataset = BSARDataset(test_queries_df, documents_df)

    # 3. Initialize the Evaluator.
    evaluator = BiEncoderEvaluator(queries=test_dataset.queries,
                                   documents=test_dataset.documents,
                                   relevant_pairs=test_dataset.one_to_many_pairs,
                                   score_fn=model.score_fn)

    # 4. Run trained model and compute scores.
    scores, wrong_pairs = evaluator(model=model,
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                       batch_size=512)

    # 5. Save results.
    os.makedirs(checkpoint_path, exist_ok=True)
    with open(join(checkpoint_path, 'test_scores_fixed.json'), 'w') as fOut:
        json.dump(scores, fOut, indent=2)

    with open(join(checkpoint_path, 'wrong_pairs_to_extrapolate.json'), 'w') as fOut:
        json.dump(wrong_pairs, fOut, indent=2)
