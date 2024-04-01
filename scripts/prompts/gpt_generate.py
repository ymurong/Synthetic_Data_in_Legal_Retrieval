from openai import OpenAI
import argparse
import tqdm
import time
import pandas as pd
import csv


def get_random_sample(df, seed, selected_indices):
    """
    Get a 10% random sample from the DataFrame without selecting previously picked rows.

    Args:
    df (pd.DataFrame): The DataFrame to sample from.
    seed (int): The seed value for reproducibility.
    selected_indices (set): Set of indices that have been previously selected.

    Returns:
    pd.DataFrame: A 5% random sample of the DataFrame.
    """
    return df.drop(selected_indices).sample(frac=0.05, random_state=seed)


def iterative_sampling(df, iterations, initial_seed=42):
    """
    Iteratively sample 10% of the DataFrame, ensuring no repetition and reproducibility.

    Args:
    df (pd.DataFrame): The DataFrame to sample from.
    iterations (int): Number of times to perform the sampling.
    initial_seed (int): The initial seed value for reproducibility.

    Returns:
    list of pd.DataFrame: A list of DataFrames, each containing a 10% random sample of the original DataFrame.
    """
    seed = initial_seed
    selected_indices = set()
    sampled_dfs = []

    for _ in range(iterations):
        sample_df = get_random_sample(df, seed, selected_indices)
        selected_indices.update(sample_df.index)
        sampled_dfs.append(sample_df)
        seed += 1

    return sampled_dfs


def generate_queries(df_sampled_articles, save_path):
    # Initialize an empty DataFrame
    final_df = pd.DataFrame()
    # generate
    ct, ignore = 0, 0

    for index, row in tqdm.tqdm(df_sampled_articles.iterrows()):
        article = row['article']
        cur_prompt = prompt.replace('{{Article}}', article)
        row['prompt'] = cur_prompt
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are an expert in statutory law."},
                    {"role": "user", "content": cur_prompt}
                ],
                temperature=1,
                max_tokens=args.max_len,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                # logprobs=40,
                n=1,
            )
            time.sleep(0.1)

            all_responses = [response.choices[i].message.content for i in range(len(response.choices))]
            row['synthetic_question'] = all_responses[0]
            df_row = row[['synthetic_question', 'id']].to_frame().T.rename(columns={'id': 'article_ids'})
            # Append the Series as a new row to the DataFrame
            final_df = pd.concat([final_df, df_row], ignore_index=True)
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(3)
            else:
                ignore += 1
                print('ignored', ignore)
                break
    final_df.to_csv(save_path, header=True, index=True, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt', type=str, default='./bsard/generate_only.txt')
    argparser.add_argument('--save_folder', type=str, default='./bsard/gpt-synthesizing/simple-ask')
    argparser.add_argument('--corpus', type=str, default='./data/articles_fr.csv')
    argparser.add_argument('--key', type=str, default='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')
    argparser.add_argument('--org_key', type=str, default='org-ViWmGBWyZw44MQvxg2djVAff')
    argparser.add_argument("--max_len", type=int, default=200)
    argparser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125')
    args = argparser.parse_args()

    client = OpenAI(
        # This is the default and can be omitted
        api_key=args.key,
        organization=args.org_key
    )

    df_articles = pd.read_csv(args.corpus)
    sampled_articles = iterative_sampling(df=df_articles, iterations=20)
    prompt = open(args.prompt).read()

    for idx, df_partial_sampled_articles in enumerate(sampled_articles):
        generate_queries(df_sampled_articles=df_partial_sampled_articles, save_path=f"{args.save_folder}/train-{idx}.csv")
