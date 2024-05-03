from openai import OpenAI
import argparse
import tqdm
import time
import pandas as pd
import csv
import math

def generate_queries(df_wrong_pairs, save_path):
    # Initialize an empty DataFrame
    final_df = pd.DataFrame()
    # generate
    ct, ignore = 0, 0

    for index, row in tqdm.tqdm(df_wrong_pairs.iterrows()):
        question = row['Question']
        article = row['Article']
        cur_prompt = prompt.replace('{{Question}}', question)
        cur_prompt = cur_prompt.replace('{{Article}}', article)
        row['prompt'] = cur_prompt
        while True:
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
                df_row = row[['synthetic_question', 'Article_Id']].to_frame().T.rename(columns={'Article_Id': 'article_ids'})
                # Append the Series as a new row to the DataFrame
                final_df = pd.concat([final_df, df_row], ignore_index=True)
                break
            except Exception as e:
                print(e)
                if ("limit" in str(e)):
                    time.sleep(3)
                else:
                    ignore += 1
                    print('ignored', ignore)
                    break
    final_df.to_csv(save_path, header=True, index=True, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')
    print(f"{save_path} generated!")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt', type=str, default='./bsard/extrapolate.txt')
    argparser.add_argument('--save_folder', type=str, default='./bsard/gpt-synthesizing/step_by_step')
    argparser.add_argument('--corpus', type=str, default='./data/wrong_pairs_to_extrapolate.csv')
    argparser.add_argument('--key', type=str, default="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")
    argparser.add_argument('--org_key', type=str, default="org-ViWmGBWyZw44MQvxg2djVAff")
    argparser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125')
    argparser.add_argument("--max_len", type=int, default=200)
    args = argparser.parse_args()

    client = OpenAI(
        # This is the default and can be omitted
        api_key=args.key,
        organization=args.org_key
    )
    # read wrong pairs file
    df_wrong_pairs = pd.read_csv(args.corpus)

    prompt = open(args.prompt).read()

    generate_queries(df_wrong_pairs=df_wrong_pairs,
                     save_path=f"{args.save_folder}/extrapolated_queries.csv")
