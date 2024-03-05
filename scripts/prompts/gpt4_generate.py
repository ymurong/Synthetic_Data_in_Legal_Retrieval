from openai import OpenAI
import json
import argparse
import tqdm
import time
import pandas as pd
import os

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt', type=str, default='./bsard/generate_control_question_type_describe.txt')
    argparser.add_argument('--save', type=str, default='./bsard/gpt4_generate_only_openai.jsonl')
    argparser.add_argument('--corpus', type=str, default='./data/articles_fr.csv')
    argparser.add_argument('--key', type=str, required=True)
    argparser.add_argument('--org_key', type=str)
    argparser.add_argument("--max_len", type=int, default=200)
    argparser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125')
    args = argparser.parse_args()

    client = OpenAI(
        # This is the default and can be omitted
        api_key=args.key,
        organization=args.org_key
    )

    df_articles = pd.read_csv(args.corpus)
    prompt = open(args.prompt).read()

    ct, ignore = 0, 0
    line = ""
    with open(args.save, 'w') as f:
        f.truncate(0)

    for index, row in tqdm.tqdm(df_articles.iterrows(), total=df_articles.shape[0]):
        article = row['article']
        cur_prompt = prompt.replace('{{Article}}', article)
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
                line += (json.dumps(row.to_dict())+ "\n")
                if ct % 10 == 0:
                    with open(args.save, 'a') as f:
                        f.write(line)
                        f.flush()
                        line = ""
                ct += 1
                break
            except Exception as e:
                print(e)
                if ("limit" in str(e)):
                    time.sleep(3)
                else:
                    ignore += 1
                    print('ignored', ignore)
                    break

    print('ignored total', ignore)

