from datasets import load_dataset

if __name__ == '__main__':
    ref_files = {"articles": "articles_fr.csv"}
    data_files = {"train": "questions_fr_train.csv", "test": "questions_fr_test.csv"}
    ref_dataset = load_dataset("maastrichtlawtech/bsard", revision="main", data_files=ref_files)
    dataset = load_dataset("maastrichtlawtech/bsard", revision="main", data_files=data_files)
    ref_dataset["articles"].to_csv("articles_fr.csv")
    dataset["train"].to_csv("questions_fr_train.csv")
    dataset["test"].to_csv("questions_fr_test.csv")

