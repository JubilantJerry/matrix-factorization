from matrix_factorization import KernelMF

import pandas as pd
import IPython

# Movie data found here https://grouplens.org/datasets/movielens/
cols = ["user_id", "item_id", "rating"]
movie_data = pd.read_csv(
    "../rankings.csv", names=cols, sep="\t", usecols=[0, 1, 2], engine="python"
)

X = movie_data[["user_id", "item_id"]]
y = movie_data["rating"]

# Initial training
matrix_fact = KernelMF(
    n_epochs=5000, n_factors=1, verbose=1, lr=0.1, reg=0.0, min_rating=0.0,
    max_rating=1.0, update_item_biases=False)
matrix_fact.fit(X, y)

metrics = ["Chatbot Arena Elo" , "HellaSwag (few-shot)" , "HellaSwag (zero-shot)" , "HellaSwag (one-shot)" , "HumanEval-Python (pass@1)" , "LAMBADA (zero-shot)" , "LAMBADA (one-shot)" , "MMLU (zero-shot)" , "MMLU (few-shot)" , "TriviaQA (zero-shot)" , "TriviaQA (one-shot)" , "WinoGrande (zero-shot)" , "WinoGrande (one-shot)" , "WinoGrande (few-shot)"]
models = ["alpaca-7b" , "alpaca-13b" , "bloom-176b" , "cerebras-gpt-7b" , "cerebras-gpt-13b" , "chatglm-6b" , "chinchilla-70b" , "codex-12b / code-cushman-001" , "codegen-16B-mono" , "codegen-16B-multi" , "codegx-13b" , "dolly-v2-12b" , "eleuther-pythia-7b" , "eleuther-pythia-12b" , "fastchat-t5-3b" , "gal-120b" , "gpt-3-7b / curie" , "gpt-3-175b / davinci" , "gpt-3.5-175b / text-davinci-003" , "gpt-3.5-175b / code-davinci-002" , "gpt-4" , "gpt4all-13b-snoozy" , "gpt-neox-20b" , "gpt-j-6b" , "koala-13b" , "llama-7b" , "llama-13b" , "llama-33b" , "llama-65b" , "mpt-7b" , "oasst-pythia-12b" , "opt-7b" , "opt-13b" , "opt-66b" , "opt-175b" , "palm-62b" , "palm-540b" , "palm-coder-540b" , "palm-2-s" , "palm-2-s*" , "palm-2-m" , "palm-2-l" , "palm-2-l-instruct" , "replit-code-v1-3b" , "stablelm-base-alpha-7b" , "stablelm-tuned-alpha-7b" , "starcoder-base-16b" , "starcoder-16b" , "vicuna-13b"]
perm_metrics = [matrix_fact.user_id_map[v] for v in metrics]
perm_models = [matrix_fact.item_id_map[v] for v in models]
ratings = (matrix_fact.item_features @ matrix_fact.user_features.T + matrix_fact.user_biases + matrix_fact.global_mean)
ratings = ratings[perm_models][:, perm_metrics]

for row in ratings:
    print('\t'.join([str(v) for v in row]))

print('\t'.join([str(v[0]) for v in matrix_fact.user_features[perm_metrics]]))
print('\n'.join([str(v[0]) for v in matrix_fact.item_features[perm_models]]))
print('\t'.join([str(v) for v in matrix_fact.user_biases[perm_metrics]]))
print(matrix_fact.global_mean)

IPython.embed()
