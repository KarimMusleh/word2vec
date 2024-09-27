## Goal
This is my implementation of the original Word2Vec paper ([Mikolov et al.](https://arxiv.org/abs/1706.03762)) in PyTorch.

## Data
I am using wikipedia articles as my training data. all the articles are in one file and are separated by `__WIKI__`. The models are trained on 200 wikipedia articles that are linked from the philosophy page.
I still haven't found the ultimate wikipedia LaTeX remover, but the following regex pattern seems to delete most of it `r' {8}.*?(?:\\displaystyle|\\textstyle).*?\n', '', o, flags=re.DOTALL)`.

TODO
- Create a better regex to remove the LaTeX from the dataset (probably can use the fact that every non-LaTeX line begins with a capital letter)
- Train a model :)
