import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer as wnl



def normalize_shakespeare():
    shake = open('../data/shakespeare.txt', 'r').read()
    shake_split = shake.split('\n\n')
    shake = '\n'.join([' '.join(x.split('\n')[1:]) for x in shake_split]).lower()

    stops = set(stopwords.words())
    stops.add('us')
    tkn = RegexpTokenizer(r'\w+')

    shake_sents = [' '.join(filter(lambda x: x not in stops, tkn.tokenize(sent))) for sent in shake.split('\n')] # remove all punctuation and stopwords

    sents = []
    lemmatizer = wnl()
    for sent in shake_sents:
        curr_sent = []
        for word in sent.split(' '):
            lemma = lemmatizer.lemmatize(word)
            if lemma == word:
                curr_sent.append(lemma)
        sents.append(tkn.tokenize(' '.join(curr_sent)))
    data = '\n'.join([' '.join(sent) for sent in sents if sent])

    with open('../data/shakespeare_normalized.txt', 'w') as w:
        w.write(data)

if __name__ == "__main__":
    normalize_shakespeare()
