import re
from nltk.corpus import stopwords

def simple_filter_tokenize(data: str):
    """
    filter and tokenize the input string.
    filter by lowercasing -> removing punctuation -> removing whitespace -> removing stopwords
    """
    filtered_data = data.lower()
    filtered_data = re.sub(r'[^a-zA-Z]', ' ', filtered_data)
    filtered_data = re.sub(r'\s+', ' ', filtered_data)

    stops = set(stopwords.words('english'))
    tokens = [word for word in filtered_data.split(' ') if word and word not in stops]
    return tokens
