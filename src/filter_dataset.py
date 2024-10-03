def filter_dataset(dataset):
    # I removed everything between braces because that's just LateX I tried using regex but couldn't get it to work
    # Then I removed empty lines and lines that don't start with an upper case (removing this results in a bunch of random letters in th
    filtered_dataset = dataset
    filtered_dataset = re.sub(r' {8}.*?(?:\\displaystyle|\\textstyle).*?\n', '', filtered_dataset, flags=re.DOTALL) # We lowercase the d
    filtered_dataset = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', '', filtered_dataset, re.DOTALL)
    # I found the regex above here https://regexr.com/37i6s
    return filtered_dataset
