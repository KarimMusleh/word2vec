# this file is a list of functions that I've written while implementing this paper. They aren't doing exactly what I want them to do.
# However I hope they might be useful in the future.
def remove_braces(text):
    result = []
    stack = []
    between, prev = '', ''
    prev_index = 0
    count_line = 0
    prev_line = 0
    i = 0
    while i < len(text):
        if text[i] == '\n':
            count_line += 1
        if text[i: i + 2] == '{\\':
            stack.append(i)  # Push the index of the opening brace
            between += '{'
        elif text[i] == '}':
            between += '}'
            if stack:
                curr = stack.pop()  # Pop the matching opening brace
            if not stack:
                # print(between.encode())
                prev_index = curr
                prev = between
                prev_line = count_line
                between = ''
            else:
                result.append(text[i])  # No matching opening brace, add to result
        elif not stack:
            result.append(text[i])  # Only add to result if not inside a pair of braces
        elif stack:
            between += text[i]
        i += 1
    print(prev, prev_line)
    print(between[:4000].encode())
    return ''.join(result)

def filter_dataset(dataset):
    # I may have filtered too hard
    # I removed everything between braces because that's just LateX I tried using regex but couldn't get it to work
    # Then I removed empty lines and lines that don't start with an upper case (removing this results in a bunch of random letters in the dataset)
    filtered_dataset = dataset
    # print(f'The dataset is size {len(filtered_dataset)} without filtering')
    # with open('latex.txt', 'w') as w: # use this to test wether it removes too much or too little
    #     w.write(''.join(re.findall(r' {6}\n {8}.*?(?:\\displaystyle|\\textstyle).*?\n', filtered_dataset, flags=re.DOTALL)))
    filtered_dataset = re.sub(r' {8}.*?(?:\\displaystyle|\\textstyle).*?\n', '', filtered_dataset, flags=re.DOTALL) # We lowercase the data and remove the LateX
    # print(f'The dataset is size {len(filtered_dataset)} without the LaTeX')
    # print(f"There are currently {len(re.findall(r'displaystyle', filtered_dataset))} LaTeX blocks that have to be manually deleted")
    # filtered_dataset = '\n'.join([line for line in filtered_dataset.splitlines() if line.strip()]) #  and line[0].isupper() and len(line) > 30
    # print(len(filter_dataset))
    filtered_dataset = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', '', filtered_dataset)
    # print(f'The dataset is size {len(filtered_dataset)} without the links')
    # I found the regex above here https://regexr.com/37i6s
    return filtered_dataset


def filter_dataset(dataset):
    dataset = '\n'.join(dataset.splitlines()[:1000])[::-1]
    i = 0
    filtered_dataset = ''
    dataset = dataset.splitlines()
    while i < len(dataset):
        # print(1, data.rstrip()[::-1])
        if dataset[i].rstrip().endswith('elytsyalpsid\\{'):
            between = dataset[i]
            i += 1
            while not re.findall(r'[A-Z]$', dataset[i], flags=re.MULTILINE):
                between += '\n' + dataset[i]
                i += 1
            if not filtered_dataset.endswith('\n\n'):
                print("NOT NN")
            else:
                filtered_dataset = filtered_dataset.removesuffix('\n\n')
            print(between.encode()[::-1])
        else:
            filtered_dataset += dataset[i] + '\n'
        i += 1
    # return filtered_dataset[::-1]

