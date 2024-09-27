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
