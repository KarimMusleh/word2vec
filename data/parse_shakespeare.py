


def main():
    shakespeare = open('shakespeare.txt', 'r').read()
    shakespeare = shakespeare.replace('--', ' ')
    shakespeare_split = shakespeare.split('\n\n')
    with open('shakespeare_normalized.txt', 'w') as w:
        w.write('\n'.join([' '.join(x.split('\n')[1:]) for x in shakespeare_split]))


if __name__ == '__main__':
    main()
