# %%
sentences = ['1 thousand devils', 'My name is 9Pasha', 'Room #125 costs $100', '888']


# r = [' '.join([''.join(list(filter(lambda s: s.isalpha(), word))) for word in sent.split()]) for sent in sentences]
# print([val.strip() for val in r])


def process(sentences):
    result = [''.join([' '.join(list(filter(lambda s: s.isalpha(), sent.split())))]) for sent in sentences]
    return result


res = process(sentences)
print(res)
