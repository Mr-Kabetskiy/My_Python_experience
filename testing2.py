# %% 5.6.8
lst = [['#', 'x', 'o'], ['x', '#', 'x'], ['o', 'o', '#']]
s1 = []


def get_even_sum(obj):
    return sum(list(filter((lambda x: type(x) == int and x % 2 == 0), obj)))


def is_string(lst):
    return all(list(map(lambda x: isinstance(x, str), lst)))


s = []
for row in lst:
    s.append(any(list(map(lambda x: x == '#', row))))

print(any(s))
