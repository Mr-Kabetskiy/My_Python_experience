# %% Добрый Python

s = ['1 0 0 0 0',
     '0 0 1 0 0',
     '0 0 0 0 0',
     '0 1 0 1 0',
     '0 0 0 0 0']

lst = []
for row in s:
    lst.append(list(map(int, row.split())))
lst_in_c = lst


def verify(lst: list):
    def is_isolate(i, j, lst):
        return sum([lst[i][j], lst[i][j + 1], lst[i + 1][j], lst[i + 1][j + 1]]) <= 1

    a = []
    for i in range(len(lst) - 1):
        for j in range(len(lst) - 1):
            if lst[i][j] == 1:
                a.append(is_isolate(i, j, lst))
    return all(a)


verify(lst_in_c)
