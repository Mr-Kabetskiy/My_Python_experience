# %% 9.6.6
lst_in = ['смартфон:120000', 'яблоко:2', 'сумка:560', 'брюки:2500', 'линейка:10', 'бумага:500']


def cheapers(d: dict):
    return [d.get(i) for i in (sorted(d))[:3]]


d = {int(i.split(':')[1]): i.split(':')[0] for i in lst_in}
print(*cheapers(d))
