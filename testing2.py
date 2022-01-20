# %% 9.5.4
s_input = 'Москва Уфа Тула Самара Омск Воронеж Владивосток Лондон Калининград Севастополь'

lst = [i for i in s_input.split()]
[print(*i) for i in zip(*[iter(lst)] * 3)]
