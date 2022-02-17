lst = ['+71234567890', '+71234567854', '+61234576890', '+52134567890', '+21235777890', '+21234567110', '+71232267890']

d = {k[:2]: [val for val in lst if val[:2] == k[:2]] for k in lst}
print(*sorted(d.items()))
# %%
lst = ['+71234567890 Сергей', '+71234567810 Сергей', '+51234567890 Михаил', '+72134567890 Николай']
d = {k.split()[1]: [val.split()[0] for val in lst if k.split()[1] == val.split()[1]] for k in lst}
print(*sorted(d.items()))
# %%
while True:
    n = float(input())
    if n == 0:
        break
    else:
        if n in d:
            print(f'значение из кэша: {round(*d[n], 2)}')
        else:
            d.update({n: [n ** 0.5]})
            print(f'{round(*d[n], 2)}')
# %%
lst_in = ['ustanovka-i-zapusk-yazyka', 'ustanovka-i-poryadok-raboty-pycharm',
          'peremennyye-operator-prisvaivaniya-tipy-dannykh', 'arifmeticheskiye-operatsii',
          'ustanovka-i-poryadok-raboty-pycharm']
d = {}
for n in lst_in:
    if n in d:
        print(f'Взято из кэша: HTML-страница для адреса {d[n]}')
    else:
        d.update({n: n})
        print(f'HTML-страница для адреса {d[n]}')
# %%
d = {' ': '-...-', 'Ё': '.', 'А': '.-', 'Б': '-...', 'В': '.--', 'Г': '--.', 'Д': '-..', 'Е': '.', 'Ж': '...-',
     'З': '--..', 'И': '..', 'Й': '.---', 'К': '-.-', 'Л': '.-..', 'М': '--', 'Н': '-.', 'О': '---', 'П': '.--.',
     'Р': '.-.', 'С': '...', 'Т': '-', 'У': '..-', 'Ф': '..-.', 'Х': '....', 'Ц': '-.-.', 'Ч': '---.', 'Ш': '----',
     'Щ': '--.-', 'Ъ': '--.--', 'Ы': '-.--', 'Ь': '-..-', 'Э': '..-..', 'Ю': '..--', 'Я': '.-.-'}

s = 'Сергей Балакирев'
print(*[d[i] for i in s.upper()], end='')
# %%
d = {' ': '-...-', 'А': '.-', 'Б': '-...', 'В': '.--', 'Г': '--.', 'Д': '-..', 'Е': '.', 'Ж': '...-', 'З': '--..',
     'И': '..', 'Й': '.---', 'К': '-.-', 'Л': '.-..', 'М': '--', 'Н': '-.', 'О': '---', 'П': '.--.', 'Р': '.-.',
     'С': '...', 'Т': '-', 'У': '..-', 'Ф': '..-.', 'Х': '....', 'Ц': '-.-.', 'Ч': '---.', 'Ш': '----', 'Щ': '--.-',
     'Ъ': '--.--', 'Ы': '-.--', 'Ь': '-..-', 'Э': '..-..', 'Ю': '..--', 'Я': '.-.-'}
s = '.-- ... . -...- .-- . .-. -. ---'.split()
for i in s:
    print(*[k for k, val in d.items() if val == i], end='')

# %% 6.2.5
s = '8 11 -4 5 2 11 4 8'.split()
d = dict.fromkeys(s)
print(*d)

# %% 6.2.6
lst_in = ['3 Сергей', '5 Николай', '4 Елена', '7 Владимир', '5 Юлия', '4 Светлана']
a = dict()
for i in lst_in:
    if i.split()[0] in a:
        a[i.split()[0]] += [i.split()[1]]
    else:
        a[i.split()[0]] = [i.split()[1]]

for i in a:
    print(f'{i}:', ', '.join(a[i]))

# %% 6.2.7
w = 10000
things = {'карандаш': 20, 'зеркальце': 100, 'зонт': 500, 'рубашка': 300, 'брюки': 1000, 'бумага': 200, 'молоток': 600,
          'пила': 400, 'удочка': 1200, 'расческа': 40, 'котелок': 820, 'палатка': 5240, 'брезент': 2130, 'спички': 10}

things = dict(sorted(things.items(), key=lambda x: -x[1]))

sw = 0
for k, val in things.items():
    sw += val
    if sw <= w:
        print(k, end=' ')
    else:
        sw -= val
        continue

# %% 6.3.3
lst = [8, 11, -5, 2]
t = (3.4, -56.7)
t += tuple(lst)
print(t)
# %% 6.3.4
t_lst = tuple(input().split())
if 'Москва' not in t_lst:
    t_lst += ('Москва',)
print(*t_lst)
# %% 6.3.5
print(*tuple([s for s in tuple(input().split()) if s != 'Ульяновск']))
# %% 6.3.6
t_lst = tuple('Петя Варвара Венера Василиса Василий Федор'.lower().split())
[print(s, end=' ') for s in t_lst if s[:2] == 'ва']
# %% 6.3.7
t_lst = tuple(map(int, input().split()))
n_t = tuple()
for i in t_lst:
    n_t += (i,) if i not in n_t else ()
print(*n_t)
# %% 6.3.8
t_lst = (5, 4, -3, 2, 4, 5, 10, 11)
# t_lst = tuple(map(int, input().split()))
for i, val in enumerate(t_lst):
    print(i, end=' ') if t_lst.count(val) > 1 else ()
# %% 6.3.9
t = ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1))
n = int(input())
for i in range(n):
    print(*t[i][:n])
print()
# %% 6.3.10
lst_in = ['Главная home', 'Python learn-python', 'Java learn-java', 'PHP learn-php']
t = ()
for val in lst_in:
    t += ((val.split()[0], val.split()[1]),)
print(t)  # %% 6.6.6
lst_in = ['Пушкин: Сказака о рыбаке и рыбке', 'Есенин: Письмо к женщине', 'Тургенев: Муму', 'Пушкин: Евгений Онегин',
          'Есенин: Русь']
lst = [[k.split(':')[0], k.split(':')[1].strip()] for k in lst_in]
print(lst)


# for i in range(len(lst)):
#     {lst[i][0]: [].append(lst[i][1])}
# %% 7.2.2
def is_triangle(x, y, z):
    return (x + y > z and x + z > y and z + y > x)


a, b, c = tuple(map(int, input().split()))
print(is_triangle(a, b, c))

# %% 7.4.4
t = {'ё': 'yo', 'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ж': 'zh',
     'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p',
     'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'ch', 'ш': 'sh',
     'щ': 'shch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'}


#
#
def translater(s, sep='-'):
    s = s.lower()
    trs = (''.join([t[i] if i in t else i for i in s])).replace(' ', sep)
    return trs


ss = 'Лучший курс по Python!'
print(translater(ss))


# %% 7.5.5
def get_data_fig(*args, **kwargs):
    if kwargs != {}:
        return sum(args), tuple(kwargs.get(k)
                                for k in ["type", "color", "closed", "width"]
                                if kwargs.get(k) != None)
    else:
        return sum(args)


# %% 7.5.5b
def get_data_fig(*args, **kwargs):
    if kwargs != {} and len(args) != 1:
        tt = (sum(args),)
        for k in ["type", "color", "closed", "width"]:
            if kwargs.get(k) is not None:
                tt += (kwargs.get(k),)
        return tt
    elif len(args) == 1:
        return args
    else:
        return sum(args)


d = [10]
res = get_data_fig(*d)
print(*res)


# %% 7.10.1
def counter_add(start=1):
    def step():
        nonlocal start
        start += 5
        return start

    return step


counter1 = counter_add(0)
print(counter1())


# %%
def set_tag(tag):
    def taggin(s):
        return print(f'<{tag}>{s}</{tag}>')

    return taggin


t1 = set_tag('h1')
t1(input())


# %% 7.11.1
def func_show(func):
    def print_msg(*args, **kwargs):
        res = func(*args, **kwargs)
        return print(f"Площадь прямоугольника: {res}")

    return print_msg


@func_show
def get_sq(width, height):
    return width * height


get_sq(3, 4)


# %% 7.11.2
def func_show(func):
    def print_msg(*args, **kwargs):
        res = func(*args, **kwargs)
        for i, val in enumerate(res):
            print(f'{i + 1}. {val}')

    return print_msg


@func_show
def get_menu(*args):
    return list(args.split())  # % ok


s = 'Главная Добавить Удалить Выйти'

get_menu(s)


# %% 7.11.3
def func_show(func):
    def print_msg(*args, **kwargs):
        res = func(*args, **kwargs)
        return sorted(res)

    return print_msg


@func_show
def get_list(s):
    return list(map(int, s.split()))


ls = '8 11 -5 4 3 10'
print(*get_list(ls))


# %% 7.11.4
def get_dict(func):
    def function(*args, **kwargs):
        s1, s2 = func(*args, **kwargs)
        d = {s1[i]: s2[i] for i in range(len(s1))}
        return d

    return function


@get_dict
def get_list(in1, in2):
    lst1 = list(in1.split())
    lst2 = list(in2.split())
    return lst1, lst2


ls1 = 'house river tree car'
ls2 = 'дом река дерево машина'

d1 = get_list(ls1, ls2)
print(*sorted(d1.items()))


# %%7.12.1
def dec_param(start=0):
    def add_sum(func):
        def wrapper(*args, **kwargs):
            res = start + sum(func(*args, **kwargs))
            return res

        return wrapper

    return add_sum


@dec_param(5)
def get_list(s):
    return list(map(int, s.split()))


st = input()
print(get_list(st))

# %% 7.12.2
from functools import wraps


def dec_param(tag=''):
    def add_tag(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            res = f"<{tag}>{func(*args, **kwargs)}</{tag}>"
            return res

        return wrapper

    return add_tag


@dec_param('div')
def get_lower_string(s):
    return s.lower()


st = input()
print(get_lower_string(st))

# %% 7.12.3
from functools import wraps


def dec_param(chars=''):
    def add_tag(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            res = (''.join(['-' if res[i] in chars else res[i] for i in range(len(res))]))
            res = res.replace('---', '-').replace('--', '-')
            return res

        return wrapper

    return add_tag


@dec_param('?!:;,. ')
def translater(s):
    trs = (''.join([t[i] if i in t else i for i in s.lower()]))
    return trs


t = {'ё': 'yo', 'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y',
     'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f',
     'х': 'h', 'ц': 'c', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'}

st = 'Декораторы - это круто!'

print(translater(st))


# %% 9.2.1
def get_sum(n):
    for _ in range(1, n + 1):
        a = range(_ + 1)
        yield sum(a)


a = get_sum(5)
print(*[i for i in a], end=' ')


# %% 9.2.2
def get_balakirev(n):
    a = [1, 1, 1]
    for _ in range(n):
        b = sum([a[i] for i in range(_, _ + 3)])
        a.append(b)
        yield b


a = get_balakirev(5)
print(*[i for i in a], end=' ')
# %% 9.3.4
s = 'dhouse=дом car=машина men=человек tree=дерево'
s_lst = s.split()


def splitting(i):
    res = (i.split('=')[0], i.split('=')[1])
    return res


print(tuple(map(splitting, s_lst)))


# %%

def get_sort(d):
    res = list({k: d.get(k) for k in sorted([val for val in d], reverse=True)}.values())
    return res


dt = {'cat': 'кот', 'horse': 'лошадь', 'tree': 'дерево', 'dog': 'собака', 'book': 'книга'}

print(get_sort(dt))
# %%

with open('dataset_3363_2.txt') as file:
    s = file.readline()
print(s)
letters = []
digits = []
cn = 0
for i in s:
    if i.isalpha():
        letters.append(i)
        cn = 0
    elif cn > 0:
        digits.append(digits.pop() + i)
    else:
        digits.append(i)
        cn += 1

res = ''.join([val * (int(digits[i])) for i, val in enumerate(letters)])
print(res)
with open('dataset_3363_2_result.txt', 'w') as otp:
    otp.write(res)

# %%
import random

random.seed(1)

lst_in = ['1 2 3 4', '5 6 7 8', '9 8 6 7']

z = list(zip(*[i.split() for i in lst_in]))
random.shuffle(z)
lst_out = zip(*z)
[print(*_) for _ in lst_out]

# %% 9.7.3
st = ['до', 'фа', 'соль', 'до', 'ре', 'фа', 'ля', 'си']
d = {'до': 1, 'ре': 2, 'ми': 3, 'фа': 4, 'соль': 5, 'ля': 6, 'си': 7}
print(*sorted(st, key=lambda n: d[n]))

# %% 9.7.4
try:
    with open('d_1.txt', encoding='utf-8') as file:
        lst_in = list(map(str.strip, file.readlines()))
except:
    print('Error')
print(lst_in)
# %%
st = list()
for i in lst_in:
    s = i.split(';')
    q = []
    for j in s:
        if j.isdigit():
            j = int(j)
        q.append(j)
    st.append(tuple(q))
st = tuple(st)
print(st)

# d = {'Имя': 1, 'Зачет': 2, 'Оценка': 3, 'Номер': 4}
t_lst = tuple(zip())

# %%
try:
    with open('dataset_3363_3.txt', encoding='utf-8') as file:
        lst_in = list(map(str.strip, file.readlines()))
except:
    print('Error')
print(lst_in)
# %%
st = ' '.join(i for i in lst_in).lower()
q = {val: st.split().count(val) for val in set(st.split())}
print(*sorted(q.items(), key=lambda x: x[-1], reverse=True)[0])

# %% 9.5.2
import sys

# считывание списка из входного потока
lst_in = list(map(str.strip, sys.stdin.readlines()))
[print(*i, sep='') for i in zip(*zip(*lst_in))]

# %% 9.5.3
lst_in = ['1 2 3 4', '5 6 7 8', '9 8 7 6']
lst = []
[lst.append(list(map(int, i.split()))) for i in lst_in]
[print(*i, sep=' ') for i in zip(*lst)]

# %% 9.5.4
s_input = 'Москва Уфа Тула Самара Омск Воронеж Владивосток Лондон Калининград Севастополь'

lst = [i for i in s_input.split()]
[print(*i) for i in zip(*[iter(lst)] * 3)]

# %% 9.6.4
input_s = '10 5 4 -3 2 0 5 10 3'
print(*sorted(list(set(map(int, input_s.split()))), reverse=True)[:4])

# %% 9.6.5
input_s1 = '7 6 4 2 6 7 9 10 4'
input_s2 = '-4 5 10 4 5 65'

lst1 = sorted(list(map(int, input_s1.split())), reverse=False)
lst2 = sorted(list(map(int, input_s2.split())), reverse=True)
print(*[l1 + l2 for l1, l2 in zip(lst1, lst2)])

# %% 9.6.6
lst_in = ['смартфон:120000', 'яблоко:2', 'сумка:560', 'брюки:2500', 'линейка:10', 'бумага:500']


def cheapers(d: dict):
    return [d.get(i) for i in (sorted(d))[:3]]


d = {int(i.split(':')[1]): i.split(':')[0] for i in lst_in}
print(*cheapers(d))

# %% 5.6.4
lst_in_c = [[1, 0, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]
p = 0
for i in range(len(lst_in_c) - 1):
    for j in range(len(lst_in_c) - 1):
        if sum([lst_in_c[i][j], lst_in_c[i][j + 1], lst_in_c[i + 1][j], lst_in_c[i + 1][j + 1]]) <= 1:
            p += 1
        else:
            p += 0
print(["НЕТ", "ДА"][p == (len(lst_in_c) - 1) ** 2])

# %% 5.6.5
lst_in = [[2, 3, 4, 5, 6], [3, 2, 7, 8, 9], [4, 7, 2, 0, 4], [5, 8, 0, 2, 1], [6, 9, 4, 1, 2]]
c = 0
for i in range(len(lst_in)):
    for j in range(i + 1, len(lst_in)):
        if lst_in[i][j] == lst_in[j][i]:
            c += 1
        else:
            c += 0
print(["НЕТ", "ДА"][c == 10])

# %% 5.6.6
s = '8 11 -53 2 10 11'
lst = list(map(int, s.split()))
for i, val in enumerate(lst):
    for j in range(i + 1, len(lst)):
        if lst[j] < lst[i]:
            lst[j], lst[i] = lst[i], lst[j]

print(*lst)

# %% 5.6.7
s = '4 5 2 0 6 3 -56 3 -1'
lst = list(map(int, s.split()))

for i in range(len(lst) - 1):
    for j in (range(len(lst) - 1)):
        if lst[j] > lst[j + 1]:
            lst[j], lst[j + 1] = lst[j + 1], lst[j]

print(*lst)

# %% 9.2.3 Генератор паролей
import random
from string import ascii_lowercase, ascii_uppercase

chars = ascii_lowercase + ascii_uppercase + "0123456789!?@#$*"


def get_pass(n: int, chr: str):
    for i in range(n):
        yield chr[random.randint(0, len(chr) - 1)]


N = int(input())
for _ in range(10):
    print(*get_pass(N, chars), sep='')

# %% 9.2.4 Генератор имейлов
import random
from string import ascii_lowercase, ascii_uppercase

random.seed(1)
chars = ascii_lowercase + ascii_uppercase


def get_random_letter(n: int, chr: str):
    for i in range(n):
        yield chr[random.randint(0, len(chr) - 1)]


N = int(input())
for _ in range(5):
    mail_name = ''.join([*get_random_letter(N, chars)])
    print(f'{mail_name}@mail.ru', sep='')

# %% 3.7.3 программирование на python
rw = {'champions', 'we', 'are', 'Stepik'}
lst = ['We are the champignons', 'We Are The Champions', 'Stepic']
rw = {w.lower() for w in rw}
lst_2 = [row.lower() for row in lst]
S = set()
for row in lst_2:
    mistakes = set(row.split()).difference(rw)
    [S.add(_) for _ in mistakes]
print(*S)

# %% 3.7.4 программирование на python
commands = ['север 10', 'запад 20', 'юг 30', 'восток 40']
x, y = 0, 0
for i in commands:
    if i.split()[0] == 'север':
        y += int(i.split()[1])
    elif i.split()[0] == 'юг':
        y -= int(i.split()[1])
    elif i.split()[0] == 'восток':
        x += int(i.split()[1])
    elif i.split()[0] == 'запад':
        x -= int(i.split()[1])
    else:
        print('Wrong destination!!!')
print(x, y)

# %% 3.7.5 программирование на python
with open('dataset_3380_5.txt') as f:
    lst = [line.rstrip('\n').split('\t') for line in f]
clss = range(1, 12)

sw = {str(key): [] for key in clss}

for row in lst:
    sw.get(row[0]).append(float(row[2]))
for k in sw:
    if sum(sw[k]) != 0:
        sw[k] = sum(sw[k]) / len(sw[k])
    else:
        sw[k] = '-'
with open('results.txt', 'w') as f:
    for k in sw:
        f.write(f'{k} {sw[k]}\n')

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


# %% 3.4.4 Файловый ввод/вывод

means, subj1, subj2, subj3 = [], [], [], []
with open('dataset_3363_4.txt') as file:
    for line in file.readlines():
        row = line.strip().split(';')
        row = list(map(lambda x: int(x) if x.isdigit() else x, row))
        means.append(sum(row[1:]) / len(row[1:]))
        subj1.append(row[1])
        subj2.append(row[2])
        subj3.append(row[3])

mean_subj1 = sum(subj1) / len(subj1)
mean_subj2 = sum(subj2) / len(subj2)
mean_subj3 = sum(subj3) / len(subj3)

with open('res.txt', 'w') as output_file:
    for m in means:
        output_file.write(f'{m}\n')
    output_file.write(f'{mean_subj1} {mean_subj2} {mean_subj3}')
