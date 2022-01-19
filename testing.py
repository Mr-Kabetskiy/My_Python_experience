lst_in = ['3 Сергей', '5 Николай', '4 Елена', '7 Владимир', '5 Юлия', '4 Светлана']
# keys = [k[0] for k in [i.split() for i in lst_in]]
# val = [k[1] for k in [i.split() for i in lst_in]]
# d = {}
# for i in range(len(lst_in)):
#     d.update({keys[i]: val[i]})
# print(d)
tp = [(i.split()[0], i.split()[1]) for i in lst_in]

a = dict()
for i in lst_in:
    if i.split()[0] in a:
        a[i.split()[0]] += [i.split()[1]]
    else:
        a[i.split()[0]] = [i.split()[1]]

for i in a:
    print(f'{i}:', ', '.join(a[i]))