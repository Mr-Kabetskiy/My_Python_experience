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
