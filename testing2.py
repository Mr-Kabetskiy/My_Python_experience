#%%
try:
    with open('dataset_3363_3.txt', encoding='utf-8') as file:
        lst_in = list(map(str.strip, file.readlines()))
except:
    print('Error')
print(lst_in)
#%%
st = ' '.join(i for i in lst_in).lower()
q = {val: st.split().count(val) for val in set(st.split())}
print(*sorted(q.items(), key= lambda x: x[-1], reverse= True)[0])