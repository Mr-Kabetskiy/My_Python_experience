# %% 5.6.8
s = '1 2 4 8 16 32 64'
n = 221
lst = list(map(int, s.split()))
p = []
for i in sorted(lst, reverse=True):
    while n >= i:
        p.append(i)
        n -= i

print(*p)
