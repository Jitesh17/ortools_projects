x = [ 1, 1, 0, 0, 0, 1, 0]


sum_all_shift_val_dec = int(''.join(map(lambda x: str(x), x)), 2)

print(x)
print(str(x))
print(sum_all_shift_val_dec)

def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

print(bool2int(x[::-1]))

from numpy import *
a = array([[ 1, 1, 0, 0, 0, 1, 0]])

b2i = 2**arange(a.shape[0]-1, -1, -1)

result = (a*b2i).sum(axis=1)  #[12  4  7 15]
print(result)