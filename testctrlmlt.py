def dec2bin (x):
    l = []
    while x!=0:
        l.append(x&1)
        x >>= 1
    l = l + [0] * (8-len(l))
    return ''.join([str(i) for i in l[::-1]])

def gcd(a,b):
    if a<b:
        a,b = b,a
    while a%b != 0:
        a, b = b, a%b
    return b

def f(x):
    return 84*x % 221

for k in range(1, 256):
    if gcd(221,k) == 1:
        print(dec2bin(k) + " ---> " + dec2bin(f(k)))
