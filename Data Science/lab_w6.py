def g(t):
    return t**(3)

def diff2(f, x, h=1E-6):
    r = (f(x-h) - 2*f(x) + f(x+h))/float(h*h)
    return r

for k in range(1,14):
    h = 10**(-k)
    print ("h=%.0e: %.5f" % (h, diff2(g,1,h)))