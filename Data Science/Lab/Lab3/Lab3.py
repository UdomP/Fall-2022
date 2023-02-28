import numpy as np
import drawing as dw

x = np.linspace(0, 30, 100)
# warm up: draw sin(x) and cos(x)
dw.DrawSin(x, 'sin.png')
dw.DrawCos(x, 'cos.png')
# draw multiple Gaussian distributions on one figure
A={10:2, 15:3, 11:1}
dw.DrawGMM(x, A, 'my_test1.png')
B={10:2, 15:3}
dw.DrawGMM(x, B, 'my_test2.png')
C={10:2}
dw.DrawGMM(x, C, 'my_test3.png')