import sys
sys.path.insert(0, 'C:/Users/serey/Desktop/test')
from my_modules import module_lab_w14 as modd
from my_modules import Yu
from my_modules import LBJ
from module_outside import CSU

CSU.f1()
CSU.f2()

Yu.f1()
Yu.f2()

LBJ.f1()
LBJ.f2()

print(modd.hello())
print(type(modd.hello()))
print(type(modd.hello))

print(modd.addXY(6, 8))