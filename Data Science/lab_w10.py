s = 'Berlin: hello 6 pm Berlin'
print(s[0])
print(s[1])
print(s[2])
print(s[-1])
print(s[-2])
print(s[-3])
print(s[8:])
print(s[8:12])
print(s[8:-1])
print(s[8:-2])
print(s.find('Berlin'))
print(s.find('hello'))
print(s.find('hel'))
print(s.find('Oslo'))

L = s.replace(s[:s.find(':')], 'Ber')
print(L)

L = s.split(':')[1].split()[3]
print(L)