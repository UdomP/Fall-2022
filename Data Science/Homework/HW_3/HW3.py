class Person:
    def __init__(self, name, age):
        # write your code here

    def __str__(self):
        return '[person\'s name: %s, age: %g]' % (self.name, self.age)

    def RiseSalary(self):
        print('SALARY RISE REFUSED')


class Doctor(Person):
    def __init__(self, name, age, speciality, salary):
        # write your code here

    def __str__(self):
        # write your code here
        
    def RiseSalary(self):
        # write your code here


class Lawyer(Person):
    def __init__(self, name, age, speciality, salary):
        # write your code here

    def __str__(self):
        # write your code here
        
    def RiseSalary(self):
        # write your code here


def RiseSalary_twice(aa):
    aa.RiseSalary()
    aa.RiseSalary()


if __name__ == '__main__':
    p = Person('Tom', 23)
    d = Doctor('James', 38, 'Pediatrician', 100000)
    l = Lawyer('Bob', 42, 'patent', 100000)
    print(p)
    print(d)
    print(l)
    RiseSalary_twice(p)
    RiseSalary_twice(d)
    RiseSalary_twice(l)