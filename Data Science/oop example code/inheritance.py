# an example for inheritance
class Animal(object):
    def __init__(self):
        self.age = 0.0

    def run(self):
        print('Animal is running...')


class Tiger(Animal):
    pass


class Dog(Animal):
    def __init__(self):
        Animal.__init__(self)
        self.interest = 'eating and playing'


class Cat(Animal):
    def eat(self):
        print('Eating meat...')


if __name__ == '__main__':
    d = Dog()
    d.run()

    c = Cat()
    c.run()
    print(c.age)

    t = Tiger()
    t.run()