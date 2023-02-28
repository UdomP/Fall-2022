# an example for polymorphism
class Animal(object):
    def run(self):
        print('Animal is running...')


class Dog(Animal):
    # overriding method
    def run(self):
        print('Dog is running...')


class Cat(Animal):
    # overriding method
    def run(self):
        print('Cat is running...')

    def eat(self):
        print('Eating meat...')


class Turtle(Animal):
    # overriding method
    def run(self, s='hello world'):
        print('Turtle is running slowly...')
        print(s)


class Tiger(Animal):
    # overriding method
    def run(self, s='I am a tiger'):
        print('Tiger is running fast...')
        print(s)


def run_twice(aa):
    aa.run()
    aa.run()


if __name__ == '__main__':
    dog = Dog()
    dog.run()
    print('----------------')
    cat = Cat()
    cat.run()
    print('----------------')
    run_twice(Animal())
    print('----------------')
    run_twice(Dog())
    print('----------------')
    run_twice(Cat())
    print('----------------')
    run_twice(Turtle())
    print('----------------')
    run_twice(Tiger())
    print('----------------')
    run_twice(dog)