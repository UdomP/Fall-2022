# class person(object):   # a "new-style" class
class person:            # a "old-style" class
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return '[person\'s name: %s, age: %g]' % (self.name, self.age)

    def hello_world(self):
        print('hello world')


class student(person):
    def __init__(self, name, age, gpa):
        person.__init__(self, name, age)
        self.gpa = gpa

    def __str__(self):
        return '[student\'s name: %s, age: %g, gpa: %g]' % (self.name, self.age, self.gpa)


class faculty(person):
    def __init__(self, name, age, title):
        # super(faculty, self).__init__(name, age)      # for "new-style" classes
        person.__init__(self, name, age)
        self.title = title

    def __str__(self):
        return '[faculty\'s name: %s, age: %g, title: %s]' % (self.name, self.age, self.title)


class PhD(student):
    def __init__(self, name, age, gpa, interest):
        student.__init__(self, name, age, gpa)
        self.interest = interest

    def __str__(self):
        s1 = student.__str__(self)
        s2 = '[research interest: %s]' % self.interest
        return s1 + ', [PhD student], ' + s2


if __name__ == '__main__':
    p = person('Allen', 30)
    s = student('Tom', 21, 3.5)
    f = faculty('James', 38, 'Associate Professor')
    print(p)
    print(s)
    print(f)
    super(faculty, f).hello_world()     # for "new-style" classes
    f.hello_world()
    x = PhD('Green', 21, 3.5, 'artificial intelligence')
    print(x)
