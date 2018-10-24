#Python yiled语法
#关系：生成器是一种特殊的迭代器，
#       迭代器是一种特殊的可迭代对象
#       可迭代对象就是可以执行迭代操作，也就是可以通过for循环来遍历的对象

#相对于列表迭代，在元素更多时，使用yiled语法可以极大的节省内存
def my_generator():
    index = 10
    while index > 0:
        yield index
        index -= 1

#DIY一个range函数
def my_range(start, end, step=1):
    my_list = []
    while start < end:
        my_list.append(start)
        start += step
    return my_list

def my_range_yield(start, end, step=1):
    while start < end:
        yield start
        start += step

#读取文件
def read_file(file_name):
    with open(file_name,'r') as file:
        line = file.readline()
        while line:
            yield line.strip('\n')
            line = file.readline()

#斐波那契数列
def fibonacci(n):
    fib_list = []
    a, b = 0, 1
    while n > 0:
        fib_list.append(b)
        a, b = b, a+b
        n -= 1
    return fib_list

def fibonacci_yield(n):
    a, b = 0, 1
    while n > 0:
        yield b
        a, b = b, a+b
        n -= 1

for item in fibonacci_yield(10):
    print(item)
