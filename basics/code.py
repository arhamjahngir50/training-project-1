# def greet():
#     print("helloworld")
#     return("returned helloworld")


# hello = greet()
# print ("what returned from the function ", hello )



##########################################
#partial functions: 


# If you frequently compute squares:

# power(x, 2)

# Instead of repeating exponent=2 every time, you can create a new function:

# from functools import partial

# square = partial(power, exponent=2)

# print(square(5))  # 25


############################################
# lambda functions 

#greet = lambda : print('Hello World')
# call the lambda
#greet()

# lambda function with parameter 
#greet_user = lambda name : print('Hey there,', name)
#greet_user('Delilah')


##########################################
#recursive functions 
# def factorial_recursive(n):
#     # Base case: 1! = 1
#     if n == 1:
#         return 1

#     # Recursive case: n! = n * (n-1)!
#     else:
#         return n * factorial_recursive(n-1)

###########################################
# monkey patching in functions 
#Monkey patching = dynamically modifying existing code at runtime.

# class Cat:
#     def speak(self):
#         return "Meow"

# c = Cat()

# def new_speak():
#     return "Roar"

# c.speak = new_speak   # Patch only this object

# print(c.speak())  # Roar

##################################################
# PYTHON LISTS 

# mutable 
# can store duplicates


# a= [10,20,"apple","banana"]
# print(a)

# print(a[3])
# a[3]=20
# print(a[3])
# print(a[1:4])

# to add to list use apend 


##############################################
#TUPLE
#Tuple unchangeable(cannot change once created) cannot add or remove elements. Allows duplicate members
# items acessed same as list 
#tup = (1, 2, 3, 4, 5)
#del tup


###############################################
#SETS 
# no indexing # no duplicates #Duplicates are automatically removed.

#my_set = {1, 2, 3, 4}
# print(my_set)
#output: {1, 3, 2, 4}   # no order guranteed 

#create empty set 
#empty_set = set()   # Correct

#my_set.add(4)   # for single elements 
#update()              # for multiple elements 

##FROZEN SET"S 
# cannot add or remove elemenelementsts
# 
#  
##############################
# Dictionary
# key value pair # key can not be changed
# can be  added , removed 
#my_dict = {"name": "Ali", "age": 25}





#####################
#STRINGS
# in pythons strings are immutable meaninng character of a string cannot be changed 

# greet = 'Hello'

# # iterating through greet string
# for letter in greet:
#     print(letter)

#print(f'{name} is from {country}')




##########################################
# ##MEMORY MANAGEMENT IN PYTHON 
# a=10
# b=[10,20]
# a in b 
# a=true


# a = False
# b = True
# c = False
# d = True

# result = a or b and c
# print(result)


# from functools import partial

# def power(val1, val2):
#     return val1 ** val2

# square = partial(power, val2=2)

# print(square(5,10)) 


# a = [10, 20, 30, 40, 50]

# for value in range(1,6):
#     print(a[value])

#########################################################################
### GENERATORS 

# FIRST SIMPLE FUNCTION EXAMPLE WITHOUT GENERATOR  
# def square_numbers(nums):
#     result = []
#     for n in nums:
#         result.append(n*n)
#     return result

# numbers = [1,2,3,4]
# print(square_numbers(numbers))


### NOW WITH GENERATOR 

# def square_numbers(nums):
#     for n in nums:
#         yield n*n

# numbers = [1,2,3,4]



# #calling generator 
# gen = square_numbers(numbers)

# print(next(gen))
# print(next(gen))
# print(next(gen))
# print(next(gen))

# # can also call in for loop 
# for value in gen:
#     print(value)
#     print("hehehhe")



# Generator expression (lazy evaluation)
# squares_gen = (x**2 for x in range(5))
# print(next(squares_gen))  # 0
# print(next(squares_gen))  # 1


#Generator pipelining means passing the output of one generator into another generator.

#This is useful for data processing pipelines.

#
##READING LARGE FILES USING GENERATOR 
#def read_large_file(file_name):
#    with open(file_name) as f:
#        for line in f:
#            yield line
#
#for line in read_large_file("large_file.txt"):
#    print(line.strip())


############################################################################################
###########################################################################################
############################################################################################

# #DECORATORS 
# def decorator_function(func):
    
#     def wrapper(a:str,b:str):
#         print("Welcome!")
#         C=a.capitalize()
#         D=b.capitalize()
#         re=func(C,D)
#         print("Goodbye!")
#         return re 
#     return wrapper


# @decorator_function
# def greet(a:str,b:str):

#     return(a+b)

# print(greet("arham","jahangir"))




# def decorator2(helpo):
#     def wraper(*args, **args2):
#         print("before func")
#         helpo(*args, **args2)
#         print("after func ")
#     return wraper

# @decorator2
# def helpo(*args, **args2):
#     print("inside func ")
#     print(args)

# helpo(2,3)



#####################
#zero copy 
# send data directly without copyin 
#memoryview


## RANDOM NUMBERS 
# import random

# # Setting a seed for reproducibility
# random.seed(44)

# print(random.random())          # Random float between 0.0 and 1.0
# print(random.randint(1, 10))     # Random integer between 1 and 10
# print(random.choice(['A', 'B'])) # Randomly pick an element from a list


# for i in range(10):
#     print(random.randint(1,10))




# random numbers using secret 
# import secrets

# # Generate a secure token for a password reset link
# token = secrets.token_hex(16) 
# # Generate a secure random integer for a dice roll in a gambling app
# secure_roll = secrets.randbelow(6) + 1



##########################################################################
#############################################################################
#NUMPY 
#Fixed Type: Python lists can hold different types (strings, ints, floats) simultaneously,
#  which makes them slow because Python has to check the type of every single element.
#  NumPy arrays are homogeneous (all elements are the same type), allowing the computer to process them much faster.
##
#
##
#
#Contiguous Memory: NumPy stores data in one continuous block of memory, 
# whereas Python lists store pointers to objects scattered all over your RAM.
#
#
#
#Vectorization: NumPy allows you to perform an operation on an entire array at once without writing a for loop.
#multiproccessing


#import numpy as np
#arr = np.array([1, 4, 9])            # 1D array
#print(np.sqrt(arr))                 # Output: [1. 2. 3.]


'''Key PEP 8 Rules:

Indentation: Use 4 spaces per indentation level (never tabs).

Variable Names: Use snake_case (e.g., my_variable_name).

Class Names: Use PascalCase (e.g., MyClassName).

Line Length: Limit all lines to a maximum of 79 characters.

Imports: Should usually be on separate lines at the top of the file.

Whitespace: Avoid extraneous whitespace inside parentheses or immediately before a comma.'''

##############################################################
################################################################
################################MULTIPROCECSSING 


# import multiprocessing
# import time

# def calculate_square(number):
#     time.sleep(1) 
#     result = number * number
#     print(f"Number {number} squared is {result}")
#     return result

# if __name__ == "__main__":
#     numbers = [1, 2, 3, 4, 5, 6, 7, 8]

#     # By default, it creates as many processes as you have CPU cores
#     with multiprocessing.Pool() as pool:
#         results = pool.map(calculate_square, numbers)

#     print(f"Final results: {results}")




##############################################################
################################################################
###############################
#MULTITHREADING


# import time
# from threading import Thread 


# def task(name):
#     print(name, "starting")
#     print(name, "finished")

# t1 = Thread(target=task, args=("Thread-1",))
# t2 = Thread(target=task, args=("Thread-2",))

# t1.start()
# t2.start()

# t1.join()
# t2.join()

# print("All threads done") 






# ABSTRACT CLASSES 

# from abc import ABC, abstractmethod

# class MyClass(ABC):

#     @abstractmethod
#     def hello(self):
#         pass


# class Child(MyClass):
#     def hello(self):
#         print("Hello implemented")

# obj = Child()
# obj.hello()


#####################
# DEPENDENCY INJECTION 
#payment example 
# Dependency Injection Version


# class PayPalProcessor:
#     def pay(self, amount):
#         print(f"Paid {amount} using PayPal")

# class StripeProcessor:
#     def pay(self, amount):
#         print(f"Paid {amount} using Stripe")

# class Order:
#     def __init__(self, payment_processor):  # inject dependency
#         self.payment_processor = payment_processor

#     def checkout(self, amount):
#         self.payment_processor.pay(amount)

# # Inject PayPal
# paypal_processor = PayPalProcessor()
# order1 = Order(paypal_processor)
# order1.checkout(100)

# # Inject Stripe
# stripe_processor = StripeProcessor()
# order2 = Order(stripe_processor)
# order2.checkout(200)

# # Inject Mock Processor for testing
# class MockProcessor:
#     def pay(self, amount):
#         print(f"Mock payment of {amount}")

# mock_processor = MockProcessor()
# test_order = Order(mock_processor)
# test_order.checkout(50)




##################################
# FACTORY PATTERN 
# class Shape:
#     def draw(self):
#         pass

# class Circle(Shape):
#     def draw(self):
#         print("Drawing Circle")

# class Square(Shape):
#     def draw(self):
#         print("Drawing Square")

# # Factory function
# def shape_factory(shape_type):
#     if shape_type == "circle":
#         return Circle()
#     elif shape_type == "square":
#         return Square()
#     else:
#         raise ValueError("Unknown shape type")

# # Using the factory
# shape1 = shape_factory("circle")
# shape1.draw()   # Output: Drawing Circle





#Adding a New Shape Without Changing Client Code
# dont need to change frontend can just add a check in factory of input and add new class 
# make code flexible and easy to maintain. 









############################################
# LIST AND DICTIONARY COMPREHENTION
#squares = [number * number for number in range(10)]
#print(squares)



#sentence = "the rocket came back from mars"
#[char for char in sentence if char in "aeiou"]
#['e', 'o', 'e', 'a', 'e', 'a', 'o', 'a']

# vowels={number: number * number for number in range(10)}
# print(vowels)



####
#Poetry 
#handles version conflicts automatically
# handles requirements.txt automatically 
# can also install dependencies 




###################
# DOCKER 
# Dockerfile
# Docker Image 
# container     

# import gc

# # See how many objects are currently in each generation
# print(gc.get_count())

# import sys

# x = [1, 2, 3]
# print(sys.getrefcount(x)) 

# y = x
# print(sys.getrefcount(x)) 

# y = None
# print(sys.getrefcount(x))



#########################################
######### CONTEXT MANAGERS 
# implement __enter__ and __exit__ 



###############################################
############# DATA WAREHOUSES 
# | Name                        | Description                                              |
# | --------------------------- | -------------------------------------------------------- |
# | **Amazon Redshift**         | Cloud data warehouse by AWS                              |
# | **Google BigQuery**         | Serverless, fast analytics data warehouse                |
# | **Snowflake**               | Cloud-native, supports structured & semi-structured data |
# | **Microsoft Azure Synapse** | Enterprise data warehouse on Azure                       |
# | **Teradata**                | Traditional on-premise warehouse for large enterprises   |''


# DATA LAKES 
# Amazon S3 (with Athena/Glue)	Object storage, can be used as a data lake
# Azure Data Lake Storage	Optimized for big data analytics
# Google Cloud Storage (GCS)	Stores raw data, can be queried with BigQuery
# Hadoop HDFS	Distributed file system for storing huge datasets
# Databricks Delta Lake	Layer on top of S3/HDFS with ACID transactions


########### MULTI THREADING AND ASYNC AWAIT DIFFERENCE 
########### Imagine fetching 1000 URLs:

# Multithreading: 1000 threads → high memory, context switching overhead

# Async/Await: 1 thread → switches between tasks when each URL waits for response → much lighter
























# from abc import ABC, abstractmethod
# class frontend(ABC):

#     @abstractmethod
#     def hello(self):
#         pass
#     @abstractmethod
#     def bye(self):
#         pass


# class child2(frontend):
#     def hello(self):
#         print("implementing hello method from real class ")

#     def bye():
#         print("implementing bye method from real class")

#     def __str__(self):
#         return "this is the print overloaded from the class "

# objj=child2()
# objj.hello()
# print(objj)





##################################################3
############# ASYNC AWAIT 
# import asyncio
# import aiofiles  # Async file operations

# async def write_file(filename, content):
#     async with aiofiles.open(filename, 'w') as f:
#         await f.write(content)
#         print(f"Written {len(content)} characters to {filename}")

# async def main():
#     tasks = [
#         write_file("file1.txt", "Hello World!"),
#         write_file("file2.txt", "Async I/O is cool!"),
#         write_file("file3.txt", "Python asyncio rocks!")
#     ]
#     await asyncio.gather(*tasks)

# asyncio.run(main())




############################### 
####33 closure 

# def outer(x):
#     def inner(y):
#         return x + y  # inner "remembers" x
#     return inner

# f = outer(10)   # outer executes and returns inner
# print(f(5))     # 15
# print(f(20))    # 30




i=0
listt=['a','b','c']
list2={listt[i]:listt[:i+1] for i in range(len(listt))}
print(list2)