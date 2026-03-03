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
##MEMORY MANAGEMENT IN PYTHON 
