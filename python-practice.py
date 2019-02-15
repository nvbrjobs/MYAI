#This is my first python programme
""" This is my first python programme
Hello everybody """
print("Welcome")
Name=input("Enter Your Name:")
Age=input("Enter Your Age:")
print()
print("'Nice to meet you",Name)
print(" you are ",Age," years old!'")
print("'Nice to meet you",Name, end='')
print(" you are ",Age," years old!'")
help()
help('math')
mary=16
joe=25
dann=49
mary,joe,dann=16,25,49
Average_goals_scored_per_game = 2.5
print(Average_goals_scored_per_game)
#Type(Average_goals_scored_per_game)
type(Average_goals_scored_per_game)
import sys
import math
import math as ma
from math import sqrt,pow
ma.sqrt(25)
math.pow(3,2)
math.sqrt(49)
x=2
y=3
z=x+y
print(z)
x,y=2,3
print(x,y)
z=x+y
x=x+1
_ + x
print(x,z)
#_+x# previous operation is print can't add
print(x)
firstname='Vijay'
lastname='Naru'
print(firstname[0])
print(firstname[0:])
print(firstname[-2])
print(firstname[2:3])
#print(firstname[10])
print(firstname + ' wiki')
lenth=len(lastname)
print(lenth)
2+3
2-3
4*3
3**2
3/2
3//2
3%2
(3+2)*3
print(3+4+2-1)
#complex number (real and imaginary)
a=3
b=6
comp=5+6j
print(comp.imag)
print(comp.real)
print(complex(a,b))
#decimal to binary (base is 2)
print(bin(a))
#binary to decimal
print(0b11)
#decimal to octal ( base is 8)
print(oct(10))
#octal to decimal
print(0o12)
#decimal to hexadecimal --base is 0-9 and a-f
print(hex(16))
#hexadecimal to decimal
print(0x10)
#lists
#empyt list
my_list=[]
print(my_list)
my_list.append('Tom')
print(my_list)
my_list.append('Mary')
my_list.extend('Vijay')
my_list.count('T')
print(my_list)

for x in (range(0,10)):
    print("welcome")
for x in (range(4,10)):
    print("welcome")
for x in (range(4,10,2)): #step up 2
    print("welcome")
"""Below, show how to append any two elements to 
my_list at run time (after compilation) A loop should be used. [3]"""
for x in (range(0,2)):
    my_list.append(x)
print(my_list)
my_list=[]
x=1
while x<=2:
    my_list.append(x)
    x=x+1
    print(my_list)
print(my_list)
secret_word=input("Enter the 3 letter work")[0:3]
print(secret_word)
del secret_word
secret_word=[]
print(secret_word)
secret_word=input("Enter any word ")
print(secret_word)
print(len(secret_word))
secret_cal=input("enter the expression")
print(secret_cal)
x=int(input("enter the value x:"))
y=int(input("enter the value y:"))
z=x+y
print(z)
result=eval(input("enter the expression"))
print(result)
def hello_return():
    " this ist the doc "
    print("hello word")
    " this ist the doc "
    return("hello")
hello_return.__doc__
hello_return()*2
def hello_noreturn():
    print("hello word")
#hello_noreturn()+2#error
hello_noreturn()
def my_shopping_list(a):
    print ('Welcome to the shopping list program...')
    input('Enter the valid option from 1 to 3')
    print(a)
print(my_shopping_list(2))

def myfun1(a,b=3):#first argument should be compulsory that means u should not assigned any default values
#def my fun1(a=4,b) this will give an error
    c=a+b
    return(c)
myfun1(2,3)

def myfun2(a,b=4):
    c=a+b
    return(c)
myfun2(2)

def myfun3(a,*arg):# arg is tuple
 #   "This is the my function 3 for any number of arguments"
    '''This is my function 3
    for number of arguments'''
    print(arg)

    print(arg[0])
    for i in arg:
        a=a+i
    return(a)
    return(None)
myfun3(2,4)
myfun3(2,4,5,7,3,8)
myfun3.__doc__# result any documentation in the function , it will always read the first line which 
#should be in double quotes
for j in range(4):
    for i in range(4):
        print("#",end="")
    print("") #print("\n") also works

for j in range(4):
    for i in range(j+1):
        print("#",end="")
    print("") #print("\n") also works

for j in range(4):
    for i in range(4-j):
        print("#",end="")
    print("") #print("\n") also works

#ARRAYS IMPORT THE ARRAY MODULE
#from array import *
import array as arr
# array will have one type of elements , so u have mention which type ur storing
# like 1 byte elements, 2 byte elements ,4 byte elements or 8 byte elements
# for that python has some type code like 'i' for signed integer -signed mean always positive
#'I' for unsiged integer,l FOR SIGNED LONG,I for unsigned long,u for characters,f for float, d for double -double also a float
# etc.,
var_arr=arr.array('i',[7,6,5,-4,3,9])
from array import *
var_arr=array('I',[9,3,5,6,2,1])
print(var_arr)
print(var_arr[0:3])
print(var_arr.buffer_info())    # size and number of elements
print(var_arr.reverse())
print(var_arr.typecode)
#new array with the same values
var_arrnew=array(var_arr.typecode,(a for a in var_arr))
print(var_arrnew)
########CLASSES AND OBJECTS#########################

class first:
    x=2
    y=3
    z=x+y
    def myfirst(a,b):
        "This is the description about myfirst function in first class"
        print("This is my first function and i am belongs to class FIRST")
        c=a**b
        print("Bye my first function first class")
        return c

class second:
    x=6
    y=3
    d=6
    z=x+y
    def mysecond(self):
        "This is the description about mysecond function in first class"
        print("This is my first function and i am belongs to class second")
        c=x+y
        return c
        print("Bye my first function first class")#this will not print because of return
    def mythird(s,t=4):
        print("sum of two number",s+t)
class third(second):
    def myfourth(self,x,y):
        "This is the description about mysecond function in first class"
        print("This is my first function and i am belongs to class second")
        c=x+y
        print(second.z)
        return c
        print("Bye my first function first class")#this will not print because of return

second.mysecond.__doc__       

first.x #you can read directly by class.variable
second.x
first.myfirst(3,2)#you can call directly by class.function
#there is an another approach to read and use methods of class , by creating objects

myobject1=first()# myobject1 is class of first
print(myobject1.x)
myobject2=second()
print(myobject2.c)
myobject3=second()
print(myobject3.mysecond())#since there is no self , will give an error, now add self to the function
myobject4=third()
myobject4.myfourth(x=9,y=8)


from datetime import *

print(datetime.now())
print(datetime.today()) 

from numpy import *

x=[4,5,6] #list 
y=[7,8,9]
type(x)

x+y  # will give contenation
from numpy import *
import numpy
x=numpy.array([6,5,4])
y=numpy.array([9,4,3])
x
print(x)
z=x+y
print(z)
type(x)
r=numpy.array(x)
type(r) #ndarray n dimension array
print(r)
print(x.shape)
print(x.size)
x=numpy.array([[2,3,4],[5,6,7]])
print(x)
print(x.shape)
print(x.size)#total number of elements
x=numpy.array([2,3,4.5])#which will make all the elements float
print(x)
print(x.dtype)
x=numpy.array([3,4,5,'6'])
print(x)
print(x.dtype)
x=numpy.array([[2,3,4],[5,6,7],[30,4,5]])
x.shape
x.size
x.min()
x.mean()
x.max()
x.sum()
x=numpy.array([3,4,5,6],int)
print(x)
x=numpy.array([3,4,5,6],float)
print(x)
x=numpy.linspace(0,15)
print(x)
x=numpy.arange(1,10)
print(x)
x=numpy.logspace(1,40,5)
print(x)
x=numpy.zeros(7,int)
print(x)

x=numpy.ones(7)
print(x)

arr=numpy.array([[3,4,5,5],[6,7,8,8],[5,6,7,3]])
print(arr)
arr2=arr+2
print(arr2)
arr3=arr+arr2
print(arr3)
arr4=arr.flatten()#multi dimensional to single dimensional
print(arr4)

arr.reshape(3,2,2)# to multi dimentionsl matrix
arr=numpy.array([[3,4,5],[6,7,8],[5,6,7]])
print(arr)
m1=matrix(arr)#matxi
print(m1)
m1=numpy.matrix('3,4,5;5,6,7;7,8,9')
print(m1)
m1.dtype
print(diagonal(arr))
#######################MULTITHREADING######################
from threading import *
from time import sleep
class Hello(Thread):
    def run(self):
        for i in range(5):
            print("Hello")
            sleep(1)
class Hi(Thread):
     def run(self):
        for i in range(5):
            print("Hi")
            sleep(1)

obj1=Hello()
obj2=Hi()

##########running the below ones the out would be sequential####So to make multiprocessing 
##use Thread in the class and instead of run use start
obj1.run() 
obj2.run()

obj1.start()
sleep(0.2)###0.2 seconds between the threads
obj2.start()

obj1.join()
obj2.join()

###still above ones not running in parallel , since the system is very high end performance one,
##u have to give a gap in between to thread, so for that use sleep()
##############################PANDAS#######################
import numpy
import pandas
data=numpy.random.randint(20,80,(30,3))#3 columns each column 30 records and values between 20 to 80
print(data)
cols=['server1','server2','server3']
rows=pandas.date_range('20190115',periods=30)
cols
rows
df=pandas.DataFrame(data,columns=cols,index=rows)
df

df['server1'] #column server1 display
df.head()#top 5 rows
df.head(10)
df.tail(10)
df.tail()#bottom 5 rows
df.describe()
df.describe().rount(4)
'''
        server1    server2    server3
count  30.000000  30.000000  30.000000
mean   52.233333  48.866667  50.800000
std    18.449714  17.600418  17.303378
min    21.000000  21.000000  20.000000
25%    39.250000  38.250000  41.000000
50%    54.500000  52.000000  47.500000
75%    69.750000  60.500000  61.250000 
max    79.000000  77.000000  79.000000 '''
df.min()
df.mean()
df.median()
df.var()
df.std()
df.kurt()
df.max()
df.skew()
df.mode()
df['server1'].mode()

df['2019-01-15':'2019-01-30']
df['2019-01-15':'2019-01-30']['server1']
df['server1']['2019-01-15':'2019-01-30']
df[df['server1']>30]['server1']
df[df['server1']>30][df['server2']>40]
df[(df['server1']>30) & (df['server2']>40)]

df[(df['server1']>30) | (df['server2']>40)]

df.to_excel(r'C:\Users\VIJAI\Desktop\AI-ML-DL-PYTHON-TRAINING-VIDEOS\DATASET-ASSIGNMENTS\dftest1.xls'
            )
df.to_csv(r'C:\Users\VIJAI\Desktop\AI-ML-DL-PYTHON-TRAINING-VIDEOS\DATASET-ASSIGNMENTS\dftest2.txt'
            )
df.to_json(r'C:\Users\VIJAI\Desktop\AI-ML-DL-PYTHON-TRAINING-VIDEOS\DATASET-ASSIGNMENTS\dftest2.jsn'
           )
import pandas
df2=pandas.read_csv(r'C:\Users\VIJAI\Desktop\AI-ML-DL-PYTHON-TRAINING-VIDEOS\DATASET-ASSIGNMENTS\datanah.wsp')
df2



#################linear regression using theano MACHINE LEARNING#################

import theano
import numpy

x=theano.tensor.fvector('x')
y=theano.tensor.fvector('y')
m=theano.shared(0.9,'m')
c=theano.shared(0.8,'c')
yhat=m*x+c
print(yhat)
#cost function
cost=theano.tensor.mean(theano.tensor.sqr(y-yhat))/2
#Gradient Descent Algorithm
LR=0.01
gradm=theano.tensor.grad(cost,m)#dj/dm
gradc=theano.tensor.grad(cost,c)#dj/dc

mn=m-LR*gradm
cn=c-LR*gradc
train=theano.function(inputs=[x,y],outputs=cost,updates=[(m,mn),(c,cn)])

##create data

area=[1.2,1.8,2.4,3.6,2.9,3.8,4.2,4.9,5.6,2.7,8.1]
price=[180,240,290,340,320,420,450,510,540,310,780]
area=numpy.array(area).astype('float32')
price=numpy.array(price).astype('float32')

for i in range(1000):
    cost_val=train(area,price)
    print(cost_val)

print(m.get_value())
print(c.get_value())
predict1=cost_val(1.9)

import pandas
import matplotlib.pyplot as plt
import numpy
from sklearn.linear_model import LinearRegression as LNR
import seaborn as sns
df=pandas.read_csv(r"C:\Users\VIJAI\Desktop\AI-ML-DL-PYTHON-TRAINING-VIDEOS\Supervised Learning - Linear Regression\combined_cycle_power_plant_dataset.csv")

print(df)
print(df.describe())
####finding the null values in the dataset
df2=df.isnull().sum()
df2
df2.shape
df2.size
df.isnull().sum()
df.shape
df.size
##dropping the duplicates
df3=df.drop_duplicates()
df3
df.drop_duplicates(inplace=True)

# finding the corelation 
cor=df.corr()
#visualization
plt.figure(figsize=(10,8))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()
# since the relations correlation between the attributes between -1 to 1, so we can go for linera regression
#X axis has all the attributes except the PE
# X AXIS HAS 4 ATTRIBUTES WITHOUT PE column
x=df.drop(['PE'],axis=1)
y=df['PE']
x.shape
y.shape
print(x)
print(y)
#now split the data into 2 parts, 1 is for train the model and 1 is for testing 
from sklearn.model_selection import train_test_split
#above function is used for split the data into train and test
#the standard split is 80 /20 or 70/30 --80 train and 20 test like that
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)#0.2 means 20%
xtrain.shape
xtest.shape
ytrain.shape
ytest.shape
model=LNR()
#train the model using fit function
model.fit(xtrain,ytrain)
print(model.coef_)
print(model.intercept_)
#test the model with some static values
ip=numpy.array([[22.99,46.93,1014.15,49.42]])
model.predict(ip)#453.34292462

##r2 score##this is used to measure how accurate the model , the maximum can be 1

ypred=model.predict(xtest)
ypred
from sklearn import metrics
metrics.r2_score(ytest,ypred)#0.9261397091650688  closeness 