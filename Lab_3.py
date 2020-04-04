import numpy as np
import random as rd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import copy
import pandas as pd

I = [[1, -1, -1, -1, 1,
     1, 1, -1, -1, 1,
     1, -1, 1, -1, 1,
     1, -1, -1, 1, 1,
     1, -1, -1, -1, 1], #N
     
     [1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1],  #E
     
     [1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, 1, 1,
     1, 1, 1, -1, 1],  #U
     
     [1, 1, 1, 1, -1,
     1, -1, -1, 1, -1,
     1, -1, -1, 1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, 1],  #R
     
     [-1, -1, 1, -1, -1, 
     -1, 1, -1, 1, -1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1,
     1, -1, -1, -1, 1], #A
     
     [1, -1, -1, -1, -1, 
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1], #L
    
    [-1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1],  #space
    
    [1, -1, -1, -1, 1,
     1, 1, -1, -1, 1,
     1, -1, 1, -1, 1,
     1, -1, -1, 1, 1,
     1, -1, -1, -1, 1], #N
    
    [1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1],  #E

     [1, 1, 1, 1, 1,
     -1, -1, 1, -1, -1,
     -1,- 1, 1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1],  #T
     
     [1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, 1, -1, 1,
     1, -1, 1, -1, 1,
     -1, 1, -1, 1, -1],  #W
     
     [-1, 1, 1, 1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     -1, 1, 1, 1, -1],  #O
     
     [1, 1, 1, 1, -1,
     1, -1, -1, 1, -1,
     1, -1, -1, 1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, 1],  #R
     
     [1, -1, -1, -1, 1,
     1, -1, -1, 1, -1,
     1, 1, 1, -1, -1,
     1, -1, -1, 1, -1,
     1, -1, -1, -1, 1]  #K

     ]


I2 = [[1, -1, -1, -1, 1,
     1, 1, 1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, 1, 1,
     1, -1, -1, -1, 1], #N
     
     [1, 1, -1, 1, -1,
     1, 1, -1, -1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1],  #E
     
     [1, -1, -1, -1, 1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, -1, 1],  #U
     
     [1, 1, 1, -1, -1,
     1, -1, -1, 1, -1,
     1, -1, 1, 1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, 1],  #R
     
     [-1, -1, 1, -1, -1, 
     -1, 1, -1, 1, -1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, -1,
     1, -1, -1, 1, 1], #A
     
     [1, -1, -1, -1, -1, 
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, -1, 1, -1, -1,
     1, 1, -1, 1, -1], #L
    
    [-1, -1, -1, -1, -1,
     -1, -1, -1, 1, -1,
     -1, -1, -1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, -1, -1, -1],  #space
    
    [1, -1, -1, -1, 1,
     1, 1, -1, 1, 1,
     1, -1, 1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1], #N
    
    [1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, -1, 1, -1,
     1, 1, -1, -1, -1,
     1, 1, 1, 1, -1],  #E

     [1, 1, 1, 1, -1,
     -1, -1, 1, -1, -1,
     -1,- 1, 1, 1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1],  #T
     
     [1, -1, -1, -1, 1,
     1, -1, -1, 1, 1,
     1, -1, 1, -1, -1,
     1, -1, 1, -1, 1,
     -1, 1, -1, 1, -1],  #W
     
     [-1, 1, 1, 1, -1,
     1, -1, -1, 1, 1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     -1, 1, 1, 1, -1],  #O
     
     [1, 1, 1, 1, -1,
     1, -1, -1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, 1,
     1, -1, -1, -1, 1],  #R
     
     [-1, -1, -1, -1, 1,
     1, -1, 1, 1, -1,
     1, 1, 1, -1, -1,
     1, -1, -1, 1, -1,
     1, -1, -1, -1, 1]  #K

     ]

I4 = [[1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     1, -1, 1, -1, -1,
     -1, -1, -1, 1, 1,
     1, -1, -1, -1, 1], #N
     
     [1, 1, 1, 1, -1,
     -1, -1, -1, -1, -1,
     1, 1, 1, 1, -1,
     -1, -1, -1, -1, -1,
     -1, 1, -1, 1, -1],  #E
     
     [-1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     1, 1, 1, -1, 1],  #U
     
     [1, 1, 1, 1, -1,
     -1, -1, -1, 1, -1,
     1, -1, -1, 1, -1,
     -1, 1, -1, -1, -1,
     1, -1, -1, -1, 1],  #R
     
     [-1, -1, 1, -1, -1, 
     -1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     -1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1], #A
     
     [1, -1, -1, -1, -1, 
     -1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1,
     -1, 1, 1, -1, -1], #L
    
    [-1, -1, -1, -1, -1,
     -1, 1, -1, 1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, -1, -1, -1],  #space
    
    [1, -1, -1, -1, 1,
     -1, 1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     1, -1, -1, 1, 1,
     -1, -1, -1, -1, 1], #N
    
    [-1, 1, 1, -1, -1,
     -1, -1, -1, -1, -1,
     1, 1, 1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1],  #E

     [1, -1, 1, 1, 1,
     -1, -1, -1, -1, -1,
     -1,- 1, -1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, -1, -1, -1],  #T
     
     [1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     1, -1, 1, -1, -1,
     -1, -1, 1, -1, 1,
     -1, 1, -1, -1, -1],  #W
     
     [-1, 1, -1, -1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1,-1,
     1, -1, -1, -1, 1,
     -1, 1, 1, -1, -1],  #O
     
     [-1, 1, -1, 1, -1,
     1, -1, -1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1,
     -1, -1, -1, -1, 1],  #R
     
     [1, -1, -1, -1, 1,
     1, -1, -1, 1, -1,
     -1, 1, -1, -1, -1,
     1, -1, -1, -1, -1,
     -1, -1, -1, -1, 1]  #K

     ]

I6 = [[1, -1, -1, -1, -1,
     1, -1, -1, 1, 1,
     1, -1, 1, -1, -1,
     -1, -1, -1, 1, 1,
     1, -1, -1, 1, 1], #N
     
     [1, 1, 1, 1, -1,
     -1, -1, -1, 1, -1,
     1, 1, 1, 1, -1,
     -1, -1, -1, 1, -1,
     -1, 1, -1, 1, -1],  #E
     
     [-1, -1, -1, -1, -1,
     1, -1, -1, 1, 1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1],  #U
     
     [1, 1, 1, 1, 1,
     -1, -1, -1, 1, 1,
     1, -1, -1, 1, -1,
     -1, 1, -1, -1, -1,
     1, -1, -1, -1, 1],  #R
     
     [-1, -1, 1, 1, -1, 
     -1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     -1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1], #A
     
     [1, -1, -1, -1, -1, 
     -1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     -1, -1, -1, -1, -1,
     -1, 1, 1, 1, -1], #L
    
    [1, -1, -1, -1, -1,
     -1, 1, -1, 1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, 1, -1,
     -1, -1, -1, -1, -1],  #space
    
    [1, -1, -1, 1, 1,
     -1, 1, -1, -1, 1,
     -1, -1, 1, -1, 1,
     1, -1, -1, 1, 1,
     -1, -1, -1, -1, 1], #N
    
    [-1, 1, 1, -1, -1,
     -1, 1, -1, -1, -1,
     1, 1, 1, -1, -1,
     1, 1, -1, -1, -1,
     1, 1, 1, 1, -1],  #E

     [1, 1, 1, 1, 1,
     -1, -1, -1, -1, -1,
     -1,- 1, -1, -1, -1,
     -1, 1, 1, -1, -1,
     -1, -1, -1, -1, -1],  #T
     
     [1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     1, 1, 1, -1, 1,
     -1, -1, 1, -1, 1,
     -1, 1, -1, -1, -1],  #W
     
     [-1, 1, -1, 1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1,-1,
     1, -1, -1, -1, 1,
     -1, 1, 1, -1, 1],  #O
     
     [-1, 1, -1, 1, 1,
     1, -1, -1, 1, -1,
     1, -1, 1, -1, -1,
     1, 1, 1, 1, -1,
     -1, -1, -1, -1, 1],  #R
     
     [1, -1, -1, -1, 1,
     1, -1, -1, 1, -1,
     -1, 1, -1, -1, -1,
     1, -1, -1, 1, -1,
     -1, -1, 1, -1, 1]  #K

     ]

Coord = [[0, 0], [0,1], [0,2], [0,3],
         [1, 0], [1,1], [1,2], [1,3],
         [2, 0], [2,1], [2,2], [2,3],
         [3, 0], [3,1], [3,2], [3,3]]
W = np.random.uniform(0,1, (16, 25))
#print(W[0])
nu0 = 0.1
t2 = 1000
sg = 2

def noise(I = []):
    print(I[0])
    I2=copy.deepcopy(I)
    for i in range(len(I)):
        for j in range(3):
            r = rd.randint(0,len(I[0])-1)
            I2[i][r]*=-1
    #print(I[0])
    return I2

def t1(): 
    return 1000/(math.log(sg))
def nu(n):
    return nu0*math.exp(-n/t2)

def sgn(n):
    return sg*math.exp(-n/t1())

def h(n,i,j) :
    return math.exp(-math.pow(-np.linalg.norm(np.array(Coord[i])-np.array(Coord[j])),2)/(2*math.pow(sgn(n),2)))

def test(I=[],W=[]):
        T = []
        for q in range(len(I)):
            n=15
            min = np.linalg.norm(I[q]-W[15])
            for i in range(15):
                j = np.linalg.norm(I[q]-W[i])
                if j < min:
                    min = j
                    n = i
            T.append(n)
        return T

f, ax=plt.subplots(1,len(I), figsize = (14,1))
for ii in range(len(I)):
    sns.heatmap(np.reshape(I[ii],(5,5)), cmap = sns.light_palette("purple"), ax=ax[ii],
                   cbar=False, yticklabels=False, xticklabels=False )

f2, ax=plt.subplots(1,len(I), figsize = (14,1))
for ii in range(len(I)):
    sns.heatmap(np.reshape(I2[ii],(5,5)), cmap = sns.light_palette("navy"), ax=ax[ii],
                   cbar=False, yticklabels=False, xticklabels=False )

f4, ax=plt.subplots(1,len(I), figsize = (14,1))
for ii in range(len(I)):
    sns.heatmap(np.reshape(I4[ii],(5,5)), cmap = sns.light_palette("seagreen"), ax=ax[ii],
                   cbar=False, yticklabels=False, xticklabels=False )

f6, ax=plt.subplots(1,len(I), figsize = (14,1))
for ii in range(len(I)):
    sns.heatmap(np.reshape(I6[ii],(5,5)), cmap = sns.light_palette("orange"), ax=ax[ii],
                   cbar=False, yticklabels=False, xticklabels=False )    

plt.show()

for k in range(1000):
    x = rd.randint(0, 11)
    min = np.linalg.norm(I[x]-W[15])
    n=15
    for i in range(15):
        j = np.linalg.norm(I[x]-W[i])
        if j < min:
            min = j;
            n = i
    for z in range(len(W)):
            W[z] = W[z] + nu(k)*h(k,n,z)*(I[x]-W[z])


for k in range(8000):
    x = rd.randint(0, 11)
    min = np.linalg.norm(I[x]-W[11])
    n=15
    for i in range(15):
        j = np.linalg.norm(I[x]-W[i])
        if j < min:
            min = j;
            n = i
    for z in range(len(W)):
            W[z] = W[z] + 0.1*h(k,n,z)*(I[x]-W[z])

Test=test(I,W)
Test2=test(I2,W)
Test4=test(I4,W)
Test6=test(I6,W)

er2 = []
er4 = []
er6 = []
    
for ii in range(len(I)):
    ht = []
    ht2 =[]
    ht4 =[]
    ht6 =[]
    x=I[ii]
    y=I2[ii]
    z=I4[ii]
    v=I6[ii]
    
    num = 0
    n = Test[ii]
    i=j=0
    for i in range(4):
        ht.append([])
        for j in range(4):
            ht[i].append(h(k,n,num))
            num+=1
    num = 0
    n = Test2[ii]
    n1 = Test4[ii]
    n2 = Test6[ii]
    i=j=0
    for i in range(4):
        ht2.append([])
        ht4.append([])
        ht6.append([])
        for j in range(4):
            ht2[i].append(h(k,n,num))
            ht4[i].append(h(k,n1,num))
            ht6[i].append(h(k,n2,num))
            num+=1
    
    if (ht==ht2): 
        er2.append(1) 
    else: er2.append(0)
    if (ht==ht4): 
        er4.append(1) 
    else: er4.append(0)
    if (ht==ht6): 
        er6.append(1) 
    else: er6.append(0)

    
words = ['N','E', 'U', 'R', 'A', 'L', ' ', 'N', 'E', 'T', 'W', 'O', 'R', 'K']    
       
df = pd.DataFrame(list(zip(er2, er4, er6)),
                  columns =['2 changes', '4 changes', '6 changes'],
                      index = words)

for col in df.columns:
    df[col] = df[col].map({1: True, 0: False})
    
print('\n',df)       

print(f'\n2 changes: {er2.count(1)} - correct | {er2.count(0)} - false | accuracy is {er2.count(1)/14:.3f}') 
print(f'4 changes: {er4.count(1)} - correct  | {er4.count(0)} - false | accuracy is {er4.count(1)/14:.3f}')
print(f'6 changes: {er6.count(1)} - correct  | {er6.count(0)} - false | accuracy is {er6.count(1)/14:.3f}')
