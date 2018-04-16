import numpy as np

# L1  = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
L1 = [0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
L2 = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# L3  = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
L3 = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
L4 = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
L5 = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
L6 = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
L7 = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
L8 = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
L9 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
L10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

U1=[0,0,0,1]
U2=[1,0,0,0]
U3=[1,1,0,0]
U4=[1,0,1,0]

L = np.array([L1, L2, L3, L4, L5, L6, L7, L8, L9, L10])
U = np.array([U1,U2,U3,U4])

ITERATIONS = 100


### TODO 1: Compute stochastic matrix M
def getM(L):
    M = np.zeros([10, 10], dtype=float)
    # number of outgoing links
    c = np.zeros([10], dtype=int)
    for i in range(len(M)):
        for j in range(len(M[i])):
            if(L[j][i]==1):
               M[i][j]= 1/np.sum(L[j])
    return M

def trustRank(tr):
    for t in range(1, ITERATIONS):
        tr = [float(i) / sum(tr) for i in tr]
        for i in range(0, len(tr)):
            v = 0
            for j in range(0, len(tr)):
                v += tr[j] * M[i, j]
            trN[i] = q * d[i] + ((1 - q) * v)
        tr = list(trN)
    tr = [float(i) / sum(tr) for i in tr]



print("Matrix L (indices)")
print(L)

M = getM(L)

print("Matrix M (stochastic matrix)")
print(M)

### TODO 2: compute pagerank with damping factor q = 0.15
### Then, sort and print: (page index (first index = 1 add +1) : pagerank)
### (use regular array + sort method + lambda function)
print("PAGERANK")

q = 0.15

pr = np.zeros([10], dtype=float)
pr = [1/len(M) for element in pr]
prN=np.zeros([10], dtype=float)

for t in range (0,ITERATIONS):
    pr = [float(i) / sum(pr) for i in pr]
    for i in range(0,len(pr)):
        v=0
        for j in range(0,len(pr)):
            v+=pr[j]*M[i,j]
        prN[i]=q+((1-q)*v)
    pr=list(prN)


pr = [float(i) / sum(pr) for i in pr]
print(sorted(enumerate(pr,1), key=lambda x: x[1]))

### TODO 3: compute trustrank with damping factor q = 0.15
### Documents that are good = 1, 2 (indexes = 0, 1)
### Then, sort and print: (page index (first index = 1, add +1) : trustrank)
### (use regular array + sort method + lambda function)
print("TRUSTRANK (DOCUMENTS 1 AND 2 ARE GOOD)")

q = 0.15

d = np.zeros([10], dtype=float)

d[0] = 1
d[1] =1
d = [float(i) / sum(d) for i in d]

tr = [v for v in d]

trN=np.zeros([10], dtype=float)

for t in range (1,ITERATIONS):
    tr = [float(i) / sum(tr) for i in tr]
    for i in range(0,len(tr)):
        v=0
        for j in range(0,len(tr)):
            v+=tr[j]*M[i,j]
        trN[i]=q*d[i]+((1-q)*v)
    tr = list(trN)



tr = [float(i) / sum(tr) for i in tr]

print(sorted(enumerate(tr,1), key=lambda x: x[1]))
### TODO 4: Repeat TODO 3 but remove the connections 3->7 and 1->5 (indexes: 2->6, 0->4)
### before computing trustrank

L[2][6]=0
L[0][4]=0

M = getM(L)

tr = [v for v in d]

trN=np.zeros([10], dtype=float)

for t in range (0,ITERATIONS):
    tr = [float(i) / sum(tr) for i in tr]
    for i in range(0,len(tr)):
        v=0
        for j in range(0,len(tr)):
            v+=tr[j]*M[i,j]
        trN[i]=q*d[i]+((1-q)*v)
    tr = list(trN)



tr = [float(i) / sum(tr) for i in tr]
print(sorted(enumerate(tr,1), key=lambda x: x[1]))