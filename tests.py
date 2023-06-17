import numpy as np 
from scipy.spatial.distance import pdist, cdist
import timeit 
import diameter
from diameter import diameter_ext
from diameter import * 

X = np.random.normal(size=(100,2))
diameter_ext.exhaustive_search(X, X, True)
# diameter_ext.exhaustive_search_parallel(X, X)

P, Q = np.array(range(X.shape[0])), np.array(range(X.shape[0]))
diameter_ext.exhaustive_index(X, P, Q)

dn = unrank_C2(np.argmax(pdist(X)), X.shape[0])
diam = np.max(pdist(X))
P = np.array(list(range(X.shape[0])))
modulo_ball(X, P, dn, 0.10*diam)

c = X[dn,:].mean(axis=0)
diameter_ext.modulo_ball(X, P, c, 0.10*diam**2)


from itertools import combinations
n = X.shape[0]
pdist

p, q = max(combinations(range(n), 2), key=lambda c: pdist(X[c,:]).item())

unrank_C2(np.argmax(pdist(X)), n=n)

len(modulo_ball(X, P, dn, 0.50*diam)[0])
diameter_ext.modulo_ball(X, P, c, 0.50*diam**2)
np.linalg.norm(X[P,:] - c, axis=1) >= diam/2.0

X = np.random.uniform(size=(5000,2))
P, Q = np.array(range(X.shape[0])), np.array(range(X.shape[0]))
diameter_naive = lambda: max(pdist(X))
diameter_exhaust = lambda: diameter_ext.exhaustive_search(X, X, False)
diameter_exhaust_parallel = lambda: diameter_ext.exhaustive_search(X, X, True)
diameter_exhaust_index = lambda: diameter_ext.exhaustive_index(X, P, Q)


timeit.timeit(diameter_naive, number=30) # 19.567014621
timeit.timeit(diameter_exhaust, number=30) # 2.1725158650000083
timeit.timeit(diameter_exhaust_parallel, number=30) # 0.5013676359999977
timeit.timeit(diameter_exhaust_index, number=30) # 2.079360650999888


X = np.random.uniform(size=(5000,2))
diameter_hull = lambda: max(pdist(X[ConvexHull(X).vertices,:]))
diameter_naive = lambda: max(pdist(X))

import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

X = np.random.normal(size=(100,2))
np.random.seed(1233)




diameter.diameter(X)

import numpy as np
import timeit 
X = np.random.uniform(size=(5000,2))
diameter_hull = lambda: max(pdist(X[ConvexHull(X).vertices,:]))
diameter_naive = lambda: max(pdist(X))
diameter_pkg = lambda: diameter.diameter(X)[1]


timeit.timeit(diameter_hull, number=30)
timeit.timeit(diameter_naive, number=30)
timeit.timeit(diameter_pkg, number=30)


import timeit
timeit.timeit(diameter_hull, number=15)/15
timeit.timeit(diameter_naive, number=1)

for d in [2, 3, 4, 5, 6, 7, 8, 9, 15]:
  X = np.random.normal(size=(1500,d))
  time_hull = timeit.timeit(diameter_hull, number=30)
  time_naive = timeit.timeit(diameter_naive, number=30)
  print(f"| {d} | {time_naive:.2f} seconds | {time_hull:.2f} seconds |")

P = np.array(range(X.shape[0]), dtype=int)



from diameter import * 
plt.scatter(*X.T)

DN, Q_empty = iterated_search(X, P)

ss = 5.5
fig = plt.figure(figsize=(6,6), dpi=200)
ax = plt.gca()
ax.scatter(*X.T, s=ss)
ax.scatter(*X[DN['ids'][-1],:].T, c='red', s=ss)
best_dn = unrank_C2(np.argmax(pdist(X)**2), X.shape[0])
ax.scatter(*X[best_dn,:].T, c='green', s=ss)
c = np.mean(X[best_dn,:], axis=0)
ax.scatter(*c, c='blue', s=ss)
circle1 = plt.Circle(c, np.max(pdist(X))/2.0, color='green', fill=False)
ax.scatter(*X[DN['P'],:].T, c='purple', s=ss)
c2 = np.mean(X[DN['ids'][-1],:], axis=0)
circle2 = plt.Circle(c2, DN['dist'][-1]/2.0, color='red', fill=False)
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.set_aspect('equal')


## Very difficult data set 
X = np.random.uniform(size=(1000,42), low=-1, high=1)
X = np.array([x / np.linalg.norm(x) for x in X])
X += np.random.uniform(size=X.shape, low=-0.001, high=0.001)
# plt.scatter(*X[:,:2].T)
# plt.gca().set_aspect('equal')

from diameter import * 
diameter(X)[1] == np.max(pdist(X))

## Lemma 2 apparently guarentees: 
## 1. X / B[pq] = \emptyset => pq = maximal segment  
## 2. pq = maximal segment !=> X / B[pq] = emptyset (!)
best_dn = unrank_C2(np.argmax(pdist(X)), X.shape[0])
modulo_ball(X, P, best_dn, np.max(pdist(X)))


from tallem.dimred import cmds 
# Z = cmds(X, 2)
plt.scatter(*X.T)
plt.scatter(*X[P_I,:].T, c='red')
plt.plot(*X[DN[1],:].T, c='green')
plt.gca().set_aspect('equal')



P = np.array(range(x.shape[0]), dtype=int)
for i in range(x.shape[0]):
  (p, q) = double_normal(0, P)
  assert np.argmax(cdist(x[[p],:], x).flatten()) == q
  assert np.argmax(cdist(x[[q],:], x).flatten()) == p




import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
np.random.seed(9)
X = np.random.normal(size=(35, 2))

ind = [14, 19] #np.random.choice(range(X.shape[0]), 2)

fig = plt.figure(figsize=(8,8), dpi=250)
ax = fig.gca()
ax.set_xlim(np.min(X[:,0])-0.45, np.max(X[:,0])+0.45)
ax.set_ylim(np.min(X[:,1])-0.45, np.max(X[:,1])+0.45)
ax.scatter(*X.T, zorder=20, c='black')
for i, (x, y) in enumerate(X): ax.text(x, y-0.05, s=str(i), va='top', ha='center')
ax.set_aspect('equal')

## Plot 1
ax.plot(*X[ind,:].T, zorder=5, c='red')
c1 = plt.Circle(np.mean(X[ind,:], axis=0), pdist(X[ind,:]).item()/2, color='#ffcccb', alpha=0.80, zorder=0)
ax.add_patch(c1)

## Plot 2
# from diameter import rank_C2, unrank_C2
# p,q = unrank_C2(np.argmax(pdist(X)), n=X.shape[0])
# ax.plot(*X[(p,q),:].T, zorder=5, c='green')
# c1 = plt.Circle(np.mean(X[(p,q),:], axis=0), pdist(X[(p,q),:]).item()/2, color='#90ee90', alpha=0.80, zorder=0)
# ax.add_patch(c1)

## Plot 3
from diameter import double_normal
DN = np.array([double_normal(X, i, range(X.shape[0]))[0] for i in range(X.shape[0])])
DN = np.c_[np.minimum(DN[:,0], DN[:,1]), np.maximum(DN[:,0], DN[:,1])]
DN = np.unique(DN, axis=0)
for dn in DN:
  if pdist(X[dn,:]).item() ==  np.max(pdist(X)):
    ax.plot(*X[dn,:].T, c='green', linewidth=2.40)
  else: 
    ax.plot(*X[dn,:].T, c='orange', linewidth=2.20, linestyle='dashed')
CHV = ConvexHull(X).vertices
ax.plot(*X[np.append(CHV, CHV[0])].T, c='black', linewidth=1.50)




## Demonstration of how memory + vectorization affects things
D = lambda idx: pdist(X[idx,:]).item()
diam1 = max(pdist(X)) # O(n^2) memory 
diam2 = max([max(cdist(X[[i],:], X).flatten()) for i in range(X.shape[0])]) # O(n) memory 
diam3 = D(max(combinations(range(n), 2), key=D)) # O(1) memory 


X[i,0]
