import numpy as np 
from diameter import double_normal, rank_C2, unrank_C2
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import ConvexHull
from diameter import diameter_ext
from diameter.diameter import modulo_ball

def test_double_normal():
  X = np.random.uniform(size=(100,2))
  V = ConvexHull(X).vertices
  for i in range(X.shape[0]):
    (p,q), d_pq, PI = double_normal(X, i)
    
    ## Assert they are in the convex hull
    assert p in V
    assert q in V

    ## Assert they are each others furthest points 
    assert q == np.argmax(cdist(X[[p],:], X))
    assert p == np.argmax(cdist(X[[q],:], X))


def test_modulo_ball():
  import matplotlib.pyplot as plt
  X = np.random.uniform(size=(50,2))
  c = X.mean(axis=0)
  r = 0.35 

  fig = plt.figure(dpi=150, figsize=(2,2))
  ax = fig.gca()
  ax.scatter(*X.T, s=1.5)
  ax.scatter(*c, c='green')
  ax.add_patch(plt.Circle(c, r, color='r', fill=False))
  ax.set_aspect('equal')
  P = np.array(list(range(X.shape[0])))
  I = np.linalg.norm(X - c, axis=1) <= r # this works
  ax.scatter(*X[I,:].T, s=1.5, c='red')

  O = np.flatnonzero(np.linalg.norm(X - c, axis=1) > r)
  O2 = diameter_ext.modulo_ball(X, P, c, (2*r))
  assert np.all(O == O2)

  Q = np.random.choice(P, 20)
  Q2 = diameter_ext.modulo_ball(X, Q, c, (2*r))
  assert np.all([q in O for q in Q2])
  

def test_diameter():
  from diameter import diameter
  X = np.random.uniform(size=(10000,32))  
  assert diameter(X) == np.max(pdist(X))

  from diameter import profile_diameter
  profile_diameter(X)
  

  timeit.timeit(lambda: diameter_ext.exhaustive_search(X, X, True), number=5)
  timeit.timeit(lambda: np.max(pdist(X)), number=5)
