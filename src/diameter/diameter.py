from array import array 
import numpy as np 
from scipy.spatial.distance import cdist, pdist
from itertools import product
from numpy.typing import ArrayLike
import diameter_ext

def rank_C2(i: int, j: int, n: int):
  i, j = (j, i) if j < i else (i, j)
  return(int(n*i - i*(i+1)/2 + j - i - 1))

def unrank_C2(x: int, n: int):
  i = int(n - 2 - np.floor(np.sqrt(-8*x + 4*n*(n-1)-7)/2.0 - 0.5))
  j = int(x + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)
  return(i,j) 

def double_normal(X, p, epsilon = 10*np.finfo(float).eps):
  Q = array('I')
  delta_c, delta_p = -1.0, 0.0
  while(np.abs(delta_c - delta_p) > epsilon):
    delta_c, delta_p = delta_p, delta_c
    Q.append(p)
    dp = cdist(X[[p],:], X).flatten()
    q = np.argmax(dp)
    if dp[q] > delta_c:
      delta_c, dn = dp[q], (p, q)
      p = q
  return(dn, delta_c, Q)

def dist_to_center(X, P, c):
  """ Returns the distance of X[P,:] to center point 'c' """
  return(cdist(np.matrix(c), X[np.array(P).flatten(),:]).flatten())

def in_ball(X, P, c, r):
  in_ball_ind = np.where(dist_to_center(X, P, c) <= r)
  if len(in_ball_ind) > 0:
    return(P[in_ball_ind])
  else: 
    return(np.array([], dtype=int))

def modulo_ball(X, P, dn, diam):
  ''' 
  Returns the indices of 'P' modulo a ball centered between (p,q)=dn with the given diameter. 
  
  That is, given a set of points 'X', indices 'P', and a tuple dn=(p,q) and a diameter 'diam', 
  returns the set difference: 
   
  Q = P \ B(c, diam / 2)

  where c = (p+q)/2 == center point between (p, q). Thus, returns the remaining indices of 'P', if any, after 
  removing points intersecting the above ball. 
  '''
  center = np.mean(X[dn,:], axis=0)
  Q = np.setdiff1d(P, in_ball(X, P, center, (diam/2) + np.finfo(float).eps))
  return(Q, center)

def intersect_ball(X, P, dn, diam):
  ''' 
  Returns the indices of 'P' intersecting a ball centered between (p,q)=dn with the given diameter. 

  That is, given a set of points 'X', indices 'P', and a tuple dn=(p,q) and a float value 'diam', 
  returns the set intersection: 
   
  Q = P \cap B(c, diam / 2)

  where c = (p+q)/2 == center point between (p, q). 
  '''
  center = np.mean(X[dn,:], axis=0)
  Q = np.intersect1d(P, in_ball(X, P, center, (diam/2) + np.finfo(float).eps))
  return(Q, center)

def iterated_search(X, P):
  """ 
  Returns the largest double-normal constructed from iteratively removing points from X via balls computed on X[P,:]. 
  
  Specifically, one starts with a set of candidate points P and a random point m \in P. 
  A double-normal (p,q) is computed w.r.t m, and then all points intersecting a ball B whose anti-podal 
  points are (p, q) are removed from P. P is then refined, and a new candidate point is chosen. 
  
  Although any point from P\B can be chosen, the heuristic used here is to choose the point 
  furthest from the center of the ball.    
  """
  m = np.random.choice(P) # TODO: heuristic: maybe just pick a random dimension and choose the max coordinate?
  Q, delta_c = P, 0.0 
  DN = { 'ids' : [], 'dist' : [], 'P' : array('I') }
  while(len(Q) != 0):
    ## Calculate a new double-normal starting a search at m
    dn, d_pq, PC = double_normal(X, m)
    DN.update(ids=DN['ids'] + [dn], dist=DN['dist'] + [d_pq])
    DN['P'].extend(PC)

    ## Check if this double-normal is the largest yet and see if we're done 
    if d_pq > delta_c:
      delta_c = d_pq # update largest distance
      # Q, center = modulo_ball(X, P, dn, d_pq)  
      center = X[dn,:].mean(axis=0)    
      Q = diameter_ext.modulo_ball(X, P, center, d_pq)
      #Q = P[np.linalg.norm(X[P,:] - center, axis=1) >= d_pq]
      if len(Q) > 0:
        m = Q[np.argmax(cdist(np.matrix(center), X[Q,:]))]
    else:
      break
  ## Returns the full set of computed double-normals + whether the last one is maximal
  return((DN, len(Q) == 0)) ## Returns (pq_{I-1}, pq_I), Boolean indicating whether d_x(pq_I) == diameter


def diameter(X: ArrayLike, segment: bool = False):
  P = np.array(range(X.shape[0]), dtype=int)
  DN, q_empty = iterated_search(X, P)
  D, I = DN['dist'], DN['ids']
  if q_empty and (len(D) == 1 or D[-1] > D[-2]):
    ## Case 1: Q is completely empty => we are done, the largest DN is a maximal segment 
    opt_dn, diam = I[-1], D[-1]
  else: 
    ## Case 2: (p_I, q_I) is not a maximal segment, but (p_{I-1}, q_{I-1}) might be
    PI = np.setdiff1d(P, DN['P'])
    #Q, _ = modulo_ball(X, PI, I[-2], D[-2])
    Q = diameter_ext.modulo_ball(X, PI, X[I[-2],:].mean(axis=0), D[-2])
    if len(Q) == 0:
      ## Case 2a: B[pq] must engulf the entire point set => pq is a maximal segment
      opt_dn, diam = I[-2], D[-2]
    else:
      ## TODO: Case 2b: Attempt to reduce the size of Q 
      opt_dn, diam = I[-2], D[-2]
      


      ## Exhaustive search over Q x P 
      if not(segment):
        diam = diameter_ext.exhaustive_search(X[Q,:], X[PI,:], True)
      else: 
        p, q, diam = diameter_ext.exhaustive_index(X, PI, Q)
        opt_dn, diam = (p,q), np.sqrt(diam)
      # for q in Q:
      #   Dq = cdist(X[[q],:], X[PI,:]).flatten()
      #   if (np.any(Dq > diam)):
      #     i = np.argmax(Dq)
      #     opt_dn, diam = (PI[i], q), Dq[i]
  
  ## By default, just return the diameter value. Optionally also return the maximal segment if requested
  return(diam if not(segment) else (opt_dn, diam))
  

def profile_diameter(X):
  from line_profiler import LineProfiler
  profiler = LineProfiler()
  profiler.add_function(diameter)
  profiler.enable_by_count()
  _ = diameter(X)
  profiler.print_stats(output_unit=1e-3)

## This doesn't work / it's not clear what to do with S 
# S = I[:-2] + I[-1:]
# while(len(S) > 0):
#   p, q = S[0]
#   Q_tmp, _ = intersect_ball(X, Q, (p,q), diam)
#   PI, _ = modulo_ball(X, np.setdiff1d(PI, Q), (p,q), diam)
#   PQ_D = np.array([dn_dist((pi,qi)) for (pi, qi) in product(PI, Q_tmp)]) # may be empty!
#   PQ = [(pi, qi)for (pi, qi) in product(PI, Q_tmp)]
#   if len(PQ_D) > 0 and np.max(PQ_D) > diam:
#     opt_dn, diam = PQ[np.argmax(PQ_D)], np.max(PQ_D)
#     S.append(opt_dn)
#   Q, _ = modulo_ball(X, Q, (p,q), dn_dist((p,q)))  
#   if len(Q) == 0:
#     return(opt_dn, diam)
#     # break
#   S = S[1:]