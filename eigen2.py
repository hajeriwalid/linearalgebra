import numpy as np
import torch

"""
A is a square matrix
Adapted from https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/2-linear-algebra-ii.ipynb 

This code performs an eigendecomposition A in the form 
𝑨 = 𝑷𝑫𝑷inv

And an SVD of A in the form : 
A=USVt
"""

print("\n ***************** \n")
A = np.array([[3, 1], [-6, -4]]) 

print("A \n", A)

#P is the square n × n matrix whose ith column is the eigenvector qi of A
#lambdas are the eigenvalues 
# see https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
lambdas, P = np.linalg.eig(A)
print("\n ***************** \n")
print(" EigenDecomposition \n")
print("\nP \n", P)
print("\nLambdas \n", lambdas)

#Inverse of matrix P
Pinv = np.linalg.inv(P)
print("\nP-1 \n", Pinv)

#D is the diagonal matrix whose diagonal entries are the eigenvalues of A (lambdas)
D = np.diag(lambdas)
print("\nD \n", D)

print ("\n 𝑷𝑫𝑷inv \n", np.dot(P, np.dot(D, Pinv)))



#A=USVt
U,S,Vt = np.linalg.svd(A)

print("\n ***************** \n")
print("SV Decomposition \n")
print("\n U \n", U)
print("\n S \n", S)
print("\n Sdiag \n", np.diag(S))
print("\n Vt \n", Vt)

#used for mxn matrices
#S = np.concatenate((np.diag(S), [[0, 0]]), axis=0)

print ("\n USVt \n", np.dot(U, np.dot(np.diag(S), Vt)) )
