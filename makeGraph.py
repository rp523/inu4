#coding: utf-8
import os, sys, shutil

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# ガンマ関数。階乗の一般化（Γ(n+1) = n!）
def Gamma(x):
    return math.gamma(x)
def PoissonVal(mean, k):
    return (mean ** k) * np.exp(- mean) / Gamma(k + 1)
def Factorial(n):
    return math.factorial(n)
def Combination(n, k):
    assert(n > 0)
    assert(n >= k)
    return Factorial(n) / (Factorial(n - k) * Factorial(k))
def BinomialVal(n, k, p):
    assert(p >= 0.0)
    assert(p <= 1.0)
    return Combination(n, k) * (p ** k) * (1 - p) ** (n - k)

def pic11_3():
    if 1:
        x = np.arange(1, 24)
        y = np.empty(x.size)
        for i in range(y.size):
            y[i] = BinomialVal(n = 24, k = x[i], p = 0.5)
        plt.bar(x/x.size, y, width  = 0.01)
        plt.xlim(0,1)
        plt.ylim(0,0.18)
        plt.title("N=24, theta=0.5")
        plt.xlabel("z/N")
        plt.ylabel("p(z/N)")
        plt.savefig("pic11_3_N24.png")
        
        plt.clf()
        x = np.arange(1, 1000)
        y = np.empty(x.size)
        for i in range(y.size):
            y[i] = BinomialVal(n = 1000, k = x[i], p = 0.5)
        plt.bar(x/x.size, y, width  = 0.001)
        plt.xlim(0,1)
        plt.ylim(0,0.18)
        plt.title("N=1000, theta=0.5")
        plt.xlabel("z/N")
        plt.ylabel("p(z/N)")
        plt.savefig("pic11_3_N1000.png")


def test():
    x = np.arange(-5,5,0.25)
    y = np.arange(-5,5,0.25)
    X, Y = np.meshgrid(x,y)
    Z = Y**2 + X**2
    assert(X.shape == Y.shape == Z.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X,Y,Z)
    plt.show()
    exit()


def pic11_5():
    max = 50
    x = np.arange(0, max, 0.01)
    y = np.empty(x.size)
    for i in range(x.size):
        y[i] = PoissonVal(mean = 24, k = x[i])
    plt.plot(x, y)
    plt.plot([24,24], [0, y[x==24]], color="red")
    plt.xlim(0, max)
    plt.ylim(0)
    plt.xlabel("N")
    plt.ylabel("p(N|λ)")
    plt.title("Poisson Distribution (λ=24)")
    plt.savefig("pic11_5_poisson.png")
    exit()
    
    ns = np.arange(0, max)
    zs = np.arange(0, max)
    
    prob = np.zeros((ns.size, zs.size))
    for n in ns[1:]:
        for z in range(0, n + 1):
            prob[n, z] = PoissonVal(mean = 24, k = n) * 1#BinomialVal(n = n, k = z, p = 0.5)
    
    N, Z = np.meshgrid(ns, zs)
    assert(N.shape == Z.shape == prob.shape)
    
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(N,Z,prob)
    #ax.set_xlim(0, 20)
    ax.set_xlabel("N")
    ax.set_ylabel("z")
    plt.show()

if "__main__" == __name__:
    pic11_5()
