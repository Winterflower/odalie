#-------------------------------------------------------------------------------
# adversariaLib - Advanced library for the evaluation of machine 
# learning algorithms and classifiers against adversarial attacks.
# 
# Copyright (C) 2013, Igino Corona, Battista Biggio, Davide Maiorca, 
# Dept. of Electrical and Electronic Engineering, University of Cagliari, Italy.
# 
# adversariaLib is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# adversariaLib is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------
import numpy as np
from math import exp
from prlib.classifier import get_score
#from util import timeit 

epsilon = 1e-20

def sigma(x,steepness):
    try:
        return 1.0/(1+exp(-steepness*x))
    except OverflowError:
        print "Overflow! x=", x
        if x < 0:
            return epsilon
        else:
            return 1-epsilon

# m = number of neurons within the hidden layer
# d = number of features
# x is the input vector (d elements)
# v is the input weight matrix (m x d elements)
# b is the bias vector (m elements, one for each neuron) 
def gk(x, v, b, k):
    res = np.dot(v[k],x)
    return res + b[k]

def deltak(x, steepness, v, b, k):
    return sigma(gk(x, v, b, k),steepness)
    
def g(x, steepness, v, w, b, outb):
    res = 0
    for k in range(len(w)):
        res += w[k]*deltak(x, steepness, v, b, k)
    return res + outb

def mlp_derive(x, steepness_output, steepness_hidden, v, w, b, outb, i):
    res = 0
    for k in range(w.shape[0]):
        dk = deltak(x,steepness_hidden, v, b, k)
        res += w[k]*dk*(1-dk)*v[k][i]
    return res


#@timeit
def gradient_mlp(classifier, pattern):
    v = classifier.v
    w = classifier.w
    b = classifier.b
    outb = classifier.outb
    s_out = classifier.activation_steepness_output
    s_hid = classifier.activation_steepness_hidden
    #sigmax = sigma(g(pattern,s_hid, v, w, b, outb),s_out)
    sigmax = 0.5*(get_score(classifier,pattern)+1)
    
    wd = np.ndarray(shape=(w.shape[0],))
    for k in range(w.shape[0]):
        dk = deltak(pattern,s_hid, v, b, k)
        wd[k] = w[k]*dk*(1-dk)
    
    #res = sigmax*(1-sigmax)*np.array([mlp_derive(pattern, s_out, s_hid,v,w,b,outb, i) for i in range(len(pattern))])
    res = sigmax*(1-sigmax)*np.array([np.dot(wd,v[:,i]) for i in range(len(pattern))])
    
    return res


