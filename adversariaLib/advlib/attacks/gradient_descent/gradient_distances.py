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
from numpy import exp
from util.distances import euclidean_dist, euclidean_dist_power2, hamming_dist

#returns: (1) the set of closest legitimate samples;
#(2) the gradient and (3) the distance value wrt to the closest sample.
def gradient_euclidean_dist(attack, leg_patterns, max_leg_patterns=10, weights=1):
    dist = [(patt, euclidean_dist(attack, patt, weights)) for patt in leg_patterns]
    dist.sort(lambda x,y: cmp(x[1],y[1]))
    if max_leg_patterns < len(dist):
        dist = dist[:max_leg_patterns]
    leg_patterns = [a[0] for a in dist]
    return leg_patterns, 2*(attack-leg_patterns[0]), dist[0][1]

def gradient_kde_euclidean_dist(attack, leg_patterns, max_leg_patterns=10, gamma=0.1, weights=1):
    kernel = [(patt, exp(-gamma*euclidean_dist_power2(attack, patt, weights))) for patt in leg_patterns]
    
    if max_leg_patterns < len(kernel):
        kernel.sort(lambda x,y: -cmp(x[1],y[1]))
        kernel = kernel[:max_leg_patterns]
        leg_patterns = [a[0] for a in kernel]
    
    kde = 0.0
    gradient_kde = 0.0
    for i,k in enumerate(kernel):
        kde += k[1]
        gradient_kde += -gamma*2*(attack-leg_patterns[i])*k[1]
    
    kde = kde / len(kernel)
    gradient_kde = gamma*gradient_kde / len(kernel) 
    
    #print -kde
    #I'm using minus since kde estimates similarity, not distance
    return leg_patterns, -gradient_kde, -kde 


def gradient_kde_hamming_dist(attack, leg_patterns, max_leg_patterns=10, gamma=0.1, weights=1):
    kernel = [(patt, exp(-gamma*hamming_dist(attack, patt, weights))) for patt in leg_patterns]
    
    if max_leg_patterns < len(kernel):
        kernel.sort(lambda x,y: -cmp(x[1],y[1]))
        kernel = kernel[:max_leg_patterns]
        leg_patterns = [a[0] for a in kernel]
    
    kde = 0.0
    gradient_kde = 0.0
    for i,k in enumerate(kernel):
        kde += k[1]
        #gradient_kde += -gamma*(attack-leg_patterns[i])*k[1] #TODO: Bat-- ho tolto la normalizzazione per non mettere lambda=milllleeeeeeeeee
        gradient_kde += -(attack-leg_patterns[i])*k[1]
    
    kde = kde / len(kernel)
    #gradient_kde = gamma*gradient_kde / len(kernel) #TODO: Bat-- ho tolto la normalizzazione per non mettere lambda=milllleeeeeeeeee
     
    #print linalg.norm(gradient_kde)
    
    #I'm using minus since kde estimates similarity, not distance
    return leg_patterns, -gradient_kde, -kde 
