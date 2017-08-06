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
import os, pickle, numpy
from scipy import sparse
from random import choice
from pyfann import libfann
from sklearn.base import BaseEstimator, ClassifierMixin
        
class MLP(BaseEstimator, ClassifierMixin):
    def __init__(self, num_neurons_hidden=3, learning_rate=0.7, 
                 activation_function_output=libfann.SIGMOID_SYMMETRIC_STEPWISE,
                 activation_steepness_hidden=0.5,
                 activation_steepness_output=0.5,
                 desired_error=0.001, max_iterations=100000, 
                 iterations_between_reports=1000, connection_rate=1):
        self.ann = libfann.neural_net()
        self.num_neurons_hidden = num_neurons_hidden
        self.learning_rate = learning_rate
        self.connection_rate = connection_rate
        self.desired_error = desired_error
        self.max_iterations = max_iterations
        self.activation_steepness_hidden = activation_steepness_hidden
        self.activation_steepness_output = activation_steepness_output
        self.iterations_between_reports = iterations_between_reports
        self.activation_function_output = activation_function_output
        
    def train_on_file(self, fname_tr, fname_net, desired_error=0.0001, max_iterations=100000, iterations_between_reports=1000):
        self.ann.train_on_file(fname_tr, max_iterations, iterations_between_reports, desired_error)
        self.ann.save(fname_net)
        
    def fit(self, X, y):
        
        #num_input = n_features, num_output = 1
        """Fit the MLP model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in
            regression)

        Returns
        -------
        self : object
            Returns self.

        Notes
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """
        if type(X) == sparse.csr.csr_matrix:
            X = X.toarray()
        X = [list(x) for x in X]
        y = [[t] for t in y]
        num_input = len(X[0])
        num_output = 1 # only one output hardcoded
        
        self.ann.create_sparse_array(self.connection_rate, (num_input, self.num_neurons_hidden, num_output))
        self.ann.set_learning_rate(self.learning_rate)
        self.ann.set_activation_function_output(self.activation_function_output)
        
        #BAT: ORA FUNZIONA! andava fatto DOPO le istruzioni sopra... forse non aveva ancora creato la rete...
        self.ann.set_activation_steepness_hidden(self.activation_steepness_hidden)
        self.ann.set_activation_steepness_output(self.activation_steepness_output)
        
        #print self.ann.get_activation_steepness(1,0)
        #print self.ann.get_activation_steepness(1,1)
        #print self.ann.get_activation_steepness(1,2)
      
        
        dset_tr = libfann.training_data()
        dset_tr.set_train_data(X,y)
        self.ann.train_on_data(dset_tr, self.max_iterations, 
                               self.iterations_between_reports, 
                               self.desired_error)
        
        state = self.__getstate__()
        self.v, self.w, self.b, self.outb = self.load_net_params(state)
        
        return self
    
    def predict(self, X, threshold=0):
        """Perform classification or regression samples in X.

        For a classification model, the predicted class for each
        sample in X is returned.  For a regression model, the function
        value of X calculated is returned.

        For an one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
        """
        # NOTE: the current wrapper for the ANN prefigures only ONE output!
        if type(X) == sparse.csr.csr_matrix:
            X = X.toarray()
        y_pred = []
        for x in X:
            if self.ann.run(x)[0] < threshold:
                y_pred.append(-1)
            else:
                y_pred.append(1)
        return numpy.array(y_pred)
    
    def decision_function(self, x):
        if type(x) == sparse.csr.csr_matrix:
            x = x.toarray()[0]
        return [self.ann.run(x)]
    
    def load_net_params(self, fann_conf):
        # TODO: are we able to get MLP parameters without using this stuff?
        """Hacking: from python bindings we are unable to get all ANN parameters"""
        """This method allow to read FANN configuration files and extract ANN parameters"""
        for line in fann_conf.split(os.linesep):
            key = 'layer_sizes='
            if key in line:
                num_input, num_hidden, num_output = [int(item) for item in line[len(key):-1].split()]
                
            key = 'connections (connected_to_neuron, weight)='
            if key in line:
                connections = []
                endidx = line.rfind(')')
                for item in line[len(key):endidx].split(") ("):
                    if item[0] == "(":
                        item = item[1:]
                    if item[-1] == " ":
                        item = item[:-2]
                    connections.append(eval(item))
        outb = connections[-1][1]
        v = dict()
        w = []
        b = []
        row = 0
        for i, conn in enumerate(connections):
            if conn[0] in range(num_input-1):
                if row not in v.keys():
                    v[row] = dict()
                v[row][conn[0]] = conn[1]
                if conn[0] == num_input-2:
                    row += 1
            elif conn[0] in range(num_input,num_input+num_hidden-1):
                w.append(conn[1])
            elif conn[0] == (num_input-1):
                b.append(conn[1])
        
        new_v = numpy.ndarray(shape=(num_hidden-1,num_input-1))
        for hid_no in range(num_hidden-1):
            for input_no in range(num_input-1):
                new_v[hid_no,input_no] = v[hid_no][input_no]
        
        w = numpy.array(w)
        b = numpy.array(b)
        return new_v, w, b, outb
        # m = number of neurons within the hidden layer
        # d = number of features
        # x is the input vector (d elements)
        # v is the input weight matrix (m x d elements)
        # b is the bias vector (m elements, one for each neuron)
    
    def __setstate__(self, state):
        """This method is called by Pickle"""
        # hacking!! just to avoid handling directly all ANN parameters
        self.ann = libfann.neural_net()
        if state:
            fname_tmp = ''.join([choice('abcdefghijklmnopqrstuvwxyz0123456789_-') for i in range(10)])
            f = open(fname_tmp, "w")
            f.write(state)
            f.close()
            self.ann.create_from_file(fname_tmp)
            os.unlink(fname_tmp)
            self.v, self.w, self.b, self.outb = self.load_net_params(state)
            self.activation_steepness_hidden = self.ann.get_activation_steepness(1,0)
            self.activation_steepness_output = self.ann.get_activation_steepness(2,0)
        else:
            raise pickle.PicklingError("It does not make sense to load an untrained MLP")
    
    def __getstate__(self):
        """This method is called by Pickle"""
        # hacking!! just to avoid handling directly all ANN parameters
        fname_tmp =''.join([choice('abcdefghijklmnopqrstuvwxyz0123456789_-') for i in range(10)])
        if self.ann.save(fname_tmp):
            f = open(fname_tmp)
            state = f.read()
            f.close()
            os.unlink(fname_tmp)
            return state
        else:
            raise pickle.PicklingError("It does not make sense to save an untrained MLP")
