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
import os, numpy
from util import log
from collections import deque
from advlib.dataset import save_value
from sklearn.svm.classes import SVC
from prlib.classifier import MLP, get_score
from prlib.dataset import split_attacks
from gradient_mlp import gradient_mlp
from sklearn.metrics import pairwise
from scipy import sparse
from gradient_distances import *
from constraints import *
from multiprocessing import Pool, Manager, cpu_count


PREC = "%.4f"

def save_attack_iter(pattern, fname, sep=','):
    if type(pattern) == sparse.csr.csr_matrix:
        pattern = pattern.toarray()[0]
    save_value(" ".join([PREC % val for val in pattern]), fname, sep)

def save_boundary_info_threadsafe(lock, p_no, boundary_info, save_attack_patterns, save_surrogate_scores, fnames):
    boundary_info.sort(lambda x,y: cmp(x[0], y[0]))
    patterns = []
    surrogate_scores = []
    targeted_scores = []
    
    for bound_no, pattern, surrogate_score, targeted_score in boundary_info:
        if save_attack_patterns:
            patterns.append(" ".join([PREC % val for val in pattern]))
        # score of the targeted classifier
        targeted_scores.append(PREC % targeted_score)
        # score of the classifier known to (or built by) the adversary
        surrogate_scores.append(PREC % surrogate_score)
    
    with lock:
        if save_surrogate_scores:
            save_value("%d %s %s" % (p_no, " ".join(surrogate_scores), os.linesep), fnames.surrogate_score)
        save_value("%d %s %s" % (p_no, " ".join(targeted_scores), os.linesep), fnames.targeted_score)
    
        if save_attack_patterns:
            save_value("%d %s %s" % (p_no, ",".join(patterns), os.linesep), fnames.data)

def gradient_svm_linear(classifier, pattern):
    return classifier.coef_[0]

def gradient_svm_rbf(classifier,pattern):
    grad = []
    dual_coef = classifier.dual_coef_
    support = classifier.support_vectors_
    gamma = classifier.get_params()['gamma']
    kernel = pairwise.rbf_kernel(support, pattern, gamma)
    for element in range(0,len(support)):
        if (grad == []):
            grad = (dual_coef[0][element]*kernel[0][element]*2*gamma*(support[element]-pattern))
        else:
            grad = grad + (dual_coef[0][element]*kernel[element][0]*2*gamma*(support[element]-pattern))
    return -grad #il bel Maiorca si e' dimenticato un meno! :)

def gradient_svm_poly(classifier, pattern):
    grad = []
    dual_coef = classifier.dual_coef_
    support = classifier.support_vectors_
    degree = classifier.get_params()['degree']
    R = classifier.get_params()['coef0']
    gamma = classifier.get_params()['gamma']
    kernel = pairwise.polynomial_kernel(support,pattern, degree-1, gamma, R)
    #log("Kernel: %s; Support: %s; Pattern: %s" % (kernel,support, pattern))
    #kernel = pairwise.rbf_kernel(support, pattern, gamma)
    for element in range(0, len(support)):
        if(grad == []):
            grad = dual_coef[0][element]*degree*kernel[element][0]*support[element]*gamma
        else:
            grad = grad + dual_coef[0][element]*degree*kernel[element][0]*support[element]*gamma
    return -grad

def evaluate_stop_criteria(obj_function_at_pattern, epsilon):
    #print obj_function_at_pattern
    if len(obj_function_at_pattern) < obj_function_at_pattern.maxlen:
        return False
    if (obj_function_at_pattern[0]-obj_function_at_pattern[-1]) < epsilon:
        return True
    return False

def update_candidate_root_patterns(candidate_root_patterns, within_boundary_constraints, **constraint_params):
    for index, candidate_root_pattern in enumerate(candidate_root_patterns):
        if not within_boundary_constraints(pattern=candidate_root_pattern, **constraint_params):
            return candidate_root_patterns[:index]
    return candidate_root_patterns

def compute_gradient(surrogate_classifier, pattern, leg_patterns, lambda_value, gradient, gradient_mimicry, **mimicry_params):
    grad = gradient(surrogate_classifier, pattern)
    if lambda_value > 0:
        closer_leg_patterns, grad_mimicry, dist = gradient_mimicry(pattern, leg_patterns, **mimicry_params)
        grad_update = grad+lambda_value*grad_mimicry
        #print numpy.linalg.norm(grad), numpy.linalg.norm(lambda_value*grad_mimicry)
        
    else:
        closer_leg_patterns = leg_patterns
        grad_update = grad
        dist = 0
    if (numpy.linalg.norm(grad_update) != 0):
        grad_update = grad_update/numpy.linalg.norm(grad_update)
        
    return closer_leg_patterns, dist, grad_update

def thread_task(lock, p_no, root_pattern, surrogate_classifier, max_boundaries, stop_criteria_window, 
                within_boundary_constraints, apply_boundary_constraints, constraint_params, leg_patterns, 
                maxiter, gradient, lambda_value, gradient_mimicry, mimicry_params, step, stop_criteria_epsilon, 
                targeted_classifier, save_attack_patterns, fnames):
    # Gradient Descent Attack in place...
    boundary_numbers = range(1,max_boundaries+1)
    boundary_numbers.reverse()
    candidate_root_patterns = [root_pattern]
    attacker_score =  get_score(surrogate_classifier, root_pattern)
    targeted_score =  get_score(targeted_classifier, root_pattern)
    boundary_info = [(0, root_pattern, attacker_score, targeted_score)]
    closer_leg_patterns, dist, grad_update = compute_gradient(surrogate_classifier, root_pattern, leg_patterns, lambda_value, gradient, gradient_mimicry, **mimicry_params) 
    
    for bound_no in boundary_numbers:
        obj_function_at_pattern = deque(maxlen=stop_criteria_window)
        candidate_root_patterns = update_candidate_root_patterns(candidate_root_patterns, within_boundary_constraints, root_pattern=root_pattern, bound_no=bound_no, **constraint_params)
        if not candidate_root_patterns:
            log('Attack pattern %d. No candidate root patterns for Bound no. %d. Gradient Descent terminated.' % (p_no, bound_no))
            break
        
        pattern = candidate_root_patterns[-1] # last pattern which satisfies the new boundary constraint 
        num_candidate_root_patterns = len(candidate_root_patterns)
        closer_leg_patterns, dist, grad_update = compute_gradient(surrogate_classifier, pattern, closer_leg_patterns, lambda_value, gradient, gradient_mimicry, **mimicry_params) 
        
        # initial values...
        attacker_score =  get_score(surrogate_classifier, pattern)
        targeted_score =  get_score(targeted_classifier, pattern)
        obj_fun_value = attacker_score+lambda_value*dist
        obj_function_at_pattern.append(obj_fun_value)
        
        for iter_no in range(num_candidate_root_patterns, maxiter):
            new_pattern = apply_boundary_constraints(root_pattern, pattern, grad_update, step, bound_no, **constraint_params)
            
            # this grad_update will be used in the next iteration
            closer_leg_patterns, dist, new_grad_update = compute_gradient(surrogate_classifier, new_pattern, closer_leg_patterns, lambda_value, gradient, gradient_mimicry, **mimicry_params) 
            
            #print pattern, surrogate_classifier.decision_function(pattern)[0], targeted_classifiers_info[0][0].decision_function(pattern)[0]
            #raw_input('iter: %d, next?' % iter_no)
            new_attacker_score =  get_score(surrogate_classifier, new_pattern)
            obj_fun_value = new_attacker_score+lambda_value*dist
            
            #print (obj_fun_value, dist, obj_fun_value-obj_function_at_pattern[-1], iter_no)
            
            # here we evaluate whether we reached a local minimum/there is a bouncing effect due to the chosen step value
            if obj_fun_value == obj_function_at_pattern[-1]:
                # stop criteria is not evaluated if obj_fun does not change at all...
                log('Attack pattern: %d, Boundary: %d, Iteration: %d. Local Minimum Reached. Obj.: %f' % (p_no, bound_no, iter_no, obj_fun_value))
                # NB: in this case, we do not update attacker_score and targeted_score
                break
            elif obj_fun_value < obj_function_at_pattern[-1]:
                obj_function_at_pattern.append(obj_fun_value)
            else:
                # it means that the step is too large to decrement the obj function...
                # in the next interation (discrete spaces) we skip the feature that has been changed
                for idx, elm in enumerate(pattern==new_pattern):
                    if not elm:
                        log('Attack pattern: %d, Boundary: %d, Jumping off the minimum: feature %d will be skipped at iteration %d' % (p_no, bound_no, idx, iter_no))
                        grad_update[idx] = 0
                        break
                continue

            
            # if we are here: no local minimum has been reached yet, thus we can compute the new_targeted_score 
            new_targeted_score =  get_score(targeted_classifier, new_pattern)
            
            # special case: we store all iterations if max_boundaries == 1
            if max_boundaries == 1:
                boundary_info.append((iter_no, new_pattern, new_attacker_score, new_targeted_score))
            
            if evaluate_stop_criteria(obj_function_at_pattern, stop_criteria_epsilon):
                log('Attack pattern: %d, Boundary: %d, Iteration: %d. Stop criteria reached. Obj.: %f' % (p_no, bound_no, iter_no, obj_function_at_pattern[-1]))
                break
            
            if not (new_pattern==candidate_root_patterns[-1]).all():
                candidate_root_patterns.append(new_pattern)
            
            # we update all values for the next iteration
            pattern = new_pattern
            grad_update = new_grad_update
            attacker_score = new_attacker_score
            targeted_score = new_targeted_score
        
        if num_candidate_root_patterns < maxiter:
            if iter_no == maxiter-1:
                log('Attack pattern: %d, Boundary: %d, Maxiter reached. Obj. value: %f' % (p_no, bound_no, obj_function_at_pattern[-1]))
        else:
            log('Attack pattern: %d, Boundary: %d, Maxiter <= Number of previously computed attack points within the new boundary (no new attack iterations performed).' % (p_no, bound_no))
        
        # if max_boundaries > 1 we store only patterns that reached a local minimum/maximum iterations 
        if max_boundaries > 1:
            boundary_info.append((bound_no, pattern, attacker_score, targeted_score))
    if surrogate_classifier == targeted_classifier:
        save_surrogate_scores = False
    else:
        save_surrogate_scores = True
    save_boundary_info_threadsafe(lock, p_no, boundary_info, save_attack_patterns, save_surrogate_scores, fnames)


def gradient_descent(setup, X, y, y_t, surrogate_training_indexes, surrogate_classifier, targeted_classifier, attack_params, fnames):
    dataset = split_attacks(X, y, attack_params.attack_class)
    if y_t is not None:
        dataset_relabeled = split_attacks(X, y_t, attack_params.attack_class)
    else:
        dataset_relabeled = dataset
    attack_patterns = dataset.X_attack
    leg_patterns = dataset_relabeled.X_benign
    
    # SELECTION OF THE ATTACK TECHNIQUE ACCORDING TO THE surrogate CLASSIFIER (EMPLOYED BY THE ATTACKER)  
    surrogate_classifier_type = type(surrogate_classifier)
    if surrogate_classifier_type == SVC and surrogate_classifier.kernel == 'linear':
        gradient = gradient_svm_linear
    elif surrogate_classifier_type == SVC and surrogate_classifier.kernel == "rbf":
        gradient = gradient_svm_rbf
    elif surrogate_classifier_type == SVC and surrogate_classifier.kernel == "poly":
        gradient = gradient_svm_poly
    elif surrogate_classifier_type == MLP:
        gradient = gradient_mlp
    else:
        log("Gradient Descent Attack: unsupported attack classifier %s." % surrogate_classifier_type)
        return

    # SELECTION OF THE MIMICRY TECHNIQUE ACCORDING TO THE CHOSEN DISTANCE MEASURE
    if attack_params.mimicry_distance == 'euclidean':
        gradient_mimicry = gradient_euclidean_dist
    elif attack_params.mimicry_distance == 'kde_euclidean':
        gradient_mimicry = gradient_kde_euclidean_dist
    elif attack_params.mimicry_distance == 'kde_hamming':
        gradient_mimicry = gradient_kde_hamming_dist
    else:
        log("Gradient Descent Attack: unsupported mimicry distance %s." % attack_params.mimicry_distance)
        return
     
    if not attack_params.constraint_function:
        apply_boundary_constraints = apply_no_constraints
        within_boundary_constraints = within_no_constraints
        attack_params.max_boundaries = 1 # we overwrite the value... it does not make sense to have multiple boundaries
    elif attack_params.constraint_function == 'box_fixed':
        apply_boundary_constraints = apply_box_fixed
        within_boundary_constraints = within_box_fixed
    elif attack_params.constraint_function == 'box':
        apply_boundary_constraints = apply_hypercube
        within_boundary_constraints = within_hypercube
    elif attack_params.constraint_function == 'hamming':
        apply_boundary_constraints = apply_hamming
        within_boundary_constraints = within_hamming
    elif attack_params.constraint_function == 'only_increment':
        apply_boundary_constraints = apply_only_increment
        within_boundary_constraints = within_only_increment
        num_features = len(attack_patterns[0])
        try:
            assert(len(setup.NORM_WEIGHTS)==num_features)
            attack_params.constraint_params['weights'] = setup.NORM_WEIGHTS
            attack_params.constraint_params['feature_upper_bound'] = 1
            log('Norm weights loaded successfully from setup!') 
        except:
            attack_params.constraint_params['weights'] = numpy.ones(num_features)/attack_params.step
            
        attack_params.mimicry_params['weights'] = attack_params.constraint_params['weights']
        attack_params.constraint_params['inv_weights'] = numpy.array([1/item for item in attack_params.constraint_params['weights']])
    else:
        log("Gradient Descent Attack: unsupported constraint function %s." % attack_params.constraint_function)
        return
    
    if attack_params.threads < 0:
        threads = cpu_count() # all cpus are employed... :-D
    
    if attack_params.threads > 1:
        log("Gradient Descent will employ %d concurrent processes." % threads)
        manager = Manager()
        lock = manager.Lock()
        pool = Pool(threads)
        for p_no, root_pattern in enumerate(attack_patterns):
            pool.apply_async(func=thread_task, 
                        args=(lock, p_no, root_pattern, surrogate_classifier, attack_params.max_boundaries, 
                              attack_params.stop_criteria_window, within_boundary_constraints, apply_boundary_constraints, 
                              attack_params.constraint_params, leg_patterns, attack_params.maxiter, gradient, 
                              attack_params.lambda_value, gradient_mimicry, attack_params.mimicry_params, attack_params.step, 
                              attack_params.stop_criteria_epsilon, targeted_classifier, attack_params.save_attack_patterns, fnames))
        pool.close()
        pool.join()
    else: # just to be able to easily block execution by keyboard interrupt... :D
        from threading import Lock
        lock = Lock()
        for p_no, root_pattern in enumerate(attack_patterns):
            thread_task(lock, p_no, root_pattern, surrogate_classifier, attack_params.max_boundaries, 
                      attack_params.stop_criteria_window, within_boundary_constraints, apply_boundary_constraints, 
                      attack_params.constraint_params, leg_patterns, attack_params.maxiter, gradient, 
                      attack_params.lambda_value, gradient_mimicry, attack_params.mimicry_params, attack_params.step, 
                      attack_params.stop_criteria_epsilon, targeted_classifier, attack_params.save_attack_patterns, fnames)
