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
from util import log, get_fname_exp, get_data, save_data, CLASSIFIERS, ATTACK
from prlib.dataset import get_testing_data
from prlib.learn import learn
from advlib.common import get_fnames_attack, get_surrogate_classifier_name
from random import sample
from sklearn import cross_validation
    
def perform_attack(setup, X, y, y_t, surrogate_training_indexes, surrogate_classifier, surrogate_classifier_name, 
                   targeted_classifier, targeted_classifier_name, split_no, attack_function, attack_params):
    if surrogate_classifier == targeted_classifier:
        log("%s attack with Perfect Knowledge against %s (split number %d)..." % (attack_function.__name__, targeted_classifier_name, split_no))
    else:
        log("%s attack with Partial Knowledge against %s (surrogate classifier: %s, split number %d)..." % (attack_function.__name__, targeted_classifier_name, surrogate_classifier_name, split_no))
    fnames = get_fnames_attack(setup, attack_function.__name__, surrogate_classifier_name, split_no)
    if fnames:
        attack_function(setup, X, y, y_t, surrogate_training_indexes, surrogate_classifier, targeted_classifier, attack_params, fnames)
    

def for_each_surrogate_classifier(setup, classifier, class_type, split_no, attack_function, verbose=False):
    attack_type = attack_function.__name__
    attack_params = setup.ATTACK_PARAMS[attack_type]
    X, y = get_testing_data(setup, split_no)
    if attack_params.relabeling:
        y_t = classifier.predict(X)
    else:
        y_t = None
    
    # Perfect Knowledge: surrogate_classifier=classifier
    perform_attack(setup, X, y, y_t, None, classifier, class_type, classifier, class_type, split_no, attack_function, attack_params)
    
    # Partial Knowledge
    # We learn each surrogate classifier and use it to attack the targeted classifier
    size = len(X)
    for surr_class_type, surr_class_params in setup.SURROGATE_CLASSIFIER_PARAMS.items():    
        surr_class_params.grid_search.update(setup.GRID_PARAMS)
        for n in attack_params.training.dataset_knowledge.samples_range:
            surr_class_params.grid_search.cv = cross_validation.KFold(n=n, n_folds=setup.NFOLDS)
            for rep_no in range(attack_params.training.dataset_knowledge.repetitions):
                indexes = sample(xrange(size), n)
                surr_class_name = get_surrogate_classifier_name(attack_params, surr_class_type, class_type, rep_no, n)
                fname_surr_classifier, exists = get_fname_exp(setup, feed_type=CLASSIFIERS, exp_type=ATTACK, attack_type=attack_type, 
                                                              class_type=surr_class_name, split_no=split_no, split_type='ts')
                if exists:
                    log("%s - dataset split: %s - It seems that a surrogate classifier has been already built." % (surr_class_name, split_no))
                    surr_classifier = get_data(fname_surr_classifier)
                else:
                    log("%s - dataset split: %s - Finding the best parameters through cross-validation." % (surr_class_name, split_no))
                    clf = learn(X[indexes], y_t[indexes], surr_class_type, surr_class_params)
                    if verbose:
                        for par, mean_score, scores in clf.grid_scores_:
                            print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std()/2, par)
                    log("Best score: %s, Best Parameters: %s" % (clf.best_score_, clf.best_params_))
                    surr_classifier = clf.best_estimator_
                    save_data(surr_classifier, fname_surr_classifier)
                    log('%s - dataset split: %s - Repetition: %d, Training samples: %d. The "best" classifier has been saved on %s' % (surr_class_name, split_no, rep_no, n, fname_surr_classifier))
                perform_attack(setup, X, y, y_t, indexes, surr_classifier, surr_class_name, classifier, class_type, split_no, attack_function, attack_params)