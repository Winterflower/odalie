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
from util import log, get_fname_exp, save_data, CLASSIFIERS, BASE
from prlib.dataset import get_training_data
from learn import learn

def learn_classifiers(setup, kfolds, verbose):
    for classifier_type, params in setup.CLASSIFIER_PARAMS.items():
        for split_no in range(setup.NSPLITS): # TODO: MULTI-THREADING?
            learn_classifier(setup, kfolds, classifier_type, params, split_no, verbose)

def learn_classifier(setup, kfolds, classifier_type, params, split_no, verbose):
    for key, val in setup.GRID_PARAMS.items():
        params.grid_search[key] = val
    params.grid_search.cv = kfolds
    
    X, y = get_training_data(setup, split_no)
    
    fname_classifier, exists = get_fname_exp(setup, feed_type=CLASSIFIERS, exp_type=BASE,
                                             class_type=classifier_type, split_no=split_no, split_type='tr')
    if exists:
        log("%s - dataset split: %s - It seems that a classifier has been already built." % (classifier_type, split_no))
        return
    
    log("%s - dataset split: %s - Finding the best parameters through cross-validation." % (classifier_type, split_no))
    
    clf = learn(X, y, classifier_type, params)
    
    if verbose:
        for par, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std()/2, par)
    log("Best score: %s, Best Parameters: %s" % (clf.best_score_, clf.best_params_))
 
    save_data(clf.best_estimator_, fname_classifier)
    log('%s - dataset split: %s - The "best" classifier has been saved on %s' % (classifier_type, split_no, fname_classifier))
