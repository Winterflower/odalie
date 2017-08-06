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
from sklearn.metrics import classification_report
from util import log, get_fname_exp, TESTS, BASE
from prlib.dataset import get_testing_data, save_lines
from prlib.classifier import get_score

def test_classifier(setup, split_no, classifier, class_type, verbose):
    log('Classifier %s (split_no: %s) under test...' % (class_type, split_no))
    
    X, y = get_testing_data(setup, split_no)
    
    y_pred = classifier.predict(X)
    lines = ["%.4f %d" % (get_score(classifier, x), y[i]) for i, x in enumerate(X)]
        
    fname_test, exists = get_fname_exp(setup, feed_type=TESTS, exp_type=BASE,
                                       class_type=class_type, split_no=split_no)
    if exists:
        log('%s - dataset split: %s - Testing results already computed.' % (class_type, split_no))
        return
    
    save_lines(lines, fname_test)
    log('%s - dataset split: %s - Testing results have been saved on %s' % (class_type, split_no, fname_test))

    if verbose:
        print "### Classification Report (class-conditional and average):"
        print classification_report(y, y_pred)
    
    
