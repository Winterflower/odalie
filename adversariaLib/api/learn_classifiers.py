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
import os, argparse, sys
from os.path import realpath, dirname, join, abspath
from sklearn import cross_validation

def main():
    this_folder = dirname(realpath(__file__))
    code_folder = abspath(join(this_folder, '../'))
    sys.path.insert(0, code_folder)
    curr_working_dir = os.getcwd()
    
    from prlib.dataset import load_dataset
    from prlib.learn import learn
    from util import log, save_data
     
    parser = argparse.ArgumentParser(description='Learns one or more classifiers given a training set and according to a setup file')
    parser.add_argument('setup_path', type=str, help='Folder containing the setup file')
    parser.add_argument('input_file', type=str, help='Training set file path.')
    parser.add_argument('output_folder', type=str, help='Output folder (each classifier will be stored here)')
    parser.add_argument('--nfolds', metavar='N', type=int, help='Number of folds used for cross-validation.', default=3)
    args = parser.parse_args()
    sys.path.insert(0, join(curr_working_dir, args.setup_path))
    setup = __import__('setup')
    
    log("Learning from %s." % args.input_file)
    X, y = load_dataset(join(curr_working_dir, args.input_file))
    kfolds = cross_validation.KFold(n=len(X), k=args.nfolds)
    
    for clf_name, params in setup.CLASSIFIER_PARAMS.items():
        for key, val in setup.GRID_PARAMS.items():
            params.grid_search[key] = val
        params.grid_search.cv = kfolds
        clf = learn(X, y, clf_name, params)
        for par, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std()/2, par)
        log("Best score: %s, Best Parameters: %s" % (clf.best_score_, clf.best_params_))
        clf_fpath = join(curr_working_dir, args.output_folder, clf_name)
        save_data(clf.best_estimator_, clf_fpath)
        log('The "best" classifier has been saved on %s' % clf_fpath)

if __name__ == "__main__":
    main()
