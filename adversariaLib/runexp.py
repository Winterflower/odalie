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
import argparse, os, sys
from prlib.learn import learn_classifiers
from prlib.dataset import build_train_test_splits, get_kfold_splits
from prlib.test import for_each_classifier, test_classifier
from util import make_exp_folders, log
from advlib.common import launch_attacks

def main(verbose=True):
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('exp_path', type=str, help='Experiment Folder (must contain a setup.py file)')
    args = parser.parse_args()
    
    currdir = os.getcwd()
    
    exp_path = args.exp_path
    if not args.exp_path.startswith('/'):
        exp_path = os.path.join(currdir,exp_path)
        
    sys.path.insert(0,exp_path)
    setup = __import__("setup")
    
    log("Starting experiment %s." % setup.EXP_PATH)
    make_exp_folders(setup.EXP_PATH)
    
    #===========================================================================
    # Build all Target classifiers
    #===========================================================================
    data = build_train_test_splits(setup)
    kfolds = get_kfold_splits(setup, data.tr_size)
    learn_classifiers(setup, kfolds, verbose)
    
    #===========================================================================
    # Standard Accuracy Test against all Target Classifiers
    #===========================================================================
    for_each_classifier(setup, test_classifier, verbose=verbose)
    
    #===========================================================================
    # Attack
    #===========================================================================
    launch_attacks(setup)
    
if __name__ == "__main__":
    main()
