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
from os.path import realpath, dirname, join, abspath, basename

def main(verbose=True):
    this_folder = dirname(realpath(__file__))
    code_folder = abspath(join(this_folder, '../'))
    sys.path.insert(0, code_folder)
    curr_working_dir = os.getcwd()
    
    from prlib.dataset import get_indexes, load_dataset
    from advlib.attacks import gradient_descent
    from util import get_data, dotdictify
    
    parser = argparse.ArgumentParser(description='Attacks a target classifier using a single test set and a Surrogate classifier available to the adversary.')
    parser.add_argument('setup_path', type=str, help='Folder containing the setup file')
    parser.add_argument('input_file', type=str, help='Test set file path.') 
    parser.add_argument('surrogate_classifier', type=str, help="Surrogate Classifier path")
    parser.add_argument('target_classifier', type=str, help="Target Classifier path")
    parser.add_argument('output_folder', type=str, help='Scores and attack iterations folder')
    parser.add_argument('--index_file', type=str, help='Indexes file path (if any).', required=False)
    args = parser.parse_args()
    
    sys.path.insert(0,join(curr_working_dir,args.setup_path))
    setup = __import__('setup')
    
    attack_params = setup.ATTACK_PARAMS.gradient_descent
    surrogate_classifier = get_data(join(curr_working_dir,args.surrogate_classifier))
    surrogate_classifier_name = basename(args.surrogate_classifier)
    target_classifier_name = basename(args.target_classifier)
    target_classifier = get_data(join(curr_working_dir,args.target_classifier))
    if args.index_file:
        indexes = get_indexes(join(curr_working_dir,args.index_file))
    else:
        indexes = None
    fname_surrogate_score = join(curr_working_dir,args.output_folder, ".".join(["surrogate_scores", surrogate_classifier_name, "txt"]))
    fname_data = join(curr_working_dir,args.output_folder, ".".join(["attack_patterns", surrogate_classifier_name, "txt"]))
    fname_targeted_score = join(curr_working_dir,args.output_folder, ".".join(["targeted_scores", surrogate_classifier_name, target_classifier_name, "txt"]))
    
    X, y = load_dataset(join(curr_working_dir, args.input_file))
    fname_target_score_exists = os.path.exists(fname_surrogate_score)
    fname_data_exists = os.path.exists(fname_data)
          
    # TODO: a more time-saving solution?
    if fname_target_score_exists:
        os.unlink(fname_surrogate_score)
    
    # TODO: a more time-saving solution?
    if attack_params.save_attack_patterns and fname_data_exists:
        os.unlink(fname_data)
    
    fnames = dotdictify({'surrogate_score': fname_surrogate_score, 'targeted_score': fname_targeted_score, 'data': fname_data})
    
    if attack_params.relabeling:
        y_t = target_classifier.predict(X)
    else:
        y_t = None
        
    gradient_descent(setup, X, y, y_t, None, surrogate_classifier, target_classifier, attack_params, fnames)
        
if __name__ == "__main__":
    main()
