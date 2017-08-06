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

def main(verbose=True):
    this_folder = dirname(realpath(__file__))
    code_folder = abspath(join(this_folder, '../'))
    sys.path.insert(0, code_folder)
    curr_working_dir = os.getcwd()
    
    from prlib.dataset import load_dataset, save_lines
    from prlib.classifier import get_score
    from util import log, get_data
     
    parser = argparse.ArgumentParser(description='Test classifiers against a single test set')
    parser.add_argument('setup_path', type=str, help='Folder containing the setup file')
    parser.add_argument('input_file', type=str, help='Test set file path.')
    parser.add_argument('input_folder', type=str, help="Classifiers folder")
    parser.add_argument('output_folder', type=str, help='Scores folder')
    args = parser.parse_args()
    
    sys.path.insert(0,join(curr_working_dir,args.setup_path))
    setup = __import__('setup')
    
    X, y = load_dataset(join(curr_working_dir, args.input_file))
    for class_type in setup.CLASSIFIER_PARAMS.keys():
        classifier = get_data(join(curr_working_dir, args.input_folder, class_type))
        log("Testing %s against dataset %s..." % (class_type, args.input_file))
        lines = ["%.4f %d" % (get_score(classifier,x), y[i]) for i, x in enumerate(X)]
        save_lines(lines, join(curr_working_dir, args.output_folder, ".".join(["scores", class_type, "txt"])))
        
if __name__ == "__main__":
    main()
