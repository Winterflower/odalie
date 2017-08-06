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
from util import log, get_classifiers

def for_each_classifier(setup, function, **kwargs):
    log("Loading target classifiers built for experiment %s." % setup.EXP_PATH)
    classifiers = get_classifiers(setup)
    if not classifiers:
        log("Did you run experiment %s? No classifiers found." % setup.EXP_PATH)
        return
    
    for class_type in classifiers.keys():
        for split_no in classifiers[class_type]:
            classifier = classifiers[class_type][split_no]
            function(setup, classifier=classifier, class_type=class_type, split_no=split_no, **kwargs)
            
