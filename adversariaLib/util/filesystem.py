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
import os
from util.log import log
from util.storage import get_data
from util.vars import BASE, ATTACK, DATA, CLASSIFIERS, TESTS

def make_folder(root_folder, name):
    try:
        d = os.path.join(root_folder, name)
        os.mkdir(d)
        log("Folder %s created successfully." % d)
    except OSError:
        pass
        #log("%s already exists." % d)
    return d
        
def make_exp_folders(exp_path):
    for D in (DATA, CLASSIFIERS, TESTS):
        d = make_folder(exp_path, D)
        make_folder(d, BASE)
        make_folder(d, ATTACK)
    
def check_fname(fname):
    return fname, os.path.exists(fname)

def get_fname_dataset(setup):
    return check_fname(os.path.join(setup.DSET_FOLDER, ".".join([setup.DSET_NAME, "txt"])))

def get_fname_exp(setup, **kwargs):
    exp_type = kwargs.get('exp_type')
    feed_type = kwargs.get('feed_type')
    
    indexes = kwargs.get('indexes')
    split_type = kwargs.get('split_type')
    split_no = kwargs.get('split_no')
    class_type = kwargs.get('class_type')
    attack_type = kwargs.get('attack_type')
    
    if indexes == 'kfold': # kfold indexes
        feed_type = DATA
        exp_type = BASE
        fname_args = [setup.DSET_NAME, "kfold"]
    elif indexes == 'split': # tr/ts split indexes
        assert(split_type in ('ts', 'tr'))
        assert(split_no is not None)
        feed_type = DATA
        exp_type = BASE
        fname_args = [setup.DSET_NAME, str(split_no), split_type, "txt"]
    else:
        assert(feed_type in (DATA, CLASSIFIERS, TESTS))
        assert(exp_type in (BASE, ATTACK))
        assert(split_no is not None)
        if feed_type == TESTS:
            split_type = 'ts'
        assert(split_type in ('ts', 'tr'))
        fname_args = [class_type, setup.DSET_NAME, str(split_no), split_type]
        if exp_type == ATTACK:
            assert(attack_type is not None)
        
        if feed_type == CLASSIFIERS:
            fname_args.append('cdat')
        else:
            fname_args.append('txt')
    
    fpath_args = [setup.EXP_PATH, feed_type, exp_type]
    fpath = os.path.join(*fpath_args)
    if exp_type == ATTACK:
        fpath = make_folder(fpath, attack_type)
    fpath = os.path.join(fpath, '.'.join(fname_args))
    
    return check_fname(fpath)

def split_fname_classifier(fname): # see previous fname_args definition !!!
    fname_split = fname.split('.')
    class_type = fname_split[0]
    split_no = fname_split[2]
    return class_type, int(split_no)

def get_classifiers(setup, exp_type=BASE, attack_type=None):
    classifiers = dict()
    folder_args = [setup.EXP_PATH, CLASSIFIERS, exp_type]
    if exp_type == ATTACK:
        assert(attack_type is not None)
        folder_args.append(attack_type)
    folder = os.path.join(*folder_args)
    for fname in os.listdir(folder):
        class_type, split_no = split_fname_classifier(fname)
        if class_type not in classifiers.keys():
            classifiers[class_type] = dict()
        classifiers[class_type][split_no] = get_data(os.path.join(folder, fname))
    return classifiers
