from prlib.test import for_each_classifier
from advlib.common import for_each_surrogate_classifier
from util import import_object

def launch_attacks(setup):
    for attack in setup.ATTACK_PARAMS.keys():
        launch_attack(setup, attack)

def launch_attack(setup, attack):
    attack_function = import_object('advlib.attacks.%s' % attack)
    for_each_classifier(setup, for_each_surrogate_classifier, attack_function=attack_function)