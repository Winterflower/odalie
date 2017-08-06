from util import dotdictify, log, get_fname_exp, DATA, TESTS, ATTACK

def get_surrogate_classifier_name(attack_params, surr_class_type, targeted_class_type, rep_no, n):
    return ''.join((surr_class_type, attack_params.fname_metachar_samples_repetitions, 
                    str(n), attack_params.fname_metachar_samples_repetitions, str(rep_no), 
                    attack_params.fname_metachar_attack_vs_target, targeted_class_type))

def get_fnames_attack(setup, attack_function_name, surrogate_classifier_name, split_no):
    fname_surrogate_score, fname_surrogate_exists = get_fname_exp(setup, feed_type=TESTS, exp_type=ATTACK,
                            attack_type=attack_function_name, class_type='surrogate_'+surrogate_classifier_name, split_no=split_no)
    fname_targeted_score, fname_targeted_exists = get_fname_exp(setup, feed_type=TESTS, exp_type=ATTACK,
                            attack_type=attack_function_name, class_type='targeted_'+surrogate_classifier_name, split_no=split_no)
    fname_data, fname_data_exists = get_fname_exp(setup, feed_type=DATA, exp_type=ATTACK,
                            attack_type=attack_function_name, class_type=surrogate_classifier_name, split_no=split_no, split_type='ts')
    if fname_surrogate_exists or fname_targeted_exists or fname_data_exists:
        log("It seems that such attack has been already performed.")
        return
    return dotdictify({'surrogate_score': fname_surrogate_score, 'targeted_score': fname_targeted_score, 'data': fname_data})