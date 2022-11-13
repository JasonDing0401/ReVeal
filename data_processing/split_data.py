import argparse, os
import json
import numpy as np


def split_and_save(name, output, buggy, non_buggy, percent, keep_original=False):
    np.random.shuffle(buggy)
    np.random.shuffle(non_buggy)
    num_bug = len(buggy)
    if keep_original:
        num_non_bug = len(non_buggy)
    else:
        num_non_bug = int(num_bug * 100 / percent)
    non_buggy_selected = non_buggy[:num_non_bug]

    train_examples = []
    valid_examples = []
    test_examples = []

    num_train_bugs = int(num_bug * 0.80)
    num_valid_bug = int(num_bug * 0.10)
    train_examples.extend(buggy[:num_train_bugs])
    valid_examples.extend(buggy[num_train_bugs:(num_train_bugs + num_valid_bug)])
    test_examples.extend(buggy[(num_train_bugs + num_valid_bug):])

    num_non_bug = len(non_buggy_selected)
    num_train_nobugs = int(num_non_bug * 0.80)
    num_valid_nobug = int(num_non_bug * 0.10)
    train_examples.extend(non_buggy_selected[:num_train_nobugs])
    valid_examples.extend(non_buggy_selected[num_train_nobugs:(num_train_nobugs + num_valid_nobug)])
    test_examples.extend(non_buggy_selected[(num_train_nobugs + num_valid_nobug):])

    final_bug_percentage = int(num_bug * 100 / (num_bug + num_non_bug))
    final_non_bug_percentage = 100 - final_bug_percentage
    file_name = os.path.join(output, name)
    if not keep_original:
        file_name = file_name + '-' + str(final_bug_percentage) + '-' + str(final_non_bug_percentage)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

    for n, examples in zip(['train', 'valid', 'test'], [train_examples, valid_examples, test_examples]):
        f_name = os.path.join(
            file_name, n + '_GGNNinput.json' )
        print('Saving to, ' + f_name)
        with open(f_name, 'w') as fp:
            json.dump(examples, fp)
            fp.close()
    pass


def split_data_main(name, input_data, percent=[50]):
    # python split_data.py --input 
    # /space2/ding/dl-vulnerability-detection/data/full_experiment_real_data_processed/bugzilla_snykio-full_graph.json 
    # --output /space2/ding/dl-vulnerability-detection/data/ggnn_input/bugzilla_snykio --name bugzilla_snykio --percent 50
    # input=f'/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/full_experiment_real_data_processed/{name}-full_graph.json'
    output=f'/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/ggnn_input/{name}'
    # input_data = json.load(open(input))
    print('Finish Reading data, #examples', len(input_data))
    buggy = []
    non_buggy = []
    for example in input_data:
        target = example['targets'][0][0]
        if target == 1:
            buggy.append(example)
        else:
            non_buggy.append(example)
    print('Buggy', len(buggy), 'Non Buggy', len(non_buggy))
    buggy_count = len(buggy)
    if not os.path.exists(output):
        os.mkdir(output)
    split_and_save(name + '-original', output, buggy, non_buggy, 0, True)
    for percent in percent:
        split_and_save(name, output, buggy, non_buggy, percent)
