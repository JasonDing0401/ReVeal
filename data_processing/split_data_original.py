import argparse, os
import json
import numpy as np
from tqdm import tqdm


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
    num_train_bugs = int(num_bug * 0.70)
    num_valid_bug = int(num_bug * 0.10)
    train_examples.extend(buggy[:num_train_bugs])
    valid_examples.extend(buggy[num_train_bugs:(num_train_bugs + num_valid_bug)])
    test_examples.extend(buggy[(num_train_bugs + num_valid_bug):])
    num_non_bug = len(non_buggy_selected)
    num_train_nobugs = int(num_non_bug * 0.70)
    num_valid_nobug = int(num_non_bug * 0.10)
    train_examples.extend(non_buggy_selected[:num_train_nobugs])
    valid_examples.extend(non_buggy_selected[num_train_nobugs:(num_train_nobugs + num_valid_nobug)])
    test_examples.extend(non_buggy_selected[(num_train_nobugs + num_valid_nobug):])
    final_bug_percentage = int(num_bug * 100 / (num_bug + num_non_bug))
    final_non_bug_percentage = 100 - final_bug_percentage
    file_name = output
    if not keep_original:
        file_name = file_name + '-' + str(final_bug_percentage) + '-' + str(final_non_bug_percentage)
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    for n, examples in zip(['train', 'valid', 'test'], [train_examples, valid_examples, test_examples]):
        f_name = os.path.join(
            file_name, n + '_GGNNinput.json' )
        with open(f_name, 'w') as fp:
            json.dump(examples, fp)
            fp.close()
    pass


if __name__ == '__main__':
    datasets = ['chrome_debian', 'devign']
    parts = ['cfg', 'dfg', 'cfg_dfg']
    for d in datasets:
        for p in parts:
            input_path = d + '_' + p + '_full.json'
            output_path = d + '/' + p
            parser = argparse.ArgumentParser()
            parser.add_argument('--input', help='Path of the input file', default=input_path)
            parser.add_argument('--output', help='Output Directory', default=output_path)
            parser.add_argument('--repeat_count', help='Number of times to be repeated', default=5, type=int)
            args = parser.parse_args()
            print(args.input, args.output)
            input_data = json.load(open(args.input))
            print('Finish Reading data, #examples', len(input_data))
            buggy = []
            non_buggy = []
            for example in tqdm(input_data):
                target = example['targets'][0][0]
                if target == 1:
                    buggy.append(example)
                else:
                    non_buggy.append(example)
            print('Buggy', len(buggy), 'Non Buggy', len(non_buggy))
            buggy_count = len(buggy)
            if not os.path.exists(args.output):
                os.mkdir(args.output)
            for r in range(1, args.repeat_count + 1):
                output = os.path.join(args.output, 'v' + str(r))
                if not os.path.exists(output):
                    os.mkdir(output)
                split_and_save(' ', output, buggy, non_buggy, 100, True)
    pass
