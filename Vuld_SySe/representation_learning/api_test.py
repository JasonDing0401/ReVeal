import argparse
import json
import numpy
import os
import sys
import torch
from representation_learning_api import RepresentationLearningModel
from sklearn.model_selection import train_test_split
from baseline_svm import SVMLearningAPI
import pickle
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset name from devign/output')
    parser.add_argument('--features', default='ggnn', choices=['ggnn', 'wo_ggnn'])
    parser.add_argument('--lambda1', default=0.5, type=float)
    parser.add_argument('--lambda2', default=0.001, type=float)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--baseline_balance', action='store_true')
    parser.add_argument('--baseline_model', default='svm')
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--pretrain', action='store_true', help='if use reveal pretrained model')
    parser.add_argument('--train_w_reveal', action='store_true', help='whether to train the model with reveal dataset again')
    numpy.random.rand(1000)
    torch.manual_seed(1000)
    args = parser.parse_args()
    dataset = args.dataset
    feature_name = args.features
    parts = ['train', 'valid', 'test']
    # if feature_name == 'ggnn':
    #     if dataset == 'chrome_debian/balanced':
    #         ds = '../../data/after_ggnn/chrome_debian/balance/v3/'
    #     elif dataset == 'chrome_debian/imbalanced':
    #         ds = '../../data/after_ggnn/chrome_debian/imbalance/v6/'
    #     elif dataset == 'devign':
    #         ds = '../../data/after_ggnn/devign/v6/'
    #     else:
    #         raise ValueError('Imvalid Dataset')
    # else:
    #     if dataset == 'chrome_debian':
    #         ds = '../../data/full_experiment_real_data_processed/chrome_debian/full_graph/v1/graph_features/'
    #     elif dataset == 'devign':
    #         ds = '../../data/full_experiment_real_data_processed/devign/full_graph/v1/graph_features/'
    #     else:
    #         raise ValueError('Imvalid Dataset')
    ds = f'../../../Devign/output/{dataset}/'
    assert isinstance(dataset, str)
    output_dir = 'results_test'
    if args.baseline:
        output_dir = 'baseline_' + args.baseline_model
        if args.baseline_balance:
            output_dir += '_balance'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file_name = output_dir + '/' + dataset.replace('/', '_') + '-' + feature_name + '-'
    if args.max_patience != 5:
        output_file_name += 'max_patience_' + str(args.max_patience) +'-'
    if args.lambda1 == 0:
        assert args.lambda2 == 0
        output_file_name += 'cross-entropy-only-layers-'+ str(args.num_layers) +'-'
    else:
        output_file_name += 'triplet-loss-layers-'+ str(args.num_layers) +'-'
    output_file_name += 'pretrained-reveal-' + str(args.pretrain) + '-'
    output_file_name += 'train_w_reveal-' + str(args.train_w_reveal)
    timestr = time.strftime("%m%d-%H%M%S")
    output_file_name += '_' + timestr + '.tsv'
    output_file = open(output_file_name, 'w')
    features = []
    targets = []
    for part in parts:
        json_data_file = open(ds + part + '_GGNNinput_graph.json')
        data = json.load(json_data_file)
        json_data_file.close()
        for d in data:
            features.append(d['graph_feature'])
            targets.append(d['target'])
        del data
    if args.train_w_reveal:
        ds_train = '../../data/after_ggnn/chrome_debian/balance/v3/'
        features_train = []
        targets_train = []
        for part in parts:
            json_data_file = open(ds_train + part + '_GGNNinput_graph.json')
            data = json.load(json_data_file)
            json_data_file.close()
            for d in data:
                features_train.append(d['graph_feature'])
                targets_train.append(d['target'])
            del data
    X = numpy.array(features)
    Y = numpy.array(targets)
    print('Dataset', X.shape, Y.shape, numpy.sum(Y), sep='\t', file=sys.stderr)
    print('=' * 100, file=sys.stderr, flush=True)
    print('Accuracy', 'Precision', 'Recall', 'F1', sep='\t', flush=True,\
        file=output_file)
    for _ in range(30):
        if args.pretrain:
            test_X, test_Y = X, Y
        else:
            if args.train_w_reveal:
                train_X, train_Y = numpy.array(features_train), numpy.array(targets_train)
                test_X, test_Y = X, Y
            else:
                train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
        # print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape, sep='\t', file=sys.stderr, flush=True)
        if args.baseline:
            model = SVMLearningAPI(True, args.baseline_balance, model_type=args.baseline_model)
        else:
            model = RepresentationLearningModel(
                lambda1=args.lambda1, lambda2=args.lambda2, batch_size=128, print=True, max_patience=args.max_patience, balance=True,
                num_layers=args.num_layers
            )
        if not args.pretrain:
            model.train(train_X, train_Y, args.dataset, args.train_w_reveal)
        else:
            model.dataset_init(test_X)
        results = model.evaluate(test_X, test_Y, args.dataset, args.pretrain)
        print(results['accuracy'], results['precision'], results['recall'], results['f1'], sep='\t', flush=True,
              file=output_file)
        print(results['accuracy'], results['precision'], results['recall'], results['f1'], sep='\t',
              file=sys.stderr, flush=True, end=('\n' + '=' * 100 + '\n'))
        if args.pretrain:
            break
    output_file.close()
    pass
