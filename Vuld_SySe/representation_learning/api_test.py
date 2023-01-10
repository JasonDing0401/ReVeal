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
    parser.add_argument('--testset', type=str, help='test dataset')
    parser.add_argument('--pretrain', type=str, default=None, help="path to load prerained model with training dataset")
    
    numpy.random.rand(1000)
    torch.manual_seed(1000)
    args = parser.parse_args()
    dataset = args.dataset
    testset = args.testset
    feature_name = args.features
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
    output_file_name += 'pretrained-' + str(args.pretrain)
    timestr = time.strftime("%m%d-%H%M%S")
    output_file_name += '_' + timestr + '.tsv'
    output_file = open(output_file_name, 'w')
    features = []
    targets = []
    for part in ['train']:
        json_data_file = open(ds + part + '_GGNNinput_graph.json')
        data = json.load(json_data_file)
        json_data_file.close()
        for d in data:
            features.append(d['graph_feature'])
            targets.append(d['target'])
        del data
    train_X = numpy.array(features)
    train_Y = numpy.array(targets)
    features = []
    targets = []
    for part in ['valid']:
        json_data_file = open(ds + part + '_GGNNinput_graph.json')
        data = json.load(json_data_file)
        json_data_file.close()
        for d in data:
            features.append(d['graph_feature'])
            targets.append(d['target'])
        del data
    valid_X = numpy.array(features)
    valid_Y = numpy.array(targets)
    features = []
    targets = []
    test_ds = f'../../../Devign/output/{testset}/'
    for part in ['test']:
        json_data_file = open(test_ds + part + '_GGNNinput_graph.json')
        data = json.load(json_data_file)
        json_data_file.close()
        for d in data:
            features.append(d['graph_feature'])
            targets.append(d['target'])
        del data
    test_X = numpy.array(features)
    test_Y = numpy.array(targets)
    print(f'Trainset {dataset} Testset {testset}', train_X.shape, valid_X.shape, test_X.shape, numpy.sum(train_Y), numpy.sum(valid_Y), numpy.sum(test_Y), sep='\t', file=sys.stderr)
    print('=' * 100, file=sys.stderr, flush=True)
    print('Accuracy', 'Precision', 'Recall', 'F1', 'TNR', 'FPR', 'FNR', sep='\t', flush=True,\
        file=output_file)
    for _ in range(5):
        if args.baseline:
            model = SVMLearningAPI(True, args.baseline_balance, model_type=args.baseline_model)
        else:
            model = RepresentationLearningModel(
                lambda1=args.lambda1, lambda2=args.lambda2, batch_size=1024, print=True, max_patience=args.max_patience, balance=True,
                num_layers=args.num_layers
            )
        if not args.pretrain:
            model.train(train_X, train_Y, valid_X, valid_Y, args.dataset)
        else:
            model.dataset_init(test_X)
        results = model.evaluate(test_X, test_Y, args.pretrain)
        print(results['accuracy'], results['precision'], results['recall'], results['f1'], results['tnr'], results['fpr'], results['fnr'], sep='\t', flush=True,
              file=output_file)
        print(results['accuracy'], results['precision'], results['recall'], results['f1'], results['tnr'], results['fpr'], results['fnr'], sep=',',
              file=sys.stderr, flush=True, end=('\n' + '=' * 100 + '\n'))
        if args.pretrain:
            break
    output_file.close()
    pass
