import numpy
import sys
import datetime
import torch
import joblib
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from graph_dataset import DataSet
from models import MetricLearningModel
from trainer import train, predict, predict_proba, evaluate as evaluate_from_model


class RepresentationLearningModel(BaseEstimator):
    def __init__(self,
                 alpha=0.5, lambda1=0.5, lambda2=0.001, hidden_dim=256,  # Model Parameters
                 dropout=0.2, batch_size=64, balance=True,   # Model Parameters
                 num_epoch=1000, max_patience=20,  # Training Parameters
                 print=False, num_layers=1
                 ):
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.dropout = dropout
        self.num_epoch = num_epoch
        self.max_patience = max_patience
        self.batch_size = batch_size
        self.balance = balance
        self.cuda = torch.cuda.is_available()
        self.print = print
        self.num_layers = num_layers
        if print:
            self.output_buffer = sys.stderr
        else:
            self.output_buffer = None
        pass

    def dataset_init(self, test_x):
        self.dataset = DataSet(self.batch_size, test_x.shape[1])
        
    def fit(self, train_x, train_y):
        self.train(train_x, train_y)

    def train(self, train_x, train_y, valid_x, valid_y, dataset_name):
        input_dim = train_x.shape[1]
        self.model = MetricLearningModel(
            input_dim=input_dim, hidden_dim=self.hidden_dim, aplha=self.alpha, lambda1=self.lambda1,
            lambda2=self.lambda2, dropout_p=self.dropout, num_layers=self.num_layers
        )
        self.optimizer = Adam(self.model.parameters())
        if self.cuda:
            self.model.cuda(device=0)
        self.dataset = DataSet(self.batch_size, train_x.shape[1])
        for _x, _y in zip(train_x, train_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), 'train')
        for _x, _y in zip(valid_x, valid_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), 'valid')
        self.dataset.initialize_dataset(balance=self.balance, output_buffer=self.output_buffer)
        time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_path = f"models/{dataset_name}_{time}.bin"
        print("model path: {}".format(model_path))
        train(
            model=self.model, dataset=self.dataset, optimizer=self.optimizer,
            num_epochs=self.num_epoch, max_patience=self.max_patience,
            cuda_device=0 if self.cuda else -1, model_path=model_path,
            output_buffer=self.output_buffer
        )
        if self.output_buffer is not None:
            print('Training Complete', file=self.output_buffer)

    def predict(self, text_x):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x in text_x:
            self.dataset.add_data_entry(_x.tolist(), 0, part='test')
        return predict(
            model=self.model, iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1,
        )

    def predict_proba(self, text_x):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x in text_x:
            self.dataset.add_data_entry(_x.tolist(), 0, part='test')
        return predict_proba(
            model=self.model, iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1
        )

    def evaluate(self, test_x, test_y, pretrain=None):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x, _y in zip(test_x, test_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), part='test')
        if pretrain:
            input_dim = test_x.shape[1]
            self.model = MetricLearningModel(
                input_dim=input_dim, hidden_dim=self.hidden_dim, aplha=self.alpha, lambda1=self.lambda1,
                lambda2=self.lambda2, dropout_p=self.dropout, num_layers=self.num_layers
            )
            self.model.load_state_dict(torch.load(f"models/{pretrain}"))
            print(f"Successfully loaded model {pretrain} from file!\n")
            self.model.cuda()
        acc, pr, rc, f1, tnr, fpr, fnr = evaluate_from_model(
            model=self.model, iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1,
            output_buffer=self.output_buffer
        )
        return {
            'accuracy': round(acc, 3),
            'precision': round(pr, 3),
            'recall': round(rc, 3),
            'f1': round(f1, 3),
            'tnr': round(tnr, 3),
            'fpr': round(fpr, 3),
            'fnr': round(fnr, 3),
        }

    def score(self, test_x, test_y):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x, _y in zip(test_x, test_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), part='test')
        _, _, _, f1 = evaluate_from_model(
            model=self.model, iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1,
            output_buffer=self.output_buffer
        )
        return f1