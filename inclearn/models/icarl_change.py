import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.nn import functional as F
from tqdm import tqdm

from inclearn.lib import factory, network, utils
from inclearn.models.base import IncrementalLearner

from sklearn.preprocessing import normalize
# from torch.utils.data.sampler import SubsetRandomSampler
# from torchvision import datasets, transforms

import torch.nn as nn
EPSILON = 1e-8


class ICarl(IncrementalLearner):
    """Implementation of iCarl.

    # References:
    - iCaRL: Incremental Classifier and Representation Learning
      Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
      https://arxiv.org/abs/1611.07725

    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args):
        super().__init__()

        self._device = args["device"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._memory_size = args["memory_size"]
        self._n_classes = 0

        self._network = network.BasicNet(args["convnet"], device=self._device, use_bias=True)

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self._herding_matrix = []
        
        # store the original data
        self._data_memory = None
        self._targets_memory = None
        
        # stroe the transform data
        self._data_transform_memory = None
        self._targets_transform_memory = None

        # add exemplar metric to store the distance
        self.metric = None
        self.metric_2 = None
        self.margin = 10.0
#         self.margin = torch.tensor([10.0])

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    # ----------
    # Public API
    # ----------

    def _before_task(self, train_loader, val_loader):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)
        print("Now {} examplars per class.".format(self._memory_per_class))

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )

        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )
    
    def _train_task(self, train_loader, val_loader):
        print("nb ", len(train_loader.dataset))

        for epoch in range(self._n_epochs):
            _loss, val_loss = 0., 0.

            self._scheduler.step()

            prog_bar = tqdm(train_loader)
            for i, (inputs, targets) in enumerate(prog_bar, start=1):
                self._optimizer.zero_grad()
                
                loss_graph = 0 
                loss_2 = 0
                feature_metric = None
                
                if self._data_transform_memory is not None:
                    features = self._network.extract(
                        self._data_transform_memory.to(self._network.device)
                    )
                    
                    feature_metric = torch.cdist(features, features)
                                        
                    print('feature_metric:', feature_metric)
                    print('feature_metric.shape:', feature_metric.shape)
                    
                    # get the upper triangle matrix to calc the loss                    
                    first_part = self.metric_2.mul(torch.pow(feature_metric, 2))
                    second_part = (1 - self.metric_2).mul(torch.pow(torch.clamp(self.margin - feature_metric, min=0.0), 2))
                    print('first_part:', first_part)
                    print('second_part:', second_part)
                    loss_graph += first_part + second_part
                    print('loss_graph:', loss_graph)
                    print('loss_graph.shape:', loss_graph.shape)
                    count = 0
                    for i in range(loss_graph.shape[0]):
                        for j in range(loss_graph.shape[1]):
                            if i >= j:
                                loss_2 += loss_graph[i][j]
                                count += 1
                    loss_2 = (loss_2 / count) / 2
                    print('loss_2:', loss_2)
                    print('loss_2.shape:', loss_2.shape)
                    loss_2.backward(retain_graph=True)

                    
                
                  
                # loss for classification and distillation
                loss = self._forward_loss(inputs, targets)
                

                if not utils._check_loss(loss):
                    import pdb
                    pdb.set_trace()

                print('loss:', loss)
                print('loss.shape:', loss.shape)
                loss.backward()
                
                self._optimizer.step()

                _loss += loss.item()

                if val_loader is not None and i == len(train_loader):
                    for inputs, targets in val_loader:
                        val_loss += self._forward_loss(inputs, targets).item()

                prog_bar.set_description(
                    "Task {}/{}, Epoch {}/{} => Clf loss: {}, Val loss: {}".format(
                        self._task + 1, self._n_tasks,
                        epoch + 1, self._n_epochs,
                        round(_loss / i, 3),
                        round(val_loss, 3)
                    )
                )

    def _forward_loss(self, inputs, targets):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        targets = utils.to_onehot(targets, self._n_classes).to(self._device)
        logits = self._network(inputs)

        return self._compute_loss(inputs, logits, targets)

    def _after_task(self, inc_dataset):
        self.build_examplars(inc_dataset)

        self._old_model = self._network.copy().freeze()

    def _eval_task(self, data_loader):
        ypred, ytrue = compute_accuracy(self._network, data_loader, self._class_means)

        return ypred, ytrue

    # -----------
    # Private API
    # -----------

    def _compute_loss(self, inputs, logits, targets):
        if self._old_model is None:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        else:
            old_targets = torch.sigmoid(self._old_model(inputs).detach())

            new_targets = targets.clone()
            new_targets[..., :-self._task_size] = old_targets
                     
            loss = F.binary_cross_entropy_with_logits(logits, new_targets)
            # todo
            # add external metric loss
            

        return loss

    def _compute_predictions(self, data_loader):
        preds = torch.zeros(self._n_train_data, self._n_classes, device=self._device)

        for idxes, inputs, _ in data_loader:
            inputs = inputs.to(self._device)
            idxes = idxes[1].to(self._device)

            preds[idxes] = self._network(inputs).detach()

        return torch.sigmoid(preds)

    def _classify(self, data_loader):
        if self._means is None:
            raise ValueError(
                "Cannot classify without built examplar means,"
                " Have you forgotten to call `before_task`?"
            )
        if self._means.shape[0] != self._n_classes:
            raise ValueError(
                "The number of examplar means ({}) is inconsistent".format(self._means.shape[0]) +
                " with the number of classes ({}).".format(self._n_classes)
            )

        ypred = []
        ytrue = []

        for _, inputs, targets in data_loader:
            inputs = inputs.to(self._device)

            features = self._network.extract(inputs).detach()
            preds = self._get_closest(self._means, F.normalize(features))

            ypred.extend(preds)
            ytrue.extend(targets)

        return np.array(ypred), np.array(ytrue)

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        return self._memory_size // self._n_classes

    # -----------------
    # Memory management
    # -----------------

    def build_examplars(self, inc_dataset):
        print("Building & updating memory.")

        self._data_memory, self._targets_memory = [], []
        self._class_means = np.zeros((100, self._network.features_dim))
        
        # store the feature
        self._feature_data_memory = []
        self._feature_targets_memory = []
        
        # store the transform data
        self._data_transform_memory = []
        self._targets_transform_memory = []
        
        for class_idx in range(self._n_classes):
            inputs, loader = inc_dataset.get_custom_loader(class_idx, mode="test")  
            
#             print('inputs.shape', inputs.shape)
#             print('inputs', inputs)
            features, targets = extract_features(
                self._network, loader
            )
            features_flipped, _ = extract_features(
                self._network, inc_dataset.get_custom_loader(class_idx, mode="flip")[1]
            )

            if class_idx >= self._n_classes - self._task_size:
                self._herding_matrix.append(select_examplars(
                    features, self._memory_per_class
                ))

            examplar_mean, alph = compute_examplar_mean(
                features, features_flipped, self._herding_matrix[class_idx], self._memory_per_class
            )
            # store the original data of exemplars
            # shape: (500, 32, 32, 3)
            self._data_memory.append(inputs[np.where(alph == 1)[0]])
            self._targets_memory.append(targets[np.where(alph == 1)[0]])
            
            exemplar_index = np.where(alph == 1)[0]
            all_data = np.zeros(inputs.shape)
            
            # get all the data in the loader (transform)
            # shape: (500, 3, 32, 32)
            for inputs_t, _ in loader:                 
                all_data = inputs_t.cpu().data.numpy()  
                
                        
    
            # store the transform data 
            self._data_transform_memory.append(all_data[np.where(alph == 1)[0]])
            self._targets_transform_memory.append(targets[np.where(alph == 1)[0]])
            
            
            # store the feature of exemplars
            self._feature_data_memory.append(features[np.where(alph == 1)[0]])
            self._feature_targets_memory.append(targets[np.where(alph == 1)[0]])
            
            # store the class mean
            self._class_means[class_idx, :] = examplar_mean

        self._data_memory = np.concatenate(self._data_memory)
        self._targets_memory = np.concatenate(self._targets_memory)
        
        self._data_transform_memory = np.concatenate(self._data_transform_memory)
        self._targets_transform_memory = np.concatenate(self._targets_transform_memory)
        # to torch
        self._data_transform_memory = torch.from_numpy(self._data_transform_memory)
        
        self._feature_data_memory = np.concatenate(self._feature_data_memory)
        self._feature_targets_memory = np.concatenate(self._feature_targets_memory)    
        
        # calc the exemplar distance
        # calc the distance metric
        self.metric = cdist(self._feature_data_memory, self._feature_data_memory, 'euclidean')
        # normalize the metric
        self.metric = normalize(self.metric, axis=1, norm='l1')
        # to torch
        self.metric = torch.from_numpy(self.metric).float().to(self._device)
        
        # use the label to cinstruct the symmetric
        self.metric_2 = torch.empty(self.metric.shape)
        for i in range(self.metric.shape[0]):
            for j in range(self.metric.shape[1]):
                if self._targets_memory[i] == self._targets_memory[j]:
                    self.metric_2[i][j] = 1
                else:
                    self.metric_2[i][j] = 0
        self.metric_2 =self.metric_2.float().to(self._device)       
#         print('self.metric_2:', self.metric_2)
#         print('self.metric_2.shape:',self.metric_2.shape)
#         print('self._targets_transform_memory:', self._targets_transform_memory)
#         print('self._targets_transform_memory.shape:', self._targets_transform_memory.shape)
#         print('self._data_transform_memory:', self._data_transform_memory)
#         print('self._data_transform_memory.shape:', self._data_transform_memory.shape)
        
#         print('self.metric:', self.metric)
#         print('self.metric.shape:', self.metric.shape)   

#         print('self._feature_data_memory:', self._feature_data_memory)        
#         print('self._feature_data_memory.shape:', self._feature_data_memory.shape)
#         print('len(self._herding_matrix[0]):', len(self._herding_matrix[0]))                
#         print('self._data_memory:', self._data_memory)
#         print('self._data_memory.shape:', self._data_memory.shape)
#         print('self._targets_memory:', self._targets_memory)
#         print('self._targets_memory.shape:', self._targets_memory.shape)
        
        

    def get_memory(self):
        return self._data_memory, self._targets_memory


def extract_features(model, loader):
    targets, features = [], []

    for _inputs, _targets in loader:
        _targets = _targets.numpy()
        _features = model.extract(_inputs.to(model.device)).detach().cpu().numpy()

        features.append(_features)
        targets.append(_targets)

    return np.concatenate(features), np.concatenate(targets)


def select_examplars(features, nb_max):
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + EPSILON)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    while not (
        np.sum(herding_matrix != 0) == min(nb_max, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    return herding_matrix


def compute_examplar_mean(feat_norm, feat_flip, herding_mat, nb_max):
    D = feat_norm.T
    D = D / (np.linalg.norm(D, axis=0) + EPSILON)

    D2 = feat_flip.T
    D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

    alph = herding_mat
    alph = (alph > 0) * (alph < nb_max + 1) * 1.

    alph_mean = alph / np.sum(alph)

    mean = (np.dot(D, alph_mean) + np.dot(D2, alph_mean)) / 2
    mean /= np.linalg.norm(mean)

    return mean, alph


def compute_accuracy(model, loader, class_means):
    features, targets_ = extract_features(model, loader)

    targets = np.zeros((targets_.shape[0], 100), np.float32)
    targets[range(len(targets_)), targets_.astype('int32')] = 1.
    features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

    # Compute score for iCaRL
    sqd = cdist(class_means, features, 'sqeuclidean')
    score_icarl = (-sqd).T

    return np.argsort(score_icarl, axis=1)[:, -1], targets_
