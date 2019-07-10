import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import functional as F

from inclearn import factory, utils
from inclearn.lib import callbacks, network
from inclearn.models.base import IncrementalLearner

tqdm.monitor_interval = 0


class End2End(IncrementalLearner):
    """Implementation of End-to-End Increment Learning.

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

        self._k = args["memory_size"]
        self._n_classes = 0

        self._temperature = args["temperature"]

        self._network = network.BasicNet(args["convnet"], use_bias=True, use_multi_fc=True,
                                         device=self._device)

        self._examplars = {}
        self._old_model = []

        self._task_idxes = []

    # ----------
    # Public API
    # ----------

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    def _before_task(self, train_loader, val_loader):
        """Set up before the task training can begin.

        1. Precomputes previous model probabilities.
        2. Extend the classifier to support new classes.

        :param train_loader: The training dataloader.
        :param val_loader: The validation dataloader.
        """
        self._network.add_classes(self._task_size)

        self._task_idxes.append([self._n_classes + i for i in range(self._task_size)])

        self._n_classes += self._task_size
        print("Now {} examplars per class.".format(self._m))

    def _train_task(self, train_loader, val_loader):
        """Train & fine-tune model.

        The scheduling is different from the paper for one reason. In the paper,
        End-to-End Incremental Learning, the authors pre-generated 12 augmentations
        per images (thus multiplying by this number the dataset size). However
        I find this inefficient for large scale datasets, thus I'm simply doing
        the augmentations online. A greater number of epochs is then needed to
        match performances.

        :param train_loader: A DataLoader.
        :param val_loader: A DataLoader, can be None.
        """
        if self._task == 0:
            epochs = 90
            optimizer = factory.get_optimizer(self._network.parameters(), self._opt_name, 0.1, 0.001)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 60], gamma=0.1)
            self._train(train_loader, val_loader, epochs, optimizer, scheduler)
            return

        # Training on all new + examplars
        print("Training")
        self._finetuning = False
        epochs = 60
        optimizer = factory.get_optimizer(self._network.parameters(), self._opt_name, 0.1, 0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 50], gamma=0.1)
        self._train(train_loader, val_loader, epochs, optimizer, scheduler)

        # Fine-tuning on sub-set new + examplars
        print("Fine-tuning")
        self._old_model = self._network.copy().freeze()

        self._finetuning = True
        self._build_examplars(train_loader,
                              n_examplars=self._k // (self._n_classes - self._task_size))
        train_loader.dataset.set_idxes(self.examplars)  # Fine-tuning only on balanced dataset

        optimizer = factory.get_optimizer(self._network.parameters(), self._opt_name, 0.01, 0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.1)
        self._train(train_loader, val_loader, 40, optimizer, scheduler)

    def _after_task(self, data_loader):
        self._reduce_examplars()
        self._build_examplars(data_loader)

        self._old_model = self._network.copy().freeze()

    def _eval_task(self, data_loader):
        ypred, ytrue = self._classify(data_loader)
        assert ypred.shape == ytrue.shape

        return ypred, ytrue

    def get_memory_indexes(self):
        return self.examplars

    # -----------
    # Private API
    # -----------

    def _train(self, train_loader, val_loader, n_epochs, optimizer, scheduler):
        self._callbacks = [
            callbacks.GaussianNoiseAnnealing(self._network.parameters()),
            #callbacks.EarlyStopping(self._network, minimize_metric=False)
        ]
        self._best_acc = float("-inf")

        print("nb ", len(train_loader.dataset))
        prog_bar = tqdm.trange(n_epochs, desc="Losses.")

        val_acc = 0.
        train_acc = 0.
        for epoch in prog_bar:
            for cb in self._callbacks:
                cb.on_epoch_begin()

            scheduler.step()

            _clf_loss, _distil_loss = 0., 0.
            c = 0

            for i, ((_, idxes), inputs, targets) in enumerate(train_loader, start=1):
                optimizer.zero_grad()

                c += 1
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)

                clf_loss, distil_loss = self._compute_loss(
                    inputs,
                    logits,
                    targets,
                    idxes,
                )

                if not utils._check_loss(clf_loss) or not utils._check_loss(distil_loss):
                    import pdb
                    pdb.set_trace()

                loss = clf_loss + distil_loss

                loss.backward()

                #if self._task != 0:
                #    for param in self._network.parameters():
                #        param.grad = param.grad * (self._temperature ** 2)
                for cb in self._callbacks:
                    cb.before_step()
                optimizer.step()

                _clf_loss += clf_loss.item()
                _distil_loss += distil_loss.item()

                if i % 10 == 0 or i >= len(train_loader):
                    prog_bar.set_description(
                        "Clf: {}; Distill: {}; Train: {}; Val: {}".format(
                            round(clf_loss.item(), 3), round(distil_loss.item(), 3),
                            round(train_acc, 3),
                            round(val_acc, 3)
                        )
                    )

            if val_loader:
                ypred, ytrue = self._classify(val_loader)
                val_acc = (ypred == ytrue).sum() / len(ytrue)
                self._best_acc = max(self._best_acc, val_acc)
                ypred, ytrue = self._classify(train_loader)
                train_acc = (ypred == ytrue).sum() / len(ytrue)

            for cb in self._callbacks:
                cb.on_epoch_end(metric=val_acc)

            prog_bar.set_description(
                "Clf: {}; Distill: {}; Train: {}; Val: {}".format(
                    round(_clf_loss / c, 3), round(_distil_loss / c, 3),
                    round(train_acc, 3),
                    round(val_acc, 3),
                )
            )

            for cb in self._callbacks:
                if not cb.in_training:
                    self._network = cb.network
                    return

        print("best", self._best_acc)

    def _compute_loss(self, inputs, logits, targets, idxes):
        """Computes the classification loss & the distillation loss.

        Distillation loss is null at the first task.

        :param logits: Logits produced the model.
        :param targets: The targets.
        :param idxes: The real indexes of the just-processed images. Needed to
                      match the previous predictions.
        :return: A tuple of the classification loss and the distillation loss.
        """
        clf_loss = F.cross_entropy(logits, targets)

        if self._task == 0:
            distil_loss = torch.zeros(1, device=self._device)
        else:
            if self._finetuning:
                # We only do distillation on current task during the distillation
                # phase:
                last_index = len(self._task_idxes)
            else:
                last_index = len(self._task_idxes) - 1

            distil_loss = 0.
            #with torch.no_grad():
            previous_logits = self._old_model(inputs)

            for i in range(last_index):
                task_idxes = self._task_idxes[i]

                distil_loss += F.binary_cross_entropy(
                    F.softmax(logits[..., task_idxes] / self._temperature, dim=1),
                    F.softmax(previous_logits[..., task_idxes] / self._temperature, dim=1)
                )

        return clf_loss, distil_loss

    def _compute_predictions(self, loader):
        """Precomputes the logits before a task.

        :param data_loader: A DataLoader.
        :return: A tensor storing the whole current dataset logits.
        """
        logits = torch.zeros(self._n_train_data, self._n_classes, device=self._device)

        for idxes, inputs, _ in loader:
            inputs = inputs.to(self._device)
            idxes = idxes[1].to(self._device)

            logits[idxes] = self._network(inputs).detach()

        return logits

    def _classify(self, loader):
        """Classify the images given by the data loader.

        :param data_loader: A DataLoader.
        :return: A numpy array of the predicted targets and a numpy array of the
                 ground-truth targets.
        """
        ypred = []
        ytrue = []

        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            logits = F.softmax(self._network(inputs), dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()

            ypred.extend(preds)
            ytrue.extend(targets)

        return np.array(ypred), np.array(ytrue)

    @property
    def _m(self):
        """Returns the number of examplars per class."""
        return self._k // self._n_classes

    def _extract_features(self, loader):
        features = []
        idxes = []

        for (real_idxes, _), inputs, _ in loader:
            inputs = inputs.to(self._device)
            features.append(self._network.extract(inputs).detach())
            idxes.extend(real_idxes.numpy().tolist())

        features = torch.cat(features)
        mean = torch.mean(features, dim=0, keepdim=False)

        return features, mean, idxes

    @staticmethod
    def _get_closest(centers, features):
        """Returns the center index being the closest to each feature.

        :param centers: Centers to compare, in this case the class means.
        :param features: A tensor of features extracted by the convnet.
        :return: A numpy array of the closest centers indexes.
        """
        pred_labels = []

        features = features
        for feature in features:
            distances = End2End._dist(centers, feature)
            pred_labels.append(distances.argmin().item())

        return np.array(pred_labels)

    @staticmethod
    def _dist(a, b):
        """Computes L2 distance between two tensors.

        :param a: A tensor.
        :param b: A tensor.
        :return: A tensor of distance being of the shape of the "biggest" input
                 tensor.
        """
        return torch.pow(a - b, 2).sum(-1)

    def _build_examplars(self, loader, n_examplars=None):
        """Builds new examplars.

        :param loader: A DataLoader.
        :param n_examplars: Maximum number of examplars to create.
        """
        n_examplars = n_examplars or self._m

        lo, hi = self._task * self._task_size, self._n_classes
        print("Building examplars for classes {} -> {}.".format(lo, hi))
        for class_idx in range(lo, hi):
            loader.dataset.set_classes_range(class_idx, class_idx)
            self._examplars[class_idx] = self._build_class_examplars(loader, n_examplars)

    def _build_class_examplars(self, loader, n_examplars):
        """Build examplars for a single class.

        Examplars are selected as the closest to the class mean.

        :param loader: DataLoader that provides images for a single class.
        :param n_examplars: Maximum number of examplars to create.
        :return: The real indexes of the chosen examplars.
        """
        features, class_mean, idxes = self._extract_features(loader)

        class_mean = F.normalize(class_mean, dim=0)
        features = F.normalize(features, dim=1)
        distances_to_mean = self._dist(class_mean, features)

        nb_examplars = min(n_examplars, len(features))

        fake_idxes = distances_to_mean.argsort().cpu().numpy()[:nb_examplars]
        return [idxes[idx] for idx in fake_idxes]

    @property
    def examplars(self):
        """Returns all the real examplars indexes.

        :return: A numpy array of indexes.
        """
        return np.array(
            [
                examplar_idx for class_examplars in self._examplars.values()
                for examplar_idx in class_examplars
            ]
        )

    def _reduce_examplars(self):
        print("Reducing examplars.")
        for class_idx in range(len(self._examplars)):
            self._examplars[class_idx] = self._examplars[class_idx][:self._m]
