# coding=utf-8


# Author: Amir Abolfazli <abolfazli@l3s.de>
# based on the implementation of the SAM-kNN by Viktor Losing


import scipy
import numpy as np
import collections
from collections import deque
import copy as cp
import logging
from scipy.stats import multivariate_normal

# scikit-learn
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score

# scikit-multiflow
from skmultiflow.lazy import libNearestNeighbor
from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import get_dimensions

# imbalanced-learn
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.metrics import geometric_mean_score


class DAM3Classifier(BaseSKMObject, ClassifierMixin):
    """ Drift-Aware Multi-Memory Model.
    Parameters
    ----------
    n_neighbors : int, optional (default=5)
        number of evaluated nearest neighbors.
    max_window_size : int, optional (default=5000)
         Maximum number of overall stored instances.
    ltm_size: float, optional (default=0.5)
        Maximum number of instances in LTM.
    min_stm_size : int, optional (default=50)
        Minimum size of the short-term memory (STM).
    wm_size : float, optional (default=0.2)
        Size of the working memory (WM).
    drift_detector_winSize : int, optional (default=min_stm_size)
        Window size of the drift detector.
    drift_detector_thr : float, optional (default=0.001)
        Signifance level of the Kolmogorov-Smirnov (K-S) test.
    pretrain_size : int, optional (default=200)
        Size of the pre-training set.
    random_state : int, optional (random_state=113355)
        random_state controls the reproducibility of the results.

    Notes
    -----
    The Drift-Aware Multi-Memory Model (DAM3) [1] mitigates the class imbalance by
    - incorporating an imbalance-sensitive drift detector,
    - preserving a balanced representation of classes in the long-term memory,
    - resolving the retroactive interference using a working memory preventing the removal of old information,
    - and weighting the classifiers induced on different memories based on their balanced accuracy.

    References
    ----------
    .. [1] Amir Abolfazli and Eirini Ntoutsi.
    "Drift-Aware Multi-Memory Model for Imbalanced Data Streams."
    2020 IEEE International Conference on Big Data (Big Data). IEEE, 2020.
    """


    def __init__(self, n_neighbors=5,
                 max_window_size=5000,
                 min_stm_size=50,
                 ltm_size=0.5,
                 wm_size=0.2,
                 drift_detector_winSize=None,
                 drift_detector_thr=0.001,
                 pretrain_size=200,
                 random_state=113355
                ):
        super().__init__()
        self.n_neighbors = n_neighbors

        self.minSTMSize = min_stm_size
        self.max_wind_size = max_window_size
        self.ltm_size = ltm_size
        self.wm_size = wm_size

        self.maxLTMSize = self.ltm_size * self.max_wind_size
        self.maxWMSize = self.wm_size * self.max_wind_size
        self.maxSTMSize = self.max_wind_size - (self.maxLTMSize + self.maxWMSize)

        if drift_detector_winSize is not None:
            self.drift_detector_winSize = drift_detector_winSize
        else:
            self.drift_detector_winSize = min_stm_size
        self.drift_detector_thr = drift_detector_thr

        self.pretrain_size = pretrain_size
        self.random_state = random_state

        self._STMinstances = None
        self._STMLabels = np.empty(shape=(0), dtype=np.int32)
        self._LTMinstances = None
        self._LTMLabels = np.empty(shape=(0), dtype=np.int32)
        self._WMinstances = None
        self._WMLabels = np.empty(shape=(0), dtype=np.int32)

        self.historySize = self.minSTMSize

        self.TL_His = deque([], maxlen= self.historySize)
        self.STM_Pred_History = deque([], maxlen= self.historySize)
        self.LTM_Pred_History = deque([], maxlen= self.historySize)
        self.STM_LTM_Pred_History = deque([], maxlen= self.historySize)

        self.trainStepCount = 0

        self.STMSizes = []
        self.LTMSizes = []
        self.WMSizes = []

        self.classifierChoice = []

        self.classes = None
        self.FB_STM = None

        self.drift_detector = None

        self.p = -1
        self.X = None
        self.y = None

    @staticmethod
    def get_distances(instance, instances):
        """Calculates distances from an instance to all instances."""
        return libNearestNeighbor.get1ToNDistances(instance, instances)


    @staticmethod
    def distanceWightedLabel(distances, labels, numNeighbours):
        """Returns the the distance weighted label of the k nearest neighbors."""
        nnIndices = libNearestNeighbor.nArgMin(numNeighbours, distances)

        sqrtDistances = np.sqrt(distances[nnIndices])
        if not isinstance(labels, type(np.array([]))):
            labels = np.asarray(labels, dtype=np.int8)
        else:
            labels = np.int8(labels)

        predLabels = libNearestNeighbor.getLinearWeightedLabels(labels[nnIndices], sqrtDistances)

        return predLabels

    def compress(self, instances, labels):
        """Performs class-wise kMeans++ clustering for given instances with corresponding labels.
        The number of instances is halved per class."""
        logging.debug('cluster Down %d' % self.trainStepCount)
        uniqueLabels = np.unique(labels)
        newinstances = np.empty(shape=(0, instances.shape[1]))
        newLabels = np.empty(shape=(0), dtype=np.int32)
        # hh = [x for x in uniqueLabels if x != 1]
        for label in uniqueLabels:
            tmpinstances = instances[labels == label]
            newLength = int(max(tmpinstances.shape[0] / 2, 1))
            clustering = KMeans(n_clusters=newLength, n_init=1, random_state=self.random_state)
            clustering.fit(tmpinstances)
            newinstances = np.vstack([newinstances, clustering.cluster_centers_])
            newLabels = np.append(newLabels, label * np.ones(shape=newLength, dtype=np.int32))
        return newinstances, newLabels


    def memorySize_check(self):
        """Makes sure that the WM does not surpass the maximum size."""
        size_reduced = False

        """Makes sure that the CM does not exceed the maximum size."""
        if len(self._STMLabels) + len(self._LTMLabels) + len(self._WMLabels) > self.maxSTMSize + self.maxLTMSize + self.maxWMSize:
            if len(self._WMLabels) > self.maxWMSize:
                WMinstances_cleaned, WMLabels_cleaned = self.cleaning(self._WMinstances, self._WMLabels, onlyLast=False,
                                                                    clean_LTM=False)

                self._LTMinstances = np.vstack([self._LTMinstances, WMinstances_cleaned])
                self._LTMLabels = np.append(self._LTMLabels, WMLabels_cleaned)

                shiftRange = range(len(self._WMLabels))
                self._WMinstances = np.delete(self._WMinstances, shiftRange, 0)
                self._WMLabels = np.delete(self._WMLabels, shiftRange, 0)

            if len(self._LTMLabels) > self.maxLTMSize:
                self._LTMinstances, self._LTMLabels = self.compress(self._LTMinstances, self._LTMLabels)
            else:
                if len(self._STMLabels) + len(self._LTMLabels) + len(self._WMLabels) > self.maxSTMSize + self.maxLTMSize + self.maxWMSize:
                    numShifts = int(self.maxLTMSize - len(self._LTMLabels) + 1)
                    shiftRange = range(numShifts)
                    self._LTMinstances = np.vstack([self._LTMinstances, self._STMinstances[:numShifts, :]])
                    self._LTMLabels = np.append(self._LTMLabels, self._STMLabels[:numShifts])
                    self._LTMinstances, self._LTMLabels = self.compress(self._LTMinstances, self._LTMLabels)
                    self._STMinstances = np.delete(self._STMinstances, shiftRange, 0)
                    self._STMLabels = np.delete(self._STMLabels, shiftRange, 0)

        return size_reduced


    def overinstance_remainingSet(self, instances, labels, kind='borderline-1'):
        """Overinstances remaining set (using BorderlineSMOTE) after a drift is detected."""
        if len(np.unique(labels)) >= 2:
            minority_class = collections.Counter(labels.tolist()).most_common()[-1][0]

            if np.sum(labels == minority_class) > self.n_neighbors:
                overinstance = BorderlineSMOTE(k_neighbors=self.n_neighbors, m_neighbors=5, kind=kind, random_state=self.random_state)
                instances, labels = overinstance.fit_sample(instances, labels)

        return instances, labels

    def shorten_STM(self, ws):
        """Shortens the STM size when a drift is detected """
        numShifts = int(len(self._STMLabels) - ws)
        shiftRange = range(numShifts)

        oldWindowSize = len(self._STMLabels)
        newWindowSize = ws


        if newWindowSize < oldWindowSize:
            delrange = range(oldWindowSize - newWindowSize)
            oldSTMinstances = self._STMinstances[delrange, :]
            oldSTMLabels = self._STMLabels[delrange]

            self._STMinstances = np.delete(self._STMinstances, shiftRange, 0)
            self._STMLabels = np.delete(self._STMLabels, shiftRange, 0)

            oldSTMinstances, oldSTMLabels = self.overinstance_remainingSet(oldSTMinstances, oldSTMLabels)

            self._LTMinstances = np.vstack([self._LTMinstances, oldSTMinstances])
            self._LTMLabels = np.append(self._LTMLabels, oldSTMLabels)

        return True

    def cleaning(self, instancesCl, labelsCl, onlyLast=True, clean_LTM=True):
        """Removes distance-based all instances from the input instances that contradict those in the STM."""
        if len(self._STMLabels) > self.n_neighbors and instancesCl.shape[0] > 0:


            if onlyLast:
                loopRange = [len(self._STMLabels) - 1]
            else:
                loopRange = range(len(self._STMLabels))
            for i in loopRange:
                if len(labelsCl) == 0:
                    break
                instancesShortened = np.delete(self._STMinstances, i, 0)
                labelsShortened = np.delete(self._STMLabels, i, 0)
                distancesSTM = DAM3Classifier.get_distances(self._STMinstances[i, :], instancesShortened)
                nnIndicesSTM = libNearestNeighbor.nArgMin(self.n_neighbors, distancesSTM)[0]

                distancesLTM = DAM3Classifier.get_distances(self._STMinstances[i, :], instancesCl)
                nnIndicesLTM = libNearestNeighbor.nArgMin(min(len(distancesLTM), self.n_neighbors), distancesLTM)[0]


                correctIndicesSTM = nnIndicesSTM[labelsShortened[nnIndicesSTM] == self._STMLabels[i]]




                if len(correctIndicesSTM) > 0:
                    distThreshold = np.max(distancesSTM[correctIndicesSTM])

                    wrongIndicesLTM = nnIndicesLTM[labelsCl[nnIndicesLTM] != self._STMLabels[i]]
                    delIndices = np.where(distancesLTM[wrongIndicesLTM] <= distThreshold)[0]

                    if clean_LTM:
                        self._WMinstances = np.vstack([self._WMinstances, instancesCl[wrongIndicesLTM[delIndices]]])
                        self._WMLabels = np.append(self._WMLabels, labelsCl[wrongIndicesLTM[delIndices]])



                        distancesWM = DAM3Classifier.get_distances(self._STMinstances[i, :], self._WMinstances)
                        nnIndicesWM = \
                        libNearestNeighbor.nArgMin(min(len(distancesWM), self.n_neighbors), distancesWM)[0]

                        correctIndicesWM = nnIndicesWM[self._WMLabels[nnIndicesWM] == self._STMLabels[i]]
                        matchIndices = np.where(distancesWM[correctIndicesWM] <= distThreshold)[0]
                        if len(matchIndices) > 0:
                            self._LTMinstances = np.vstack([self._LTMinstances, self._WMinstances[correctIndicesWM[matchIndices]]])
                            self._LTMLabels = np.append(self._LTMLabels, self._WMLabels[correctIndicesWM[matchIndices]])

                            self._WMinstances = np.delete(self._WMinstances, correctIndicesWM[matchIndices], 0)
                            self._WMLabels = np.delete(self._WMLabels, correctIndicesWM[matchIndices], 0)


                    instancesCl = np.delete(instancesCl, wrongIndicesLTM[delIndices], 0)
                    labelsCl = np.delete(labelsCl, wrongIndicesLTM[delIndices], 0)
        return instancesCl, labelsCl


    def noise_removal(self, instances, labels):
        # The idea is to keep the WM inconsistent with LTM all the time.
        # Due to the transfer of instances from LTM (WM) to WM (LTM), some noisy instances might be formed
        # An instance is considered to be 'noisy' in WM, if it could be correctly classified by LTM classifier.

        if len(self._LTMLabels) > self.n_neighbors and instances.shape[0] > 0:
            idx_l = []
            loopRange = range(len(labels))
            for i in loopRange:
                if len(labels) == 0:
                    break
                pred, case = self._predict_by_LTM(instances[i, :])

                if pred == labels[i] and case:
                    idx_l.append(i)

            if len(idx_l) > 0:
                self._LTMinstances = np.vstack([self._LTMinstances, instances[np.array(idx_l)]])
                self._LTMLabels = np.append(self._LTMLabels, labels[np.array(idx_l)])

                instances = np.delete(instances, np.array(idx_l), 0)
                labels = np.delete(labels, np.array(idx_l), 0)

        return instances, labels


    def partial_fit(self, X, y, classes=None, instance_weight=None):

        r, c = get_dimensions(X)

        if self._STMinstances is None:
            self._STMinstances = np.empty(shape=(0, c))
            self._LTMinstances = np.empty(shape=(0, c))
            self._WMinstances = np.empty(shape=(0, c))


            self.drift_detector = DAM3_DriftDetector(window=self.drift_detector_winSize, thr=self.drift_detector_thr)


        if self.p == -1:
            self.X = np.zeros((self.pretrain_size, c))
            self.y = np.zeros(self.pretrain_size)
            self.p = 0

            # fill up the datasets chunk
            for i, x in enumerate(X):
                self.X[self.p] = X[i]
                self.y[self.p] = y[i]
                self.p += 1

            if self.p == self.pretrain_size:
                # reset the pointer
                self.p = -5
                data = np.hstack((self.X, self.y.reshape((-1, 1))))
                mlnd_0 = MultiNormalClassDistribution(data, 0)
                mlnd_1 = MultiNormalClassDistribution(data, 1)
                self.FB_STM = MAP(mlnd_0, mlnd_1)

        for i in range(r):
            new_pred = self.FB_STM.predict(X[i, :].reshape(1, -1))

            self.drift_detector.add_prediction(new_pred)
            self.drift_detector.add_trueLabel(y[i])
            self.drift_detector.update_balancedAccuracy()

            if self.drift_detector.detected_drift():
                self.shorten_STM(self.minSTMSize)

            self._partial_fit(X[i, :], y[i])

        return self


    def _partial_fit(self, x, y):
        """Processes a new instance."""
        distancesSTM = DAM3Classifier.get_distances(x, self._STMinstances)

        self._partial_fit_by_all_memories(x, y, distancesSTM)
        # self._partial_fit_by_GM(x, y,distancesSTM)

        self.trainStepCount += 1

        self._STMinstances = np.vstack([self._STMinstances, x])
        self._STMLabels = np.append(self._STMLabels, y)
        self._WMinstances, self._WMLabels = self.noise_removal(self._WMinstances, self._WMLabels)
        self._LTMinstances, self._LTMLabels = self.cleaning(self._LTMinstances, self._LTMLabels, onlyLast=True, clean_LTM=True)

        self.memorySize_check()

        self.STMSizes.append(len(self._STMLabels))
        self.LTMSizes.append(len(self._LTMLabels))
        self.WMSizes.append(len(self._WMLabels))


    def _partial_fit_by_all_memories(self, instance, label, distancesSTM):
        """ Updating the prediction history by predicting the label of a given instance by using the STM, LTM and the CM."""
        predictedLabelLTM = 0
        predictedLabelWM = 0
        predictedLabelSTM_WM = 0
        predictedLabelSTM = 0
        predictedLabelSTM_LTM = 0

        predictedLabelGNB = 0

        classifierChoice = 0
        predictedLabel = 0

        if len(self._STMLabels) == 0:
            predictedLabel = predictedLabelSTM

        else:
            if len(self._STMLabels) < self.n_neighbors or len(np.unique(self._STMLabels)) < 2:
                predictedLabelSTM = DAM3Classifier.distanceWightedLabel(distancesSTM, self._STMLabels, len(self._STMLabels))[0]
                predictedLabel = predictedLabelSTM
            else:

                temp_data = np.hstack((self._STMinstances, self._STMLabels.reshape((-1, 1))))
                mlnd_0 = MultiNormalClassDistribution(temp_data, 0)
                mlnd_1 = MultiNormalClassDistribution(temp_data, 1)
                self.FB_STM = MAP(mlnd_0, mlnd_1)
                predictedLabelSTM = self.FB_STM.predict(instance.reshape(1, -1))

                if len(self._LTMLabels) >= self.n_neighbors and len(self._WMLabels) >= self.n_neighbors:

                    distancesLTM = DAM3Classifier.get_distances(instance, self._LTMinstances)

                    predictedLabelSTM_LTM = \
                        DAM3Classifier.distanceWightedLabel(np.append(distancesSTM, distancesLTM),
                                                            np.append(self._STMLabels, self._LTMLabels),
                                                            self.n_neighbors)[0]

                    predictedLabelLTM = DAM3Classifier.distanceWightedLabel(distancesLTM, self._LTMLabels, self.n_neighbors)[0]
                    correctSTM = Performance.BalancedAccuracy(self.TL_His, self.STM_Pred_History)
                    correctLTM = Performance.BalancedAccuracy(self.TL_His, self.LTM_Pred_History)
                    correctSTM_LTM = Performance.BalancedAccuracy(self.TL_His, self.STM_LTM_Pred_History)

                    classifierChoice = np.argmax([correctSTM, correctLTM, correctSTM_LTM])


        self.TL_His.append(label)
        self.STM_Pred_History.append(predictedLabelSTM)
        self.LTM_Pred_History.append(predictedLabelLTM)
        self.STM_LTM_Pred_History.append(predictedLabelSTM_LTM)
        self.classifierChoice.append(classifierChoice)


    def predict(self, X):
        r, c = get_dimensions(X)
        predicted_label = []

        mem_choice = {
            0: "STM",
            1: "LTM",
            2: "STM_LTM"
        }

        if self._STMinstances is None:
            self._STMinstances = np.empty(shape=(0, c))
            self._LTMinstances = np.empty(shape=(0, c))
            self._WMinstances = np.empty(shape=(0, c))

        for i in range(r):
            distancesSTM = DAM3Classifier.get_distances(X[i], self._STMinstances)
            pred, mem = self._predict_by_all_memories(X[i], distancesSTM)
            predicted_label.append(pred)


        return np.asarray(predicted_label)


    def _predict_by_all_memories(self, instance, distancesSTM):
        predictedLabelLTM = 0
        predictedLabelSTM = 0
        predictedLabelSTM_LTM = 0

        classifierChoice = 0
        predictedLabel = None

        if len(self._STMLabels) == 0:
            predictedLabel = predictedLabelSTM
            classifierChoice = 0

        else:

            if len(self._STMLabels) < self.n_neighbors or len(np.unique(self._STMLabels)) < 2:
                predictedLabelSTM = DAM3Classifier.distanceWightedLabel(distancesSTM, self._STMLabels, len(self._STMLabels))[0]
                predictedLabel = predictedLabelSTM
                classifierChoice = 0
            else:
                temp_data = np.hstack((self._STMinstances, self._STMLabels.reshape((-1, 1))))
                mlnd_0 = MultiNormalClassDistribution(temp_data, 0)
                mlnd_1 = MultiNormalClassDistribution(temp_data, 1)
                self.FB_STM = MAP(mlnd_0, mlnd_1)
                predictedLabelSTM = self.FB_STM.predict(instance.reshape(1, -1))


                if len(self._LTMLabels) >= self.n_neighbors and len(self._WMLabels) >= self.n_neighbors:
                    distancesLTM = DAM3Classifier.get_distances(instance, self._LTMinstances)
                    predictedLabelLTM = DAM3Classifier.distanceWightedLabel(distancesLTM, self._LTMLabels, self.n_neighbors)[0]


                    distances_new = cp.deepcopy(distancesSTM)
                    stm_labels_new = cp.deepcopy(self._STMLabels)
                    predictedLabelSTM_LTM = \
                        DAM3Classifier.distanceWightedLabel(np.append(distances_new, distancesLTM),
                                                            np.append(stm_labels_new, self._LTMLabels),
                                                            self.n_neighbors)[0]

                    correctSTM = Performance.BalancedAccuracy(self.TL_His, self.STM_Pred_History)
                    correctLTM = Performance.BalancedAccuracy(self.TL_His, self.LTM_Pred_History)
                    correctSTM_LTM = Performance.BalancedAccuracy(self.TL_His, self.STM_LTM_Pred_History)

                    labels = [predictedLabelSTM, predictedLabelLTM, predictedLabelSTM_LTM]

                    classifierChoice = np.argmax([correctSTM, correctLTM, correctSTM_LTM])

                    predictedLabel = labels[int(classifierChoice)]

                else:
                    predictedLabel = predictedLabelSTM
                    classifierChoice = 0

        return predictedLabel, classifierChoice


    def _predict_by_LTM(self, instance):
        predictedLabelLTM = 0
        distancesLTM = DAM3Classifier.get_distances(instance, self._LTMinstances)

        case = False

        if len(self._LTMLabels) >= self.n_neighbors:
            predictedLabelLTM = DAM3Classifier.distanceWightedLabel(distancesLTM, self._LTMLabels, self.n_neighbors)[0]
            case = True
        return predictedLabelLTM, case


    def predict_proba(self, X):
        raise NotImplementedError


class Performance(object):
    """Utility class to claculate the performance"""

    @staticmethod
    def BalancedAccuracy(labels, predLabels):
        """Calculates the balanced accuracy."""
        return balanced_accuracy_score(labels, predLabels)

    @staticmethod
    def GMean(labels, predLabels):
        """Calculates the geometric mean."""
        return geometric_mean_score(labels, predLabels)


class DAM3_DriftDetector(object):
    # Imbalance-sensitive drift detector
    def __init__(self, window, thr):
        self.balanced_accuracy_history  = []
        self.predictions = []
        self.true_labels = []
        self.window = window
        self.threshold = thr
        self.idx = 0

        self.data = []

    def add_data(self, instance):
        self.data.append(instance)

    def add_prediction(self, instance):
        self.predictions.append(instance)

    def add_trueLabel(self, instance):
        self.true_labels.append(instance)

    def update_balancedAccuracy(self):
        bacc = Performance.BalancedAccuracy(self.true_labels, self.predictions)
        self.balanced_accuracy_history.append(round(bacc, 3))

    def detected_drift(self):
        bacc_values = np.array(self.balanced_accuracy_history)
        window = self.window

        if len(bacc_values) < 2 * window:
            self.idx += 1
            return (False)

        newConcept_window = bacc_values[-window:]
        prevConept_window = bacc_values[-(window * 2):-window]

        ht = scipy.stats.ks_2samp(newConcept_window, prevConept_window)
        pval = ht[1]
        has_change = pval < self.threshold

        if has_change:
            self.idx += 1
            self.balanced_accuracy_history = list(bacc_values[-window:])
            return (True)
        else:
            self.idx += 1
            return (False)



    def reset(self):
        """ reset

        Resets the change detector parameters.
        """
        self.balanced_accuracy_history  = []
        self.predictions = []
        self.true_labels = []
        self.idx = 0
        self.data = []



# Implementation of FullBayes Classifier
#### The full Bayes classifier assumes that the distribution of data
#### can be modeled with a multivariate Gaussian distribution

class MultiNormalClassDistribution():
    def __init__(self, data, class_label):
        """
        A class that encapsulates the relevant parameters (mean, covariance matrix) for a class conditinoal multivariate normal distribution.
        The mean and covariance matrix will be computed from a given datasets set.

        Input
        - dataset: The dataset to compute the stats
        - class_label : The class label
        """
        self.data = data
        self.class_label = class_label
        class_rows = data[data[:, -1] == class_label]
        self.mean = []
        count = np.shape(class_rows)[0]
        for param in range(0, (np.shape(class_rows)[1] - 1)):
            self.mean.append(np.sum(class_rows[:, param]) / count)

        self.cov = np.cov(class_rows[:, 0:-1], rowvar=False)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        class_column = self.data[:, -1]
        freq = np.count_nonzero(class_column == self.class_label)
        count = np.shape(class_column)[0]
        return freq / count

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        return multi_normal_pdf(x, self.mean, self.cov)

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior() * self.get_instance_likelihood(x)

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multivariable normal desnity function for a given x, mean and covarince matrix.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - cov:  The covariance matrix.

    Returns the normal distribution pdf according to the given mean and var for the given x.
    """
    var = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
    return var.pdf(x)


class MAP:
    def __init__(self, cd0, cd1):
        """
        A Maximum a postreiori classifier.
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predict the instance
        by the class that outputs the highest posterior probability for that given instance.
        Input
            - ccd0 : An object containing the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object containing the relevant parameters and methods for the distribution of class 1.
        """
        # class distribution of class 0
        self.cd0 = cd0
        # class distribution of class 1
        self.cd1 = cd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects.
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher; 1 otherwise.
        """
        return 1 if self.cd0.get_instance_posterior(x) < self.cd1.get_instance_posterior(x) else 0


