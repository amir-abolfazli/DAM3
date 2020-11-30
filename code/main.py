import matplotlib
matplotlib.use('TkAgg')

from skmultiflow.data import FileStream
from skmultiflow.meta.online_rus_boost import OnlineRUSBoostClassifier
from skmultiflow.meta.online_adac2 import OnlineAdaC2Classifier
from skmultiflow.meta.leverage_bagging import LeveragingBaggingClassifier
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingADWINClassifier
from skmultiflow.lazy import SAMKNNClassifier

from meta.dam3 import DAM3Classifier
from evaluation import EvaluatePrequential

r_state = 113355


if __name__ == '__main__':

    # Setup the File Stream

    ################### Real-world datasets ###################
    stream = FileStream('datasets/real/weather.csv')
    # stream = FileStream('./datasets/real/electricity.csv')
    # stream = FileStream('./datasets/real/pima.csv')

    ################### Synthetic datasets ###################
    # stream = FileStream('./datasets/synthetic/HyperFast.csv')
    # stream = FileStream('./datasets/synthetic/HyperSlow.csv')
    # stream = FileStream('./datasets/synthetic/SEA_S.csv')
    # stream = FileStream('./datasets/synthetic/SEA_G.csv')




    OBA = OzaBaggingADWINClassifier(random_state=r_state)
    LB = LeveragingBaggingClassifier(random_state=r_state)
    ORUSBoost = OnlineRUSBoostClassifier(random_state=r_state)
    OAdaC2 = OnlineAdaC2Classifier(random_state=r_state)
    samknn = SAMKNNClassifier(n_neighbors=5,
                              min_stm_size=50,
                              max_window_size=5000)
    dam3 = DAM3Classifier(n_neighbors=5,
                          min_stm_size=50,
                          wm_size=0.3,
                          ltm_size=0.5,
                          max_window_size=5000,
                          drift_detector_winSize=100,
                          drift_detector_thr=0.001,
                          pretrain_size=200,
                          random_state=r_state
                          )


    models = [dam3, samknn]
    models_names = ["DAM3", "SAMkNN"]

    # models = [dam3, samknn, OAdaC2, ORUSBoost ,LB, OBA]
    # models_names = ["DAM3", "SAMkNN", "OAdaC2", "ORUSBoost","LB", "OBA"]



    # Setup the evaluator

    evaluator = EvaluatePrequential(max_samples=100000,
                                    max_time=10000000000,
                                    pretrain_size=200,
                                    show_plot=True,
                                    metrics=['balanced_accuracy', 'gmean'])
                                    # metrics=['balanced_accuracy', 'gmean', 'recall', 'specificity'])

    # Evaluate
    evaluator.evaluate(stream=stream, model=models, model_names=models_names)


