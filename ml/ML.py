import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class MachineLearning():

    def __init__(self):

        print("Loading dataset ...")

        self.counter = 0

        self.flow_dataset = pd.read_csv('../controller/data/FlowStatsfile.csv')

        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace(
            '.', '', regex=False)
        self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace(
            '.', '', regex=False)
        self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace(
            '.', '', regex=False)

        self.X_flow = self.flow_dataset.iloc[:, :-1].values
        self.X_flow = self.X_flow.astype('float64')

        self.y_flow = self.flow_dataset.iloc[:, -1].values

        self.X_flow_train, self.X_flow_test, self.y_flow_train, self.y_flow_test = train_test_split(self.X_flow,
                                                                                                    self.y_flow,
                                                                                                    test_size=0.25,
                                                                                                    random_state=0)

    def check_if_model_exist(sel, file_name):
        if os.path.exists(file_name):
            print("Model already exist")
            return True
        else:
            print("Model does not exist i will train it")
            return False

    def LR(self):

        print(
            "------------------------------------------------------------------------------")
        print("Logistic Regression ...")

        self.classifier = LogisticRegression(
            solver='liblinear', random_state=0)
        if self.check_if_model_exist("TrainedModels/lr_model.pkl"):
            self.flow_model = joblib.load("TrainedModels/lr_model.pkl")
        else:
            self.flow_model = self.classifier.fit(
                self.X_flow_train, self.y_flow_train)
            joblib.dump(self.flow_model, "TrainedModels/lr_model.pkl")
        self.Confusion_matrix(self.flow_model)

    def KNN(self):

        print(
            "------------------------------------------------------------------------------")
        print("K-NEAREST NEIGHBORS ...")

        self.classifier = KNeighborsClassifier(
            n_neighbors=5, metric='minkowski', p=2)
        if self.check_if_model_exist("TrainedModels/knn_model.pkl"):
            self.flow_model = joblib.load("TrainedModels/knn_model.pkl")
        else:
            self.flow_model = self.classifier.fit(
                self.X_flow_train, self.y_flow_train)
            joblib.dump(self.flow_model, "TrainedModels/knn_model.pkl")
        self.Confusion_matrix(self.flow_model)

    def SVM(self):

        print(
            "------------------------------------------------------------------------------")
        print("SUPPORT-VECTOR MACHINE ...")

        self.classifier = SVC(kernel='rbf', random_state=0)
        if self.check_if_model_exist("TrainedModels/svm_model.pkl"):
            self.flow_model = joblib.load("TrainedModels/svm_model.pkl")
        else:
            self.flow_model = self.classifier.fit(
                self.X_flow_train, self.y_flow_train)
            joblib.dump(self.flow_model, "TrainedModels/svm_model.pkl")
        self.Confusion_matrix(self.flow_model)

    def NB(self):

        print(
            "------------------------------------------------------------------------------")
        print("NAIVE-BAYES ...")

        self.classifier = GaussianNB()
        if self.check_if_model_exist("TrainedModels/nb_model.pkl"):
            self.flow_model = joblib.load("TrainedModels/nb_model.pkl")
        else:
            self.flow_model = self.classifier.fit(
                self.X_flow_train, self.y_flow_train)
            joblib.dump(self.flow_model, "TrainedModels/nb_model.pkl")
        self.Confusion_matrix(self.flow_model)

    def DT(self):

        print(
            "------------------------------------------------------------------------------")
        print("DECISION TREE ...")

        self.classifier = DecisionTreeClassifier(
            criterion='entropy', random_state=0)
        if self.check_if_model_exist("TrainedModels/dt_model.pkl"):
            self.flow_model = joblib.load("TrainedModels/dt_model.pkl")
        else:
            self.flow_model = self.classifier.fit(
                self.X_flow_train, self.y_flow_train)
            joblib.dump(self.flow_model, "TrainedModels/dt_model.pkl")
        self.Confusion_matrix(self.flow_model)

    def RF(self):

        print(
            "------------------------------------------------------------------------------")
        print("RANDOM FOREST ...")

        self.classifier = RandomForestClassifier(
            n_estimators=10, criterion="entropy", random_state=0)
        if self.check_if_model_exist("TrainedModels/rf_model.pkl"):
            self.flow_model = joblib.load("TrainedModels/rf_model.pkl")
        else:
            self.flow_model = self.classifier.fit(
                self.X_flow_train, self.y_flow_train)
            joblib.dump(self.flow_model, "TrainedModels/rf_model.pkl")
        self.Confusion_matrix(self.flow_model)

    def Confusion_matrix(self, flow_model):
        self.counter += 1

        # self.flow_model = self.classifier.fit(self.X_flow_train, self.y_flow_train)
        self.flow_model = flow_model
        self.y_flow_pred = self.flow_model.predict(self.X_flow_test)

        print(
            "------------------------------------------------------------------------------")

        print("confusion matrix")
        cm = confusion_matrix(self.y_flow_test, self.y_flow_pred)
        print(cm)

        acc = accuracy_score(self.y_flow_test, self.y_flow_pred)

        print("succes accuracy = {0:.2f} %".format(acc * 100))
        fail = 1.0 - acc
        print("fail accuracy = {0:.2f} %".format(fail * 100))
        print(
            "------------------------------------------------------------------------------")

        x = ['TP', 'FP', 'FN', 'TN']
        x_indexes = np.arange(len(x))
        width = 0.10
        plt.xticks(ticks=x_indexes, labels=x)
        plt.title("Algorithm results ")
        plt.xlabel('Prediction Class')
        plt.ylabel('Number of streams')
        plt.tight_layout()
        plt.style.use("seaborn-v0_8-darkgrid")
        # plt.style.use("dark_background")
        # plt.style.use("ggplot")
        if self.counter == 1:
            y1 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
            plt.bar(x_indexes - 2 * width, y1, width=width,
                    color="#1b7021", label='LR')
            plt.legend()
        if self.counter == 2:
            y2 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
            plt.bar(x_indexes - width, y2, width=width,
                    color="#e46e6e", label='KNN')
            plt.legend()
        if self.counter == 3:
            y3 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
            plt.bar(x_indexes, y3, width=width, color="#000999", label='SVM')
            plt.legend()
        if self.counter == 4:
            y4 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
            plt.bar(x_indexes, y4, width=width, color="#0000ff", label='NB')
            plt.legend()
        if self.counter == 5:
            y5 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
            plt.bar(x_indexes + width, y5, width=width,
                    color="#e0d692", label='DT')
            plt.legend()
        if self.counter == 6:
            y6 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
            plt.bar(x_indexes + 2 * width, y6, width=width,
                    color="#000000", label='RF')
            plt.legend()
            plt.show()


def main():
    start_script = datetime.now()

    ml = MachineLearning()

    start = datetime.now()
    ml.LR()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    start = datetime.now()
    ml.KNN()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    start = datetime.now()
    ml.SVM()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    start = datetime.now()
    ml.NB()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    start = datetime.now()
    ml.DT()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    start = datetime.now()
    ml.RF()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    end_script = datetime.now()
    print("Script Time: ", (end_script - start_script))


if __name__ == "__main__":
    main()
