import os
from datetime import datetime

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

"""
P(A|B) = B’nin doğru olduğu bilindiğinde A’nın olma olasılığı

P(B|A) = A’nın doğru olduğu bilindiğinde B’nin olma olasılığı

P(A) = A’nın olma olasılığı

P(B) = B’nin olma olasılığı

Naive Bayes 18.yy’da Thomas Bayes’in Bayes Teoremi temel alarak geliştirilimiştir. 
Hemen bir örnekle açıklayalım.


"""


def check_if_model_exist():
    if os.path.exists("TrainedModels/nb_model.pkl"):
        print("Model already exist")
        return True
    else:
        print("Model does not exist i will train it")
        return False


class MachineLearning():

    def __init__(self):
        print("Loading dataset ...")

        self.flow_dataset = pd.read_csv('../controller/data/FlowStatsfile.csv')

        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace(
            '.', '', regex=False)
        self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace(
            '.', '', regex=False)
        self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace(
            '.', '', regex=False)

    def flow_training(self):
        print("Flow Training ...")

        X_flow = self.flow_dataset.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')

        y_flow = self.flow_dataset.iloc[:, -1].values

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25,
                                                                                random_state=0)

        # create the model
        classifier = GaussianNB()

        # check if model exist
        if check_if_model_exist():
            # load the model
            flow_model = joblib.load('TrainedModels/nb_model.pkl')
        else:
            # train the model
            flow_model = classifier.fit(X_flow_train, y_flow_train)
            # save the model
            joblib.dump(flow_model, 'TrainedModels/nb_model.pkl')

        # predict the test set results
        y_flow_pred = flow_model.predict(X_flow_test)

        print(
            "------------------------------------------------------------------------------")

        print("confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        print(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)

        print("succes accuracy = {0:.2f} %".format(acc * 100))
        fail = 1.0 - acc
        print("fail accuracy = {0:.2f} %".format(fail * 100))
        print(
            "------------------------------------------------------------------------------")

        x = ['TP', 'FP', 'FN', 'TN']
        plt.title("Naive Bayes")
        plt.xlabel('Prediction Class')
        plt.ylabel('Number of Flows')
        plt.tight_layout()
        plt.style.use("seaborn-v0_8-darkgrid")
        y = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        plt.bar(x, y, color="#0000ff", label='NB')
        plt.legend()
        plt.show()

        plt.pie(y, labels=x, wedgeprops={'edgecolor': 'black'})
        plt.show()
def main():
    start = datetime.now()

    ml = MachineLearning()
    ml.flow_training()

    end = datetime.now()
    print("Training time: ", (end - start))


if __name__ == "__main__":
    main()
