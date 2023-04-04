import os
from datetime import datetime

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def check_if_model_exist():
    if os.path.exists("TrainedModels/knn_model.pkl"):
        print("Model already exist")
        return True
    else:
        print("Model does not exist i will train it")
        return False


"""
Sınıfı bilinmeyen verilerin, eğitim setindeki diğer veriler ile karşılaştırılıp bir uzaklık ölçümü 
gerçekleştirilmesi sonucu hesaplanan uzaklığa göre henüz bir sınıfa atanamamış verinin, en ideal 
(optimal) sınıfa atanarak sınıflandırılması olarak ifade edilmektedir.
"""


class MachineLearning():

    def __init__(self):
        print("Loading dataset ...")

        self.flow_dataset = pd.read_csv('../controller/data/FlowStatsfile.csv')
        # get flow_id column and replace . with nothing
        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace(
            '.', '', regex=False)
        self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace(
            '.', '', regex=False)
        self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace(
            '.', '', regex=False)

    def flow_training(self):
        print("Flow Training ...")

        X_flow = self.flow_dataset.iloc[:,
                 :-1].values  # get all columns except the last one , the last one is label which show is it ddos on normal traffic
        X_flow = X_flow.astype('float64')  # convert to float64

        # get the last column which is label
        y_flow = self.flow_dataset.iloc[:, -1].values

        # split the dataset into training and testing set , test_size = 0.25 means 25% of the dataset will be used for testing
        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25,
                                                                                random_state=0)

        # create the model
        classifier = KNeighborsClassifier(
            n_neighbors=5, metric='minkowski', p=2)

        # check if model exist
        if check_if_model_exist():
            # load the model
            flow_model = joblib.load('TrainedModels/knn_model.pkl')
        else:
            # train the model
            flow_model = classifier.fit(X_flow_train, y_flow_train)
            # save the model
            joblib.dump(flow_model, 'TrainedModels/knn_model.pkl')

        # test the model
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

        # plot the confusion matrix
        x = ['TP', 'FP', 'FN', 'TN']
        plt.title("KNN")
        plt.xlabel('Prediction Class')
        plt.ylabel('Number of Flows')
        plt.tight_layout()
        plt.style.use("seaborn-v0_8-darkgrid")
        y = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        plt.bar(x, y, color="#e46e6e", label='KNN')
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
