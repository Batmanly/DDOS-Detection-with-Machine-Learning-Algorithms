import os
from datetime import datetime

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

"""
Linear Regression Nedir ? Linear regresyon, gözlemlenen verilere doğrusal bir denklem uydurarak iki
 değişken arasındaki ilişkiyi modellemeye çalışır. Bir başka deyişle en uygun çizgiyi kullanarak, giriş
  değeri olan x ve sonuç değeri olan y arasında uygun bir ilişki kurarak tahmin yapmamızı sağlar
"""


def check_if_model_exist():
    if os.path.exists("TrainedModels/lr_model.pkl"):
        print("Model already exist")
        return True
    else:
        print("Model does not exist i will train it")
        return False


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
        classifier = LogisticRegression(solver='liblinear', random_state=0)

        # check if model exist
        if check_if_model_exist():
            # load the model
            flow_model = joblib.load('TrainedModels/lr_model.pkl')
        else:
            # train the model
            flow_model = classifier.fit(X_flow_train, y_flow_train)
            # save the model
            joblib.dump(flow_model, 'TrainedModels/lr_model.pkl')

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

        benin = 0
        ddos = 0
        for i in y_flow:
            if i == 0:
                benin += 1
            elif i == 1:
                ddos += 1

        print("benin = ", benin)
        print("ddos = ", ddos)
        print(
            "------------------------------------------------------------------------------")
        #  code counts the number of "normal" and "DDoS" flows in the dataset and creates
        #  a bar chart and pie chart to visualize the results.

        plt.title("Dataset")
        plt.xlabel('Flow Type')
        plt.ylabel('Number of Flows')
        plt.tight_layout()
        # plt.style.use("seaborn-v0_8-darkgrid")
        plt.bar(['NORMAL', 'DDOS'], [benin, ddos])
        # plt.legend()
        plt.show()

        # code counts the number of flows for each type of protocol (TCP, UDP, ICMP) and
        # creates a pie chart to visualize the results.
        explode = [0, 0.1]

        plt.pie([benin, ddos], labels=['NORMAL', 'DDoS'], wedgeprops={
            'edgecolor': 'black'}, explode=explode, autopct="%1.2f%%")
        plt.show()

        # code counts the number of "normal" and "DDoS" flows for each type of protocol
        # (TCP, UDP, ICMP) and creates a pie chart to visualize the results.
        icmp = 0
        tcp = 0
        udp = 0

        proto = self.flow_dataset.iloc[:, 7].values
        proto = proto.astype('int')
        for i in proto:
            if i == 6:
                tcp += 1
            elif i == 17:
                udp += 1
            elif i == 1:
                icmp += 1

        print("tcp = ", tcp)
        print("udp = ", udp)
        print("icmp = ", icmp)

        plt.title("Dataset")
        plt.xlabel('Protocols')
        plt.ylabel('Number of Flows')
        plt.tight_layout()
        # plt.style.use("seaborn-v0_8-darkgrid")
        plt.bar(['ICMP', 'TCP', 'UDP'], [icmp, tcp, udp])
        # plt.legend()
        plt.show()

        explode = [0, 0.1, 0.1]

        plt.pie([icmp, tcp, udp], labels=['ICMP', 'TCP', 'UDP'], wedgeprops={
            'edgecolor': 'black'}, explode=explode, autopct="%1.2f%%")
        plt.show()

        #

        icmp_normal = 0
        tcp_normal = 0
        udp_normal = 0
        icmp_ddos = 0
        tcp_ddos = 0
        udp_ddos = 0

        proto = self.flow_dataset.iloc[:, [7, -1]].values
        proto = proto.astype('int')

        for i in proto:
            if i[0] == 6 and i[1] == 0:
                tcp_normal += 1
            elif i[0] == 6 and i[1] == 1:
                tcp_ddos += 1

            if i[0] == 17 and i[1] == 0:
                udp_normal += 1
            elif i[0] == 17 and i[1] == 1:
                udp_ddos += 1

            if i[0] == 1 and i[1] == 0:
                icmp_normal += 1
            elif i[0] == 1 and i[1] == 1:
                icmp_ddos += 1

        print("tcp_normal = ", tcp_normal)
        print("tcp_ddos = ", tcp_ddos)
        print("udp_normal = ", udp_normal)
        print("udp_ddos = ", udp_ddos)
        print("icmp_normal = ", icmp_normal)
        print("icmp_ddos = ", icmp_ddos)
        # create a line graph of the network traffic flows over time.
        # display number of flows for different protocls
        plt.title("Dataset")
        plt.xlabel('Protocols')
        plt.ylabel('Number of Flows')
        plt.tight_layout()
        # plt.style.use("seaborn-v0_8-darkgrid")
        plt.bar(['ICMP_N', 'ICMP_D', 'TCP_N', 'TCP_D', 'UDP_N', 'UDP_D'],
                [icmp_normal, icmp_ddos, tcp_normal, tcp_ddos, udp_normal, udp_ddos])
        # plt.legend()
        plt.show()

        explode = [0, 0.1, 0.1, 0.1, 0.1, 0.1]

        plt.pie([icmp_normal, icmp_ddos, tcp_normal, tcp_ddos, udp_normal, udp_ddos],
                labels=['ICMP_Normal', 'ICMP_DDoS', 'TCP_Normal',
                        'TCP_DDoS', 'UDP_Normal', 'UDP_DDoS'],
                wedgeprops={'edgecolor': 'black'}, explode=explode, autopct="%1.2f%%")
        plt.show()

        # be creating a bar chart or pie chart to visualize the results of a logistic
        # regression algorithm.
        plt.title("Dataset")
        plt.xlabel('generation time')
        plt.ylabel('flow type')
        plt.tight_layout()
        # plt.style.use("seaborn-v0_8-darkgrid")
        plt.plot(X_flow[:, 0], y_flow)
        # plt.legend()
        plt.show()

        # creating a bar chart and a pie chart to visualize a confusion matrix for a
        # logistic regression model.
        x = ['TP', 'FP', 'FN', 'TN']
        plt.title("logistic regression")
        plt.xlabel('Prediction Class')
        plt.ylabel('Number of Flows')
        plt.tight_layout()
        # plt.style.use("seaborn-v0_8-darkgrid")
        y = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        plt.bar(x, y, color="#1b7021", label='LR')
        # plt.legend()
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
