import os
import sys
from datetime import datetime

import joblib
import pandas as pd
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import switch


# load and make prediction or just load model , don't load dataset and don't make prediction if it's already trained.
def get_command_variables():
    return os.getenv('RYU_ARG')


class SimpleMonitor13(switch.SimpleSwitch13):
    # SimpleSwitch13 class, and creates an empty dictionary self.datapaths to store information
    # about switches connected to the controller. It also creates a hub thread to run the
    # _monitor method which will periodically request flow statistics from the connected switches,
    # and predict the flows with a machine learning model. Additionally, it calls the flow_training
    # method to train the machine learning model.
    def __init__(self, *args, **kwargs):

        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.flow_model = None
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        start = datetime.now()

        self.flow_training()

        end = datetime.now()
        print("Training time: ", (end - start))

    # Ryu event handler that gets called whenever there is a change in the state of a switch connected to the controller.
    # If the switch is newly connected, it is added to the self.datapaths dictionary,
    # and if it has been disconnected, it is removed from the dictionary.
    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    # method is the main loop of the monitoring module. It requests flow statistics from each switch in self.
    # datapaths dictionary every 10 seconds, and calls the
    # flow_predict method to predict the flows with the trained machine learning model.
    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)

            self.flow_predict()

    # The _request_stats method sends a flow statistics request to a switch.
    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    # Ryu event handler that gets called whenever a flow statistics reply message is received from a switch.
    # It processes the flow statistics data, and saves it to a CSV file named PredictFlowStatsfile.csv.
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):

        timestamp = datetime.now()
        timestamp = timestamp.timestamp()

        file0 = open("data/PredictFlowStatsfile.csv", "w")
        file0.write(
            'timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
        body = ev.msg.body
        icmp_code = -1
        icmp_type = -1
        tp_src = 0
        tp_dst = 0

        for stat in sorted([flow for flow in body if (flow.priority == 1)], key=lambda flow:
        (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):

            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']

            if stat.match['ip_proto'] == 1:
                icmp_code = stat.match['icmpv4_code']
                icmp_type = stat.match['icmpv4_type']

            elif stat.match['ip_proto'] == 6:
                tp_src = stat.match['tcp_src']
                tp_dst = stat.match['tcp_dst']

            elif stat.match['ip_proto'] == 17:
                tp_src = stat.match['udp_src']
                tp_dst = stat.match['udp_dst']
            # The flow_id is a unique identifier for each flow.
            # It is created by concatenating the source IP address,
            flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + \
                      str(tp_dst) + str(ip_proto)

            try:
                packet_count_per_second = stat.packet_count / stat.duration_sec
                packet_count_per_nsecond = stat.packet_count / stat.duration_nsec
            except:
                packet_count_per_second = 0
                packet_count_per_nsecond = 0

            try:
                byte_count_per_second = stat.byte_count / stat.duration_sec
                byte_count_per_nsecond = stat.byte_count / stat.duration_nsec
            except:
                byte_count_per_second = 0
                byte_count_per_nsecond = 0

            file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                        .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src, ip_dst, tp_dst,
                                stat.match['ip_proto'], icmp_code, icmp_type,
                                stat.duration_sec, stat.duration_nsec,
                                stat.idle_timeout, stat.hard_timeout,
                                stat.flags, stat.packet_count, stat.byte_count,
                                packet_count_per_second, packet_count_per_nsecond,
                                byte_count_per_second, byte_count_per_nsecond))

        file0.close()

    # Start Training , training model
    def flow_training(self):
        classifier = DecisionTreeClassifier(
            criterion='entropy', random_state=0)

        if 'train' == get_command_variables():
            self.logger.info('Training Model ... , if model doesn\' \nexist it will be created , model will be train')
            self.logger.info("Flow Training ...")

            flow_dataset = pd.read_csv('data/FlowStatsfile.csv')
            # get flow_id column and replace . with nothing
            flow_dataset.iloc[:, 2] = flow_dataset.iloc[:,
                                      2].str.replace('.', '', regex=False)
            flow_dataset.iloc[:, 3] = flow_dataset.iloc[:,
                                      3].str.replace('.', '', regex=False)
            flow_dataset.iloc[:, 5] = flow_dataset.iloc[:,
                                      5].str.replace('.', '', regex=False)

            X_flow = flow_dataset.iloc[:, :-1].values
            X_flow = X_flow.astype('float64')

            y_flow = flow_dataset.iloc[:, -1].values

            X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25,
                                                                                    random_state=0)

            # Training the Random Forest Classification model on the Training set
            # check if model exist
            if os.path.exists('../ml/TrainedModels/dt_model.pkl'):
                self.logger.info("Flow Model exist")
                self.flow_model = joblib.load('../ml/TrainedModels/dt_model.pkl')
            else:
                self.logger.info("Flow Model not exist")
                self.logger.info("Flow Model training")
                self.flow_model = classifier.fit(X_flow_train, y_flow_train)
                joblib.dump(self.flow_model, '../ml/TrainedModels/dt_model.pkl')

            # Predicting the Test set results
            y_flow_pred = self.flow_model.predict(X_flow_test)

            self.logger.info(
                "------------------------------------------------------------------------------")

            self.logger.info("confusion matrix")
            cm = confusion_matrix(y_flow_test, y_flow_pred)
            self.logger.info(cm)

            acc = accuracy_score(y_flow_test, y_flow_pred)

            self.logger.info("succes accuracy = {0:.2f} %".format(acc * 100))
            fail = 1.0 - acc
            self.logger.info("fail accuracy = {0:.2f} %".format(fail * 100))
            self.logger.info(
                "------------------------------------------------------------------------------")
        elif 'load' == get_command_variables():
            self.logger.info("Flow Model exist")
            self.flow_model = joblib.load('../ml/TrainedModels/dt_model.pkl')
            self.logger.info("Flow Model loaded")
        else:
            self.logger.info('you must choose between train or load RYU_ARG=load or RYU_ARG=train')
            sys.exit(1)

    # Start Predicting , predicting model, test flow if it is legitimate or ddos
    def flow_predict(self):
        try:
            self.logger.info("Flow Predicting ...")
            self.logger.info("Waiting for 10 seconds to collect data ...")
            predict_flow_dataset = pd.read_csv('data/PredictFlowStatsfile.csv')

            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace(
                '.', '', regex=False)
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace(
                '.', '', regex=False)
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace(
                '.', '', regex=False)

            X_predict_flow = predict_flow_dataset.iloc[:, :].values
            X_predict_flow = X_predict_flow.astype('float64')

            y_flow_pred = self.flow_model.predict(X_predict_flow)

            legitimate_trafic = 0
            ddos_trafic = 0

            for i in y_flow_pred:
                if i == 0:
                    legitimate_trafic = legitimate_trafic + 1
                else:
                    ddos_trafic = ddos_trafic + 1
                    victim = int(predict_flow_dataset.iloc[i, 5]) % 20

            self.logger.info(
                "------------------------------------------------------------------------------")
            if (legitimate_trafic / len(y_flow_pred) * 100) > 80:
                self.logger.info("legitimate trafic ...")
            else:
                self.logger.info("ddos trafic ...")
                self.logger.info("victim is host: h{}".format(victim))

            self.logger.info(
                "------------------------------------------------------------------------------")

            file0 = open("data/PredictFlowStatsfile.csv", "w")

            file0.write(
                'timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
            file0.close()

        except:
            pass
