import os
from datetime import datetime

from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch


def get_command_variables():
    return os.getenv('RYU_ARG')


def check_if_file_exist():
    folder = os.path.exists("data/FlowStatsfile.csv")
    if folder:
        pass
    else:
        # Create csv file if it doesn't exist , add flow header
        file0 = open("data/FlowStatsfile.csv", "w")
        file0.write(
            'timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond,label\n')
        file0.close()


"""
CollectTrainingStatsApp that extends the SimpleSwitch13 class from the Ryu controller framework for Software-Defined Networking (SDN). 
The purpose of this class is to collect network flow statistics from the SDN switches in the network and write them to a CSV file called "FlowStatsfile.csv".

The purpose of this class is to collect network flow statistics from the SDN switches in the network and write them to a CSV file called "FlowStatsfile.csv".


The state_change_handler() method is an event handler that is called when the state of a switch in the network changes. It registers or unregisters the switch in the datapaths dictionary,
 depending on whether the switch is being added or removed from the network.
"""


# stored in a CSV file. class CollectTrainingStatsApp(simple_switch_13.SimpleSwitch13):
class CollectTrainingStatsApp(switch.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(CollectTrainingStatsApp, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self.monitor)
        check_if_file_exist()

        # file0 = open("FlowStatsfile.csv", "w")
        # file0.write(
        #     'timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond,label\n')
        # file0.close()

    # Asynchronous message
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath

        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    """
    The monitor() method is a thread that sends flow statistics requests to each switch in the network every 10 seconds.
    
    """

    def monitor(self):
        while True:
            for dp in self.datapaths.values():
                self.request_stats(dp)
            hub.sleep(10)

    # request_stats method sends a request to the switch to get statistics on its flows.
    # It takes in a datapath parameter which is an object representing the connection to the switch.
    def request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)

        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """
        The _flow_stats_reply_handler() method is an event handler that is called when a flow statistics reply message is received from a switch in response to a request sent
        by request_stats(). It extracts the relevant flow statistics from the message and writes them to the CSV file.
        :param ev:
        :return:

        """
        # get the timestamp
        timestamp = datetime.now()
        timestamp = timestamp.timestamp()
        icmp_code = -1
        icmp_type = -1
        tp_src = 0
        tp_dst = 0

        file0 = open("data/FlowStatsfile.csv", "a+")

        # get stats from the switch
        body = ev.msg.body
        for stat in sorted([flow for flow in body if (flow.priority == 1)], key=lambda flow:
        (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):

            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']
            # check if the flow is ICMP
            if stat.match['ip_proto'] == 1:
                icmp_code = stat.match['icmpv4_code']
                icmp_type = stat.match['icmpv4_type']
            # check if the flow is TCP
            elif stat.match['ip_proto'] == 6:
                tp_src = stat.match['tcp_src']
                tp_dst = stat.match['tcp_dst']
            # check if the flow is UDP
            elif stat.match['ip_proto'] == 17:
                tp_src = stat.match['udp_src']
                tp_dst = stat.match['udp_dst']
            # create a unique flow ID
            flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + str(tp_dst) + str(ip_proto)

            try:
                # calculate the number of packets and bytes per second and nanosecond
                packet_count_per_second = stat.packet_count / stat.duration_sec
                packet_count_per_nsecond = stat.packet_count / stat.duration_nsec
            except:
                # if the flow duration is 0, set the number of packets and bytes per second and nanosecond to 0
                packet_count_per_second = 0
                packet_count_per_nsecond = 0

            try:
                #
                byte_count_per_second = stat.byte_count / stat.duration_sec
                byte_count_per_nsecond = stat.byte_count / stat.duration_nsec
            except:
                byte_count_per_second = 0
                byte_count_per_nsecond = 0
            """
            The CSV file contains the following fields for each flow:

            timestamp: the time when the flow statistics were collected
            datapath_id: the ID of the switch that sent the flow statistics
            flow_id: a unique identifier for the flow based on its source and destination IP addresses, transport protocol, and port numbers
            ip_src: the source IP address of the flow
            tp_src: the source port number of the flow (if applicable)
            ip_dst: the destination IP address of the flow
            tp_dst: the destination port number of the flow (if applicable)
            ip_proto: the transport protocol used by the flow (TCP, UDP, or ICMP)
            icmp_code: the ICMP code (if applicable)
            icmp_type: the ICMP type (if applicable)
            flow_duration_sec: the duration of the flow in seconds
            flow_duration_nsec: the duration of the flow in nanoseconds
            idle_timeout: the number of seconds before the flow is removed due to inactivity
            hard_timeout: the number of seconds before the flow is forcibly removed
            flags: flags associated with the flow (e.g., whether it is being monitored)
            packet_count: the number of packets in the flow
            byte_count: the number of bytes in the flow
            packet_count_per_second: the average number of packets per second in the flow
            packet_count_per_nsecond: the average number of packets per nanosecond in the flow
            byte_count_per_second: the average number of bytes per second in the flow
            byte_count_per_nsecond: the average number of bytes per nanosecond in the flow
            label: a label for the flow (not currently used in the code)
            """
            # check argument if it's dos or normal
            if get_command_variables() == "norm":
                # write flow as normal
                file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                            .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src, ip_dst, tp_dst,
                                    stat.match['ip_proto'], icmp_code, icmp_type,
                                    stat.duration_sec, stat.duration_nsec,
                                    stat.idle_timeout, stat.hard_timeout,
                                    stat.flags, stat.packet_count, stat.byte_count,
                                    packet_count_per_second, packet_count_per_nsecond,
                                    byte_count_per_second, byte_count_per_nsecond, 0))
            elif get_command_variables() == "dos":
                # write flow as dos
                file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                            .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src, ip_dst, tp_dst,
                                    stat.match['ip_proto'], icmp_code, icmp_type,
                                    stat.duration_sec, stat.duration_nsec,
                                    stat.idle_timeout, stat.hard_timeout,
                                    stat.flags, stat.packet_count, stat.byte_count,
                                    packet_count_per_second, packet_count_per_nsecond,
                                    byte_count_per_second, byte_count_per_nsecond, 1))
            else:
                print(
                    "Give me Argument to label your data if it's dos or normal {ARGUMNET ARG=norm or ARG=dos ryu-manager }")
                exit()
        file0.close()
