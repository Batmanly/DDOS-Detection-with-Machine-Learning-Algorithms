import os
from datetime import datetime
from random import randrange, choice
from time import sleep

from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.net import Mininet
from mininet.node import RemoteController

from topology import MyTopo


def ip_generator():
    """
    Generate random IP address between 10.0.0.0 and 10.0.0.19

   :returns: IP address
   :parameter: None

    """
    ip = ".".join(["10", "0", "0", str(randrange(1, 19))])
    return ip


def generate_hosts(net):
    """
    Generate hosts and add them to the network

    :returns: hosts
    :parameter: net: network

    """
    info("Generating hosts\n")
    # print "Generating hosts"
    hosts = {}
    for i in range(1, 19):
        hosts[f'h{i}'] = net.get(f'h{i}')
    # convert dictionary to list
    hosts = list(hosts.values())
    return hosts


def generate_traffic(h1):
    """
    Generate traffic between hosts

    :returns: None
    :parameter: h1: host

    """
    info('--------------------------------------------------------------------------------\n')
    info("Generating traffic\n")
    # print "Generating traffic"
    # execute commands on hosts
    h1.cmd('cd /home/mininet/webserver')
    # run webserver on h1
    h1.cmd('python3 -m http.server 80 &')
    # run iperf server on h1 , check speed
    h1.cmd('iperf -s -p 5050 &')
    h1.cmd('iperf -s -u -p 5051 &')
    sleep(2)


def generate_icmp_traffic(hosts, h1):
    """
    Generate ICMP traffic between hosts

    :returns: None
    :parameter: hosts: list of hosts, h1: host

    """
    info('--------------------------------------------------------------------------------\n')
    info("Generating ICMP traffic\n")
    # choose random host
    # chose src if it's not h1
    src = choice(hosts)
    # choose random destination
    dst = ip_generator()
    # run ping and iperf command
    info("generating ICMP traffic between %s and h%s and TCP/UDP traffic between %s and h1\n" % (
        src, ((dst.split('.'))[3]), src))
    src.cmd("ping {} -c 100 &".format(dst))
    src.cmd("iperf -p 5050 -c 10.0.0.1")
    src.cmd("iperf -p 5051 -u -c 10.0.0.1")
    # download files from h1
    info("%s Downloading index.html from h1\n" % src)
    src.cmd("wget http://10.0.0.1/index.html")
    info("%s Downloading test.zip from h1\n" % src)
    src.cmd("wget http://10.0.0.1/test.zip")
    # remove files from h1
    # h1.cmd("rm -f *.* /home/mininet/Downloads")
    h1.cmd("rm -f /home/mininet/Downloads")


def startNetwork():
    """
      Start Mininet with a remote controller
      Use Ryu as the controller

      :returns: None

      :parameter: None

    """
    info("Starting Network\n")
    # print "Starting Network"
    topo = MyTopo()  # Create topology

    c0 = RemoteController('c0', ip='192.168.55.138', port=6653)  # create controller and connect to it
    # don't forget to open firewall with this sudo ufw allow 6653/tcp
    net = Mininet(topo=topo, link=TCLink, controller=c0)
    # create network and add controller to it

    net.start()  # start network

    # get hosts
    hosts = generate_hosts(net)
    h1 = hosts[0]
    # generate traffic
    generate_traffic(h1)

    for h in hosts:
        # run for each host
        h.cmd('cd /home/mininet/Downloads')
    for i in range(600):

        info("--------------------------------------------------------------------------------\n")
        info("Iteration n {} ...\n".format(i + 1))
        info("--------------------------------------------------------------------------------\n")

        for j in range(10):
            # generate ICMP traffic
            generate_icmp_traffic(hosts, h1)
            sleep(1)

    info("--------------------------------------------------------------------------------\n")
    # stop network
    # CLI(net)
    net.stop()


if __name__ == '__main__':
    try:
        # starting time
        start = datetime.now()

        # set log level
        setLogLevel('info')

        # start network
        startNetwork()

        # ending time
        end = datetime.now()
        # total time taken
        print(end - start)

    except KeyboardInterrupt:
        print("Ctrl + C pressed. Program terminated.")
        # execute command
        # sudo mn -c
        os.system("sudo mn -c")
