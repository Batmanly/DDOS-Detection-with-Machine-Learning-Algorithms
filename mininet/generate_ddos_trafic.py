import os
from datetime import datetime
from random import randrange, choice
from time import sleep

from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.net import Mininet
from mininet.node import RemoteController

from topology import MyTopo


# generate ip between 10.0.0.1 and 10.0.0.19
def ip_generator():
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


def generate_ddos_traffic(hosts):
    """
    Generate DDOS traffic between hosts

    :returns: None
    :parameter: hosts: list of hosts, h1: host

    """
    icmp_ping_flood(hosts)

    udp_flood(hosts)

    tcp_syn(hosts)

    land_attack(hosts)


def land_attack(hosts):
    src = choice(hosts)
    dst = ip_generator()
    info("--------------------------------------------------------------------------------\n")
    info("Performing LAND Attack\n")
    info("--------------------------------------------------------------------------------\n")
    src.cmd("timeout 20s hping3 -1 -V -d 120 -w 64 --rand-source --flood -a {} {}".format(dst, dst))
    sleep(100)
    info("--------------------------------------------------------------------------------\n")


def tcp_syn(hosts):
    src = choice(hosts)
    dst = ip_generator()
    info("--------------------------------------------------------------------------------\n")
    info("Performing TCP-SYN Flood\n")
    info("--------------------------------------------------------------------------------\n")
    src.cmd('timeout 20s hping3 -S -V -d 120 -w 64 -p 80 --rand-source --flood 10.0.0.1')
    sleep(100)


def udp_flood(hosts):
    src = choice(hosts)
    dst = ip_generator()
    info("--------------------------------------------------------------------------------\n")
    info("Performing UDP Flood\n")
    info("--------------------------------------------------------------------------------\n")
    src.cmd("timeout 20s hping3 -2 -V -d 120 -w 64 --rand-source --flood {}".format(dst))
    sleep(100)


def icmp_ping_flood(hosts):
    src = choice(hosts)
    dst = ip_generator()
    info("--------------------------------------------------------------------------------\n")
    info("Performing ICMP (Ping) Flood\n")
    info("--------------------------------------------------------------------------------\n")
    src.cmd("timeout 20s hping3 -1 -V -d 120 -w 64 -p 80 --rand-source --flood {}".format(dst))
    sleep(100)


def startNetwork():
    # info "Starting Network"
    topo = MyTopo()
    # net = Mininet( topo=topo, host=CPULimitedHost, link=TCLink, controller=None )
    # net.addController( 'c0', controller=RemoteController, ip='192.168.43.55', port=6653 )

    c0 = RemoteController('c0', ip='192.168.55.138', port=6653)
    net = Mininet(topo=topo, link=TCLink, controller=c0)

    net.start()

    hosts = generate_hosts(net)
    h1 = hosts[0]
    # write hosts to info
    info("--------------------------------------------------------------------------------\n")
    info("Hosts: {}\n".format(hosts))
    info("--------------------------------------------------------------------------------\n")
    # write h1 ip
    info("--------------------------------------------------------------------------------\n")
    info("h1 ip: {}\n".format(h1.IP()))
    info("--------------------------------------------------------------------------------\n")
    h1.cmd('cd /home/mininet/webserver')
    h1.cmd('python3 -m http.server 80 &')

    generate_ddos_traffic(hosts)
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
