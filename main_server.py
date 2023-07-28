# -*- coding: utf-8 -*-

"""
This is the main script for running the video streaming server.
It creates a Server object and runs it.

@author: Tommy Azzino [tommy.azzino@gmail.com]
"""

from stream_utils import utils
from server import Server, PyTorchServer

SERVER_IP_PORT = 8000
SERVER_IP_ADDRESS = "10.18.166.39"  # NYU's wifi network SERVER address
CLIENT_PORT = 8001
CLIENT_IP_ADDRESS = "10.18.177.33"  # NYU's wifi network CLIENT address

IS_WIFI = True # True if the video streaming runs over a WiFi Network, False otherwise
DEBUG = True
INFERENCE = True  # Enable object detection inference at the server
FEC = False  # Enable forward error correction
FEEDBACK_FREQ = 0.1  # frequency of link rate estimates from Server to Client [s]

# get correct interface name for the packet sniffer
if IS_WIFI:
    IFACE = utils.get_wireless_iface()
else:
    IFACE = utils.get_eth_iface() # assuming streaming runs over a Ethernet connection

iface_type = "Wi-Fi" if IS_WIFI else "Ethernet"
assert IFACE, iface_type + " connection interface was not found"
print("Using interface: ", IFACE)

USE_LEGACY_SERVER = True
if USE_LEGACY_SERVER:
    server = Server(server_ip_address=SERVER_IP_ADDRESS, server_port=SERVER_IP_PORT, client_ip_address=CLIENT_IP_ADDRESS, client_port=CLIENT_PORT, 
                    feedback_freq=FEEDBACK_FREQ, iface=IFACE, fec=FEC, debug=DEBUG, infer=INFERENCE, encoding="h264", sink_type='video')
else:
    server = PyTorchServer(server_ip_address=SERVER_IP_ADDRESS, server_port=SERVER_IP_PORT, client_ip_address=CLIENT_IP_ADDRESS, client_port=CLIENT_PORT, 
                           feedback_freq=FEEDBACK_FREQ, iface=IFACE, fec=FEC, debug=DEBUG, infer=INFERENCE, encoding="h264", sink_type='video')
try:
    server.run()
except KeyboardInterrupt:
    print("Interrupted")
