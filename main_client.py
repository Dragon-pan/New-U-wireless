# -*- coding: utf-8 -*-

"""
This is the main script for running the video streaming client.
It creates a Client object and runs it.

@author: Tommy Azzino [tommy.azzino@gmail.com]
"""

from stream_utils import utils
from client import Client

SERVER_PORT = 8000
SERVER_IP_ADDRESS = "10.18.166.39"   # NYU's wifi network SERVER address
CLIENT_PORT = 8001
CLIENT_IP_ADDRESS = "10.18.173.235"  # NYU's wifi network CLIENT address

I_FRAME_INTER = 8
BITRATE = 15e6
IS_WIFI = True     # True if the video streaming runs over a WiFi Network, False otherwise
DEBUG = True       # True to enable debug logs, False otherwise 
FEC = False        # Enable forward error correction (FEC)
ADAPTATION = True  # Enable bitrate and resolution adaptation
FEC_OVERHEAD = 20

# get correct interface name for the packet sniffer
if IS_WIFI:
    IFACE = utils.get_wireless_iface()
else:
    IFACE = utils.get_eth_iface() # assuming streaming runs over a Ethernet connection

iface_type = "Wi-Fi" if IS_WIFI else "Ethernet"
assert IFACE, iface_type + " connection interface was not found"
print("Using interface: ", IFACE)

# get correct device ID for the Arducam
CAM_DEV_ID = utils.get_cam()
assert CAM_DEV_ID, "Arducam was not found"
print("Using Arducam with device id: ", CAM_DEV_ID)

client = Client(SERVER_IP_ADDRESS, SERVER_PORT, CLIENT_IP_ADDRESS, CLIENT_PORT, CAM_DEV_ID,
        bitrate=BITRATE, ifi=I_FRAME_INTER, iface=IFACE, fec=FEC, encoding="h264",
        adaptation=ADAPTATION, fec_over=FEC_OVERHEAD, debug=DEBUG)
client.run()
