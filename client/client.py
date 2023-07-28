# -*- coding: utf-8 -*-

"""
This is the Client class for the video streaming.
It creates the video data stream towards the server.
It receives feedbacks from the server: detected obejcts and bitrate estimates.

@author: Tommy Azzino [tommy.azzino@gmail.com]
"""

import os
import csv
import sys
import socket
import struct
import time
import logging
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
from multiprocessing import Queue
from threading import Thread, Event
from scapy.all import ETH_P_ALL
from stream_utils import MessagingClient
import numpy as np

logging.basicConfig()
GObject.threads_init()
Gst.init(None)

class ClientSniffer(Thread):
    def __init__(self, iface, ip_server, port_server, sender=None, sniff_q=None, bw_q=None, als_q=None, 
                 debug=False, bw_delta = float(2/1000), als_delta = float(3.5/1000)):
        super().__init__()
        self.iface = iface
        self.ip_server = ip_server
        self.port_server = port_server
        self.sender = sender
        self.bw_q = bw_q    # queue for bandwidth estimates
        self.als_q = als_q  # queue for obj detection feebacks
        self.sniff_q = sniff_q  # queue for exchanging data with messaging client
        self.debug = debug
        self.start_time = time.time()
        self.pkt_count = 0
        self.overhead = 20 + 8  # overhead of IP + UDP headers [bytes]
        self.pkt_data = []
        self.bw_delta = bw_delta*1e9     # convert to ns
        self.als_delta =  als_delta*1e9  # convert to ns

        # create and initialize L2 socket [listening to this socket is deprecated]
        '''self.socket = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**30)
        self.socket.bind((self.iface, ETH_P_ALL))'''

    def __process_ipframe(self, ip_header):
        if len(ip_header) < 20:
            return
        iph = struct.unpack('!BBHHHBBH4s4s', ip_header)
        ip_size = iph[2]
        ip_dst = socket.inet_ntoa(iph[9])
        if ip_dst == self.ip_dest:  
            self.pkt_data.append([time.time()-self.start_time, ip_size-self.overhead])
            self.pkt_count += 1
            if self.pkt_count % 100 == 0:
                # write 500 packets at a time (calling the writer for every single RX pkt is slow)
                self.writer.writerows(self.pkt_data)
                self.pkt_data = []
                self.pkt_count = 0

    def run(self):
        '''try:
            while True:
                pkt, _  = self.socket.recvfrom(65565)
                self.__process_ipframe(pkt[14:34])
        except KeyboardInterrupt:
            self.file.close()''' # this is deprecated
        try:
            while True:
                message = self.sniff_q.get()
                decoded_message = message.split('|')
                now = time.time_ns()

                for i, elem in enumerate(decoded_message):
                    if elem == "BW":
                        if len(decoded_message) > i+2:
                            if (float(decoded_message[i+1]) - now) <= self.bw_delta:
                                self.bw_q.put([time.time()-self.start_time, now, float(decoded_message[i+2])])
                    elif elem == "ALS":
                        if len(decoded_message) > i+2:
                                if (float(decoded_message[i+1]) - now) <= self.bw_delta:
                                    self.als_q.put([time.time()-self.start_time, now, decoded_message[i+2]])
                                    print(decoded_message[i+2])
                    else:
                        continue

        except KeyboardInterrupt:
            print("Interrupted")

class Client:
    def __init__(self, server_ip_address, server_port, client_ip_address, client_port, dev_id, msg_port=9000,
                 w="1920", h="1080", fps="30", ifi=30, bitrate=10e6, mtu_size=1290, iface="wlan0", 
                 encoding="h264", fec=False, fec_over=None, adaptation=True, debug=False):

        # Create GStreamer pipeline
        self.pipeline = Gst.Pipeline()

        # Create source
        self.src = Gst.ElementFactory.make("v4l2src", None)
        self.src.set_property("device", dev_id)

        # Create a caps filter
        self.caps1 = Gst.ElementFactory.make("capsfilter", None)
        self.caps1.set_property("caps", Gst.Caps.from_string("image/jpeg, width="+w+", height="+h+", framerate="+fps+"/1"))

        # Create jpdec decoder
        self.decoder = Gst.ElementFactory.make("jpegdec", None)

        # Create the videoscaler
        self.videoscale = Gst.ElementFactory.make("videoscale", None)
        self.videoscale.set_property("method", 1)
        self.videoscale.set_property("n-threads", 2)
        self.caps_resize = Gst.ElementFactory.make("capsfilter", None)
        self.caps_resize.set_property("caps", Gst.Caps.from_string("video/x-raw, width=1920, height=1080, framerate="+fps+"/1"))

        # Create the nvidia accelareted encoder
        # self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", None)
        if encoding == "h264":
            print("Using H264 encoder")
            self.encoder = Gst.ElementFactory.make("x264enc", None)
        elif encoding == "h265":
            print("Using H265 encoder")
            self.encoder = Gst.ElementFactory.make("x265enc", None)
        else:
        	raise ValueError(encoding + " is not supported")
        self.encoder.set_property("bitrate", int(bitrate/1e3)) # x26Xenc expects input in Kbit/sec
        self.encoder.set_property("tune", "zerolatency")
        self.encoder.set_property("speed-preset", "veryfast")
        self.encoder.set_property("pass", "cbr") # ONLY for H264 encoder
        self.encoder.set_property("insert-vui", True) # ONLY for H264 encoder
        # self.encoder.set_property("iframeinterval", ifi)
        # self.encoder.set_property("control-rate", 2)
        # self.encoder.set_property("maxperf-enable", True)
        # self.encoder.set_property("insert-sps-pps", True)
        # self.encoder.set_property("insert-aud", True)
        # self.encoder.set_property('"preset-level", 0) 
        
        # Create a caps filter
        self.caps2 = Gst.ElementFactory.make("capsfilter", None)
        self.caps2.set_property("caps", Gst.Caps.from_string("video/x-"+encoding+",stream-format=byte-stream,alignment=(string)au,width=1920,height=1080"))
        
        if encoding == "h264":
            self.parser = Gst.ElementFactory.make("h264parse", None)
            self.rtppay = Gst.ElementFactory.make("rtph264pay", None)
        elif encoding == "h265":
            self.parser = Gst.ElementFactory.make("h265parse", None)
            self.rtppay = Gst.ElementFactory.make("rtph265pay", None)
        else:
            raise ValueError(encoding + " is not supported")

        self.rtppay.set_property("mtu", mtu_size)
        self.rtppay.set_property("pt", 96)

        if fec:
            if not fec_over:
                raise ValueError("FEC overhead must be specified")
            # apply forward error correction to RTP packets
            print("Using FEC \n")
            self.rtpfec = Gst.ElementFactory.make("rtpulpfecenc", None)
            self.rtpfec.set_property("percentage", fec_over)
            self.rtpfec.set_property("percentage-important", 100)
            self.rtpfec.set_property("pt", 122)

        # Create the UDP sink
        self.sink = Gst.ElementFactory.make("udpsink", None)
        self.sink.set_property("host", server_ip_address)
        self.sink.set_property("port", server_port)
        self.sink.set_property("bind-port", 43450)
        self.sink.set_property("auto-multicast", False)
        self.sink.set_property("async", False)
        self.sink.set_property("sync", False)

        # Add elements to pipeline
        self.pipeline.add(self.src)
        self.pipeline.add(self.caps1)
        self.pipeline.add(self.decoder)
        self.pipeline.add(self.videoscale)
        self.pipeline.add(self.caps_resize)
        # self.pipeline.add(self.nvvidconv)
        self.pipeline.add(self.encoder)
        self.pipeline.add(self.caps2)
        self.pipeline.add(self.parser)
        self.pipeline.add(self.rtppay)
        if fec:
            self.pipeline.add(self.rtpfec)
        self.pipeline.add(self.sink)

        # Link elements in the pipeline
        self.src.link(self.caps1)
        self.caps1.link(self.decoder)
        self.decoder.link(self.videoscale)
        self.videoscale.link(self.caps_resize)
        # self.caps_resize.link(self.nvvidconv)
        self.caps_resize.link(self.encoder)
        self.encoder.link(self.caps2)
        self.caps2.link(self.parser)
        self.parser.link(self.rtppay)
        if fec:
            self.rtppay.link(self.rtpfec)
            self.rtpfec.link(self.sink)
        else:
            self.rtppay.link(self.sink)

        self.debug = debug
        self.enable_adaptation = adaptation
        self.default_bitrate = bitrate
        self.server_address = server_ip_address
        self.server_port = server_port
        self.steps = 1

        self.enable_res_change = encoding == "h264"  # currently h265 does not support dynamic changes of resolution
        self.available_res_raw = ["video/x-raw, width=640, height=480, framerate="+fps+"/1", 
                                    "video/x-raw, width=1280, height=720, framerate="+fps+"/1",
                                    "video/x-raw, width=1920, height=1080, framerate="+fps+"/1"]
        self.available_res_enc = ["video/x-"+encoding+",stream-format=byte-stream,alignment=(string)au,width=640,height=480",
                                    "video/x-"+encoding+",stream-format=byte-stream,alignment=(string)au,width=1280,height=720",
                                    "video/x-"+encoding+",stream-format=byte-stream,alignment=(string)au,width=1920,height=1080"]

        self.filename1 = "./results/rx_feedbacks.csv"
        if(self.debug and os.path.exists(self.filename1) and os.path.isfile(self.filename1)):
            os.remove(self.filename1)
            print(self.filename1 + " deleted")
        self.file1 = open(self.filename1, "w")
        self.writer1 = csv.writer(self.file1)

        '''self.filename2 = "./results/tx_packets.csv"
        if(self.debug and os.path.exists(self.filename2) and os.path.isfile(self.filename2)):
            os.remove(self.filename2)
            print(self.filename2 + " deleted")
        self.file2 = open(self.filename2, "w")
        self.writer2 = csv.writer(self.file2)'''

        self.capacity = [20, 10, 5, 2, 10, 100] # Mbps
        self.client_ip = client_ip_address
        self.client_port = client_port
        self.overhead = 20 + 8  # overhead of IP + UDP headers [bytes]
        self.rx_feedbacks = []
        self.current_bitrate = self.default_bitrate
        self.current_resolution = 2
        if fec:
            self.fec_multiplier = float(100 + fec_over)/100
        else:
            self.fec_multiplier = 1
        self.utilization_factor = 0.95
        self.lower_margin = 0.5e6  # lower bitrate margin for triggering an encoder bitrate change [bps]
        self.higher_margin = 0.5e6  # higher bitrate margin for triggering an encoder bitrate change [bps]

        # create queue for exchanging messages among threads
        self.bw_q = Queue(maxsize=0)
        self.als_q = Queue(maxsize=0)
        self.sniff_q = Queue(maxsize=0)

        # create messaging client
        self.messaging_client = MessagingClient(ip=server_ip_address, port=msg_port, msg_q=self.sniff_q)

        # create packet sniffer to listen for feedback
        self.sniffer = ClientSniffer(iface=iface, ip_server=server_ip_address, port_server=server_port, sniff_q=self.sniff_q, 
                                     bw_q=self.bw_q, als_q=self.als_q, debug=self.debug, bw_delta=float(2/1000), als_delta=float(3.5/1000))

        # create a recursive least square (RLS) available bandwidth (ABW) predictor
        self.pred_type = "RLS"
        self.M = 5
        self.lam = 0.999
        self.theta = 0.001
        # initialize RLS predictor
        self.P = np.eye(self.M) / self.theta
        self.w = np.zeros((self.M, 1))
        self.c = np.zeros((self.M, 1))

    def _change_bitrate(self, new_bitrate):
        self.encoder.set_state(Gst.State.READY)
        self.pipeline.set_state(Gst.State.PAUSED)
        self.encoder.set_property("bitrate", int(new_bitrate/1e3))
        self.pipeline.set_state(Gst.State.PLAYING)
        self.encoder.set_state(Gst.State.PLAYING)
        self.current_bitrate = new_bitrate
        if self.debug:
            print("*"*50)
            print("The new encoder bitrate is: {:.2f} Mbps".format(self.encoder.get_property("bitrate")/1e3))
            print("*"*50)
        return True

    def _change_resolution(self, new_resolution):
        self.encoder.set_state(Gst.State.READY)
        self.pipeline.set_state(Gst.State.PAUSED)
        self.caps_resize.set_property("caps", Gst.Caps.from_string(self.available_res_raw[new_resolution]))
        self.caps2.set_property("caps", Gst.Caps.from_string(self.available_res_enc[new_resolution]))
        self.pipeline.set_state(Gst.State.PLAYING)
        self.encoder.set_state(Gst.State.PLAYING)
        # Force a key frame so that the resolution can change immediately
        # time.sleep(1/25)  # Need to wait a little before sending the force-IDR signal
        # self.encoder.emit("force-IDR")
        self.current_resolution = new_resolution
        if self.debug:
            print("*"*50)
            print("The new resolution is: ", self.available_res_raw[self.current_resolution])
            print("*"*50)

        return True

    def handle_tx_packets(self):
        pkt_data = []
        pkt_count = 0
        start_time = time.time()
        while True:
            _, _, pkt = self.q_tx_pkts.get()
            ip_header = pkt[14:34]
            if len(ip_header) < 20:
                return

            iph = struct.unpack('!BBHHHBBH4s4s', ip_header)
            # print(iph)
            ip_size = iph[2]
            ip_dst = socket.inet_ntoa(iph[9])
            ip_src = socket.inet_ntoa(iph[8])
            if ip_src == self.client_ip:
                pkt_data.append([time.time()-start_time, ip_size-self.overhead])
                pkt_count += 1
                if pkt_count % 50 == 0:
                    # write 50 packets at a time (calling the writer for every single RX pkt is slow)
                    self.writer2.writerows(pkt_data)
                    pkt_data = []
                    pkt_count = 0

    def trigger_change(self):
        while True:
            t, now, est_rate = self.bw_q.get()
            pkt_info = [t, est_rate]
            self.rx_feedbacks.append(pkt_info)
            # if self.debug:
                # print("The measured delay is: {:.2f} ms".format(delay/1e6))
                # self.writer1.writerow([t, data[0], delay])
                
            if self.enable_adaptation:
                pred_rate = None
                if self.pred_type == "RLS":
                    # use Recursive Least Square predictor
                    # get latest feeback and scale by the amount of FEC overhead
                    last_feedback = float(self.rx_feedbacks[-1][1]) / self.fec_multiplier
                    # compute the gain vector
                    g = (1/self.lam) * self.P @ self.c / (1. + (1/self.lam) * self.c.T @ self.P @ self.c)
                    # compute a priori prediction error
                    eps = last_feedback - self.w.T @ self.c
                    self.w = self.w + eps * g
                    self.P = (1/self.lam) * (self.P - g @ self.c.T @ self.P)
                    self.c = np.roll(self.c, -1)
                    self.c[-1] = last_feedback
                    pred_rate = self.w.T @ self.c
                else:
                    # use Average predictor
                    if len(self.rx_feedbacks) > self.M:
                        tot_rate = 0
                        # scale the predicted link-rate by the amount of FEC overhead
                        for _, rate in self.rx_feedbacks[-self.M:]: tot_rate += float(rate) / self.fec_multiplier
                        # predict next link-rate as the average of the last M estimates
                        pred_rate = tot_rate / self.M

                if pred_rate:
                # convert pred_rate to bps and scale according to utilization factor
                    pred_rate = pred_rate * 1e3 * self.utilization_factor
                    if self.debug:
                        print("Predicted bitrate for adaptation (before capping): {:.2f} Mbps".format(float(pred_rate)/1e6))
                    # cap the pred_rate to a maximum value
                    pred_rate = min(pred_rate, 25e6)
                    pred_rate = max(1e3, pred_rate)
                        
                    # avoid too frequent changes of bitrate (TODO: I believe there's some issue with the Jetson TX2)
                    if pred_rate < self.current_bitrate - self.lower_margin or pred_rate > self.current_bitrate + self.higher_margin:
                        # do_nothing = 1
                        self._change_bitrate(pred_rate)
                        # adjust the resolution based on the predicted link-rate
                        if self.enable_res_change:
                            # based on the Access paper
                            if self.current_bitrate <= 350e3:
                                target_resolution = 0
                            elif self.current_bitrate > 350e3 and self.current_bitrate <= 6e6:
                                target_resolution = 1
                            else:
                                target_resolution = 2
                            if self.current_resolution != target_resolution:
                                # do_nothing = 1
                                self._change_resolution(target_resolution)

            # debug bitrate adaptation
            '''if self.steps % 100 == 0:
                # change link capacity every 20 seconds
                rate = self.capacity.pop(0)
                if len(self.capacity) == 0:
                    self.capacity = [100]
                # sudo tc qdisc add dev wlan0 root handle 1: htb default 10
                # sudo tc class add dev wlan0 parent 1: classid 1:10 htb rate 1000Mbit
                os.system("tc class replace dev wlan0 parent 1: classid 1:10 htb rate "+str(rate)+"Mbit")
                print("Link capacity changed to: "+str(rate)+" Mbps")
            self.steps += 1'''

            # debug dynamic resolution
            if self.steps % 50 == 0:
                # new_resolution = 2 if self.current_resolution == 1 else 1  # switching between FullHD and HD resolution
                if self.current_resolution == 0:
                    new_resolution = 1
                elif self.current_resolution == 1:
                    new_resolution = 2
                else:
                    new_resolution = 0
                self._change_resolution(new_resolution)
            self.steps += 1
            
    def print_something(self):
        print("ok")
        return True
    
    def run(self):
        self.thread_change = Thread(target=self.trigger_change, daemon=True)
        # self.thread_tx_pkts = Thread(target=self.handle_tx_packets, daemon=True)

        # Create bus to receive events from GStreamer pipeline
        self.bus = self.pipeline.get_bus()
        # self.bus.add_signal_watch()
        # self.bus.connect("message::any", self.on_error)
        print("Starting pipeline \n")
        self.pipeline.set_state(Gst.State.PLAYING)
        # ret = GLib.timeout_add_seconds(5, self._change_resolution)
        # ret1 = GLib.timeout_add(100, self.print_something) # calls a function every 100 ms

        # start sniffers on a separate process
        self.sniffer.start()
        self.thread_change.start()
        self.messaging_client.start()
        # self.tx_pkts_sniffer.start()
        # self.thread_tx_pkts.start()

        try:
            while True:
                if self.bus.have_pending():  # Working without glib mainloop
                    message = self.bus.pop()
                    if message.type == Gst.MessageType.EOS:
                        self.close()
                    elif message.type == Gst.MessageType.ERROR:
                        print("ERROR", message)
                        self.close()
                        break
        except KeyboardInterrupt:
            self.close()

    def on_error(self, error):
        print(error)

    def close(self):
        print("\nDeleting streaming")
        self.pipeline.set_state(Gst.State.NULL)
        print("Closing files")
        self.file1.close()
