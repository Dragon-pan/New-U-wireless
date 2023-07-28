# -*- coding: utf-8 -*-

"""
This is the Server class for the video streaming.
It receives video data, performs object detection and rate estimation,
and sends back feedbacks to the client.

@author: Tommy Azzino [tommy.azzino@gmail.com]
"""

import os
import csv
import cv2
import sys
import socket
import struct
import time
import logging
import gi
import platform
import pyds
import numpy as np
import datetime as datetime
import contextlib
import torch, torchvision

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
from threading import Thread
from multiprocessing import Queue
from scapy.all import ETH_P_ALL
from stream_utils import MessagingServer
from .ghetto_nvds import *
from PIL import Image
from track import Hloc

# Uncomment this imports ONLY if using RA-Yolo and have it configured on your machine
# sys.path.insert(0, '../ra-yolo-v7')
# from utils.torch_utils import select_device
# from utils.general import non_max_suppression
# from utils.plots import plot_one_box
# from models.experimental import attempt_load
# import random
# import matplotlib.pyplot as plt
# plt.switch_backend('TkAgg')

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print("Using GPU device: ", device)

logging.basicConfig()
GObject.threads_init()
Gst.init(None)

PGIE_CLASS_ID_VEHICLE = 2  # Yolov7: 2
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 0   # Yolov7: 0
PGIE_CLASS_ID_TRAFFIC_LIGHT = 9
PGIE_CLASSES = [PGIE_CLASS_ID_PERSON, PGIE_CLASS_ID_BICYCLE, PGIE_CLASS_ID_VEHICLE, PGIE_CLASS_ID_TRAFFIC_LIGHT]
is_aarch64 = platform.uname()[4] == "aarch64"

@contextlib.contextmanager
def nvtx_range(msg):
    depth = torch.cuda.nvtx.range_push(msg)
    try:
        yield depth
    finally:
        torch.cuda.nvtx.range_pop()

class BitRateEstimator(Thread):
    def __init__(self, delta, window=1, sniff_q=None, msg_q=None, debug=False):
        super().__init__()
        self.sniff_q = sniff_q
        self.msg_q = msg_q
        self.T = window
        self.delta = delta
        self.rx_pkts = []
        self.burst_interval = 10e-3  # packet burts duration [s]
        self.if_interval = 30e-3  # time-interval between consecutive frames
        self.feedback_elapsed_time = time.time()
        self.num_iter = int(10 / self.delta)
        self.curr_iter = 0  # current iteration index
        self.start_time = None
        self.debug = debug
        self.pkt_data = []
        self.pkt_count = 0
        self.capacity = [100, 20, 10, 5, 2, 10, 100]  # Mbps, sequence of link capacity values for debugging

        self.filename = "./results/est_band.csv"
        if(self.debug and os.path.exists(self.filename) and os.path.isfile(self.filename)):
            os.remove(self.filename)
            print(self.filename + " deleted")
        self.file = open(self.filename, "w")
        self.writer = csv.writer(self.file)

    def estimate_bitrate(self, now):
        self.curr_iter += 1
        elapsed_time = now - self.feedback_elapsed_time
        self.feedback_elapsed_time = now
        # print("The elapsed time between two consecutive bandwidth estimations is: {:.3f} s".format(elapsed_time)) 
        if not self.rx_pkts:
            # list of received packets is empty
            return
        rx_pkts_np = np.array(self.rx_pkts)
        last_time = rx_pkts_np[-1, 0]  # get time of last packet to arrive
        min_window = np.maximum(0, last_time - self.T)
        idx_win = rx_pkts_np[:, 0] > min_window
        w_pkts = rx_pkts_np[idx_win]  # get packets inside window
        it_pkts = w_pkts[1:, 0] - w_pkts[:-1, 0]
        idxs = it_pkts < self.if_interval  # keep only packets belonging to the same frame
        A_n = sum(it_pkts[idxs])  # sum of inter-arrival times
        Z_n = sum(w_pkts[1:, 1][idxs])
        if A_n < self.burst_interval:
            # this packet train is a burst, discard it
            print("Discarding measurement")
            return
        est_bandwidth = Z_n*8/A_n  # estimated link bandwidth [bps]
        print("The estimated link bandwidth is: {:.3f} Mbps".format(est_bandwidth/1e6))
        est_data = [now, est_bandwidth/1e6]
        if self.debug:
            self.writer.writerow(est_data)

        # clean list storing received packets
        if self.curr_iter == self.num_iter:
            self.rx_pkts = rx_pkts_np[idx_win].tolist()
            self.curr_iter = 0

        # send feeback with estimated bandwidth to Client
        self.msg_q.put("|BW|"+ str(time.time_ns()) + "|" + str(est_bandwidth/1e3))

    def run(self):
        self.start_time = time.time()
        try:
            while True:
                pkt_info = self.sniff_q.get()
                self.rx_pkts.append(pkt_info)
                now = time.time()
                if now < self.T:
                    # not enough packets have been received
                    continue
                else:
                    if now >= self.start_time + self.delta:
                        self.estimate_bitrate(now)
                        self.start_time = now
        except KeyboardInterrupt:
            self.file.close()


class ServerSniffer(Thread):
    def __init__(self, iface, ip_server, port_server, sniff_q=None, sender=None, debug=False):
        super().__init__()
        self.iface = iface
        self.ip_server = ip_server
        self.port_server = port_server
        self.sender = sender
        self.debug = debug
        self.start_time = time.time()
        self.pkt_count = 0
        self.overhead = 8  # overhead of UDP header [bytes]
        self.pkt_data = []
        self.rx_pkts = []
        self.sniff_q = sniff_q
        
        self.filename = "./results/rx_pkt_trace.csv"
        if(os.path.exists(self.filename) and os.path.isfile(self.filename)):
            os.remove(self.filename)
            print(self.filename + " deleted")
        self.file = open(self.filename, "w")
        self.writer = csv.writer(self.file)

        # create and initialize L2 socket
        self.socket = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**30)
        self.socket.bind((self.iface, ETH_P_ALL))

    def __process_rxpkt(self, udp_header):
        if len(udp_header) < 8:
            # corrupted UDP header
            return
        _, dest_port, pkt_len, _ = struct.unpack(">HHHH", udp_header)
        if dest_port == self.port_server:
            # store received packets for bandwidh estimation
            pkt_info = [time.time()-self.start_time, pkt_len-self.overhead]
            self.sniff_q.put(pkt_info)
            self.pkt_data.append(pkt_info)

            if self.debug:
                self.pkt_count += 1
                if self.pkt_count % 100 == 0:
                    # write 100 packets at a time (calling the writer for every single RX might be slow)
                    self.writer.writerows(self.pkt_data)
                    self.pkt_data = []
                    self.pkt_count = 0

    def run(self):
        try:
            while True:
                pkt, _  = self.socket.recvfrom(65565)
                self.__process_rxpkt(pkt[34:42])
        except KeyboardInterrupt:
            self.file.close()

class Server:
    def __init__(self, server_ip_address, server_port, client_ip_address, client_port, msg_port=9000,
                 feedback_freq=None, infer=False, sink_type="video", iface="wlan0", encoding="h264", fec=False, debug=False):
        # Create GStreamer pipeline
        if fec:
            print("Using FEC \n")
            self.pipeline= Gst.parse_launch("""udpsrc auto-multicast=False buffer-size=2147483647 address="""+server_ip_address+""" port="""+str(server_port)+""" caps="application/x-rtp, payload=96, clock-rate=90000" ! rtpstorage size-time=500000000 ! 
                rtpssrcdemux ! application/x-rtp, payload=96, clock-rate=90000, media=video, encoding-name="""+encoding.upper()+""" ! rtpjitterbuffer do-lost=1 latency=100 ! rtpulpfecdec pt=122 ! rtp"""+encoding+"""depay""")
        else:    
            self.pipeline = Gst.parse_launch("""udpsrc auto-multicast=False buffer-size=2147483647 address="""+server_ip_address+""" port="""+str(server_port)+""" caps="application/x-rtp, payload=96, clock-rate=90000" ! rtp"""+encoding+"""depay""")

        # Create payload decoder and parser
        if encoding == "h264":
            print("Using h264 decoder")
            self.rtpdepay = self.pipeline.get_by_name("rtph264depay0")
            if not self.rtpdepay:
                sys.stderr.write("Unable to get the rtp depay element \n")
            self.parser = Gst.ElementFactory.make("h264parse", None)
        elif encoding == "h265":
            print("Using h265 decoder")
            self.rtpdepay = self.pipeline.get_by_name("rtph265depay0")
            if not self.rtpdepay:
                sys.stderr.write("Unable to get the rtp depay element \n")
            self.parser = Gst.ElementFactory.make("h265parse", None)
        else:
            raise ValueError(encoding + " is not supported")

        '''self.pipeline = Gst.Pipeline()
        self.src = Gst.ElementFactory.make("udpsrc", None)
        self.src.set_property("auto-multicast", False)
        self.src.set_property("buffer-size", 2147483647)
        self.src.set_property("address", server_ip_address)
        self.src.set_property("port", server_port)
        self.src.set_property("caps", Gst.Caps.from_string("application/x-rtp, payload=96, clock-rate=90000"))
        self.rtpdepay = Gst.ElementFactory.make("rtp"+encoding+"depay", None)
        self.parser = Gst.ElementFactory.make(encoding+"parse", None)'''

        if infer:
            # Use avdec
            if encoding == "h264":
                self.nvdecoder = Gst.ElementFactory.make("avdec_h264", None)
            elif encoding == "h265":
                self.nvdecoder = Gst.ElementFactory.make("avdec_h265", None)
            else:
                raise ValueError(encoding + " is not supported")

            self.vidconv1 = Gst.ElementFactory.make("nvvideoconvert", None)

            # Create nvstreammux instance to form batches from one or more sources.
            self.streammux = Gst.ElementFactory.make("nvstreammux", None)

            # Use nvinfer to run inferencing on decoder's output,
            # behaviour of inferencing is set through config file
            self.pgie = Gst.ElementFactory.make("nvinfer", None)

            # Use convertor to convert from NV12 to RGBA as required by nvosd
            self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", None)

            # Create OSD to draw on the converted RGBA buffer
            self.nvosd = Gst.ElementFactory.make("nvdsosd", None)
            # self.nvosd.set_property("process-mode", 0)  # processing with CPU

            # Finally render the osd output
            if is_aarch64:
                print("Using transform \n")
                self.transform = Gst.ElementFactory.make("nvegltransform",None)

            print("Creating Sink \n")
            if sink_type == "video":
                self.queue = Gst.ElementFactory.make("queue", None)
                self.sink = Gst.ElementFactory.make("nveglglessink", None)
                self.sink.set_property("sync", False)
                self.sink.set_property("async", False)
            elif sink_type == "file":
                self.videoconv = Gst.ElementFactory.make("nvvideoconvert", None)
                self.encoder = Gst.ElementFactory.make("nvv4l2h264enc", None)
                self.parser2 = Gst.ElementFactory.make("h264parse", None)
                self.queue = Gst.ElementFactory.make("queue", None)
                self.mux = Gst.ElementFactory.make("matroskamux", None)
                self.sink = Gst.ElementFactory.make("filesink", None)
                self.sink.set_property("location", "../Videos/output.mkv")
            else:
                self.sink = Gst.ElementFactory.make("fakesink", None)

            self.streammux.set_property("width", 1920)
            self.streammux.set_property("height", 1080)
            self.streammux.set_property("batch-size", 1)
            self.streammux.set_property("batched-push-timeout", 4000000)
            self.streammux.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_DEVICE))
            self.streammux.set_property("live-source", True)
            # self.streammux.set_property("enable-padding", True)
            # self.pgie.set_property("config-file-path", "/opt/nvidia/deepstream/deepstream-6.1/sources/objectDetector_Yolo/config_infer_primary_yoloV3.txt")
            self.pgie.set_property("config-file-path", "../DeepStream-Yolo/config_infer_primary_yolonas.txt")
        
        else:
            if sink_type == "video":
                # create video decoder
                if encoding == "h264":
                    self.videodec = Gst.ElementFactory.make("avdec_h264", None)
                elif encoding == "h265":
                    self.videodec = Gst.ElementFactory.make("avdec_h265", None)
                else:
                    raise ValueError(encoding + " is not supported")

                # create the autovideo converter
                self.autoconv = Gst.ElementFactory.make("autovideoconvert", None)
                self.sink = Gst.ElementFactory.make("xvimagesink", None)
                self.sink.set_property("sync", False)
                self.sink.set_property("async", False)

            elif sink_type == "file":
                self.mux = Gst.ElementFactory.make("matroskamux", None)
                self.sink = Gst.ElementFactory.make("filesink", None)
                self.sink.set_property("location", "../Videos/output.mkv")
            else:
                # create a fakesink
                self.sink = Gst.ElementFactory.make("fakesink", None)

        # Add elements to pipeline
        self.pipeline.add(self.parser)
        if infer:
            self.pipeline.add(self.nvdecoder)
            self.pipeline.add(self.vidconv1)
            self.pipeline.add(self.streammux)
            self.pipeline.add(self.pgie)
            self.pipeline.add(self.nvvidconv)
            self.pipeline.add(self.nvosd)
            if sink_type == "video":
                self.pipeline.add(self.queue)
                self.pipeline.add(self.sink)
            elif sink_type == "file":
                self.pipeline.add(self.videoconv)
                self.pipeline.add(self.encoder)
                self.pipeline.add(self.parser2)
                self.pipeline.add(self.queue)
                self.pipeline.add(self.mux)
                self.pipeline.add(self.sink)
            else:
                self.pipeline.add(self.sink)
            if is_aarch64:
                self.pipeline.add(self.transform)
        else:
            if sink_type == "video":
                self.pipeline.add(self.videodec)
                self.pipeline.add(self.autoconv)
                self.pipeline.add(self.sink)
            elif sink_type == "file":
                self.pipeline.add(self.mux)
                self.pipeline.add(self.sink)
            else:
                self.pipeline.add(self.sink)

        # Link elements in the pipeline
        self.rtpdepay.link(self.parser)

        if infer:
            self.parser.link(self.nvdecoder)
            self.nvdecoder.link(self.vidconv1)
            self.sinkpad = self.streammux.get_request_pad("sink_0")
            if not self.sinkpad:
                sys.stderr.write("Unable to get the sink pad of streammux \n")
            self.srcpad = self.vidconv1.get_static_pad("src")
            if not self.srcpad:
                sys.stderr.write("Unable to get source pad of decoder \n")
            self.srcpad.link(self.sinkpad)
            self.streammux.link(self.pgie)
            self.pgie.link(self.nvvidconv)
            self.nvvidconv.link(self.nvosd)
            if is_aarch64:
                self.nvosd.link(self.transform)
                if sink_type == "video":
                    self.transform.link(self.queue)
                    self.queue.link(self.sink)
                elif sink_type == "file":
                    self.transform.link(self.videoconv)
                    self.videoconv.lin(self.encoder)
                    self.encoder.link(self.parser2)
                    self.parser2.link(self.queue)
                    self.queue.link(self.mux)
                    self.mux.link(self.sink)
                else:
                    self.transform.link(self.sink)
            else:
                if sink_type == "video":
                    self.nvosd.link(self.queue)
                    self.queue.link(self.sink)
                elif sink_type == "file":
                    self.nvosd.link(self.videoconv)
                    self.videoconv.link(self.encoder)
                    self.encoder.link(self.parser2)
                    self.parser2.link(self.queue)
                    self.queue.link(self.mux)
                    self.mux.link(self.sink)
                else:
                    self.nvosd.link(self.sink)

        else:
            if sink_type == "video":
                self.parser.link(self.videodec)
                self.videodec.link(self.autoconv)
                self.autoconv.link(self.sink)
            elif sink_type == "file":
                self.parser.link(self.mux)
                self.mux.link(self.sink)
            else:
                self.parser.link(self.sink)

        self.debug = debug
        self.infer = infer
        self.server_ip_address = server_ip_address
        self.server_port = server_port
        self.loop = None
        self.step = 0

        # Create shared queues among different threads
        self.sniff_q = Queue(maxsize=0)
        self.msg_q = Queue(maxsize=0)        

        # Create Messaging channel
        self.messaging_server = MessagingServer(ip=server_ip_address, port=msg_port, msg_q=self.msg_q, debug=self.debug)

        # Create RX packet sniffer
        self.sniffer = ServerSniffer(iface=iface, ip_server=self.server_ip_address, port_server=self.server_port,
                                     debug=self.debug, sniff_q=self.sniff_q)
        # Create BitRate estimator
        self.estimator = BitRateEstimator(delta=feedback_freq, sniff_q=self.sniff_q, msg_q=self.msg_q, debug=self.debug)

    def print_something(self):
        print("ok")
        return True

    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        frame_number=0
        #Intiallizing object counter with 0.
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE:0,
            PGIE_CLASS_ID_PERSON:0,
            PGIE_CLASS_ID_BICYCLE:0,
            PGIE_CLASS_ID_TRAFFIC_LIGHT:0
        }
        num_rects=0

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer")
            return

        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.glist_get_nvds_frame_meta()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                #frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_number=frame_meta.frame_num
            num_rects = frame_meta.num_obj_meta
            l_obj=frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    #obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                    obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # confidence = obj_meta.confidence
                if obj_meta.class_id in PGIE_CLASSES:
                    obj_counter[obj_meta.class_id] += 1
                    obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.0)
                try: 
                    l_obj=l_obj.next
                except StopIteration:
                    break

            # Acquiring a display meta object. The memory ownership remains in
            # the C code so downstream plugins can still access it. Otherwise
            # the garbage collector will claim it when this probe function exits.
            display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            py_nvosd_text_params = display_meta.text_params[0]
            # Setting display text to be shown on screen
            # Note that the pyds module allocates a buffer for the string, and the
            # memory will not be claimed by the garbage collector.
            # Reading the display_text field here will return the C address of the
            # allocated string. Use pyds.get_string() to get the string content.
            
            frame_als_msg = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])
            self.msg_q.put("|ALS|" + str(time.time_ns()) + "|" + str(frame_als_msg)) # should change the datetime.now() to the actual frame number
            py_nvosd_text_params.display_text = frame_als_msg

            # Now set the offsets where the string should appear
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 12

            # Font , font-color and font-size
            py_nvosd_text_params.font_params.font_name = "Serif"
            py_nvosd_text_params.font_params.font_size = 10
            # set(red, green, blue, alpha); set to White
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

            # Text background color
            py_nvosd_text_params.set_bg_clr = 1
            # set(red, green, blue, alpha); set to Black
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
            # Using pyds.get_string() to get display_text as string
            # print(pyds.get_string(py_nvosd_text_params.display_text))
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            try:
                l_frame=l_frame.next
            except StopIteration:
                break
                
        return Gst.PadProbeReturn.OK
    
    def run(self):

        self.loop = GLib.MainLoop()
        # Create bus to receive events from GStreamer pipeline
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message::error", self.on_error)

        if self.infer:
            # Lets add probe to get informed of the meta data generated, we add probe to
            # the sink pad of the osd element, since by that time, the buffer would have
            # had got all the metadata.
            osdsinkpad = self.nvosd.get_static_pad("sink")
            if not osdsinkpad:
                sys.stderr.write("Unable to get sink pad of nvosd \n")
            osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        print("Starting pipeline \n")
        self.pipeline.set_state(Gst.State.PLAYING)
        # ret = GLib.timeout_add_seconds(5, self.print_something)
        # ret1 = GLib.timeout_add(100, self.print_something) # calls a function every 100 ms

        # start RX packet sniffer on a separate process
        self.sniffer.start()
        self.estimator.start()
        self.messaging_server.start()
        # start GStreamer main loop
        try:
            self.loop.run()
        except:
            pass
        self.pipeline.set_state(Gst.State.NULL)
        
    def on_error(self, bus, msg):
        print("on_error():", msg.parse_error())

# def resize(self):
#     return self;
class PyTorchServer:

    def __init__(self, server_ip_address, server_port, client_ip_address, client_port, msg_port=9000,
                 feedback_freq=None, infer=False, sink_type="video", iface="wlan0", encoding="h264", model_path=None, fec=False, debug=False):
        # Create GStreamer pipeline
        if fec:
            print("Using FEC \n")
            self.pipeline= Gst.parse_launch("""udpsrc auto-multicast=False buffer-size=2147483647 address="""+server_ip_address+""" port="""+str(server_port)+""" caps="application/x-rtp, payload=96, clock-rate=90000" ! rtpstorage size-time=500000000 ! 
                rtpssrcdemux ! application/x-rtp, payload=96, clock-rate=90000, media=video, encoding-name="""+encoding.upper()+""" ! rtpjitterbuffer do-lost=1 latency=100 ! rtpulpfecdec pt=122 ! rtp"""+encoding+"""depay""")
        else:    
            self.pipeline = Gst.parse_launch("""udpsrc auto-multicast=False buffer-size=2147483647 address="""+server_ip_address+""" port="""+str(server_port)+""" caps="application/x-rtp, payload=96, clock-rate=90000" ! rtp"""+encoding+"""depay""")

        # Create payload decoder and parser
        if encoding == "h264":
            print("Using h264 decoder")
            self.rtpdepay = self.pipeline.get_by_name("rtph264depay0")
            if not self.rtpdepay:
                sys.stderr.write("Unable to get the rtp depay element \n")
            self.parser = Gst.ElementFactory.make("h264parse", None)
        elif encoding == "h265":
            print("Using h265 decoder")
            self.rtpdepay = self.pipeline.get_by_name("rtph265depay0")
            if not self.rtpdepay:
                sys.stderr.write("Unable to get the rtp depay element \n")
            self.parser = Gst.ElementFactory.make("h265parse", None)
        else:
            raise ValueError(encoding + " is not supported")

        if infer:
            # Create video decoder
            if encoding == "h264":
                self.decoder = Gst.ElementFactory.make("avdec_h264", None)
            elif encoding == "h265":
                self.decoder = Gst.ElementFactory.make("avdec_h265", None)
            else:
                raise ValueError(encoding + " is not supported")

            # Create videoconvert element
            self.vidconv = Gst.ElementFactory.make("nvvideoconvert", None)

            # Create caps filter
            self.caps = Gst.ElementFactory.make("capsfilter", None)
            self.caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))

            # Create fakesink element
            self.fakesink = Gst.ElementFactory.make("fakesink", None)
            
        else:
            if sink_type == "video":
                # create video decoder
                if encoding == "h264":
                    self.videodec = Gst.ElementFactory.make("avdec_h264", None)
                elif encoding == "h265":
                    self.videodec = Gst.ElementFactory.make("avdec_h265", None)
                else:
                    raise ValueError(encoding + " is not supported")

                # create the autovideo converter
                self.autoconv = Gst.ElementFactory.make("autovideoconvert", None)
                self.sink = Gst.ElementFactory.make("xvimagesink", None)
                self.sink.set_property("sync", False)
                self.sink.set_property("async", False)

            elif sink_type == "file":
                self.mux = Gst.ElementFactory.make("matroskamux", None)
                self.sink = Gst.ElementFactory.make("filesink", None)
                self.sink.set_property("location", "../Videos/output.mkv")
            else:
                # create a fakesink
                self.sink = Gst.ElementFactory.make("fakesink", None)

        # Add elements to pipeline
        self.pipeline.add(self.parser)
        if infer:
            self.pipeline.add(self.decoder)
            self.pipeline.add(self.vidconv)
            self.pipeline.add(self.caps)
            self.pipeline.add(self.fakesink)
        else:
            if sink_type == "video":
                self.pipeline.add(self.videodec)
                self.pipeline.add(self.autoconv)
                self.pipeline.add(self.sink)
            elif sink_type == "file":
                self.pipeline.add(self.mux)
                self.pipeline.add(self.sink)
            else:
                self.pipeline.add(self.sink)

        # Link elements in the pipeline
        self.rtpdepay.link(self.parser)

        if infer:
            self.parser.link(self.decoder)
            self.decoder.link(self.vidconv)
            self.vidconv.link(self.caps)
            self.caps.link(self.fakesink)

        else:
            if sink_type == "video":
                self.parser.link(self.videodec)
                self.videodec.link(self.autoconv)
                self.autoconv.link(self.sink)
            elif sink_type == "file":
                self.parser.link(self.mux)
                self.mux.link(self.sink)
            else:
                self.parser.link(self.sink)

        self.debug = debug
        self.infer = infer
        self.server_ip_address = server_ip_address
        self.server_port = server_port
        self.loop = None
        self.step = 0
        self.channels = 3
        self.frame_num = 0

        # Create shared queues among different threads
        self.sniff_q = Queue(maxsize=0)
        self.msg_q = Queue(maxsize=0)        

        # Create Messaging channel
        self.messaging_server = MessagingServer(ip=server_ip_address, port=msg_port, msg_q=self.msg_q, debug=self.debug)

        # Create RX packet sniffer
        self.sniffer = ServerSniffer(iface=iface, ip_server=self.server_ip_address, port_server=self.server_port,
                                     debug=self.debug, sniff_q=self.sniff_q)
        # Create BitRate estimator
        self.estimator = BitRateEstimator(delta=feedback_freq, sniff_q=self.sniff_q, msg_q=self.msg_q, debug=self.debug)

        self.hloc = Hloc

        if infer:
            # Load RA-Yolo-V7 model
            if model_path is None:
                self.model_path = "../ra-yolo-v7/best.pt"
            else:
                self.model_path = model_path

            self.model = None
            try:
                self.model = torch.load(self.model_path)['model'].to(device)
            except:
                pass

            if self.model is not None:
                self.model = self.model.half().eval()
                if self.debug:
                    print(self.model)
                self.inference_time_1080 = []
                self.inference_time_720 = []
                self.inference_time_480 = []
                self.inference_samples = 200

                # Class names and color for plotting boxes
                self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
                self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

            '''# Load YoloV8 model
            self.model = YOLO("yolov8n.pt").to(device)
            self.transform = torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((640, 640))])'''

    def _buffer_to_image_tensor(self, buf, caps):
        with nvtx_range('buffer_to_image_tensor'):
            caps_structure = caps.get_structure(0)
            height, width = caps_structure.get_value('height'), caps_structure.get_value('width')

            is_mapped, map_info = buf.map(Gst.MapFlags.READ)
            if is_mapped:
                try:
                    source_surface = NvBufSurface(map_info)
                    torch_surface = NvBufSurface(map_info)

                    dest_tensor = torch.zeros(
                        (torch_surface.surfaceList[0].height, torch_surface.surfaceList[0].width, 4),
                        dtype=torch.uint8,
                        device=device
                    )

                    torch_surface.struct_copy_from(source_surface)
                    assert(source_surface.numFilled == 1)
                    # assert(source_surface.surfaceList[0].colorFormat == 27) # RGB
                    assert(source_surface.surfaceList[0].colorFormat == 19) # RGBA

                    # make torch_surface map to dest_tensor memory
                    torch_surface.surfaceList[0].dataPtr = dest_tensor.data_ptr()

                    # copy decoded GPU buffer (source_surface) into Pytorch tensor (torch_surface -> dest_tensor)
                    torch_surface.mem_copy_from(source_surface)
                finally:
                    buf.unmap(map_info)

                return dest_tensor[:, :, :self.channels]

    def on_frame_probe(self, pad, info):
        t0 = time.time()
        buf = info.get_buffer()
        image_tensor = self._buffer_to_image_tensor(buf, pad.get_current_caps())
        image_tensor = image_tensor.permute(2,0,1).unsqueeze(0).half()
        image_tensor = image_tensor / 255.

        # print("The current frame number is {}".format(self.frame_num))
        # h, w = image_tensor.shape[2], image_tensor.shape[3]
        # print("The current image resolution is: {}x{}".format(w, h))
        # print(image_tensor)
        # img = resize(img)
        pose = self.hloc.get_location(image_tensor)
        self.msg_q.put(str(pose) + str(t0))
        # inference with RA-Yolo
        '''if h == 1080:
            # pad_h, pad_w = (128 - h%128)//2, (128 - w%128)//2
            # pad_h, pad_w = 36, 0   # Padding on both sides, width is multiple of 128 no need padding
            image_tensor = torch.nn.functional.pad(image_tensor, (0,0,36,36), 'constant')   # zero padding to multiple of 128
        elif h == 720:
            image_tensor = torch.nn.functional.pad(image_tensor, (0,0,24,24), 'constant')
        elif h == 480:
            image_tensor = torch.nn.functional.pad(image_tensor, (0,0,16,16), 'constant')
        else:
            raise NotImplemented 
        # print(image_tensor)
        # print(image_tensor.shape)
        # print(image_tensor[0,0,37:37+8,0:8])
        
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = non_max_suppression(output[0], 0.25, 0.45, classes=None, agnostic=False)

        image_np = image_tensor.to(torch.float32).cpu().numpy().squeeze()
        image_np = np.moveaxis(image_np, 0, -1)
        image_np_orig = image_np.copy()
        image_np = image_np.copy()
        for *xyxy, conf, cls_ in reversed(pred[0]):
            label = f'{self.names[int(cls_)]} {conf:.2f}'
            color = self.colors[int(cls_)]
            plot_one_box(xyxy, image_np, label=label, color=color)

        # saving frames for debugging
        image_np_orig = cv2.cvtColor((image_np_orig*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite("./debug/orig/frame_number_"+str(self.frame_num)+".jpeg", image_np_orig)
        image_np = cv2.cvtColor((image_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite("./debug/pred/frame_number_"+str(self.frame_num)+".jpeg", image_np)
        
        # self.inference_time_record.append(t_inf)
        # if len(self.inference_time_record) > 50:
        #     self.inference_time_record.pop(0)
        # avg_inf_time = np.mean(self.inference_time_record)

        t_inf = 0
        if h == 1080:
            self.inference_time_1080.append(t_inf)
        elif h == 720:
            self.inference_time_720.append(t_inf)
        elif h == 480:
            self.inference_time_480.append(t_inf)
        else:
            raise NotImplemented

        if len(self.inference_time_1080) > self.inference_samples:
            self.inference_time_1080.pop(0)
        if len(self.inference_time_720) > self.inference_samples:
            self.inference_time_720.pop(0)
        if len(self.inference_time_480) > self.inference_samples:
            self.inference_time_480.pop(0)

        if self.debug and len(self.inference_time_1080) > 0 and len(self.inference_time_720) > 0 and len(self.inference_time_480) > 0:
            # avg_inf_time = np.mean(self.inference_time_record)
            print("\nAvg Inference Time 1080 [ms]: {:.2f}".format(np.mean(self.inference_time_1080)*1e3))
            print("Avg Inference Time 720  [ms]: {:.2f}".format(np.mean(self.inference_time_720)*1e3))
            print("Avg Inference Time 480  [ms]: {:.2f}\n".format(np.mean(self.inference_time_480)*1e3))

            print("Avg Inference FPS 1080 [fps]: {:.2f}".format(1. / np.mean(self.inference_time_1080)))
            print("Avg Inference FPS 720  [fps]: {:.2f}".format(1. / np.mean(self.inference_time_720)))
            print("Avg Inference FPS 480  [fps]: {:.2f}\n".format(1. / np.mean(self.inference_time_480)))

        # saving frames for debugging
        #image_np = image_tensor.to(torch.uint8).cpu().numpy().squeeze()
        #image_np = np.moveaxis(np_img, 0, -1)
        #img = Image.fromarray(image_np/255, "RGB")
        #img_filename = "./debug/frame_number_"+str(self.frame_num)+".jpeg"
        #img.save(img_filename)

        t_inf = time.time() - t0
        print("The current inference time is: {:.2f} ms".format(t_inf*1e3))'''

        print(50*"-")
        self.frame_num += 1
        return Gst.PadProbeReturn.OK, pose

    def run(self):

        self.loop = GLib.MainLoop()
        # Create bus to receive events from GStreamer pipeline
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message::error", self.on_error)

        if self.infer:
            fakesinkpad = self.fakesink.get_static_pad("sink")
            if not fakesinkpad:
                sys.stderr.write("Unable to get sink pad of fakesink \n")
            fakesinkpad.add_probe(Gst.PadProbeType.BUFFER, self.on_frame_probe)
            print("Probing sink pad of fakesink")

        print("Starting pipeline \n")
        self.pipeline.set_state(Gst.State.PLAYING)
        # ret = GLib.timeout_add_seconds(5, self.print_something)
        # ret1 = GLib.timeout_add(100, self.print_something) # calls a function every 100 ms

        # start RX packet sniffer on a separate process
        self.sniffer.start()
        self.estimator.start()
        self.messaging_server.start()
        # start GStreamer main loop
        try:
            self.loop.run()
        except:
            pass
        self.pipeline.set_state(Gst.State.NULL)
        
    def on_error(self, bus, msg):
        print("on_error():", msg.parse_error())


# class Connected_Client(threading.Thread):
#     def __init__(self, socket=None, address='128.122.136.173', hloc=None, trajectory=None, connections=None,
#                  destinations=None, map_scale=1, log_dir=None, logger=None):
#         threading.Thread.__init__(self)
#         self.socket = socket
#         self.address = address
#         self.id = len(connections)
#         self.connections = connections
#         self.signal = True
#         self.total_connections = 0
#         self.hloc = hloc
#         self.trajectory = trajectory
#         self.destination = destinations
#         self.destinations_dicts = {}
#         for k0, v0 in destinations.items():
#             building_dicts = {}
#             for k1, v1 in v0.items():
#                 floor_dicts = {}
#                 for k2, v2 in v1.items():
#                     list0 = []
#                     for v3 in v2:
#                         list0.append(list(v3.keys())[0])
#                     floor_dicts.update({k2: list0})
#                 building_dicts.update({k1: floor_dicts})
#             self.destinations_dicts.update({k0: building_dicts})
#         self.log_dir = log_dir
#         self.logger = logger
#         self.map_scale = map_scale
#
#     def __str__(self):
#         return str(self.id) + " " + str(self.address)
#
#     def recvall(self, sock, count):
#         buf = b''
#         while count:
#             newbuf = sock.recv(count)
#             if not newbuf:
#                 return None
#             buf += newbuf
#             count -= len(newbuf)
#         return buf
#
#     def date(self, s):
#         return [s.year, s.month, s.day, s.hour, s.minute, s.second]
#
#     def run(self):
#         while self.signal:
#             try:
#                 number = self.recvall(self.socket, 4)
#                 if not number:
#                     continue
#                 command = int.from_bytes(number, 'big')
#                 if command == 1:
#                     self.logger.info('===========Loading image===========')
#                     length = self.recvall(self.socket, 4)
#                     data = self.recvall(self.socket, int.from_bytes(length, 'big'))
#                     if not data:
#                         continue
#                     nparr = np.frombuffer(data, np.uint8)
#                     img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#                     Destination = self.socket.recv(4096)
#                     Destination = jpysocket.jpydecode(Destination)
#                     Place, Building, Floor, Destination_ = Destination.split(',')
#                     dicts = self.destination[Place][Building][Floor]
#                     for i in dicts:
#                         for k, v in i.items():
#                             if k == Destination_:
#                                 destination_id = v
#                     self.logger.info('=========Received one image=========')
#                     pose = self.hloc.get_location(img)  # img - pose
#                     image_destination = join(
#                         self.log_dir, destination_id, 'images')
#                     if not exists(image_destination):
#                         makedirs(image_destination)
#                     message_destination = join(
#                         self.log_dir, destination_id, 'logs')
#                     if not exists(message_destination):
#                         mkdir(message_destination)
#                     current_time = time()
#                     readable_date = datetime.fromtimestamp(current_time)
#                     formatted_date = readable_date.strftime('%Y-%m-%d_%H-%M-%S')
#
#                     image_num = len(self.hloc.list_2d)
#                     cv2.imwrite(
#                         join(image_destination, formatted_date + '.png'), img)
#                     if pose:
#
#                         self.logger.info(
#                             f"===============================================\n                                                       Estimated location: x: %d, y: %d, ang: %d\n                                                       Used {image_num} images for localization\n                                                       ===============================================" % (
#                                 pose[0], pose[1], pose[2]))
#                         path_list = self.trajectory.calculate_path(pose[:2], destination_id)
#
#                         if image_num == 1:
#                             message = command_type0(pose, path_list, self.map_scale)
#                         else:
#                             message = command_type0(pose, path_list, self.map_scale)
#
#                         self.logger.info(
#                             f"===============================================\n                                                       {message}\n                                                       ===============================================")
#                         self.socket.sendall(bytes(message, 'UTF-8'))
#
#                         with open(join(message_destination, formatted_date + '.txt'), "w") as file:
#                             file.write(str(pose[0]) + ', ' + str(pose[1]) + '\n')
#                             file.write(message)
#
#                     else:
#                         pass
#
#                 elif command == 0:
#                     self.logger.info('=====Send destination to Client=====')
#                     destination_dicts = str(self.destinations_dicts) + '\n'
#                     self.socket.sendall(bytes(destination_dicts, 'UTF-8'))
#             except:
#                 self.logger.warning("Client " + str(self.address) + " has disconnected")
#                 self.signal = False
#                 self.connections.remove(self)
#                 break