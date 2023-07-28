################################################################################
# Gstreamer UDP sender pipeline code.
# Uses the Arducam to get the MJPEG compressed video format.
# Decodes and re-encodes it in H264  with hardware accelration encoder plugins.
# The encoded/compressed videostream is further parsed and RTP payload is added.
# Then the H264 stream is sent to a receiver with an UDP sink.
#
# When the pipeline is executed, the function "bitrate_change" is called every 1s and 
# the encoder bitrate is halved. Minimum bitrate is 1 Mbps. Every 20s the encoder 
# bitrate is brought back to 20 Mbps and the same process starts.
#
# Need the basic Gstreamer librariers.
# Accelerated plugins - OMX plugins for encoding and decoding.
#
# The program can be run as follows:
# python3 gs_tx_pipeline.py -ip RX_IP_ADDRESS -b 10000000
#################################################################################

import argparse
import sys
sys.path.append('../')
from threading import Timer
from time import sleep
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst, GstVideo

def main(args):
    # Standard GStreamer initialization
    global k 
    k = 1
    Gst.init(None)
    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element is the Arducam
    source = Gst.ElementFactory.make("v4l2src", None)
    if not source:
        sys.stderr.write(" Unable to create Source \n")
    source.set_property('device','/dev/video0')
        
    # Create a caps filter
    caps1 = Gst.ElementFactory.make("capsfilter", None)
    caps1.set_property("caps", Gst.Caps.from_string("image/jpeg, width=1920, height=1080, framerate=30/1"))

    # Create the jpdec decoder
    decoder = Gst.ElementFactory.make("jpegdec", None)
    if not decoder:
        sys.stderr.write(" Unable to create jpegdec decoder")
        
    # Create the autovideo converter for automatic video format conversion between plugins
    autoconv = Gst.ElementFactory.make("autovideoconvert", None)
    if not autoconv:
        sys.stderr.write(" Unable to create autovideo converter")

    # Create the hardware accelarated encoder
    encoder = Gst.ElementFactory.make("omxh264enc", None)
    print("Creating H264 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property('bitrate', bitrate)
    encoder.set_property('iframeinterval', iframe_interval)
    encoder.set_property('control-rate', 2)
    encoder.set_property('EnableStringentBitrate', True)
    # encoder.set_property('insert-sps-pps', True)
    # encoder.set_property('EnableTwopassCBR', True)
    # encoder.set_property('preset-level', 0)
    # encoder.set_property('low-latency', False)
    # encoder.set_property('rc-mode','cbr')
    
    # Create a caps filter
    caps2 = Gst.ElementFactory.make("capsfilter", None)
    caps2.set_property("caps", Gst.Caps.from_string("video/x-h264,profile=baseline,stream-format=byte-stream"))
    
    # Since the data format in the input file is elementary h264 stream, we need a h264parser
    h264parser = Gst.ElementFactory.make("h264parse", None)
    if not h264parser:
        sys.stderr.write("Unable to create h264 parser \n")

    # Make the payload-encode video into RTP packets
    # Set MTU to be lower than 1390,
    # Default is 1400. It seems that 1390 is the magic number.
    rtppay = Gst.ElementFactory.make("rtph264pay", None)
    if not rtppay:
        sys.stderr.write("Unable to create rtppay")
    rtppay.set_property('mtu', mtu_size)

    # Create the UDP sink
    updsink_port_num = 8000
    sink = Gst.ElementFactory.make("udpsink", None)
    print("Creating UDP sink element \n")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")
    sink.set_property('host', rx_ip_address)
    sink.set_property('port', updsink_port_num)
    sink.set_property('async', False)
    sink.set_property('sync', 1)

    # Adding elements to the pipeline
    print("Adding elements to the Pipeline \n")
    pipeline.add(source)
    pipeline.add(caps1)
    pipeline.add(decoder)
    pipeline.add(autoconv)
    pipeline.add(encoder)
    pipeline.add(caps2)
    pipeline.add(h264parser)
    pipeline.add(rtppay)
    pipeline.add(sink)

    # Link elements in the pipeline
    print("Linking elements in the Pipeline \n")
    source.link(caps1)
    caps1.link(decoder)
    decoder.link(autoconv)
    autoconv.link(encoder)
    encoder.link(caps2)
    caps2.link(h264parser)
    h264parser.link(rtppay)
    rtppay.link(sink)

    # Create an event loop and feed gstreamer bus messages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    #bus.connect ("message", bus_call, loop)
    ## omitted the bus for compiling purposes but need to create a
    # bus system for stable internal messaging and error handling purposes

    def bitrate_change(): 
        global k
        encoder.set_state(Gst.State.READY)
        pipeline.set_state(Gst.State.PAUSED)
        
        new_bitrate = max(encoder.get_property('bitrate')/1.5, 1e6)
        if k % 20 == 0:
            new_bitrate = 20000000
        # print(f"changing from {encoder.get_property('bitrate')} to {new_bitrate}")
        encoder.set_property('bitrate', int(new_bitrate))
        # encoder.set_property('peak-bitrate', int(new_birate*1.2))
        pipeline.set_state(Gst.State.PLAYING)
        encoder.set_state(Gst.State.PLAYING)

        #event = GstVideo.video_event_new_downstream_force_key_unit(pipeline.get_pipeline_clock().get_time(), 0, 0, True, 0)
        #pipeline.send_event(event)
        print("the new encoder bitrate is: {:.2f} Mbps".format(encoder.get_property('bitrate')/1e6))
        print('*'*50)
        #try:
        # loop.run()
        #except:
        # pass
        # cleanup
        k = k + 1
        print(k)
        return True

    # Start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    # can send arguments for the functions
    # creating a dummy argument
    ret = GLib.timeout_add_seconds(1, bitrate_change)
    loop.run()
    # pipeline.set_state(Gst.State.NULL)
    # t = Timer(60.0, bitrate_change(pipeline,encoder,loop))
    # t.start()

def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help')
    parser.add_argument("-ip", "--rx", default="localhost", help="Set the receiver IP address")
    parser.add_argument("-b", "--bitrate", default=5000000, help="Set the encoder bitrate", type=int)
    parser.add_argument("-ifi", "--ifinter", default=32, help="Set I-frame interarrival interval", type=int)

    # Check input arguments
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    global rx_ip_address
    global bitrate
    global stream_path
    global iframe_interval
    rx_ip_address = args.rx
    bitrate = args.bitrate
    #stream_path = args.input
    iframe_interval = args.ifinter

    global mtu_size
    mtu_size = 1390
    return 0

if __name__ == '__main__':
    parse_args()
    sys.exit(main(sys.argv))