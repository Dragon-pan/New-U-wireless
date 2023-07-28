# -*- coding: utf-8 -*-

"""
Messaging classes.

@author: Tommy Azzino [tommy.azzino@gmail.com]
"""

import socket
from threading import Thread

class MessagingServer(Thread):
    def __init__(self, ip, port, msg_q=None, debug=False):
        super().__init__()
        self.ip = ip
        self.port = port
        self.msg_q = msg_q
        self.debug = debug
        self.message_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        self.message_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("Binding socket to IP: {}, PORT: {}".format(self.ip, self.port))
        self.message_sock.bind((self.ip, self.port))

    def run(self):
        self.message_sock.listen()
        conn, addr = self.message_sock.accept()
        with conn:
            while True:
                message = self.msg_q.get()
                if self.debug:
                    print("Sending message: ", message)
                conn.sendall(message.encode("utf-8"))

class MessagingClient(Thread):
    def __init__(self, ip, port, msg_q=None, debug=False):
        super().__init__()
        self.ip = ip
        self.port = port
        self.msg_q = msg_q
        self.debug = debug
        self.message_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        print("Connecting socket to IP: {}, PORT: {}".format(self.ip, self.port))
        self.message_sock.connect((self.ip, self.port))

    def run(self):
        while True:
            message = self.message_sock.recv(1024)
            if self.debug:
                print("Receveid message: ", message)
            self.msg_q.put(message.decode("utf-8"))
