# client.py

import socket
import multiprocessing
from multiprocessing import Process, Queue

BW_buffer = Queue(maxsize=0)
ALS_buffer = Queue(maxsize=0)
other_buffer = Queue(maxsize=0) ## buffers for incoming data before being serviced

message_buffer = Queue(maxsize = 0) # outgoing data buffer

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 9000  # The port used by the server

class Messaging(Process):
    def __init__(self, bw=Queue, als=Queue, oth=Queue, q=Queue, host=str,port=int):
        super().__init__()
        self.bwq = bw
        self.alsq = als
        self.othq = oth
        self.q = q
        self.host = host
        self.host_port = port
        self.message_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  

    def __unwrap(message):
         print(message)
         #check for time
         #check for msg type

    def run(self):
        self.message_socket.connect((self.host,self.host_port))
        self.message_socket.send("hello".encode("utf-8"))
        while True:
                if (not(self.q.empty())):
                    message = self.q.get()
                    self.message_socket.sendall(message.encode("utf-8"))
                else:
                    incoming_msg = self.message_socket.recv(4096)
                    #self.__unwrap(incoming_msg)
                    print(incoming_msg)

if __name__ == '__main__':    
    Msg = Messaging(BW_buffer, ALS_buffer, other_buffer, message_buffer, HOST, PORT)
    Msg.start()
    Msg.join()
