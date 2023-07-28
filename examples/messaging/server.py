import socket
from multiprocessing import Process, Queue
from datetime import datetime

message_buffer = Queue(maxsize=0)

class Messaging(Process):
    def __init__(self,queue=None, debug=False):
        super().__init__()
        self.q = queue
        self.message_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        self.message_sock.bind(("127.0.0.1", 9000))

        ## for testing purposes
        message_buffer.put("ALS |" + str(datetime.now()) + "|" + " class1 12 23 23 ,class 2 23 45 56, class3 23 32 31")
        message_buffer.put("ALS |" + str(datetime.now()) + "|" + " class1 12 23 23 ,class 2 23 45 56, class3 23 32 31")
        message_buffer.put("BW |" + str(datetime.now()) + "|" + "5000000")
        message_buffer.put("OTH |" + str(datetime.now()) + "|" + " Hello World")


    def __dewrap(message):
        print(message)

    def run(self):
        self.message_sock.listen()
        conn, addr = self.message_sock.accept()
        with conn:
            while True:
                if (not(self.q.empty())):
                    message = self.q.get()
                    conn.sendall(message.encode("utf-8"))
                else:
                    incoming_msg = conn.recv(4096)
                    #self.__dewrap(incoming_msg)
                    print(incoming_msg)

if __name__ == '__main__':  
    Msg = Messaging(message_buffer)
    Msg.start()
    Msg.join()
