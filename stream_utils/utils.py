# -*- coding: utf-8 -*-

"""
Video streaming utils

@author: Tommy Azzino [tommy.azzino@gmail.com]
"""

import pyric.pyw as pyw
from subprocess import Popen, PIPE


def get_wireless_iface():
    target_if_idx = None
    wifs = pyw.winterfaces()
    for i, wif in enumerate(wifs):
        try:
            w = pyw.getcard(wif)
            if pyw.isup(w):
                target_if_idx = i
        except:
            target_if_idx = i
            continue
    if target_if_idx is not None:
        return wifs[target_if_idx]
    return target_if_idx

def get_eth_iface():
    target_if = None
    n = 0
    while n < 10:
        iface = "eth"+str(n)
        if pyw.isinterface(iface):
            try:
                e = pyw.getcard(iface)
                if pyw.isconnected(e):
                    target_if = iface 
                    break
            except:
                target_if = iface
                break
        n+=1

    if target_if is None:
        interfaces = pyw.interfaces()
        for interface in interfaces:
            if "en" in interface:
                target_if = interface

    return target_if

def get_cam():
    cmd = ["/usr/bin/v4l2-ctl", "--list-devices"]
    out, err = Popen(cmd, stdout=PIPE, stderr=PIPE).communicate()
    out, err = out.strip(), err.strip()
    for l in [i.split("\n\t") for i in out.decode().split("\n\n")]:
        if "Arducam" in l[0]:
            return l[1]
    return None

