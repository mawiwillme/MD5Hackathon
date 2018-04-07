
import pyshark
#from sklearn import *

def sanitizeData():
    counter = 0
    total = 0
    cap = pyshark.FileCapture('mstp_2018_Bad1.cap')
    for packet in cap:
        total += 1
        layer = packet.mstp
        frameType = layer.frame_type
        if(frameType == '1'):
            print("HERE")
            counter += 1
    print(counter)
    print(total)
    print("Sanitizing")
    # Sanitizes the data so it can be passed into a scikit leanr classifier

def classifier():
    print("classifying")
    # Runs scikit leanr classifier to learn patterns in data sets

sanitizeData()
