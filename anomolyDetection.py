
import pyshark
#from sklearn import *

def sanitizeData():
    cap = pyshark.FileCapture('mstp_2018_Bad1.cap')
    for packet in cap:
        print(packet)
    print("Sanitizing")
    # Sanitizes the data so it can be passed into a scikit leanr classifier

def classifier():
    print("classifying")
    # Runs scikit leanr classifier to learn patterns in data sets

sanitizeData()
