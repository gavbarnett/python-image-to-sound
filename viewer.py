import pyaudio
import numpy as np
import cv2
import time
import math
import copy

quant_v = 10
quant_h = 10

def main():
    p = pyaudio.PyAudio()
    fs = 42000 # sampling rate, Hz, must be integer

    cap = cv2.VideoCapture(0)
    #generate tones
    fmax = 10000
    fmin = 500
    fdiv = (math.log(fmax)-math.log(fmin))/quant_v
    t = []
    for n in reversed(range(quant_v)):
        t.append(tonemaker(math.exp(math.log(fmin) + fdiv*n), fs))
  
    stream = p.open(format=pyaudio.paFloat32,
                channels=2,
                rate=fs,
                output=True)
    oldrect = []
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        horizontal_img = gray #cv2.flip(gray, 1)
        height, width = gray.shape

        h = round(height/quant_v)
        w = round(width/quant_h)
        rect = []
        rect_sum = []
        t_tot = []
        f_tot = [[],[]]
        for nw in range(quant_h):
            rect.append([])
        
            rect[nw] = np.power(255-np.mean(np.mean(np.mean(horizontal_img[np.meshgrid(np.arange(0,height-h,h),np.arange(h,height,h)),w*nw:w*nw+w],3),2),0),5)/(math.pow(255,5)*quant_v)
            rect_sum.append(sum(rect[nw]))
            t_tot.append(0)

            if not(oldrect):
                for nv in range(quant_v-1):
                    t_tot[nw] += rect[nw][nv]*t[nv]
            else:
                for nv in range(quant_v-1):
                    temp = copy.copy(t[nv])
                    temp *= np.ogrid[oldrect[nw][nv]:rect[nw][nv]:1j*fs]
                    t_tot[nw] += temp
                    #print(np.ogrid[oldrect[nw][nv]:rect[nw][nv]:42000j], t[nv])
            t_tot[nw] /= 0.5*quant_h # this could be improved with an exponent.

        oldrect = copy.deepcopy(rect)

        #t_tot = stereo_zipper(tonewindow(t_tot[0],4), tonewindow(t_tot[1],4))
        f_tot[0] = t_tot[0]
        f_tot[1] = t_tot[quant_h-1]
        for nw in range(1,quant_h-2): # this could be improved with an exponent.
            f_tot[0] += t_tot[nw]*nw/quant_h
            f_tot[1] += t_tot[nw]*(1-nw/quant_h)
        t_tot = stereo_zipper(f_tot[0], f_tot[1])
        stream.write(t_tot)

        cv2.imshow('frame1',horizontal_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stream.stop_stream()
            stream.close()
            p.terminate()
            break

    cap.release()
    cv2.destroyAllWindows()

def tonemaker(f, fs):
    duration = 1   # in seconds, may be float
    samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
    #samples *= np.hanning(fs*duration)
    return (samples)

def stereo_zipper(left, right):

    samples = np.vstack((left,right)).reshape((-1,),order='F')
    return(samples)
    
main()