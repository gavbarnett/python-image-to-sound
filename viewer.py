import pyaudio
import numpy as np
import cv2
import time
import math

quant_v = 10

def main():
    p = pyaudio.PyAudio()
    fs = 42000       # sampling rate, Hz, must be integer

    cap = cv2.VideoCapture(0)
    #generate tones
    fmax = 10000
    fmin = 500
    fdiv = (fmax-fmin)/quant_v
    t = []
    for n in reversed(range(quant_v)):
        t.append(tonemaker(fmin + fdiv*n, fs))
  
    stream = p.open(format=pyaudio.paFloat32,
                channels=2,
                rate=fs,
                output=True)

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        horizontal_img = gray #cv2.flip(gray, 1)
        height, width = horizontal_img.shape

        h = round(height/quant_v)
        w = round(width/2)
        rect = []
        rect_sum = []
        t_tot = []
        for nw in range(2):
            #linedw = cv2.line(horizontal_img, (w*nw, 0), (w*nw, width), (50), 2)
            rect.append([])
           
            rect[nw] = np.power(255-np.mean(np.mean(np.mean(horizontal_img[np.meshgrid(np.arange(0,height-h,h),np.arange(h,height,h)),w*nw:w*nw+w],2),1),0),5)/(math.pow(255,5)*quant_v)
            #print(rect[nw])

            rect_sum.append(sum(rect[nw]))
            t_tot.append(0)
            for nv in range(quant_v-1):
                t_tot[nw] += rect[nw][nv]*t[nv]

        #t_tot = stereo_zipper(tonewindow(t_tot[0],4), tonewindow(t_tot[1],4))
        t_sum = sum(t_tot[0]) + sum(t_tot[1])
        t_tot = stereo_zipper(t_tot[0], t_tot[1])
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
    return (samples)

def stereo_zipper(left, right):

    samples = np.vstack((left,right)).reshape((-1,),order='F')
    return(samples)
    
main()