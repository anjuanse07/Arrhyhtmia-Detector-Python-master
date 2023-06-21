import asyncio
import serial
import time
import csv
import matplotlib.pyplot as plt
from paho.mqtt import client as mqtt_client
import json
import anfis1 as anfis
import mfDerivs
import membershipfunction
import numpy as np
import pickle
from datetime import datetime
from openpyxl import load_workbook
import pandas as pd
import numpy
import scipy.signal
import scipy.io
import math
import neurokit2 as nk
from scipy.fftpack import fft
from scipy import signal

with open('Model_Anfis_ClassifyTrainlabel_8N', 'rb') as modelku:
    model = pickle.load(modelku)

timerecord = 10
data = []

packedData = {
    "id" : "1",
    "rr": None,
    "rr_stdev": None,
    "pr": None,
    "pr_stdev": None,
    "qs": None,
    "qs_stdev": None,
    "qt": None,
    "qt_stdev": None,
    "st": None,
    "st_stdev": None,
    "heartrate": None,
    "classification": None,
    "ecg_graph": None
}

ecg_grap = {
    "ecg_graph": None
}

def mqtt_publish(data):
    mqtt_broker = "localhost"
    mqtt_port = 1883
    mqtt_topic = "ecg/analysisparsed"
    
    client = mqtt_client.Client()
    client.connect(mqtt_broker, mqtt_port)
    client.publish(mqtt_topic, str(data))
    client.disconnect()

def convertToVolt(n):
    return (n / 1024 - (1 / 2)) * 3.3 / 1100 * 1000

def preprocess_data(device_id, received_data):
    ecgmvconv = list(map(convertToVolt, received_data))
    ecgmv = data
    detr_restingecg1 = scipy.signal.detrend(ecgmvconv, axis=-1, type='linear', bp=0, overwrite_data=False)
    detr_restingecg = scipy.signal.detrend(ecgmv, axis=-1, type='linear', bp=0, overwrite_data=False)

    y1 = [e for e in detr_restingecg1]
    y = [e for e in detr_restingecg1]
    N = len(y)
    Fs = int(N / timerecord) + 1
    T = 1.0 / Fs
    x = np.linspace(0.0, N * T, N)

    yf1 = scipy.fftpack.fft(y1)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    Wn1 = np.around(26 / (Fs / 2), 2)
    Ts = 1 / Fs
    Ohm1 = (2 / Ts) * (math.tan(24 * Ts / 2))
    Ohm2 = (2 / Ts) * (math.tan(30 * Ts / 2))
    Ohmr = Ohm2 / Ohm1
    orde = math.log10(((10 ** 1) - 1) / ((10 ** 0.2) - 1)) / (2 * math.log10(Ohmr))
    Norde = np.ceil(orde)

    # Compute filtering co-efficients to eliminate 50hz brum noise
    b1, a1 = signal.butter(4, 0.6, 'low')
    # Compute filtered signal
    tempf1 = signal.filtfilt(b1, a1, y1)

    #FIR Filter
    nyq_rate = Fs / 2
    # The desired width of the transition from pass to stop.
    width = 5.0 / nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0
    # Compute the order and Kaiser parameter for the FIR filter.
    O1, beta1 = signal.kaiserord(ripple_db, width)

    if (O1 % 2) == 0:
        O1 = O1 + 1
    else:
        O1 = O1

    # The cutoff frequency of the filter.
    cutoff_hz = 4.0
    # Use firwin with a Kaiser window to create a lowpass FIR filter.###
    taps1 = signal.firwin(O1, cutoff_hz/nyq_rate, window=('kaiser', beta1), pass_zero=False)
    # taps = signal.firwin(O, 0.06, window=('kaiser', beta), pass_zero=False)
    # Use lfilter to filter x with the FIR filter.
    y_filt1 = signal.lfilter(taps1, 1.0, tempf1)
    y_filter1 = y_filt1.tolist()

    try:

     # Find Peaks
     _, rpeaks = nk.ecg_peaks(y_filt1, sampling_rate=Fs)
     # signal_peak, waves_peak = nk.ecg_delineate(y_filt, rpeaks, sampling_rate=1000, method="peak")
     signal_dwt, waves_dwt = nk.ecg_delineate(y_filt1, rpeaks, sampling_rate=Fs, method="dwt")

    except:

       Hasil = 'Electrodes are not installed properly'
       packedData = {
           "id":device_id,
            "rr_avg": 0,
            "rr_dev": 0,
            "pr_avg": 0,
            "pr_dev": 0,
            "qs_avg": 0,
            "qs_dev": 0,
            "qt_avg": 0,
            "qt_dev": 0,
            "st_avg": 0,
            "st_dev": 0,
            "classification_result": Hasil,
            "heart_rate": 0,
            "ecg_graph": data
        }
       print('Diagnosa =', Hasil)
       print("frequency sampling = %.01f\n\n\n" % (Fs))  # Round off to 1 decimal and print


    else:

        y = [e for e in detr_restingecg]
        N = len(y)
        Fs = int(N / timerecord) + 1
        T = 1.0 / Fs
        x = np.linspace(0.0, N * T, N)

        yf = scipy.fftpack.fft(y)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        # Norde = 4
        # Wn1 = 0.6
        Wn1 = np.around(26 / (Fs / 2), 2)
        Ts = 1 / Fs
        Ohm1 = (2 / Ts) * (math.tan(24 * Ts / 2))
        Ohm2 = (2 / Ts) * (math.tan(30 * Ts / 2))
        Ohmr = Ohm2 / Ohm1
        orde = math.log10(((10 ** 1) - 1) / ((10 ** 0.2) - 1)) / (2 * math.log10(Ohmr))
        Norde = np.ceil(orde)

        # Compute filtering co-efficients to eliminate 50hz brum noise
        b, a = signal.butter(Norde, Wn1, 'low')
        # Compute filtered signal
        tempf = signal.filtfilt(b, a, y)
        tempf2 = signal.filtfilt(b, a, y1)

        # FIR Filter
        # Compute Kaiser window co-effs to eliminate baseline drift noise
        nyq_rate = Fs / 2
        # The desired width of the transition from pass to stop.
        width = 5.0 / nyq_rate
        # The desired attenuation in the stop band, in dB.
        ripple_db = 60.0
        # Compute the order and Kaiser parameter for the FIR filter.
        O, beta = signal.kaiserord(ripple_db, width)

        if (O % 2) == 0:
            O = O + 1
        else:
            O = O

        # The cutoff frequency of the filter.
        cutoff_hz = 4.0
        # Use firwin with a Kaiser window to create a lowpass FIR filter.###
        taps = signal.firwin(O, cutoff_hz / nyq_rate, window=('kaiser', beta), pass_zero=False)
        # taps = signal.firwin(O, 0.06, window=('kaiser', beta), pass_zero=False)
        # Use lfilter to filter x with the FIR filter.
        y_filt = signal.lfilter(taps, 1.0, tempf)
        y_filt2 = signal.lfilter(taps, 1.0, tempf2)
        y_filter2 = y_filt2.tolist()

        # Find Peaks
        _, rpeaks = nk.ecg_peaks(y_filt, sampling_rate=Fs)
        # signal_peak, waves_peak = nk.ecg_delineate(y_filt, rpeaks, sampling_rate=1000, method="peak")
        signal_dwt, waves_dwt = nk.ecg_delineate(y_filt, rpeaks, sampling_rate=Fs, method="dwt")

        # Remove Nan and change to ndarray int
        rpeaks['ECG_R_Peaks'] = np.array([x for x in rpeaks['ECG_R_Peaks'] if math.isnan(x) is False]).astype(int)
        waves_dwt['ECG_P_Peaks'] = np.array([x for x in waves_dwt['ECG_P_Peaks'] if math.isnan(x) is False]).astype(int)
        waves_dwt['ECG_Q_Peaks'] = np.array([x for x in waves_dwt['ECG_Q_Peaks'] if math.isnan(x) is False]).astype(int)
        waves_dwt['ECG_S_Peaks'] = np.array([x for x in waves_dwt['ECG_S_Peaks'] if math.isnan(x) is False]).astype(int)
        waves_dwt['ECG_T_Peaks'] = np.array([x for x in waves_dwt['ECG_T_Peaks'] if math.isnan(x) is False]).astype(int)
        waves_dwt['ECG_P_Onsets'] = np.array([x for x in waves_dwt['ECG_P_Onsets'] if math.isnan(x) is False]).astype(int)
        waves_dwt['ECG_P_Offsets'] = np.array([x for x in waves_dwt['ECG_P_Offsets'] if math.isnan(x) is False]).astype(int)
        waves_dwt['ECG_R_Onsets'] = np.array([x for x in waves_dwt['ECG_R_Onsets'] if math.isnan(x) is False]).astype(int)
        waves_dwt['ECG_R_Offsets'] = np.array([x for x in waves_dwt['ECG_R_Offsets'] if math.isnan(x) is False]).astype(int)
        waves_dwt['ECG_T_Onsets'] = np.array([x for x in waves_dwt['ECG_T_Onsets'] if math.isnan(x) is False]).astype(int)
        waves_dwt['ECG_T_Offsets'] = np.array([x for x in waves_dwt['ECG_T_Offsets'] if math.isnan(x) is False]).astype(int)

        # Correcting first cycle
        if rpeaks['ECG_R_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
            rpeaks['ECG_R_Peaks'] = np.delete(rpeaks['ECG_R_Peaks'], 0)
        if waves_dwt['ECG_P_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
            waves_dwt['ECG_P_Peaks'] = np.delete(waves_dwt['ECG_P_Peaks'], 0)
        if waves_dwt['ECG_Q_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
            waves_dwt['ECG_Q_Peaks'] = np.delete(waves_dwt['ECG_Q_Peaks'], 0)
        if waves_dwt['ECG_S_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
            waves_dwt['ECG_S_Peaks'] = np.delete(waves_dwt['ECG_S_Peaks'], 0)
        if waves_dwt['ECG_T_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
            waves_dwt['ECG_T_Peaks'] = np.delete(waves_dwt['ECG_T_Peaks'], 0)
        if waves_dwt['ECG_P_Offsets'][0] < waves_dwt['ECG_P_Onsets'][0]:
            waves_dwt['ECG_P_Offsets'] = np.delete(waves_dwt['ECG_P_Offsets'], 0)
        if waves_dwt['ECG_R_Offsets'][0] < waves_dwt['ECG_P_Onsets'][0]:
            waves_dwt['ECG_R_Offsets'] = np.delete(waves_dwt['ECG_R_Offsets'], 0)
        if waves_dwt['ECG_T_Offsets'][0] < waves_dwt['ECG_P_Onsets'][0]:
            waves_dwt['ECG_T_Offsets'] = np.delete(waves_dwt['ECG_T_Offsets'], 0)
        if waves_dwt['ECG_R_Onsets'][0] < waves_dwt['ECG_P_Onsets'][0]:
            waves_dwt['ECG_R_Onsets'] = np.delete(waves_dwt['ECG_R_Onsets'], 0)
        if waves_dwt['ECG_T_Onsets'][0] < waves_dwt['ECG_P_Onsets'][0]:
            waves_dwt['ECG_T_Onsets'] = np.delete(waves_dwt['ECG_T_Onsets'], 0)
        if waves_dwt['ECG_R_Offsets'][0] < rpeaks['ECG_R_Peaks'][0]:
            waves_dwt['ECG_R_Offsets'] = np.delete(waves_dwt['ECG_R_Offsets'], 0)
        if waves_dwt['ECG_T_Offsets'][0] < rpeaks['ECG_R_Peaks'][0]:
            waves_dwt['ECG_T_Offsets'] = np.delete(waves_dwt['ECG_T_Offsets'], 0)
        if waves_dwt['ECG_T_Onsets'][0] < rpeaks['ECG_R_Peaks'][0]:
            waves_dwt['ECG_T_Onsets'] = np.delete(waves_dwt['ECG_T_Onsets'], 0)
        if waves_dwt['ECG_S_Peaks'][0] < rpeaks['ECG_R_Peaks'][0]:
            waves_dwt['ECG_S_Peaks'] = np.delete(waves_dwt['ECG_S_Peaks'], 0)
        if waves_dwt['ECG_T_Peaks'][0] < rpeaks['ECG_R_Peaks'][0]:
            waves_dwt['ECG_T_Peaks'] = np.delete(waves_dwt['ECG_T_Peaks'], 0)

        if y_filt[rpeaks['ECG_R_Peaks']][0] < y_filt[rpeaks['ECG_R_Peaks']][1] / 2:
            rpeaks['ECG_R_Peaks'] = np.delete(rpeaks['ECG_R_Peaks'], 0)
            waves_dwt['ECG_P_Peaks'] = np.delete(waves_dwt['ECG_P_Peaks'], 0)
            waves_dwt['ECG_Q_Peaks'] = np.delete(waves_dwt['ECG_Q_Peaks'], 0)
            waves_dwt['ECG_S_Peaks'] = np.delete(waves_dwt['ECG_S_Peaks'], 0)
            waves_dwt['ECG_T_Peaks'] = np.delete(waves_dwt['ECG_T_Peaks'], 0)
            waves_dwt['ECG_R_Onsets'] = np.delete(waves_dwt['ECG_R_Onsets'], 0)
            waves_dwt['ECG_R_Offsets'] = np.delete(waves_dwt['ECG_R_Offsets'], 0)
            waves_dwt['ECG_P_Onsets'] = np.delete(waves_dwt['ECG_P_Onsets'], 0)
            waves_dwt['ECG_P_Offsets'] = np.delete(waves_dwt['ECG_P_Offsets'], 0)
            waves_dwt['ECG_T_Onsets'] = np.delete(waves_dwt['ECG_T_Onsets'], 0)
            waves_dwt['ECG_T_Offsets'] = np.delete(waves_dwt['ECG_T_Offsets'], 0)

        # Correcting last cycle
        if rpeaks['ECG_R_Peaks'][len(rpeaks['ECG_R_Peaks']) - 1] > waves_dwt['ECG_T_Offsets'][
            len(waves_dwt['ECG_T_Offsets']) - 1]:
            rpeaks['ECG_R_Peaks'] = np.delete(rpeaks['ECG_R_Peaks'], (len(rpeaks['ECG_R_Peaks']) - 1))
        if waves_dwt['ECG_P_Peaks'][len(waves_dwt['ECG_P_Peaks']) - 1] > waves_dwt['ECG_T_Offsets'][
            len(waves_dwt['ECG_T_Offsets']) - 1]:
            waves_dwt['ECG_P_Peaks'] = np.delete(waves_dwt['ECG_P_Peaks'], (len(waves_dwt['ECG_P_Peaks']) - 1))
        if waves_dwt['ECG_Q_Peaks'][len(waves_dwt['ECG_Q_Peaks']) - 1] > waves_dwt['ECG_T_Offsets'][
            len(waves_dwt['ECG_T_Offsets']) - 1]:
            waves_dwt['ECG_Q_Peaks'] = np.delete(waves_dwt['ECG_Q_Peaks'], (len(waves_dwt['ECG_Q_Peaks']) - 1))
        if waves_dwt['ECG_S_Peaks'][len(waves_dwt['ECG_S_Peaks']) - 1] > waves_dwt['ECG_T_Offsets'][
            len(waves_dwt['ECG_T_Offsets']) - 1]:
            waves_dwt['ECG_S_Peaks'] = np.delete(waves_dwt['ECG_S_Peaks'], (len(waves_dwt['ECG_S_Peaks']) - 1))
        if waves_dwt['ECG_T_Peaks'][len(waves_dwt['ECG_T_Peaks']) - 1] > waves_dwt['ECG_T_Offsets'][
            len(waves_dwt['ECG_T_Offsets']) - 1]:
            waves_dwt['ECG_T_Peaks'] = np.delete(waves_dwt['ECG_T_Peaks'], (len(waves_dwt['ECG_T_Peaks']) - 1))
        if waves_dwt['ECG_P_Onsets'][len(waves_dwt['ECG_P_Onsets']) - 1] > waves_dwt['ECG_T_Offsets'][
            len(waves_dwt['ECG_T_Offsets']) - 1]:
            waves_dwt['ECG_P_Onsets'] = np.delete(waves_dwt['ECG_P_Onsets'], (len(waves_dwt['ECG_P_Onsets']) - 1))
        if waves_dwt['ECG_T_Offsets'][len(waves_dwt['ECG_T_Offsets']) - 1] > waves_dwt['ECG_T_Offsets'][
            len(waves_dwt['ECG_T_Offsets']) - 1]:
            waves_dwt['ECG_T_Offsets'] = np.delete(waves_dwt['ECG_T_Offsets'], (len(waves_dwt['ECG_T_Offsets']) - 1))
        if waves_dwt['ECG_T_Onsets'][len(waves_dwt['ECG_T_Onsets']) - 1] > waves_dwt['ECG_T_Offsets'][
            len(waves_dwt['ECG_T_Offsets']) - 1]:
            waves_dwt['ECG_T_Onsets'] = np.delete(waves_dwt['ECG_T_Onsets'], (len(waves_dwt['ECG_T_Onsets']) - 1))
        if waves_dwt['ECG_R_Onsets'][len(waves_dwt['ECG_R_Onsets']) - 1] > waves_dwt['ECG_T_Offsets'][
            len(waves_dwt['ECG_T_Offsets']) - 1]:
            waves_dwt['ECG_R_Onsets'] = np.delete(waves_dwt['ECG_R_Onsets'], (len(waves_dwt['ECG_R_Onsets']) - 1))
        if waves_dwt['ECG_R_Offsets'][len(waves_dwt['ECG_R_Offsets']) - 1] > waves_dwt['ECG_T_Offsets'][
            len(waves_dwt['ECG_T_Offsets']) - 1]:
            waves_dwt['ECG_R_Offsets'] = np.delete(waves_dwt['ECG_R_Offsets'], (len(waves_dwt['ECG_R_Offsets']) - 1))

        if waves_dwt['ECG_P_Peaks'][len(waves_dwt['ECG_P_Peaks']) - 1] > rpeaks['ECG_R_Peaks'][
            len(rpeaks['ECG_R_Peaks']) - 1]:
            waves_dwt['ECG_P_Peaks'] = np.delete(waves_dwt['ECG_P_Peaks'], (len(waves_dwt['ECG_P_Peaks']) - 1))
        if waves_dwt['ECG_Q_Peaks'][len(waves_dwt['ECG_Q_Peaks']) - 1] > rpeaks['ECG_R_Peaks'][
            len(rpeaks['ECG_R_Peaks']) - 1]:
            waves_dwt['ECG_Q_Peaks'] = np.delete(waves_dwt['ECG_Q_Peaks'], (len(waves_dwt['ECG_Q_Peaks']) - 1))
        if waves_dwt['ECG_P_Onsets'][len(waves_dwt['ECG_P_Onsets']) - 1] > rpeaks['ECG_R_Peaks'][
            len(rpeaks['ECG_R_Peaks']) - 1]:
            waves_dwt['ECG_P_Onsets'] = np.delete(waves_dwt['ECG_P_Onsets'], (len(waves_dwt['ECG_P_Onsets']) - 1))
        if waves_dwt['ECG_P_Offsets'][len(waves_dwt['ECG_P_Offsets']) - 1] > rpeaks['ECG_R_Peaks'][
            len(rpeaks['ECG_R_Peaks']) - 1]:
            waves_dwt['ECG_P_Offsets'] = np.delete(waves_dwt['ECG_P_Offsets'], (len(waves_dwt['ECG_P_Offsets']) - 1))
        if waves_dwt['ECG_R_Onsets'][len(waves_dwt['ECG_R_Onsets']) - 1] > rpeaks['ECG_R_Peaks'][
            len(rpeaks['ECG_R_Peaks']) - 1]:
            waves_dwt['ECG_R_Onsets'] = np.delete(waves_dwt['ECG_R_Onsets'], (len(waves_dwt['ECG_R_Onsets']) - 1))

        ##R-R interval##
        print('\n====RR Interval====')
        RR_list = []

        cnt = 0
        while (cnt < (len(rpeaks['ECG_R_Peaks']) - 1)):
            RR_interval = (rpeaks['ECG_R_Peaks'][cnt + 1] - rpeaks['ECG_R_Peaks'][
                cnt])  # Calculate distance between beats in # of samples
            RRms_dist = ((RR_interval / Fs) * 1000.0)  # Convert sample distances to ms distances
            RR_list.append(RRms_dist)
            cnt += 1

        dfRR = pd.DataFrame(RR_list)
        dfRR = dfRR.fillna(0)
        print('SD =', dfRR.std())
        RRstdev = np.std(RR_list, axis=None)  # Save Average to RRstdev
        sum = 0.0
        count = 0.0
        for index in range(len(RR_list)):

            if (np.isnan(RR_list[index]) == True):
                continue
            else:
                sum += RR_list[index]
                count += 1
            # print(sum / count)
            RRavg = (sum / count)
            # print('RR_avg_distance =', sum / count)
        print('RR_avg_distance =', RRavg)

        ##cal heart rate manual##
        bpm = 60000 / np.mean(RR_list)  # 60000 ms (1 minute) / average R-R interval of signal
        print("\nAverage Heart Beat is: %.01f\n" % (bpm))  # Round off to 1 decimal and print

        print('\n====PR Interval====')
        PR_peak_list = []
        try:
            idex = ([x for x in range(0, len(waves_dwt['ECG_R_Onsets']) - 1)])
            for i in idex:
                if waves_dwt['ECG_R_Onsets'][i] < waves_dwt['ECG_P_Onsets'][i]:
                    cnt = 0
                    while (cnt < (len(waves_dwt['ECG_R_Onsets']) - 1)):
                        PR_peak_interval = (waves_dwt['ECG_Q_Peaks'][cnt] - waves_dwt['ECG_P_Onsets'][cnt])
                        ms_dist = ((PR_peak_interval / Fs) * 1000.0)  # Convert sample distances to ms distances
                        PR_peak_list.append(ms_dist)
                        cnt += 1
                else:
                    cnt = 0
                    while (cnt < (len(waves_dwt['ECG_R_Onsets']) - 1)):
                        PR_peak_interval = (waves_dwt['ECG_R_Onsets'][cnt] - waves_dwt['ECG_P_Onsets'][cnt])
                        ms_dist = ((PR_peak_interval / Fs) * 1000.0)  # Convert sample distances to ms distances
                        PR_peak_list.append(ms_dist)
                        cnt += 1
        except:
            print('PR Interval Error')

        dfPR = pd.DataFrame(PR_peak_list)
        dfPR = dfPR.fillna(0)
        print('SD =', dfPR.std())
        PRdev = np.std(PR_peak_list, axis=None)  # Save Average to RRstdev
        sum = 0.0
        count = 0.0
        for index in range(len(PR_peak_list)):

            if (np.isnan(PR_peak_list[index]) == True):
                continue
            else:
                sum += PR_peak_list[index]
                count += 1
            PRavg = (sum / count)  # Save Average to RRavg
            # print('PR_avg_distance =', sum / count)
        print('PR_avg_distance =', PRavg)


        print('\n===QRS Width===')
        QS_peak_list = []

        try:
            idex = ([x for x in range(0, len(waves_dwt['ECG_S_Peaks']) - 1)])
            for i in idex:
                if waves_dwt['ECG_S_Peaks'][i] < waves_dwt['ECG_Q_Peaks'][i]:
                    QRS_complex = (waves_dwt['ECG_S_Peaks'][i + 1] - waves_dwt['ECG_Q_Peaks'][i])
                    ms_dist = ((QRS_complex / Fs) * 1000.0)  # Convert sample distances to ms distances
                    QS_peak_list.append(ms_dist)
                else:
                    QRS_complex = (waves_dwt['ECG_S_Peaks'][i] - waves_dwt['ECG_Q_Peaks'][i])
                    ms_dist = ((QRS_complex / Fs) * 1000.0)  # Convert sample distances to ms distances
                    QS_peak_list.append(ms_dist)
        except:
            print('QRS Interval Error')

        dfQS = pd.DataFrame(QS_peak_list)
        dfQS = dfQS.fillna(0)
        print('SD =', dfQS.std())
        QSdev = np.std(QS_peak_list, axis=None)  # Save Average to QSstdev
        sum = 0.0
        count = 0.0
        for index in range(len(QS_peak_list)):

            if (np.isnan(QS_peak_list[index]) == True):
                continue
            else:
                sum += QS_peak_list[index]
                count += 1
            QSavg = (sum / count)  # Save Average to RRavg
            # print('QRS_avg_distance =', sum / count)
        print('QRS_avg_distance =', QSavg)

        ##PR-QRS-ST-QT##
        print('\n====ST Interval====')
        ST_peak_list = []
        try:
            idex = ([x for x in range(0, len(waves_dwt['ECG_T_Offsets']) - 1)])
            for i in idex:
                if waves_dwt['ECG_T_Offsets'][i] < waves_dwt['ECG_R_Offsets'][i]:
                    cnt = 0
                    while (cnt < (len(waves_dwt['ECG_T_Offsets']) - 1)):
                        ST_peak_interval = (waves_dwt['ECG_T_Offsets'][cnt + 1] - waves_dwt['ECG_R_Offsets'][cnt])
                        ms_dist = ((ST_peak_interval / Fs) * 1000.0)  # Convert sample distances to ms distances
                        ST_peak_list.append(ms_dist)
                        cnt += 1
                else:
                    cnt = 0
                    while (cnt < (len(waves_dwt['ECG_T_Offsets']) - 1)):
                        ST_peak_interval = (waves_dwt['ECG_T_Offsets'][cnt] - waves_dwt['ECG_R_Offsets'][cnt])
                        ms_dist = ((ST_peak_interval / Fs) * 1000.0)  # Convert sample distances to ms distances
                        ST_peak_list.append(ms_dist)
                        cnt += 1
        except:
            print('ST Interval Error')

        dfST = pd.DataFrame(ST_peak_list)
        dfST = dfST.fillna(0)
        print('SD =', dfST.std())
        STdev = np.std(ST_peak_list, axis=None)  # Save Average to RRstdev
        sum = 0.0
        count = 0.0
        for index in range(len(ST_peak_list)):

            if (np.isnan(ST_peak_list[index]) == True):
                continue
            else:
                sum += ST_peak_list[index]
                count += 1
            STavg = (sum / count)  # Save Average to RRavg
            # print('ST_avg_distance =', sum / count)
        print('ST_avg_distance =', STavg)


        print('\n===QT Interval===')
        QT_peak_list = []

        try:
            idex = ([x for x in range(0, len(waves_dwt['ECG_T_Offsets']) - 1)])
            for i in idex:
                if waves_dwt['ECG_T_Offsets'][i] < waves_dwt['ECG_R_Onsets'][i]:
                    QTdeff = (waves_dwt['ECG_T_Offsets'][i + 1] - waves_dwt['ECG_R_Onsets'][i])
                    ms_dist = ((QTdeff / Fs) * 1000.0)  # Convert sample distances to ms distances
                    QT_peak_list.append(ms_dist)
                else:
                    QTdeff = (waves_dwt['ECG_T_Offsets'][i] - waves_dwt['ECG_R_Onsets'][i])
                    ms_dist = ((QTdeff / Fs) * 1000.0)  # Convert sample distances to ms distances
                    QT_peak_list.append(ms_dist)

        except:
            print("QT Interval Error")

        dfQT = pd.DataFrame(QT_peak_list)
        dfQT = dfQT.fillna(0)
        print('SD =', dfQT.std())
        QTdev = np.std(QT_peak_list, axis=None)  # Save Average to QTstdev
        sum = 0.0
        count = 0.0
        for index in range(len(QT_peak_list)):

            if (np.isnan(QT_peak_list[index]) == True):
                continue
            else:
                sum += QT_peak_list[index]
                count += 1
            QTavg = (sum / count)  # Save Average to RRavg
            # print('QT_avg_distance =', sum / count)
        print('QT_avg_distance =', QTavg)
        QTc = QTavg / (math.sqrt(60 / bpm))
        print('\nQTc =', QTc)
        
        atest = numpy.asarray([[RRavg, PRavg, QSavg, STavg, QTc]])
        C = float(anfis.predict(model, atest))

        print(C)

        if C >= 0.5 and C <= 1.5:
            Hasil = 'Normal'
        elif C > 1.5 and C <= 2.5:
            Hasil = 'Abnormal'
        elif C < 0:
            Hasil = 'Abnormal'
        elif C > 2.5:
            Hasil = 'Abnormal'
            
        # packedData["id"] = device_id
        # packedData["ecg_graph"] = ecgmvconv

        # Update the ecg_grap dictionary based on device ID
        ecg_grap["ecg_graph"] = ecgmvconv

        packedData = {
            "user_id": device_id,
            "rr_avg": RRavg,
            "rr_dev": RRstdev,
            "pr_avg": PRavg,
            "pr_dev": PRdev,
            "qs_avg": QSavg,
            "qs_dev": QSdev,
            "qt_avg": QTavg,
            "qt_dev": QTdev,
            "st_avg": STavg,
            "st_dev": STdev,
            "classification_result": Hasil,
            "heart_rate": bpm,
            "ecg_graph": data
            # "ecg_raw_on" : data
        }
        
        # ecg_grap = {
        #     "ecg_graph": data
        #     # "ecg_graph": ecgmv
        # }
    ecgAnalysisParsed = json.dumps(packedData)
    mqtt_publish(ecgAnalysisParsed) 

def record_data(device_id, serial_port, recording_duration):
    global data

    start_time = time.time()
    end_time = start_time + recording_duration

    while time.time() < end_time:
        try:
            received_data = serial_port.readline().decode("utf-8").splitlines()[0]

            if received_data != '':
                if int(received_data) > 0 and int(received_data) < 7023:
                    data.append(int(received_data))

        except Exception as e:
            print(f"Error reading data from serial port: {str(e)}")
            
    preprocess_data(device_id, data)
    data = []


#============================ MAIN FUNCTION ========================================================================    
def main(userid, serial_port, recording_duration):
    # device = "COM13"  
    # userid = 1
    # recording_duration = 10 

    try:
        # serial_port = serial.Serial(device, baudrate=9600, timeout=1)
        while True:
            record_data(userid, serial_port, recording_duration)
    except serial.SerialException:
        print(f"Failed to open serial port {serial_port}")

if __name__ == "__main__":
    main()