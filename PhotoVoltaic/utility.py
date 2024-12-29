import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime


def DataNormalization(target, meanV=None, stdV=None):
    nData, nDim = target.shape[0], target.shape[1]
    if meanV is None:
        meanV = target.mean(0)
        stdV = target.std(0)
        output = (target-meanV)/stdV
    else:
        output = (target-meanV)/stdV

    return output, meanV, stdV

def MinMaxNormalization(target, maxV=None, minV=None):
    if maxV is None:
        maxV = target.max(0).values

    if minV is None:
        minV = target.min(0).values

    output = (target - minV[None,:])/(maxV[None,:]-minV[None,:])

    return output, maxV, minV


def path2date(path):
    date_str = path.split('/')[-2] + '/' + path.split('/')[-1].replace('.xlsx', '')
    date = datetime.strptime(date_str, '%Y_%m/%d')
    return date

def list_up_solar(solar_directory):
    # build photovoltaic data list
    solar_dir = os.listdir(solar_directory)
    #solar_dir.sort()
    solar_list = []
    for folder in solar_dir:
        mlist = os.listdir(solar_directory+'/'+folder)
        mlist = [file for file in mlist if file.find('xlsx') > 0]
        mlist = sorted(mlist, key=lambda x:int(x.split('.')[0]))
        for file in mlist:
            path = solar_directory + '/' + folder + '/' + file
            solar_list.append(path)

    solar_list.sort(key=path2date)

    # find period
    first_ = solar_list[0].split('.')[2].split('/')
    first_year, first_month = first_[-2].split('_')
    first_day = str("%02d"%int(first_[-1]))
    first_date = first_year+first_month+first_day

    last_ = solar_list[-1].split('.')[2].split('/')
    last_year, last_month = last_[-2].split('_')
    last_day = str("%02d"%int(last_[-1]))
    last_date = last_year+last_month+last_day
    #print('Training with data from %s to %s.'%(first_date, last_date))

    return solar_list, first_date, last_date


def list_up_weather(weather_directory, first_date, last_date):
    # build weather data list
    weather_dir = os.listdir(weather_directory)
    weather_dir.sort()
    weather_list = []
    stridx, endidx, cnt = -1, -1, -1
    for folder in weather_dir:
        wlist = os.listdir(weather_directory+'/'+folder)
        wlist = [file for file in wlist if file.find('csv') > 0]
        wlist.sort()
        for file in wlist:
            path = weather_directory + '/' + folder + '/' + file	
            weather_list.append(path)
            cnt += 1
            if path.find(first_date) > 0:
                stridx = cnt
            if path.find(last_date) > 0:
                endidx = cnt

    weather_list = weather_list[stridx:endidx+1]
    
    return weather_list


def list_up_irradiance(sr_directory):
    # build weather data list
    sr_dir = os.listdir(sr_directory)
    sr_dir.sort()
    sr_list = []
    for folder in sr_dir:
        srlist = os.listdir(sr_directory+'/'+folder)
        srlist = [file for file in srlist if file.find('xlsx') > 0]
        srlist.sort()
        for file in srlist:
            path = sr_directory + '/' + folder + '/' + file	
            sr_list.append(path)
    
    sr_list.sort(key=path2date)
    return sr_list

