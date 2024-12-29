import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import argparse

from model import *
from data_loader2 import WPD
from torch.utils.data import DataLoader
from utility import *

def hyper_params():
    # Default setting
    model_params = {
        'modelType': 'RNN',
        'nPrevSeq': 30,
        'seqLeng': 3,
        'slideLeng': 1,
        'input_dim': 8,
        'hidden_dim': 64,
        'nLayers': 1,
        'output_dim': 1,
        'dropout': 0.0,
    }

    learning_params = {
        'BatchSize': 24,
        'lr'    : 1.0e-3,
        'max_epoch': 1000,
        'loss_func': 'mse',
    }

    hparams = {
        'model' : model_params,
        'learning' : learning_params,

        # system flags
        'loss_plot_flag': False,
        'save_losses': True,
        'save_result': True,

        'capacity' : 1.0,
        'val_capacity': 1.0,
        'tst_capacity': 1.0,
    }

    return hparams

def parse_flags(hparams):
    parser = argparse.ArgumentParser(description='Photovoltaic estimation')

    # Flags common to all modes
    all_modes_group = parser.add_argument_group('Flags common to all modes')
    all_modes_group.add_argument('--mode', type=str, choices=['train', 'test'], required=True)

    # Flags for training only
    training_group = parser.add_argument_group('Flags for training only')
    training_group.add_argument('--modelType', type=str, default='RNN')
    training_group.add_argument('--save_dir', type=str, default='')
    training_group.add_argument('--aws_dir', type=str, default='../dataset/AWS/')
    training_group.add_argument('--asos_dir', type=str, default='../dataset/ASOS/')
    training_group.add_argument('--sr_dir', type=str, default='../dataset/SR/GWNU_C9/')
    training_group.add_argument('--solar_dir', type=str, default='../dataset/photovoltaic/GWNU_C9/')    # C9 capacity: 70.84
    training_group.add_argument('--capacity', type=float, default=70.84)
    training_group.add_argument('--loc_ID', type=int, default=678)

    # Flags for validation only
    validation_group = parser.add_argument_group('Flags for validation only')
    validation_group.add_argument('--val_aws_dir', type=str, default='../dataset/AWS/')
    validation_group.add_argument('--val_asos_dir', type=str, default='../dataset/ASOS/')
    validation_group.add_argument('--val_sr_dir', type=str, default='../dataset/SR/GWNU_PS/')
    validation_group.add_argument('--val_solar_dir', type=str, default='../dataset/photovoltaic/GWNU_PS/')  # C3 capacity: 48.6  // Preschool capacity: 25.35
    validation_group.add_argument('--val_capacity', type=float, default=25.35)

    # Flags for test only
    test_group = parser.add_argument_group('Flags for test only')
    test_group.add_argument('--load_path', type=str, default='')
    test_group.add_argument('--tst_aws_dir', type=str, default='../dataset/AWS/')
    test_group.add_argument('--tst_asos_dir', type=str, default='../dataset/ASOS/')
    test_group.add_argument('--tst_sr_dir', type=str, default='')
    test_group.add_argument('--tst_solar_dir', type=str, default='')        # knu(samseok): 55.24 (guess)
    test_group.add_argument('--tst_capacity', type=float, default=1.0)
    test_group.add_argument('--tst_loc_ID', type=int, default=678)

    # Flags for training params
    trn_param_set = parser.add_argument_group('Flags for training paramters')
    trn_param_set.add_argument('--nLayers', type=int, default=hparams['model']['nLayers'])
    trn_param_set.add_argument('--seqLeng', type=int, default=hparams['model']['seqLeng'])
    trn_param_set.add_argument('--slideLeng', type=int, default=hparams['model']['slideLeng'])
    trn_param_set.add_argument('--hidden_dim', type=int, default=hparams['model']['hidden_dim'])
    trn_param_set.add_argument('--nBatch', type=int, default=hparams['learning']['BatchSize'])
    trn_param_set.add_argument('--max_epoch', type=int, default=hparams['learning']['max_epoch'])
    trn_param_set.add_argument('--dropout', type=float, default=hparams['model']['dropout'])
    trn_param_set.add_argument('--loss_func', type=str, default=hparams['learning']['loss_func'])

    flags = parser.parse_args()

    # update parse
    hparams['model']['nLayers'] = flags.nLayers
    hparams['model']['seqLeng'] = flags.seqLeng
    hparams['model']['slideLeng'] = flags.slideLeng
    hparams['model']['hidden_dim'] = flags.hidden_dim
    hparams['model']['dropout'] = flags.dropout
    hparams['model']['modelType'] = flags.modelType
    hparams['learning']['nBatch'] = flags.nBatch
    hparams['learning']['max_epoch'] = flags.max_epoch
    hparams['learning']['loss_func'] = flags.loss_func
    hparams['capacity'] = flags.capacity
    hparams['val_capacity'] = flags.val_capacity
    hparams['tst_capacity'] = flags.tst_capacity

    # Additional per-mode validation
    try:
        if flags.mode == 'train':
            assert flags.save_dir, 'Must specify --save_dir'
        elif flags.mode == 'test':
            assert flags.load_path, 'Must specify --load_path'

    except AssertionError as e:
        print('\nError: ', e, '\n')
        parser.print_help()
        sys.exit(1)

    return flags, hparams


def train(hparams):
    model_params = hparams['model']
    learning_params = hparams['learning']
    modelType = model_params['modelType']

    trnset  = WPD(hparams['aws_list'], hparams['asos_list'], hparams['solar_list'], hparams['loc_ID'], sr_list=hparams['sr_list'], input_dim=model_params['input_dim'])
    valset  = WPD(hparams['val_aws_list'], hparams['val_asos_list'], hparams['val_solar_list'], hparams['loc_ID'], sr_list=hparams['val_sr_list'], input_dim=model_params['input_dim'])

    trnloader = DataLoader(trnset, batch_size=1, shuffle=False, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, drop_last=True)

    input_dim  = model_params['input_dim']
    hidden_dim = model_params['hidden_dim']
    output_dim = model_params['output_dim']
    nLayers    = model_params['nLayers']

    if modelType == 'LSTM':
        model = LSTM(input_dim, hidden_dim, output_dim, nLayers, dropout=model_params['dropout'])
    elif modelType == 'GRU':
        model = GRU(input_dim, hidden_dim, output_dim, nLayers, dropout=model_params['dropout'])
    elif modelType == 'RNN':
        model = RNN(input_dim, hidden_dim, output_dim, nLayers, dropout=model_params['dropout'])
    elif modelType == 'DSLSTM':
        model = DSLSTM(input_dim, hidden_dim, output_dim, nLayers, dropout=model_params['dropout'])
    elif modelType == 'DSM':
        model = Vanilla_DSM(input_dim, hidden_dim, output_dim, modelType, dropout=model_params['dropout'])
    model.cuda()

    criterion = torch.nn.MSELoss(reduction = 'sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_params['lr'])

    max_epoch = learning_params['max_epoch']
    seqLeng = model_params['seqLeng']
    slideLeng = model_params['slideLeng']
    nPrevSeq = model_params['nPrevSeq']
    BatchSize  = learning_params['BatchSize']

    capacity = hparams['capacity']
    val_capacity = hparams['val_capacity']

    losses = []
    val_losses = []


    ################################## data_loading ###################################
    # Loading training data
    weather_data    = []
    irradiance_data = []
    power_data      = []
    for trn_i, (w, r, p) in enumerate(trnloader):
        x = w.squeeze().cuda()
        r = r.squeeze().cuda()
        y = p.squeeze().cuda()

        irradiance_data.append(r)
        power_data.append(y)

        weather_temp = []
        for j in range(24):
            stridx = (j+1)*60 - nPrevSeq
            endidx = (j+1)*60 
            weather_temp.append(x[stridx:endidx,:].view(1,nPrevSeq, input_dim))
        weather_temp = torch.cat(weather_temp, dim=0).mean(1).squeeze()
        weather_data.append(weather_temp)

    weather_data    = torch.cat(weather_data)
    weather_data, mV, sV = DataNormalization(weather_data)
    irradiance_data = torch.cat(irradiance_data)
    power_data      = torch.cat(power_data)

    # Sequential training data
    nLeng, nFeat = weather_data.shape
    nData = np.floor((nLeng-seqLeng)/slideLeng).astype(int)

    TrnData  = []
    TrnLabel = []
    AuxLabel = []
    for j in range(nData):
        stridx = j*slideLeng
        endidx = j*slideLeng + seqLeng
        TrnData.append(weather_data[stridx:endidx,:].view(1,seqLeng,nFeat))
        TrnLabel.append(power_data[stridx:endidx].view(1,seqLeng))
        AuxLabel.append(irradiance_data[stridx:endidx].view(1,seqLeng))
    TrnData  = torch.cat(TrnData)
    TrnLabel = torch.cat(TrnLabel)
    TrnAuxLabel = torch.cat(AuxLabel)

    # Loading validation data
    weather_data    = []
    irradiance_data = []
    power_data      = []
    for val_i, (w, r, p) in enumerate(valloader):
        x = w.squeeze().cuda()
        r = r.squeeze().cuda()
        y = p.squeeze().cuda()

        irradiance_data.append(r)
        power_data.append(y)

        weather_temp = []
        for j in range(24):
            stridx = (j+1)*60 - nPrevSeq
            endidx = (j+1)*60
            weather_temp.append(x[stridx:endidx,:].view(1,nPrevSeq, input_dim))
        weather_temp = torch.cat(weather_temp, dim=0).mean(1).squeeze()
        weather_data.append(weather_temp)

    weather_data    = torch.cat(weather_data)
    weather_data, _, _ = DataNormalization(weather_data, mV, sV)
    irradiance_data = torch.cat(irradiance_data)
    power_data      = torch.cat(power_data)

    # Sequential validation data
    nLeng, nFeat = weather_data.shape
    nData = np.floor((nLeng-seqLeng)/slideLeng).astype(int)

    ValData  = []
    ValLabel = []
    AuxLabel = []
    for j in range(nData):
        stridx = j*slideLeng
        endidx = j*slideLeng + seqLeng
        ValData.append(weather_data[stridx:endidx,:].view(1,seqLeng,nFeat))
        ValLabel.append(power_data[stridx:endidx].view(1,seqLeng))
        AuxLabel.append(irradiance_data[stridx:endidx].view(1,seqLeng))
    ValData  = torch.cat(ValData)
    ValLabel = torch.cat(ValLabel)
    ValAuxLabel = torch.cat(AuxLabel)


    ############################### Training & Validation ##############################
    nBatch = np.floor((nData-BatchSize)/4).astype(int)

    nTrnData, _, _ = TrnData.shape
    shuffle_idx    = np.arange(nTrnData)
    prev_loss = np.inf
    for epoch in range(max_epoch):
        loss = 0
        model.train()

        # shuffle TrnData
        np.random.shuffle(shuffle_idx)
        TrnData     = TrnData[shuffle_idx]
        TrnLabel    = TrnLabel[shuffle_idx]
        TrnAuxLabel = TrnAuxLabel[shuffle_idx]
        for bter in range(nBatch):
            stridx = bter*4
            endidx = bter*4 + BatchSize
            batch_data  = TrnData[stridx:endidx,:,:]
            batch_label = TrnLabel[stridx:endidx,:]
            batch_auxil = TrnAuxLabel[stridx:endidx,:]
        
            output = model(batch_data)
            output = capacity*output.squeeze()

            loss += criterion(output, batch_label)

        loss /= (trn_i*nBatch*seqLeng)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        model.eval()
        # Batch data
        val_loss = 0
        nBatch = np.floor((nData-BatchSize)/4).astype(int)
        for bter in range(nBatch):
            stridx = bter*4
            endidx = bter*4 + BatchSize
            batch_data  = ValData[stridx:endidx,:,:]
            batch_label = ValLabel[stridx:endidx,:]
            batch_auxil = ValAuxLabel[stridx:endidx,:]
        
            output = model(batch_data)
            output = val_capacity*output.squeeze()

            val_loss += criterion(output, batch_label)
        val_loss /= (val_i*nBatch*seqLeng)

        if val_loss < prev_loss:
            savePath = os.path.join(hparams['save_dir'], f'model_{epoch}')	# overwrite
            model_dict = {
                'kwargs'   : model_params,
                'paramSet' : model.state_dict(),
                'statistics': [mV, sV]
            }
            #torch.save(model_dict, savePath)
            savePath = os.path.join(hparams['save_dir'], 'best_model')
            torch.save(model_dict, savePath)
            prev_loss = val_loss

        losses.append(loss.item())
        val_losses.append(val_loss.item())
        print(f'Epoch [{epoch+1}/{max_epoch}], MSE: {loss.item():.4f}, {val_loss.item():.4f}')

    if hparams['save_losses']:
        trn_loss = np.array(losses)
        val_loss = np.array(val_losses)
        savepath = os.path.join(hparams['save_dir'], 'trn_loss.npy')
        np.save(savepath, trn_loss)
        savepath = os.path.join(hparams['save_dir'], 'val_loss.npy')
        np.save(savepath, val_loss)
 

    if hparams['loss_plot_flag']:
        plt.plot(range(max_epoch), losses, 'b', range(max_epoch), val_losses, 'r')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()


def test(hparams):
    model_params = hparams['model']
    learning_params = hparams['learning']

    modelPath = hparams['load_path']
    if modelPath.find('DSLSTM')>0:
        modelType = 'DSLSTM'
    elif modelPath.find('RNN')>0:
        modelType = 'RNN'
    elif modelPath.find('LSTM')>0:
        modelType = 'LSTM'
    elif modelPath.find('GRU')>0:
        modelType = 'GRU'
    elif modelPath.find('RL')>0:
        modelType = 'RL'
    elif modelPath.find('RG')>0:
        modelType = 'RG'
    elif modelPath.find('LR')>0:
        modelType = 'LR'
    elif modelPath.find('LG')>0:
        modelType = 'LG'
    elif modelPath.find('GR')>0:
        modelType = 'GR'
    elif modelPath.find('GL')>0:
        modelType = 'GL'

    try:
        ckpt = torch.load(modelPath)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Loaded object from {modelPath} is not a dictionary.")
        if 'kwargs' not in ckpt or 'paramSet' not in ckpt:
            raise ValueError(f"Dictionary from {modelPath} does not contain expected keys.")
        model_conf = ckpt['kwargs']
        paramSet = ckpt['paramSet']
        statistics = ckpt['statistics']
    except Exception as e:
        print(f"Error occurred while loading model from {modelPath}")
        print(f"Error: {e}")

    input_dim  = model_conf['input_dim']
    hidden_dim = model_conf['hidden_dim']
    output_dim = model_conf['output_dim']
    nLayers    = model_conf['nLayers']
    
    if modelType == 'LSTM':
        model = LSTM(input_dim, hidden_dim, output_dim, nLayers, dropout=model_params['dropout'])
    elif modelType == 'GRU':
        model = GRU(input_dim, hidden_dim, output_dim, nLayers, dropout=model_params['dropout'])
    elif modelType == 'RNN':
        model = RNN(input_dim, hidden_dim, output_dim, nLayers, dropout=model_params['dropout'])
    elif modelType == 'DSLSTM':
        model = DSLSTM(input_dim, hidden_dim, output_dim, nLayers, dropout=model_params['dropout'])
    elif modelType == 'DSM':
        model = Vanilla_DSM(input_dim, hidden_dim, output_dim, modelType, dropout=model_params['dropout'])
    
    model.load_state_dict(paramSet)
    model.cuda()
    model.eval()

    tstset  = WPD(hparams['aws_list'], hparams['asos_list'], hparams['solar_list'], hparams['loc_ID'], sr_list=hparams['sr_list'])
    tstloader = DataLoader(tstset, batch_size=1, shuffle=False, drop_last=True)

    seqLeng = model_params['seqLeng']
    nPrevSeq = model_params['nPrevSeq']
    slideLeng = model_params['slideLeng']
    BatchSize = learning_params['BatchSize']
    tst_capacity = hparams['tst_capacity']

    criterion = torch.nn.MSELoss(reduction='sum')

    ################################## Test ####################################
    model.eval()
    # Loading data
    weather_data    = []
    irradiance_data = []
    power_data      = []
    for i, (w, r, p) in enumerate(tstloader):
        x = w.squeeze().cuda()
        r = r.squeeze().cuda()
        y = p.squeeze().cuda()

        irradiance_data.append(r)
        power_data.append(y)

        weather_temp = []
        for j in range(24):
            stridx = (j+1)*60 - nPrevSeq
            endidx = (j+1)*60
            weather_temp.append(x[stridx:endidx,:].view(1,nPrevSeq, input_dim))
        weather_temp = torch.cat(weather_temp, dim=0).mean(1).squeeze()
        weather_data.append(weather_temp)

    weather_data    = torch.cat(weather_data)
    weather_data, _, _ = DataNormalization(weather_data, statistics[0], statistics[1])
    irradiance_data = torch.cat(irradiance_data)
    power_data      = torch.cat(power_data)

    # Sequential data
    nLeng, nFeat = weather_data.shape
    nData = np.floor((nLeng-seqLeng)/slideLeng).astype(int)

    TstData  = []
    TstLabel = []
    AuxLabel = []
    for j in range(nData):
        stridx = j*slideLeng
        endidx = j*slideLeng + seqLeng
        TstData.append(weather_data[stridx:endidx,:].view(1,seqLeng,nFeat))
        TstLabel.append(power_data[stridx:endidx].view(1,seqLeng))
        AuxLabel.append(irradiance_data[stridx:endidx].view(1,seqLeng))
    TstData  = torch.cat(TstData)
    TstLabel = torch.cat(TstLabel)
    AuxLabel = torch.cat(AuxLabel)

    # Batch data
    loss = 0
    result = []
    labs   = []
    nBatch = np.floor(nData/BatchSize).astype(int)
    for bter in range(nBatch):
        stridx = bter*BatchSize
        endidx = bter*BatchSize + BatchSize
        batch_data  = TstData[stridx:endidx,:,:]
        batch_label = TstLabel[stridx:endidx,:]
        batch_auxil = AuxLabel[stridx:endidx,:]
        
        output = model(batch_data)
        output = tst_capacity*output.squeeze()

        loss += criterion(output, batch_label)
        result.append(output.detach().cpu().numpy())
        labs.append(batch_label.detach().cpu().numpy())

    loss /= (i*nBatch*seqLeng)
    print(f'[{modelPath}] MSE: {loss.item():.4f}')
    
    if hparams['save_result']:
        result_npy = np.array(result)
        label_npy  = np.array(labs)
        #label_npy  = TstLabel.detach().cpu().numpy()
        data_npy   = TstData.detach().cpu().numpy()
        solar_dir = hparams['solar_dir']
        loc = solar_dir.split('/')[-1]

        saveDir = './Results/'+modelPath.split('/')[-2]
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)

        np.save(saveDir+f'/prediction_{loc}.npy', result_npy)
        #np.save(saveDir+f'/data_{loc}.npy', data_npy)
        np.save(saveDir+f'/label_{loc}.npy', label_npy)

if __name__=='__main__':
    hp    = hyper_params()
    flags, hp = parse_flags(hp)

    if flags.mode == 'train':
        print(hp)
        #=============================== training data list ====================================#
        # build photovoltaic data list
        solar_list, first_date, last_date = list_up_solar(flags.solar_dir)
        aws_list = list_up_weather(flags.aws_dir, first_date, last_date)
        asos_list = list_up_weather(flags.asos_dir, first_date, last_date)
        if flags.sr_dir is not None:
            sr_list = list_up_irradiance(flags.sr_dir)
        else:
            sr_list = None
        print('Training on the interval from %s to %s.'%(first_date, last_date))
        #=============================== validation data list ===================================#		
        # build photovoltaic data list
        val_solar_list, first_date, last_date = list_up_solar(flags.val_solar_dir)
        val_aws_list = list_up_weather(flags.val_aws_dir, first_date, last_date)
        val_asos_list = list_up_weather(flags.val_asos_dir, first_date, last_date)
        if flags.val_sr_dir is not None:
            val_sr_list = list_up_irradiance(flags.val_sr_dir)
        else:
            val_sr_list = None
        print('Validating on the interval from %s to %s.'%(first_date, last_date))
        #========================================================================================#

        hp.update({"aws_list": aws_list})
        hp.update({"val_aws_list": val_aws_list})
        hp.update({"asos_list": asos_list})
        hp.update({"val_asos_list": val_asos_list})
        hp.update({"sr_list": sr_list})
        hp.update({"val_sr_list": val_sr_list})
        hp.update({"solar_list": solar_list})
        hp.update({"val_solar_list": val_solar_list})
        hp.update({"save_dir": flags.save_dir})
        hp.update({"loc_ID": flags.loc_ID})

        if not os.path.isdir(flags.save_dir):
            os.makedirs(flags.save_dir)

        train(hp)

    elif flags.mode == 'test':
        hp.update({"load_path": flags.load_path})
        hp.update({"loc_ID": flags.tst_loc_ID})

        #=============================== test data list ====================================#
        # build photovoltaic data list
        solar_list, first_date, last_date = list_up_solar(flags.tst_solar_dir)
        aws_list = list_up_weather(flags.tst_aws_dir, first_date, last_date)
        asos_list = list_up_weather(flags.tst_asos_dir, first_date, last_date)
        if flags.tst_sr_dir is not None:
            sr_list = list_up_irradiance(flags.tst_sr_dir)
        else:
            sr_list = None
        print('Testing on the interval from %s to %s.'%(first_date, last_date))
        #========================================================================================#

        hp.update({"aws_list": aws_list})
        hp.update({"asos_list": asos_list})
        hp.update({"solar_list": solar_list})
        hp.update({"sr_list": sr_list})
        hp.update({"solar_dir": flags.tst_solar_dir})

        test(hp)
