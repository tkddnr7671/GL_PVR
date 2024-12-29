import torch
from torch import nn

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nLayers=1, dropout=0.0, **kwargs):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers = nLayers, batch_first=True, dropout=dropout, dtype=torch.double)
        self.fc  = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        self.fc.weight.data = self.fc.weight.data.double()
        self.fc.bias.data = self.fc.bias.data.double()

    def load_state_dict(self, state_dict):
        self.rnn.load_state_dict(state_dict['rnn'])
        self.fc.load_state_dict(state_dict['fc'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'rnn': self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc' : self.fc.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({'rnn': self.rnn.state_dict()})
        parameters.update({'fc': self.fc.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        out, hn = self.rnn(x)
        #out = out[:, -1, :]
        out = self.fc(out)
        out = self.relu(out)
        return out

class GRU2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0, **kwargs):
        super(GRU2, self).__init__()
        self.rnn1 = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        self.rnn2 = nn.GRU(input_size=input_dim+1, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        self.fc1  = nn.Linear(hidden_dim, output_dim)   # for solar irradiance
        self.fc2  = nn.Linear(hidden_dim, output_dim)   # for solar power
        self.relu = nn.ReLU()

        self.fc1.weight.data = self.fc1.weight.data.double()
        self.fc1.bias.data   = self.fc1.bias.data.double()
        self.fc2.weight.data = self.fc2.weight.data.double()
        self.fc2.bias.data = self.fc2.bias.data.double()

    def load_state_dict(self, state_dict):
        self.rnn1.load_state_dict(state_dict['rnn1'])
        self.rnn2.load_state_dict(state_dict['rnn2'])
        self.fc1.load_state_dict(state_dict['fc1'])
        self.fc2.load_state_dict(state_dict['fc2'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'rnn1': self.rnn1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'rnn2': self.rnn2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc1' : self.fc1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc2' : self.fc2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({'rnn1': self.rnn1.state_dict()})
        parameters.update({'rnn2': self.rnn2.state_dict()})
        parameters.update({'fc1': self.fc1.state_dict()})
        parameters.update({'fc2': self.fc2.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        bs, sl, feat = x.shape
        o1, _ = self.rnn1(x)
        sr = self.relu(self.fc1(o1))
        x = torch.cat((x,sr.view(bs,sl,1)),dim=2)

        o2, _ = self.rnn2(x)
        out = self.fc2(o2)
        out = self.relu(out)
        return out, sr


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nLayers=1, dropout=0.0, **kwargs):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers = nLayers, batch_first=True, dropout=dropout, dtype=torch.double)
        self.fc  = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        self.fc.weight.data = self.fc.weight.data.double()
        self.fc.bias.data = self.fc.bias.data.double()

    def load_state_dict(self, state_dict):
        self.rnn.load_state_dict(state_dict['rnn'])
        self.fc.load_state_dict(state_dict['fc'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'rnn': self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc' : self.fc.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({'rnn': self.rnn.state_dict()})
        parameters.update({'fc': self.fc.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        out, hn = self.rnn(x)
        #out = out[:, -1, :]
        out = self.fc(out)
        out = self.relu(out)
        return out


class RNN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0, **kwargs):
        super(RNN2, self).__init__()
        self.rnn1 = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        self.rnn2 = nn.RNN(input_size=input_dim+1, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        self.fc1  = nn.Linear(hidden_dim, output_dim)   # for solar irradiance
        self.fc2  = nn.Linear(hidden_dim, output_dim)   # for solar power
        self.relu = nn.ReLU()

        self.fc1.weight.data = self.fc1.weight.data.double()
        self.fc1.bias.data   = self.fc1.bias.data.double()
        self.fc2.weight.data = self.fc2.weight.data.double()
        self.fc2.bias.data = self.fc2.bias.data.double()

    def load_state_dict(self, state_dict):
        self.rnn1.load_state_dict(state_dict['rnn1'])
        self.rnn2.load_state_dict(state_dict['rnn2'])
        self.fc1.load_state_dict(state_dict['fc1'])
        self.fc2.load_state_dict(state_dict['fc2'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'rnn1': self.rnn1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'rnn2': self.rnn2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc1' : self.fc1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc2' : self.fc2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({'rnn1': self.rnn1.state_dict()})
        parameters.update({'rnn2': self.rnn2.state_dict()})
        parameters.update({'fc1': self.fc1.state_dict()})
        parameters.update({'fc2': self.fc2.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        bs, sl, feat = x.shape
        o1, _ = self.rnn1(x)
        sr = self.relu(self.fc1(o1))
        x = torch.cat((x,sr.view(bs,sl,1)),dim=2)

        o2, _ = self.rnn2(x)
        out = self.fc2(o2)
        out = self.relu(out)
        return out, sr



class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nLayers = 1, dropout=0.0):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=nLayers, batch_first=True, dropout=dropout, dtype=torch.double)
        self.fc  = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        self.fc.weight.data = self.fc.weight.data.double()
        self.fc.bias.data = self.fc.bias.data.double()

    def load_state_dict(self, state_dict):
        self.lstm.load_state_dict(state_dict['lstm'])
        self.fc.load_state_dict(state_dict['fc'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'lstm': self.lstm.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc' : self.fc.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({'lstm': self.lstm.state_dict()})
        parameters.update({'fc': self.fc.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        out, hn = self.lstm(x)
        #out = out[:, -1, :]
        out = self.fc(out)
        out = self.relu(out)
        return out


class LSTM2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0, **kwargs):
        super(LSTM2, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        self.lstm2 = nn.LSTM(input_size=input_dim+1, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        self.fc1  = nn.Linear(hidden_dim, output_dim)   # for solar irradiance
        self.fc2  = nn.Linear(hidden_dim, output_dim)   # for solar power
        self.relu = nn.ReLU()

        self.fc1.weight.data = self.fc1.weight.data.double()
        self.fc1.bias.data   = self.fc1.bias.data.double()
        self.fc2.weight.data = self.fc2.weight.data.double()
        self.fc2.bias.data = self.fc2.bias.data.double()

    def load_state_dict(self, state_dict):
        self.lstm1.load_state_dict(state_dict['lstm1'])
        self.lstm2.load_state_dict(state_dict['lstm2'])
        self.fc1.load_state_dict(state_dict['fc1'])
        self.fc2.load_state_dict(state_dict['fc2'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'lstm1': self.lstm1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'lstm2': self.lstm2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc1' : self.fc1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc2' : self.fc2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({'lstm1': self.lstm1.state_dict()})
        parameters.update({'lstm2': self.lstm2.state_dict()})
        parameters.update({'fc1': self.fc1.state_dict()})
        parameters.update({'fc2': self.fc2.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        bs, sl, feat = x.shape
        o1, _ = self.lstm1(x)
        sr = self.relu(self.fc1(o1))
        x = torch.cat((x,sr.view(bs,sl,1)),dim=2)

        o2, _ = self.lstm2(x)
        out = self.fc2(o2)
        out = self.relu(out)
        return out, sr



class Vanilla_DSM(nn.Module): # Vanilla Double-stacking model
    def __init__(self, input_dim, hidden_dim, output_dim, modelType, dropout=0.0, **kwargs):
        super(Vanilla_DSM, self).__init__()
        self.modelType = modelType
        if modelType == 'RNN':
            self.rec1 = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'LSTM':
            self.rec1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'GRU':
            self.rec1 = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'RL':
            self.rec1 = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'LR':
            self.rec1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'RG':
            self.rec1 = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'LG':
            self.rec1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'GR':
            self.rec1 = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'GL':
            self.rec1 = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
  
        self.fc1  = nn.Linear(hidden_dim, output_dim, dtype=torch.double)   # for solar power
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.bn1  = nn.BatchNorm1d(hidden_dim, dtype=torch.double)

    def load_state_dict(self, state_dict):
        keys = state_dict.keys()

        self.rec1.load_state_dict(state_dict['rec1'])
        self.rec2.load_state_dict(state_dict['rec2'])
        self.fc1.load_state_dict(state_dict['fc1'])
        if 'bn1' in keys:
            self.bn1.load_state_dict(state_dict['bn1'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'rec1': self.rec1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'rec2': self.rec2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc1' : self.fc1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'bn1' : self.bn1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({'rec1': self.rec1.state_dict()})
        parameters.update({'rec2': self.rec2.state_dict()})
        parameters.update({'fc1': self.fc1.state_dict()})
        parameters.update({'bn1': self.bn1.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        bs, sl, feat = x.shape
        o1, _ = self.rec1(x)
        o1 = self.tanh(o1)
        o2, _ = self.rec2(self.dropout(o1))
        out = self.relu(self.fc1(o2))
        return out


class DSLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nLayers, dropout=0.2, **kwargs):
        super(DSLSTM, self).__init__()
        self.dconv1 = nn.Conv1d(1, 1, 3, dilation=1, padding='same', dtype=torch.double)
        self.dconv2 = nn.Conv1d(1, 1, 3, dilation=2, padding='same', dtype=torch.double)
        self.dconv3 = nn.Conv1d(1, 1, 3, dilation=4, padding='same', dtype=torch.double)
        self.rconv  = nn.Conv1d(1, 1, 1, padding='same', dtype=torch.double)
        
        self.rec = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=nLayers, batch_first=True, dropout=dropout, dtype=torch.double)
        self.fc  = nn.Linear(hidden_dim, output_dim, dtype=torch.double)
        self.relu = nn.ReLU()

    def load_state_dict(self, state_dict):
        self.dconv1.load_state_dict(state_dict['dconv1'])
        self.dconv2.load_state_dict(state_dict['dconv2'])
        self.dconv3.load_state_dict(state_dict['dconv3'])
        self.rconv.load_state_dict(state_dict['rconv'])
        self.rec.load_state_dict(state_dict['rec'])
        self.fc.load_state_dict(state_dict['fc'])


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'dconv1': self.dconv1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'dconv2': self.dconv2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'dconv3' : self.dconv3.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'rconv' : self.rconv.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'rec' : self.rec.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc' : self.fc.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({'dconv1': self.dconv1.state_dict()})
        parameters.update({'dconv2': self.dconv2.state_dict()})
        parameters.update({'dconv3': self.dconv3.state_dict()})
        parameters.update({'rconv': self.rconv.state_dict()})
        parameters.update({'rec': self.rec.state_dict()})
        parameters.update({'fc': self.fc.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        bs, sl, feat = x.shape

        x_in = x.reshape(bs*sl,1,feat)
        cout = self.relu(self.dconv1(x_in))
        cout = self.relu(self.dconv2(cout))
        cout = self.relu(self.dconv3(cout))
        rout = self.relu(self.rconv(x_in))

        r_in = cout + rout
        r_in = r_in.reshape(bs,sl,feat)

        r_out, _ = self.rec(r_in)
        output = self.relu(self.fc(r_out))

        return output
        


class GMT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, modelType, dropout=0.0, **kwargs):
        super(GMT, self).__init__()
        self.modelType = modelType
        if modelType == 'RNN':
            self.rec1 = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.RNN(input_size=input_dim+1, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'LSTM':
            self.rec1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.LSTM(input_size=input_dim+1, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'GRU':
            self.rec1 = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.GRU(input_size=input_dim+1, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'RL':
            self.rec1 = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.LSTM(input_size=input_dim+1, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'RG':
            self.rec1 = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.GRU(input_size=input_dim+1, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'LR':
            self.rec1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.RNN(input_size=input_dim+1, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'LG':
            self.rec1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.GRU(input_size=input_dim+1, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'GR':
            self.rec1 = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.RNN(input_size=input_dim+1, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        elif modelType == 'GL':
            self.rec1 = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
            self.rec2 = nn.LSTM(input_size=input_dim+1, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
  
        self.fc1  = nn.Linear(hidden_dim, output_dim, dtype=torch.double)   # for solar irradiance
        self.fc2  = nn.Linear(hidden_dim, output_dim, dtype=torch.double)   # for solar power
        self.fc3  = nn.Linear(hidden_dim, output_dim, dtype=torch.double)   # for weight
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.bn1  = nn.BatchNorm1d(hidden_dim, dtype=torch.double)

    def load_state_dict(self, state_dict):
        keys = state_dict.keys()

        self.rec1.load_state_dict(state_dict['rec1'])
        self.rec2.load_state_dict(state_dict['rec2'])
        self.fc1.load_state_dict(state_dict['fc1'])
        self.fc2.load_state_dict(state_dict['fc2'])
        self.fc3.load_state_dict(state_dict['fc3'])
        if 'bn1' in keys:
            self.bn1.load_state_dict(state_dict['bn1'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'rec1': self.rec1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'rec2': self.rec2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc1' : self.fc1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc2' : self.fc2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc3' : self.fc3.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'bn1' : self.bn1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({'rec1': self.rec1.state_dict()})
        parameters.update({'rec2': self.rec2.state_dict()})
        parameters.update({'fc1': self.fc1.state_dict()})
        parameters.update({'fc2': self.fc2.state_dict()})
        parameters.update({'fc3': self.fc3.state_dict()})
        parameters.update({'bn1': self.bn1.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        bs, sl, feat = x.shape
        o1, _ = self.rec1(x)
        o1 = self.bn1(torch.permute(o1,(0,2,1)))
        o1 = torch.permute(o1,(0,2,1))
        sr = self.dropout(self.relu(self.fc1(o1)))
        w  = torch.abs(self.sigmoid(self.dropout(self.fc2(o1))))
        ssr = sr*(w+1)
        x = torch.cat((x,ssr.view(bs,sl,1)),dim=2)

        o2, _ = self.rec2(x)
        out = self.fc3(o2)
        out = self.relu(out)
        return out, ssr



class GMT_DSLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nLayers, dropout=0.2, **kwargs):
        super(GMT_DSLSTM, self).__init__()
        self.dconv1 = nn.Conv1d(1, 1, 3, dilation=1, padding='same', dtype=torch.double)
        self.dconv2 = nn.Conv1d(1, 1, 3, dilation=2, padding='same', dtype=torch.double)
        self.dconv3 = nn.Conv1d(1, 1, 3, dilation=4, padding='same', dtype=torch.double)
        self.rconv  = nn.Conv1d(1, 1, 1, padding='same', dtype=torch.double)

        self.rec1 = nn.LSTM(input_size=input_dim, hidden_size = hidden_dim, batch_first=True, dtype=torch.double)
        self.rec2 = nn.LSTM(input_size=input_dim+1, hidden_size=hidden_dim, num_layers=nLayers-1, batch_first=True, dropout=dropout, dtype=torch.double)
        self.fc1  = nn.Linear(hidden_dim, output_dim, dtype=torch.double)
        self.fc2  = nn.Linear(hidden_dim, output_dim, dtype=torch.double)
        self.fc3  = nn.Linear(hidden_dim, output_dim, dtype=torch.double)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn1  = nn.BatchNorm1d(hidden_dim, dtype=torch.double)
        self.dropout = nn.Dropout(p=dropout)

    def load_state_dict(self, state_dict):
        self.dconv1.load_state_dict(state_dict['dconv1'])
        self.dconv2.load_state_dict(state_dict['dconv2'])
        self.dconv3.load_state_dict(state_dict['dconv3'])
        self.rconv.load_state_dict(state_dict['rconv'])
        self.rec1.load_state_dict(state_dict['rec1'])
        self.rec2.load_state_dict(state_dict['rec2'])
        self.fc1.load_state_dict(state_dict['fc1'])
        self.fc2.load_state_dict(state_dict['fc2'])
        self.fc3.load_state_dict(state_dict['fc3'])


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'dconv1': self.dconv1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'dconv2': self.dconv2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'dconv3' : self.dconv3.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'rconv' : self.rconv.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'rec1' : self.rec1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'rec2' : self.rec2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc1' : self.fc1.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc2' : self.fc2.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
            'fc3' : self.fc3.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
        }
        return state_dict

    def load_parameters(self, filename, save_flag=False):
        parameters = {}
        parameters.update({'dconv1': self.dconv1.state_dict()})
        parameters.update({'dconv2': self.dconv2.state_dict()})
        parameters.update({'dconv3': self.dconv3.state_dict()})
        parameters.update({'rconv': self.rconv.state_dict()})
        parameters.update({'rec1': self.rec1.state_dict()})
        parameters.update({'rec2': self.rec2.state_dict()})
        parameters.update({'fc1': self.fc1.state_dict()})
        parameters.update({'fc2': self.fc2.state_dict()})
        parameters.update({'fc3': self.fc3.state_dict()})
        if save_flag:
            torch.save(parameters, filename)
        return parameters

    def forward(self, x):
        bs, sl, feat = x.shape

        x_in = x.reshape(bs*sl,1,feat)
        cout = self.relu(self.dconv1(x_in))
        cout = self.relu(self.dconv2(cout))
        cout = self.relu(self.dconv3(cout))
        rout = self.relu(self.rconv(x_in))

        r_in = cout + rout
        r_in = r_in.reshape(bs,sl,feat)

        o1, _ = self.rec1(r_in)
        o1 = self.bn1(torch.permute(o1,(0,2,1)))
        o1 = torch.permute(o1, (0,2,1))
        sr = self.dropout(self.relu(self.fc1(o1)))
        w  = torch.abs(self.sigmoid(self.dropout(self.fc2(o1))))
        ssr = sr*(w+1)
        x = torch.cat((r_in,ssr.view(bs,sl,1)),dim=2)

        o2, _ = self.rec2(x)
        out = self.fc3(o2)
        out = self.relu(out)
        return out, ssr
 
