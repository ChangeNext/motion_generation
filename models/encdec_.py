import torch.nn as nn
from models.resnet import Resnet1D, Resnet1D_decoder

class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        self.conv1 = nn.Conv1d(input_emb_width, width, 3, 1, 1)
        self.relu = nn.ReLU()
        self.encoder_layers = nn.ModuleList()
        
        
        for i in range(down_t):
            input_dim = width
            encoder_layer = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            self.encoder_layers.append(encoder_layer)
            
        self.conv2 = nn.Conv1d(width, output_emb_width, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None, 
                 zq_ch=None, 
                 add_conv=False):
        super().__init__()
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        
        self.conv1 = nn.Conv1d(output_emb_width, width, 3, 1, 1)
        self.relu = nn.ReLU()
        self.decoder_layers = nn.ModuleList()

        out_dim = width
        self.resnet= Resnet1D_decoder(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm, zq_ch=zq_ch, add_conv=add_conv)
        for i in range(down_t):
            out_dim = width
            decoder_layer = nn.Sequential(
                # Resnet1D_decoder(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm, zq_ch=zq_ch,
                #                        add_conv=add_conv),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            self.decoder_layers.append(decoder_layer)
            
        self.conv2 = nn.Conv1d(width, width, 3, 1, 1)
        self.conv3 = nn.Conv1d(width, input_emb_width, 3, 1, 1)

    def forward(self, x, zq):
        # print("0:", x.shape)
        x = self.conv1(x)
        # print("1:", x.shape)
        x = self.relu(x)
        # print("2:", x.shape)
        for layer in self.decoder_layers:
            x = self.resnet(x, zq)
            x = layer(x)
        # print("3", x.shape)    
        x = self.conv2(x)
        # print("4", x.shape)    
        x = self.relu(x)
        x = self.conv3(x)
        # print("5", x.shape)    
        
        return x
    
