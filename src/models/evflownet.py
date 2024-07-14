import torch
from torch import nn
from src.models.base import *
from typing import Dict, Any

_BASE_CHANNELS = 64

class EVFlowNet(nn.Module):
    def __init__(self, args):
        super(EVFlowNet,self).__init__()
        self._args = args

        self.encoder1 = general_conv2d(in_channels = 8, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder2 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder3 = general_conv2d(in_channels = 2*_BASE_CHANNELS, out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder4 = general_conv2d(in_channels = 4*_BASE_CHANNELS, out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.resnet_block = nn.Sequential(*[build_resnet_block(8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm) for i in range(2)])

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
                        out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
                        out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
                        out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
                        out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self._args.no_batch_norm)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        #print(inputs.size(0),inputs[1])
        batch_size=inputs.size(0)
        batch_slice=[i for i in range(batch_size)]
        #flows=[]
        for i in range(batch_size - 1):
            # バッチの中から連続する2枚の画像を取り出し、チャネル方向で結合
            batch_slice[i] = torch.cat((inputs[i], inputs[i+1]), dim=0)
        # 最後のペアは同じ画像を2回使用
        batch_slice[batch_size - 1] = torch.cat((inputs[-1], inputs[-1]), dim=0)
            
        # encoder
        skip_connections = {}
        inputs = self.encoder1(batch_slice)
        skip_connections['skip0'] = inputs.clone()
        inputs = self.encoder2(inputs)
        skip_connections['skip1'] = inputs.clone()
        inputs = self.encoder3(inputs)
        skip_connections['skip2'] = inputs.clone()
        inputs = self.encoder4(inputs)
        skip_connections['skip3'] = inputs.clone()

        # transition
        inputs = self.resnet_block(inputs)

        # decoder

        flow_dict = {}
        inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        inputs, flow = self.decoder1(inputs)
        flow_dict['flow0'] = flow.clone()
        
        inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        inputs, flow = self.decoder2(inputs)
        flow_dict['flow1'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        inputs, flow = self.decoder3(inputs)
        flow_dict['flow2'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        inputs, flow = self.decoder4(inputs)
        flow_dict['flow3'] = flow.clone()

        return flow
        '''    
        # encoder
        skip_connections = {}
        inputs = self.encoder1(inputs)
        skip_connections['skip0'] = inputs.clone()
        inputs = self.encoder2(inputs)
        skip_connections['skip1'] = inputs.clone()
        inputs = self.encoder3(inputs)
        skip_connections['skip2'] = inputs.clone()
        inputs = self.encoder4(inputs)
        skip_connections['skip3'] = inputs.clone()

        # transition
        inputs = self.resnet_block(inputs)

        # decoder
        #total_flow=0
        flow_dict = {}
        inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        inputs, flow0 = self.decoder1(inputs)
        flow_dict['flow0'] = flow0.clone()
        #total_flow+=flow
        #print(flow.shape)
        
        inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        inputs, flow1 = self.decoder2(inputs)
        flow_dict['flow1'] = flow1.clone()
        #total_flow+=flow
        #print(flow.shape)

        inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        inputs, flow2 = self.decoder3(inputs)
        flow_dict['flow2'] = flow2.clone()
        #total_flow+=flow
        #print(flow.shape)

        inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        inputs, flow3 = self.decoder4(inputs)
        flow_dict['flow3'] = flow3.clone()
        #total_flow+=flow
        #print(flow.shape)
        
        return flow3
        '''

# if __name__ == "__main__":
#     from config import configs
#     import time
#     from data_loader import EventData
#     '''
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     input_ = torch.rand(8,4,256,256).cuda()
#     a = time.time()
#     output = model(input_)
#     b = time.time()
#     print(b-a)
#     print(output['flow0'].shape, output['flow1'].shape, output['flow2'].shape, output['flow3'].shape)
#     #print(model.state_dict().keys())
#     #print(model)
#     '''
#     import numpy as np
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     EventDataset = EventData(args.data_path, 'train')
#     EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)
#     #model = nn.DataParallel(model)
#     #model.load_state_dict(torch.load(args.load_path+'/model18'))
#     for input_, _, _, _ in EventDataLoader:
#         input_ = input_.cuda()
#         a = time.time()
#         (model(input_))
#         b = time.time()
#         print(b-a)
