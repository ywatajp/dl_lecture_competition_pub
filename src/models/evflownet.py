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
        self.encoder5 = general_conv2d(in_channels = 8*_BASE_CHANNELS, out_channels=16*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.resnet_block = nn.Sequential(*[build_resnet_block(16*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm) for i in range(2)])

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=32*_BASE_CHANNELS,
                        out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS+2,
                        out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
                        out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
                        out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder5 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
                        out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self._args.no_batch_norm)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: 
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
        inputs = self.encoder5(inputs)
        skip_connections['skip4'] = inputs.clone()

        # transition
        inputs = self.resnet_block(inputs)

        # decoder
        flow_dict = {}
        inputs = torch.cat([inputs, skip_connections['skip4']], dim=1)
        inputs, flow = self.decoder1(inputs)
        shape = flow.shape
        flow = nn.functional.interpolate(flow,size=[shape[2]*16,shape[3]*16],mode='bilinear', align_corners=False)
        flow_dict['flow0'] = flow.clone()
        
        inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        inputs, flow = self.decoder2(inputs)
        shape = flow.shape
        flow = nn.functional.interpolate(flow,size=[shape[2]*8,shape[3]*8],mode='bilinear', align_corners=False)
        flow_dict['flow1'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        inputs, flow = self.decoder3(inputs)
        shape = flow.shape
        flow = nn.functional.interpolate(flow,size=[shape[2]*4,shape[3]*4],mode='bilinear', align_corners=False)
        flow_dict['flow2'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        inputs, flow = self.decoder4(inputs)
        shape = flow.shape
        flow = nn.functional.interpolate(flow,size=[shape[2]*2,shape[3]*2],mode='bilinear', align_corners=False)
        flow_dict['flow3'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        inputs, flow = self.decoder5(inputs)
        flow_dict['flow4'] = flow.clone()
        
        return skip_connections, flow_dict

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
