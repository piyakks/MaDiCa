import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.single_dinat_change import vssm_small as backbone
from .decoders.DinatDecoder import Dinat_Decoder

class Model(nn.Module):
    def __init__(self,img_size=352):
        super(Model, self).__init__()
        self.backbone = backbone(img_size=[352,352])
        self.channels = [96, 192, 384, 768]
        self.decoder = Dinat_Decoder(img_size=[img_size, img_size], 
                                    in_channels=self.channels, 
                                    num_classes=1, 
                                    depths=[4,4,4,4],#1,1,1,1
                                    embed_dim=self.channels[0], 
                                    deep_supervision=False,
                                    )
    def forward(self, rgb):
        
        orisize = rgb.shape
        x, saliency = self.backbone(rgb)
        
        out = self.decoder.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        saliency = F.interpolate(saliency, size=orisize[2:], mode='bilinear', align_corners=False)
        # background = F.interpolate(background, size=orisize[2:], mode='bilinear', align_corners=False)
        return out, saliency
    
    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

if __name__ == '__main__':
    model = Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = flow = depth = torch.ones([2, 3, 448, 448]).to(device)
    out = model(image, flow)
    print(model)
    print(out.shape)
        