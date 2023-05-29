import resnet_50
import torch
from torchsummary import summary
from torchviz import make_dot

import torchvision.models as models

net = resnet_50.ResNet_50().cuda()
summary(net, (512,512,3))
class EfficientNetRomp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        net = models.efficientnet_v2_s(pretrained=False, num_classes=10000).cuda()
        self.feature_extractor = net.features
        self.deconv_layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1280, 256, (3,3,), 2, 1, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(256, 128, (3,3,), 2, 1, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 64, (3,3,), 2, 1, 1),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.deconv_layers(self.feature_extractor(x))

    def image_preprocess(self, x):
        x = BHWC_to_BCHW(x)/255.
        #x = F.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],inplace=True).contiguous() # for pytorch version>1.8.0
        x = torch.stack(list(map(lambda x:F.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],inplace=False),x)))
        #x = ((BHWC_to_BCHW(x)/ 255.) * 2.0 - 1.0).contiguous()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
    
    def load_pretrain_params(self):
        if os.path.exists(args().resnet_pretrain):
            print("USE PRETRAINED")
            success_layer = copy_state_dict(self.state_dict(), torch.load(args().efficientnet_pretrain), prefix = '', fix_loaded=True)
    
        
    


net = EfficientNetRomp().cuda()
summary(net, (3,512,512))


# Use the torchsummary library to get a summary of your network



# # Create a visualization of your network using torchviz
# dot = make_dot(net(torch.rand(size=(1,3,512,512,)).cuda()), params=dict(net.named_parameters()))

# dot.render(filename='net_graph', format='pdf')

# # Print out the number of parameters in your network
# num_params = sum(p.numel() for p in net.parameters())
# print(f"Number of parameters in network: {num_params}")