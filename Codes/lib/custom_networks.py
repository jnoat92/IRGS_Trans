'''
No@
17/02/2022
'''

from turtle import forward
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision import models
import torch.nn.functional as F

from icecream import ic
import math
from functools import partial
import numpy as np
from timm.models.layers import trunc_normal_

from .DeiT import deit_small_patch16_224 as deit
from .vision_transformer import VisionTransformer

class Encoder(nn.Module):
    def __init__(self, backbone, in_chans, drop_rate, output_stride):
        super(Encoder, self).__init__()

        cnn_func = getattr(models, backbone)
        self.enc = cnn_func()

        if backbone == 'resnet18':
            layers = [2, 2, 2, 2]           # number of layers in residual blocks
        elif backbone == 'resnet34':
            layers = [3, 4, 6, 3]
        
        # Custom strides
        self.strides = []
        for i in range(5):
            if output_stride > 1:
                self.strides.append(2)
                output_stride //= 2
            else:
                self.strides.append(1)

        self.enc.inplanes = 64
        self.enc.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=self.strides[0], padding=3, bias=False)
        if self.strides[1] > 1:
            self.enc.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        else: self.enc.maxpool = nn.Identity()
        self.enc.layer1 = self.enc._make_layer(BasicBlock, 64 , layers[0])
        self.enc.layer2 = self.enc._make_layer(BasicBlock, 128, layers[1], stride=self.strides[2])
        self.enc.layer3 = self.enc._make_layer(BasicBlock, 256, layers[2], stride=self.strides[3])
        self.enc.layer4 = self.enc._make_layer(BasicBlock, 512, layers[3], stride=self.strides[4])

        self.drop = nn.Dropout2d(drop_rate)
    
    def forward(self, x):
        x0 = self.enc.conv1(x)
        x0 = self.enc.bn1(x0)
        x0 = self.enc.relu(x0)
        x1 = self.enc.maxpool(x0)

        layer1 = self.enc.layer1(x1)
        layer1 = self.drop(layer1)

        layer2 = self.enc.layer2(layer1)
        layer2 = self.drop(layer2)

        layer3 = self.enc.layer3(layer2)
        layer3 = self.drop(layer3)

        layer4 = self.enc.layer4(layer3)
        layer4 = self.drop(layer4)

        return [x, x0, layer1, layer2, layer3, layer4]

class Decoder(nn.Module):
    def __init__(self, num_classes, strides, drop_rate):
        super(Decoder, self).__init__()

        self.upsample4 = nn.ConvTranspose2d(512, 256, 3, stride=strides[4], padding=1)
        self.upsample3 = nn.ConvTranspose2d(256, 128, 3, stride=strides[3], padding=1)
        self.upsample2 = nn.ConvTranspose2d(128, 64 , 3, stride=strides[2], padding=1)
        self.upsample1 = nn.ConvTranspose2d(64 , 64 , 3, stride=strides[1], padding=1)
        self.upsample0 = nn.ConvTranspose2d(32 , 32 , 3, stride=strides[0], padding=1)

        res_layer = models.resnet.ResNet(BasicBlock, [1, 1, 1, 1])
        self.decoder_Residual3 = res_layer._make_layer(BasicBlock, 256, 1)
        self.decoder_Residual2 = res_layer._make_layer(BasicBlock, 128, 1)
        self.decoder_Residual1 = res_layer._make_layer(BasicBlock, 64, 1)
        res_layer.inplanes = 128
        self.decoder_Residual0 = res_layer._make_layer(BasicBlock, 32, 1)

        self.drop = nn.Dropout2d(drop_rate)
        self.logits = nn.Conv2d(32, num_classes, 3, padding=1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.upsample4(x[-1], output_size=x[-2].size())
        out = torch.cat((out, x[-2]), dim=1) 
        out = self.decoder_Residual3(out)
        out = self.drop(out)

        out = self.upsample3(out, output_size=x[-3].size())
        out = torch.cat((out, x[-3]), dim=1) 
        out = self.decoder_Residual2(out)
        out = self.drop(out)

        out = self.upsample2(out, output_size=x[-4].size())
        out = torch.cat((out, x[-4]), dim=1) 
        out = self.decoder_Residual1(out)
        out = self.drop(out)

        out = self.upsample1(out, output_size=x[-5].size())
        out = torch.cat((out, x[-5]), dim=1) 
        out = self.decoder_Residual0(out)
        out = self.drop(out)

        out = self.upsample0(out, output_size=x[-6].size())
        out = self.logits(out)
        # out = self.softmax(out)

        return out

class ResUNet(nn.Module):
    def __init__(self, resnet_backbone: str, in_chans = 2,
                 num_classes: int = 2, output_stride = 4, drop_rate=0.1):

        super(ResUNet, self).__init__()

        self.net_name = 'ResUNet'

        self.enc = Encoder(resnet_backbone, in_chans, drop_rate, output_stride)
        self.dec = Decoder(num_classes, self.enc.strides, drop_rate)

        self.init_weights()

    def forward(self, x):

        neck = self.enc(x)
        out  = self.dec(neck)

        return out

    def init_weights(self, zero_init_residual=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


class VisionTrans(nn.Module):
    def __init__(self, img_size = 320, patch_size = 16, num_heads=6, depth=8, 
                       embed_dim=384, in_chans=3, num_classes=1,
                       drop_rate=0.2, pretrained=False):
        super(VisionTrans, self).__init__()

        self.net_name = 'ViT'

        self.transformer = deit(pretrained=pretrained, in_chans=in_chans, embed_dim=embed_dim,
                                img_size=img_size, patch_size=patch_size, num_heads=num_heads,
                                depth=depth)
        
        self.upsample = nn.ModuleList([
                                        # nn.ConvTranspose2d(embed_dim   , embed_dim   , 3, stride=2, padding=1),
                                        # nn.ConvTranspose2d(embed_dim   , embed_dim//2, 3, stride=2, padding=1),
                                        # nn.ConvTranspose2d(embed_dim//2, embed_dim//4, 3, stride=2, padding=1),
                                        # nn.ConvTranspose2d(embed_dim//4, embed_dim//8, 3, stride=2, padding=1)
                                      ])
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.drop = nn.Dropout2d(drop_rate)
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()
    
    def forward(self, img):

        x = self.transformer(img)
        x = torch.transpose(x, 1, 2)
        x = x.view(x.shape[0], -1, int(np.sqrt(x.shape[2])), int(np.sqrt(x.shape[2])))
        x = self.drop(x)

        for up in self.upsample:
            x = up(x, [2*x.shape[-2], 2*x.shape[-1]])

        logits = self.head(x)
        # out = self.softmax(logits)

        return logits

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


def Create_mask(sz, pad, device='cuda'):

    # # mask example test
    # att_at_v = np.ones((1, 2, 3, 4))  # (batch, n_heads, n_tokens, embed_dim)
    # att_at_v[:,:,:,-1] = 0
    # w = np.transpose(q, (0, 2, 1, 3)).reshape((1, 3, -1)) # (batch, n_tokens, n_heads x embed_dim)

    pad = pad.long()
    mask = torch.ones(sz).to(device)
    attn_mask = torch.ones((sz[0], sz[1], sz[1])).to(device)
    for i in range(pad.shape[0]):
        if pad[i] > 0:
            mask[i, -pad[i]:] = 0
            attn_mask[i,:,-pad[i]:] = 0
            attn_mask[i,-pad[i]:,:] = 0   # not necessary if we use a mask in the loss function
    
    '''
    # Attention Mask after softmax
    # |             |             |
    # |   Valid     |    padded   |
    # |   Tokens    |    Tokens   |
    # |             |       0     |
    # |_____________|_____________|
    # |             |             |
    # |    padded   |    padded   |
    # |    Tokens   |    Tokens   |
    # |      0      |      0      |
    # |             |             |
    '''
    return mask, attn_mask

class Trans_no_patch_embed(VisionTransformer):
    def __init__(self, n_in_feat, embed_dim=768, **kwargs):
        super(Trans_no_patch_embed, self).__init__(embed_dim=embed_dim, mlp_ratio=4,
                                                   qkv_bias=True, drop_rate=0.2, 
                                                   norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                                   **kwargs)

        self.net_name = 'transformer'
        self.embed = nn.Linear(n_in_feat, embed_dim)

        self.init_weights()
    
    def forward(self, x, mask=None):

        x = self.embed(x)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x, mask=mask)
        x = self.norm(x)
        logits = self.head(x)

        return logits

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


class IRGS_Trans(nn.Module):
    def __init__(self, cnn, transformer, max_length, mix_images, random_tokens):

        super(IRGS_Trans, self).__init__()
        self.net_name = 'IRGS_trans'

        self.cnn = cnn
        self.transformer = transformer
        self.max_length = max_length
        self.mix_images = mix_images
        self.random_tokens = random_tokens


    def Extract_tokens(self, features, gts, segments, n_tokens, device='cuda'):
        
        B, _, _, _ = features.shape
        
        # CREATE EXCLUSIVE TOKEN IDS AMONG IMAGES SAMPLES
        s = 0
        tokens_ids = torch.zeros((n_tokens.sum())).to(device).long()
        for i in range(B):
            segments[i] += s + 1
            if self.random_tokens:
                tokens_ids[s:s+n_tokens[i]] = torch.randperm(n_tokens[i]).to(device).long() + s + 1
            else:
                tokens_ids[s:s+n_tokens[i]] = torch.arange  (n_tokens[i]).to(device).long() + s + 1
            s += n_tokens[i]


        if self.mix_images and self.random_tokens:
            ids = torch.randperm(len(tokens_ids)).to(device)
            tokens_ids = tokens_ids[ids]
        
        # CALCULATE TOKENS
        temp_seq, temp_lb, tokens, super_labels, pads = 5 * [torch.Tensor([]).to(device)]
        i_sample, cont = 0, 0

        for i in range(len(tokens_ids)):
            pos = torch.where(segments==tokens_ids[i])
            assert len(torch.unique(pos[0])) == 1, "Tokens ids are not unique among image samples"

            tk = features[pos[0], :,pos[1], pos[2]].mean(0)
            lb = torch.mode(gts[pos[0], pos[1], pos[2]])[0]

            if len(temp_seq)==0:
                temp_seq = tk.unsqueeze(0)
                temp_lb = lb.unsqueeze(0)
            else:
                temp_seq = torch.cat((temp_seq, tk.unsqueeze(0)), 0)
                temp_lb = torch.cat((temp_lb, lb.unsqueeze(0)), 0)
            cont += 1
                        
            pad = 0
            if not self.mix_images:
                if cont == n_tokens[i_sample]:
                    pad = self.max_length - len(temp_seq)
                    temp_seq = F.pad(temp_seq, (0, 0, 0, pad))
                    temp_lb = F.pad(temp_lb, (0, pad))
                    i_sample += 1
                    cont = 0

            # Creating batch
            if len(temp_seq) == self.max_length:
                if len(tokens)==0:
                    tokens = temp_seq.unsqueeze(0)
                    super_labels = temp_lb.unsqueeze(0)
                    pads = torch.Tensor([[pad]]).to(device)
                else:
                    tokens = torch.cat((tokens, temp_seq.unsqueeze(0)), 0)
                    super_labels = torch.cat((super_labels, temp_lb.unsqueeze(0)), 0)
                    pads = torch.cat((pads, torch.Tensor([[pad]]).to(device)), 0)
                
                temp_seq, temp_lb = 2 * [torch.Tensor([]).to(device)]
            
        # Last sequence
        if len(temp_seq):
            pad = self.max_length - len(temp_seq)
            temp_seq = F.pad(temp_seq, (0, 0, 0, pad))
            temp_lb = F.pad(temp_lb, (0, pad))

            if len(tokens)==0:
                tokens = temp_seq.unsqueeze(0)
                super_labels = temp_lb.unsqueeze(0)
                pads = torch.Tensor([[pad]]).to(device)
            else:
                tokens = torch.cat((tokens, temp_seq.unsqueeze(0)), 0)
                super_labels = torch.cat((super_labels, temp_lb.unsqueeze(0)), 0)
                pads = torch.cat((pads, torch.Tensor([[pad]]).to(device)), 0)

        return tokens, super_labels, pads, tokens_ids, segments
    
    def forward(self, img, gts, segments, n_tokens, stage=None, device='cuda'):

        cnn_logits, features = self.cnn(img)        

        if stage == 'cnn': return cnn_logits, 0, 0, 0, 0, 0

        tokens, super_labels, pads, tokens_ids, segments = self.Extract_tokens(features, gts, segments, 
                                                                               n_tokens, device=device)

        mask, attn_mask = Create_mask(super_labels.shape, pads, device=device)

        trans_logits = self.transformer(tokens, attn_mask)

        return cnn_logits, trans_logits, super_labels, mask, tokens_ids, segments


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
