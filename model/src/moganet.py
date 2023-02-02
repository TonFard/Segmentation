import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F
from torch import Tensor
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
from parcNext import DropPath, trunc_normal_


def build_act_layer(act_type: str):
    """
        
        Build activation layer
    """
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'GELU':
        return nn.GELU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.SiLU()

def build_norm_layer(norm_type: str, embed_dims: int):
    """
        Build normalization layer
    """
    
    assert norm_type in ['GN', 'BN', 'LN2d', 'SyncBN']
    if norm_type == 'GN':
        return nn.GroupNorm(embed_dims, embed_dims, eps=1e-5)
    elif norm_type == 'BN':
        return nn.BatchNorm2d(embed_dims, eps=1e-5)
    elif norm_type == 'LN2d':
        return LayerNorm2d(embed_dims, eps=1e-6)
    else:
        return nn.SyncBatchNorm(embed_dims, eps=1e-5)

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape: str, eps: float = 1e-6, data_format: str = 'channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
        
    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class ElementScale(nn.Module):
    def __init__(self, embed_dims: int, init_values: float, requires_grad: bool = True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_values * torch.ones((1, embed_dims, 1, 1)),
            requires_grad = requires_grad
        )
    def forward(self, x):
        return x * self.scale
    

class ConvPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dims, kernel_size=3, stride=2, norm_type='BN'):
        super(ConvPatchEmbed, self).__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dims,
            kernel_size=kernel_size,
            stride=stride, padding=kernel_size//2
        )
        self.norm = build_norm_layer(norm_type, embed_dims)
    
    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size
    
class StackConvPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dims, kernel_size=3, stride=2, act_type='GELU', norm_type='BN'):
        super(StackConvPatchEmbed, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels, embed_dims // 2, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2
            ),
            build_norm_layer(norm_type, embed_dims // 2),
            build_act_layer(act_type),
            nn.Conv2d(
                embed_dims // 2, embed_dims, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2,
            ),
            build_norm_layer(norm_type, embed_dims)
        )
        
    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size
    
class ChannelAggregationFFN(nn.Module):
    """
        Structure of Channel aggregation block
        -------------------------------------------
        Input
        -------------------------------------------
        layer1: Conv1x1
        layer2: DWConv3x3
        layer3: GELU
        layer4: Channel Aggregation(feat_decompose)
        layer5: Conv1x1
        -------------------------------------------
        output 
        -------------------------------------------
                         
    """
    def __init__(self, embed_dims, feedforward_channels, kernel_size=3, act_type='GELU', ffn_drop=.0):
        super(ChannelAggregationFFN, self).__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.fc1 = nn.Conv2d(in_channels=embed_dims, out_channels=self.feedforward_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(self.feedforward_channels, self.feedforward_channels, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=True, groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(self.feedforward_channels, embed_dims, kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)
        # C -> 1
        self.decompose = nn.Conv2d(self.feedforward_channels, 1, kernel_size=1)
        self.sigma = ElementScale(self.feedforward_channels, init_values=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)
    
    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        return x + self.sigma(x - self.decompose_act(self.decompose(x)))
    
    def forward(self, x):
        # proj 1
        x = self.fc1(x)                        
        x = self.dwconv(x)               
        x = self.act(x)                     
        x = self.drop(x)              
        # proj 2 
        x = self.feat_decompose(x)             
        x = self.fc2(x)                  
        x = self.drop(x)
        return x
    

class MultiOrderDWConv(nn.Module):
    """
        Structure of Spatial aggregation block
        -------------------------------------------
        Input
        -------------------------------------------
        layer1: DWConv5x5
        layer2: Concat([layer1 * 1/8, DWConv5x5(layer1) * 3/8, DWConv7x7(layer1) * 1/2])
        layer3: Conv1x1
        -------------------------------------------
        output 
        -------------------------------------------
                         
    """
    def __init__(self, embed_dims, dw_dilation=[1, 2, 3], channel_split=[1, 3, 4]):
        super(MultiOrderDWConv, self).__init__()
        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0
        
        self.DW_conv0 = nn.Conv2d(
            self.embed_dims, self.embed_dims,
            kernel_size=5, padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims, stride=1, dilation=dw_dilation[0]
        )
        
        self.DW_conv1 = nn.Conv2d(
            self.embed_dims_1, self.embed_dims_1,
            kernel_size=5, padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1]
        )
        self.DW_conv2 = nn.Conv2d(
            self.embed_dims_2, self.embed_dims_2,
            kernel_size=7, padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2]
        )
        self.PW_conv = nn.Conv2d(
            self.embed_dims, self.embed_dims,
            kernel_size=1
        )
    
    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...]
        )
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims - self.embed_dims_2:, ...]
        )
        x = torch.cat(
            [x_0[:, :self.embed_dims_0, ...], x_1, x_2],
            dim=1
        )
        x = self.PW_conv(x)
        return x
    
    
class MultiOrderGatedAggregation(nn.Module):
    """
        Structure of Spatial aggregation block
        -------------------------------------------
        Input
        -------------------------------------------
        layer1: feat_decompose
        layer2: GELU
        layer3: MultiOrderGatedAggregation
            1. path1: 
                layer1: Conv1x1
                layer2: SiLU
            2. path2:
                layer1: MultiOrderDWConv
                layer2: Conv1x1
                layer3: SiLU
            3. Conv1x1(path1_output * path2_output)
            
        -------------------------------------------
        output 
        -------------------------------------------
                         
    """
    def __init__(self, embed_dims, attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4],
                attn_act_type='SiLU', attn_force_fp32=False):
        super(MultiOrderGatedAggregation, self).__init__()
        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            embed_dims, embed_dims,
            kernel_size=1
        )
        self.gate = nn.Conv2d(
            embed_dims, embed_dims,
            kernel_size=1
        )
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split
        )
        self.proj_2 = nn.Conv2d(
            embed_dims, embed_dims,
            kernel_size=1
        )
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        self.sigma = ElementScale(
            embed_dims, init_values=1e-5, requires_grad=True
        )
    
    def feat_decompose(self, x):
        x = self.proj_1(x)
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x
    
    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))
    
    def forward(self, x):
        shortcut = x.clone()
        x = self.feat_decompose(x)
        
        g = self.gate(x)
        v = self.value(x)
        
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x        

    
    

class MogaBlock(nn.Module):
    def __init__(self, embed_dims, ffn_ratio=4., drop_rate=0., drop_path_rate=0., act_type='GELU', norm_type='BN', init_value=1e-5,
                attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4], attn_act_type='SiLU', attn_force_fp32=False):
        super(MogaBlock, self).__init__()
        self.out_channels = embed_dims
        self.norm1 = build_norm_layer(norm_type, embed_dims)
        self.attn = MultiOrderGatedAggregation(
            embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split,
            attn_act_type=attn_act_type,
            attn_force_fp32=attn_force_fp32
        )
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        self.norm2 = build_norm_layer(norm_type, embed_dims)
        
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(
            embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_type=attn_act_type,
            ffn_drop=drop_rate
        )
        
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
    
    def forward(self, x):
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x     
    
class MogaNet(nn.Module):
    arch_zoo = {
        **dict.fromkeys(['xt', 'x-tiny'],
                        {'embed_dims': [32, 64, 96, 192],
                         'depths': [3, 3, 10, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': [32, 64, 128, 256],
                         'depths': [3, 3, 12, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [2, 3, 12, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': [64, 160, 320, 512],
                         'depths': [4, 6, 22, 3],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': [64, 160, 320, 640],
                         'depths': [4, 6, 44, 4],
                         'ffn_ratios': [8, 8, 4, 4]}),
    }  # yapf: disable
    def __init__(self, arch='tiny',
                 in_channels=3,
                 num_classes=1000,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 init_value=1e-5,
                 head_init_scale=1.,
                 patch_sizes=[3, 3, 3, 3],
                 stem_norm_type='BN',
                 conv_norm_type='BN',
                 patchembed_types=['ConvEmbed', 'Conv', 'Conv', 'Conv',],
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_final_dilation=True,
                 attn_force_fp32=False,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'ffn_ratios'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.ffn_ratios = self.arch_settings['ffn_ratios']
        self.num_stages = len(self.depths)
        self.attn_force_fp32 = attn_force_fp32
        self.use_layer_norm = stem_norm_type == 'LN'
        assert len(patchembed_types) == self.num_stages
        self.fork_feat = fork_feat

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            if i == 0 and patchembed_types[i] == "ConvEmbed":
                assert patch_sizes[i] <= 3
                patch_embed = StackConvPatchEmbed(
                    in_channels=in_channels,
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    act_type='GELU',
                    norm_type=conv_norm_type,
                )
            else:
                patch_embed = ConvPatchEmbed(
                    in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    norm_type=conv_norm_type)

            if i == self.num_stages - 1 and not attn_final_dilation:
                attn_dw_dilation = [1, 2, 1]
            blocks = nn.ModuleList([
                MogaBlock(
                    embed_dims=self.embed_dims[i],
                    ffn_ratio=self.ffn_ratios[i],
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur_block_idx + j],
                    norm_type=conv_norm_type,
                    init_value=init_value,
                    attn_dw_dilation=attn_dw_dilation,
                    attn_channel_split=attn_channel_split,
                    attn_act_type=attn_act_type,
                    attn_force_fp32=attn_force_fp32,
                ) for j in range(depth)
            ])
            cur_block_idx += depth
            norm = build_norm_layer(stem_norm_type, self.embed_dims[i])

            self.add_module(f'patch_embed{i + 1}', patch_embed)
            self.add_module(f'blocks{i + 1}', blocks)
            self.add_module(f'norm{i + 1}', norm)

        if self.fork_feat:
            self.head = nn.Identity()
        else:
            # Classifier head
            self.num_classes = num_classes
            self.head = nn.Linear(self.embed_dims[-1], num_classes) \
                if num_classes > 0 else nn.Identity()

            # init for classification
            self.apply(self._init_weights)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model 
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights(pretrained)

    def _init_weights(self, m):
        """ Init for timm image classification """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        """ Init for mmdetection or mmsegmentation by loading pre-trained weights """
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            if self.init_cfg is not None:
                assert 'checkpoint' in self.init_cfg, f'Only support specify ' \
                                                      f'`Pretrained` in `init_cfg` in ' \
                                                      f'{self.__class__.__name__} '
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            blocks = getattr(self, f'blocks{i + 1}')
            norm = getattr(self, f'norm{i + 1}')

            x, hw_shape = patch_embed(x)
            for block in blocks:
                x = block(x)
            if self.use_layer_norm:
                x = x.flatten(2).transpose(1, 2)
                x = norm(x)
                x = x.reshape(-1, *hw_shape,
                            block.out_channels).permute(0, 3, 1, 2).contiguous()
            else:
                x = norm(x)
            if self.fork_feat:
                outs.append(x)

        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        else:
            # output only the last layer for image classification
            return x.mean(dim=[2, 3])

    def forward(self, x):
        x = self.forward_features(x)
        if self.fork_feat:
            # for dense prediction
            return x
        else:
            # for image classification
            x = self.head(x)
            return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': 0.90, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'moganet_xt': _cfg(crop_pct=0.9),
    'moganet_t': _cfg(crop_pct=0.9),
    'moganet_s': _cfg(crop_pct=0.9),
    'moganet_b': _cfg(crop_pct=0.9),
    'moganet_l': _cfg(crop_pct=0.9),
}    

def moganet_xtiny(pretrained=False, **kwargs):
    model = MogaNet(arch='x-tiny', **kwargs)
    model.default_cfg = default_cfgs['moganet_xt']
    return model

