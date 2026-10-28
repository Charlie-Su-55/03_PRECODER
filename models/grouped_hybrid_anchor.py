"""Independent grouped hybrid with user-exclusive anchors and committed power priors."""
from __future__ import annotations

import math
from types import SimpleNamespace
import torch
from torch import nn

from configs.grouped_hybrid_anchor_train_config import protocol, validate_config
from .adaptive_hybrid import AdaptiveHybridPrecoder
from .adaptive_hybrid_joint_swin import _coordinates, _initialize_extra, _run, channel_coupling
from .baseline_common import inverse_angular_delay_transform
from .grouped_codec_precoder import (
    LegacyUE, GroupedJointSwinDecoder, GroupedObservationMemory, MappedJointLayer, MappedAttention,
)
from .swin_precoder import SwinCFNetDecoder


class ExclusiveAnchor(nn.Module):
    """Own exclusive soft symbols/error/coordinates, never truth or another user's symbols."""
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(9,128)
        self.output = nn.Linear(128,128)
        nn.init.normal_(self.input.weight,std=.02)
        nn.init.zeros_(self.input.bias)
        nn.init.normal_(self.output.weight,std=.001)
        nn.init.zeros_(self.output.bias)

    def forward(self,soft,error_feature,query_fixed,exclusive_mask):
        return protocol.anchor_context(soft,error_feature,query_fixed,exclusive_mask,self.input,self.output)


class HybridJointSwinDecoder(GroupedJointSwinDecoder):
    """Same complete BS parameter names; local geometry and one explicit anchor addition."""
    def __init__(self,layout,seed):
        nn.Module.__init__(self)
        self.layout=layout
        # Equivalent common module construction; no fake old LayoutSpec or global patch.
        self.codec=SwinCFNetDecoder(SimpleNamespace(SWIN_NC=256,ANTENNAS=32,SWIN_PATCH_SIZE=4,
            SWIN_WINDOW_SIZE=4,SWIN_EMBED_DIM=128,SWIN_NUM_HEADS=4,SWIN_MLP_RATIO=4.,
            SWIN_NUM_BLOCKS=4,SWIN_CODEWORD_DIM=512))
        self.spatial_position=nn.Linear(10,128)
        self.observation_memory=GroupedObservationMemory()
        self.query_projection=nn.Linear(15,128)
        self.joint_layers=nn.ModuleList([MappedJointLayer() for _ in range(2)])
        self.overlap_bias=nn.Parameter(torch.full((4,),.02))
        self.coupling_bias=nn.Parameter(torch.full((4,),.02))
        self.resource_cross_bias=nn.Parameter(torch.full((4,),.02))
        self.residual_norm=nn.LayerNorm(128)
        self.residual_head=nn.Linear(128,2)
        self.spatial_query_norm=nn.LayerNorm(128)
        self.spatial_context_norm=nn.LayerNorm(128)
        self.spatial_attention=MappedAttention()
        for name,module in self.named_modules():
            if name=='codec' or name.startswith('codec.'): continue
            _initialize_extra(module)
        nn.init.normal_(self.residual_head.weight,std=.001)
        nn.init.normal_(self.spatial_attention.out.weight,std=.001)
        for name,value in protocol.tensor_geometry(layout).items():
            self.register_buffer(name,value,persistent=False)
        resources=_coordinates(torch.arange(256),256)
        one_hot=torch.eye(4)
        slots=resources[None].expand(4,256,3)
        physical=_coordinates(self.support_sc.clamp_min(0),256)
        roles=torch.stack((self.exclusive_mask,self.shared_mask),-1).float()
        self.register_buffer('query_fixed',torch.cat((physical,slots,roles,
            one_hot[:,None].expand(4,256,4)),-1),persistent=False)
        rows,columns=_coordinates(torch.arange(64),64),_coordinates(torch.arange(8),8)
        spatial=torch.cat((rows[None,:,None].expand(4,64,8,3),columns[None,None].expand(4,64,8,3),
                           one_hot[:,None,None].expand(4,64,8,4)),-1)
        self.register_buffer('resource_coordinates',resources,persistent=False)
        self.register_buffer('spatial_coordinates',spatial.reshape(4,512,10),persistent=False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(protocol.RandomAddress(
                protocol.namespace_seed(seed,'initialization'),'init',0,'anchor').seed_value)
            self.anchor=ExclusiveAnchor()

    def forward(self,z_hat,error_feature,Y,H_ul_est,gamma_db):
        batch=Y.shape[0]
        if (tuple(Y.shape)!=(batch,32,256) or tuple(H_ul_est.shape)!=(batch,4,32,256)
                or tuple(z_hat.shape)!=(batch,4,256) or error_feature.shape!=z_hat.shape):
            raise ValueError('Hybrid public decoder shape mismatch')
        device=self.codec.fc.weight.device
        for name,value in (('soft',z_hat),('Y',Y),('UL',H_ul_est)):
            if value.dtype!=torch.complex64 or value.device!=device or not torch.isfinite(value).all():
                raise ValueError(name+' must be finite complex64 on decoder device')
        if (error_feature.dtype!=torch.float32 or error_feature.device!=device
                or not torch.isfinite(error_feature).all() or (error_feature<0).any()):
            raise ValueError('Finite nonnegative transformed linear-error feature required')
        protocol.noise_variance(gamma_db)
        mask=self.active_mask[None]
        soft=z_hat.masked_fill(~mask,0)
        proxy=error_feature.masked_fill(~mask,0)
        grid=proxy.new_zeros(batch,4,256).scatter(-1,self.physical_supports[None].expand(batch,-1,-1),
            proxy.index_select(-1,self.active_slots)).transpose(1,2)
        gamma=proxy.new_full((1,1,1),float(gamma_db)/25)
        memory=_run(self.observation_memory,Y,H_ul_est,grid,self.physical_mask,
                    self.resource_coordinates,gamma,enabled=self.training)
        features=torch.cat((torch.view_as_real(soft.contiguous()),self.query_fixed[None].expand(batch,-1,-1,-1),
                            proxy[...,None]),-1)
        anchor=self.anchor(soft,proxy,self.query_fixed,self.exclusive_mask)
        context=((self.query_projection(features)+anchor[:,:,None,:])*mask[...,None]).reshape(batch,1024,128)
        coupling=channel_coupling(H_ul_est)
        for layer in self.joint_layers:
            context=_run(layer,context,memory,self.support_sc.flatten(),self.active_mask.flatten(),coupling,
                self.overlap_bias,self.coupling_bias,self.resource_cross_bias,enabled=self.training)
        residual=self.residual_head(self.residual_norm(context)).reshape(batch,4,256,2)
        correction=torch.complex(residual[...,0],residual[...,1]).masked_fill(~mask,0)
        corrected=(soft+correction).masked_fill(~mask,0)
        real=torch.cat((corrected.real,corrected.imag),-1)
        spatial=self.codec.fc(real.reshape(batch*4,512)).reshape(batch,4,512,128)
        spatial=spatial+self.spatial_position(self.spatial_coordinates)[None]
        flat=spatial.reshape(batch,2048,128)
        spatial=flat+_run(self._spatial_cross,flat,context,enabled=self.training)
        tokens=spatial.reshape(batch*4,512,128)
        for block in self.codec.blocks:
            tokens=_run(block,tokens,64,8,enabled=self.training)
        patches=self.codec.final_proj(self.codec.norm(tokens)).reshape(batch*4,64,8,2,4,4)
        H_ad=patches.permute(0,3,1,4,2,5).contiguous().reshape(batch,4,2,256,32)
        if not torch.isfinite(H_ad).all() or not torch.isfinite(corrected).all():
            raise FloatingPointError('Nonfinite hybrid decoder output')
        execution=dict(memory_shape=list(memory.shape),context_shape=list(context.shape),
            spatial_shape=list(spatial.shape),angular_delay_shape=list(H_ad.shape))
        return H_ad,corrected,execution,anchor


class GroupedHybridAnchorPrecoder(nn.Module):
    zf_solver=AdaptiveHybridPrecoder.zf_solver

    def __init__(self,cfg):
        super().__init__()
        validate_config(cfg)
        self.cfg,self.arm,self.layout=cfg,cfg.arm,protocol.LayoutSpec(cfg.arm)
        self.ue_frozen=False
        seed=protocol.namespace_seed(cfg.SEED,'initialization')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(protocol.RandomAddress(seed,'init',0,'ue_legacy').seed_value)
            self.ue=LegacyUE(cfg)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(protocol.RandomAddress(seed,'init',0,'bs').seed_value)
            self.decoder=HybridJointSwinDecoder(self.layout,cfg.SEED)
        self.power_prior=protocol.make_power_prior(self.layout)

    @property
    def architecture_metadata(self): return protocol.architecture_metadata(self.arm)
    def ue_parameters(self): yield from self.ue.parameters()
    def bs_parameters(self): yield from self.decoder.parameters()
    def inactive_parameter_names(self):
        return tuple(name for name,_ in self.named_parameters()
            if name.startswith(('ue.encoder.fdma_','ue.encoder.fc_fdma.')))

    def _forward(self,H_dl,H_ul,gamma_db,*,standard_noise,H_ul_est=None,diagnostics=False,statistics_step=None):
        batch=H_dl.shape[0]
        protocol._complex(H_dl,'DL',(batch,4,32,256))
        protocol._complex(H_ul,'UL',(batch,4,32,256))
        if H_dl.device!=H_ul.device or batch<1: raise ValueError('Invalid paired channels')
        if H_ul_est is not None and not torch.equal(H_ul_est,H_ul): raise ValueError('Perfect UL required')
        N0=protocol.noise_variance(gamma_db)
        prior=self.power_prior.read(gamma_db)
        raw=self.ue(H_dl,gamma_db,self.layout.active_per_user)
        X,truth,energy=protocol.normalize_and_map(raw,self.layout)
        Y=protocol.transmit(H_ul,X,self.layout,N0=N0,standard_noise=standard_noise)
        recovered=protocol.recover(Y,H_ul,self.layout,N0=N0,prior_power=prior)
        H_ad,corrected,execution,anchor=self.decoder(recovered['z_hat'],recovered['error_feature'],Y,H_ul,gamma_db)
        H_hat=inverse_angular_delay_transform(H_ad,256)
        W=self.zf_solver(H_hat)
        W=W*torch.sqrt(1/(W.abs().square().sum((1,2),keepdim=True)+1e-9))
        pending=None if statistics_step is None else self.power_prior.pending(truth,gamma_db,statistics_step)
        detail=None
        if diagnostics:
            exclusive=self.decoder.exclusive_mask;shared=self.decoder.shared_mask
            power=truth.detach().abs().square()
            detail=dict(z_hat=recovered['z_hat'].detach(),z_corrected=corrected.detach(),true_z=truth.detach(),
                error_proxy=recovered['error_proxy'].detach(),prior_power=prior.detach(),
                error_feature=recovered['error_feature'].detach(),active_mask=self.decoder.active_mask.detach(),
                exclusive_mask=exclusive.detach(),shared_mask=shared.detach(),energy=energy.detach(),
                energy_exclusive=power.masked_fill(~exclusive[None],0).sum(-1),
                energy_shared=power.masked_fill(~shared[None],0).sum(-1),
                anchor_rms=anchor.detach().square().mean(-1).sqrt(),
                noise_power=(standard_noise*math.sqrt(N0/2)).abs().square().mean((1,2)).detach(),
                N0=N0,gamma_db=float(gamma_db),prior_update_count=int(self.power_prior.update_count.item()),execution=execution)
        return W,H_hat,pending,detail

    def forward_with_representation(self,H_dl,H_ul,gamma_db,*,standard_noise,H_ul_est=None):
        W,H,_,_=self._forward(H_dl,H_ul,gamma_db,standard_noise=standard_noise,H_ul_est=H_ul_est)
        return W,H
    def forward_with_diagnostics(self,H_dl,H_ul,gamma_db,*,standard_noise,H_ul_est=None):
        W,H,_,detail=self._forward(H_dl,H_ul,gamma_db,standard_noise=standard_noise,H_ul_est=H_ul_est,diagnostics=True)
        return dict(precoder=W,channel_surrogate=H,diagnostics=detail)
    def training_with_statistics(self,H_dl,H_ul,gamma_db,*,standard_noise,global_step,H_ul_est=None):
        if not self.training: raise ValueError('Training statistics require training mode')
        W,H,stats,_=self._forward(H_dl,H_ul,gamma_db,standard_noise=standard_noise,H_ul_est=H_ul_est,
                                statistics_step=global_step)
        return W,H,stats
    def post_optimizer_commit(self,stats,global_step):
        if type(global_step) is not int or stats.get('step')!=global_step: raise ValueError('Optimizer/statistics step mismatch')
        self.power_prior.commit(stats)
    def forward(self,H_dl,H_ul,gamma_db,*,standard_noise,H_ul_est=None):
        W,H=self.forward_with_representation(H_dl,H_ul,gamma_db,standard_noise=standard_noise,H_ul_est=H_ul_est)
        return (W,H) if self.training else W


def migrate_legacy_fdma_state_dict(model,source_state):
    """Explicit F-to-hybrid initialization; ordinary resume must load the full new state."""
    if not isinstance(model,GroupedHybridAnchorPrecoder): raise TypeError('Hybrid target model required')
    model.power_prior.validate_state(expected_step=0)
    expected=model.state_dict()
    new_parameters={name for name,_ in model.named_parameters() if name.startswith('decoder.anchor.')}
    new_buffers={name for name in expected if name.startswith('power_prior.')}
    common=set(expected)-new_parameters-new_buffers
    if not isinstance(source_state,dict) or set(source_state)!=common:
        missing=sorted(common-set(source_state)) if isinstance(source_state,dict) else sorted(common)
        extra=sorted(set(source_state)-common) if isinstance(source_state,dict) else []
        raise ValueError(f'Complete F source keys required; missing={missing}, extra={extra}')
    assembled={};mapping=[]
    for name,reference in expected.items():
        if name in common:
            value=source_state[name]
            if not torch.is_tensor(value) or value.shape!=reference.shape or value.dtype!=reference.dtype:
                raise ValueError('Source shape/dtype mismatch: '+name)
            if not torch.isfinite(value).all(): raise ValueError('Nonfinite source tensor: '+name)
            assembled[name]=value
            mapping.append(dict(source=name,target=name,action='copied_exactly',shape=list(value.shape),dtype=str(value.dtype)))
        else:
            assembled[name]=reference
            mapping.append(dict(source=None,target=name,action='new_anchor_initialization' if name in new_parameters else 'new_uniform_historical_prior',
                shape=list(reference.shape),dtype=str(reference.dtype)))
    model.load_state_dict(assembled,strict=True)
    for name in common:
        if not torch.equal(model.state_dict()[name].cpu(),source_state[name].detach().cpu()): raise RuntimeError('Migration altered old state: '+name)
    return dict(strict_loaded=True,old_key_count=len(common),copied_keys=sorted(common),
        new_parameter_keys=sorted(new_parameters),new_buffer_keys=sorted(new_buffers),per_key=mapping,
        full_forward_equivalence_claimed=False)
