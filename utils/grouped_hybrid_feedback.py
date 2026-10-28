"""Hybrid exclusive/shared feedback and historical power priors (lazy Torch)."""
from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
import math
import random

NAMESPACE = 'grouped_hybrid_anchor_v1'
ARMS = ('fdma_control', 'hybrid_f16_a96', 'hybrid_f32_a64')
ALLOCATIONS = {'fdma_control': (64, 0), 'hybrid_f16_a96': (16, 96), 'hybrid_f32_a64': (32, 64)}
NODES = (0, 5, 10, 15, 20, 25)
EMA_BETA, UNIFORM_SHRINK, PI_FLOOR, RIDGE = .01, .05, 1e-8, 1e-6


@dataclass(frozen=True)
class LayoutSpec:
    arm: str

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError('Unknown grouped hybrid arm')
        if 4*self.df+2*self.da != 256 or self.df%4 or self.da%4:
            raise ValueError('Hybrid budget must satisfy 4 Df + 2 Da = 256')
        occupancy = [tuple(k for k,s in enumerate(self.supports) if q in s) for q in range(256)]
        if any(len(users) not in (1,2) or len({k//2 for k in users})!=1 for users in occupancy):
            raise ValueError('Exclusive/shared/group-orthogonal occupancy violation')
        if sum(len(u)==1 for u in occupancy)!=4*self.df or sum(len(u)==2 for u in occupancy)!=2*self.da:
            raise ValueError('Actual exclusive/shared occupancy differs from budget')

    @property
    def df(self): return ALLOCATIONS[self.arm][0]
    @property
    def da(self): return ALLOCATIONS[self.arm][1]
    @property
    def layout(self): return self.arm
    @property
    def active_per_user(self): return self.df+self.da
    @property
    def occupied_resources(self): return 256
    @property
    def active_slots(self): return tuple(sb*64+j for sb in range(4) for j in range(self.active_per_user//4))
    @property
    def exclusive_slots(self): return tuple(sb*64+j for sb in range(4) for j in range(self.df//4))
    @property
    def shared_slots(self): return tuple(sb*64+j for sb in range(4) for j in range(self.df//4,self.active_per_user//4))
    @cached_property
    def supports(self):
        users=[]; e=self.df//4
        for k in range(4):
            support=[]
            for sb in range(4):
                group=[q for q in range(sb*64,(sb+1)*64) if q%4//2==k//2]
                support += [group[2*j+k%2] for j in range(e)]+group[2*e:]
            users.append(tuple(support))
        return tuple(users)
    @cached_property
    def canonical_to_physical(self):
        rows=[]
        for support in self.supports:
            row=[-1]*256
            for slot,q in zip(self.active_slots,support): row[slot]=q
            rows.append(tuple(row))
        return tuple(rows)
    @cached_property
    def physical_to_canonical(self):
        rows=[]
        for mapping in self.canonical_to_physical:
            row=[-1]*256
            for slot,q in enumerate(mapping):
                if q>=0: row[q]=slot
            rows.append(tuple(row))
        return tuple(rows)
    @cached_property
    def receive_groups(self):
        groups={}
        for q in range(256):
            users=tuple(k for k,s in enumerate(self.supports) if q in s)
            groups.setdefault(users,[]).append(q)
        return tuple((users,tuple(resources)) for users,resources in groups.items())
    def to_dict(self):
        return dict(arm=self.arm,D_f_per_ue=self.df,D_a_per_group=self.da,
            budget='4*Df+2*Da=256',groups=[[0,1],[2,3]],occupied_resources=256,
            canonical_slots_per_user=256,active_per_user=self.active_per_user,
            active_slots=list(self.active_slots),exclusive_slots=list(self.exclusive_slots),shared_slots=list(self.shared_slots),
            physical_supports=[list(x) for x in self.supports],
            canonical_to_physical=[list(x) for x in self.canonical_to_physical],
            physical_to_canonical=[list(x) for x in self.physical_to_canonical],
            mapping='each subband: group q%4//2; first 2e RE alternate exclusive users, remaining RE shared; canonical exclusive then shared')


def layout_for_arm(arm): return LayoutSpec(arm)


def namespace_seed(seed,stream):
    if type(seed) is not int or not 0<=seed<2**63 or stream not in ('channel','initialization','channel_test','channel_validation'):
        raise ValueError('Invalid hybrid seed namespace')
    body=json.dumps(dict(namespace=NAMESPACE,seed=seed,stream=stream),sort_keys=True,separators=(',',':'))
    return int.from_bytes(hashlib.sha256(body.encode()).digest()[:4],'big')


@dataclass(frozen=True)
class RandomAddress:
    seed: int
    split: str
    index: int
    stream: str
    def __post_init__(self):
        if type(self.seed) is not int or not 0<=self.seed<2**63 or type(self.index) is not int or self.index<0:
            raise ValueError('Invalid hybrid random address')
        if self.split not in ('init','train','validation','test','preflight') or self.stream not in ('gamma','standard_noise','ue_legacy','bs','anchor'):
            raise ValueError('Unknown hybrid random stream')
    def to_dict(self): return dict(namespace=NAMESPACE,seed=self.seed,split=self.split,index=self.index,stream=self.stream)
    @property
    def seed_value(self):
        value=json.dumps(self.to_dict(),sort_keys=True,separators=(',',':')).encode()
        return int.from_bytes(hashlib.sha256(value).digest()[:8],'big')&((1<<63)-1)


def sample_gamma(seed,global_step):
    return random.Random(RandomAddress(seed,'train',global_step,'gamma').seed_value).uniform(0,25)


def standard_noise(seed,split,index,batch_size,device='cpu'):
    import torch
    if type(batch_size) is not int or batch_size<1: raise ValueError('Positive batch size required')
    generator=torch.Generator(device='cpu').manual_seed(RandomAddress(seed,split,index,'standard_noise').seed_value)
    return torch.view_as_complex(torch.randn(batch_size,32,256,2,generator=generator,dtype=torch.float32)).to(device)


def noise_variance(gamma):
    if isinstance(gamma,bool) or not isinstance(gamma,(int,float)) or not math.isfinite(gamma) or not 0<=gamma<=25:
        raise ValueError('Public gamma must lie in [0,25]')
    return .25*10**(-float(gamma)/10)


def interpolation(gamma):
    noise_variance(gamma)
    low=min(int(gamma//5),5)
    if low==5 or gamma==NODES[low]: return ((low,1.),)
    upper=(float(gamma)-NODES[low])/5
    return ((low,1-upper),(low+1,upper))


def prior_metadata():
    return dict(nodes=list(NODES),ema_beta=EMA_BETA,uniform_shrink=UNIFORM_SHRINK,
        initial='active-coordinate uniform, sum64 per UE at every node; inactive0',
        read='linear interpolation between adjacent committed nodes',
        update='target=.95*detached mean batch normalized-coordinate power+.05*uniform; rate=.01*interpolation_weight; normalize each node/user to64',
        timing='forward reads committed t-1; successful optimizer update then explicit commit t',
        prior_floor=PI_FLOOR,ridge=RIDGE,error_feature='log1p(error_proxy / pi.clamp_min(1e-8))',
        error_proxy='diag((LG-I)diag(pi)(LG-I)^H + N0 L L^H)',posterior_variance=False,
        registered_buffers=['pi_nodes','update_count','node_update_mass'])


def protocol_metadata():
    return dict(schema=NAMESPACE,energy_per_frame_per_ue=64.,occupied_resources=256,
        noise_variance='0.25*10**(-gamma/10)',gamma='block-average transmit reference, not received feedback SNR',
        normalization='joint exclusive/shared active vector; 8*u/sqrt(sum(abs(u)**2)), no epsilon',
        one_full_mac=True,one_noise_tensor=True,perfect_ul=True,pilots=False,power_control=False,
        groups=[[0,1],[2,3]],prior=prior_metadata(),receiver='solve(G^H G+N0*diag(1/pi)+ridge I,G^H)',
        random_namespace=NAMESPACE,training_data_reuses_previous_namespace=False)


def architecture_metadata(arm):
    from configs.grouped_codec_train_config import protocol as old
    base=old.architecture_metadata('legacy_all4')
    base['bs']=dict(base['bs'])
    base['bs']['memory_features']=[
        'log1p(error_proxy/pi.clamp_min(1e-8))' if name=='log1p_reliability_proxy' else name
        for name in base['bs']['memory_features']]
    base['bs']['public_reliability']=(
        'log1p(error_proxy/pi.clamp_min(1e-8)); full linear-estimation error proxy '
        'under historical per-user coordinate power; not calibrated posterior variance or precision')
    base.update(architecture=NAMESPACE,model_class='GroupedHybridAnchorPrecoder',arm=arm,
        layout_spec=LayoutSpec(arm).to_dict(),protocol=protocol_metadata(),
        anchor=dict(input='own exclusive soft complex codeword/error feature/canonical and physical coordinates',
            width=128,hidden=128,input_width=9,pooling='masked exclusive mean',output_weight_std=.001,
            inject='user-specific addition to own Joint query features before the two Joint layers',gate=False),
        power_statistics_buffer=True,prior=prior_metadata(),
        initialization='same-seed F best common weights; independent new-namespace anchor initialization; uniform priors')
    return base


def tensor_geometry(layout,device='cpu'):
    import torch
    if not isinstance(layout,LayoutSpec): raise TypeError('Hybrid LayoutSpec required')
    support=torch.tensor(layout.canonical_to_physical,dtype=torch.long,device=device)
    physical=torch.zeros(4,256,dtype=torch.bool,device=device)
    physical.scatter_(1,torch.tensor(layout.supports,device=device),True)
    exclusive=torch.zeros_like(physical)
    exclusive[:,list(layout.exclusive_slots)]=True
    return dict(support_sc=support,active_mask=support>=0,physical_mask=physical,
        active_slots=torch.tensor(layout.active_slots,dtype=torch.long,device=device),
        physical_supports=torch.tensor(layout.supports,dtype=torch.long,device=device),
        exclusive_mask=exclusive,shared_mask=(support>=0)&~exclusive)


def _complex(value,name,shape=None):
    import torch
    if not torch.is_tensor(value) or value.dtype!=torch.complex64 or not torch.isfinite(value).all():
        raise ValueError(name+' must be finite complex64')
    if shape is not None and tuple(value.shape)!=tuple(shape): raise ValueError(name+' shape mismatch')


def normalize_and_map(raw,layout):
    import torch
    _complex(raw,'raw')
    if raw.ndim!=3 or tuple(raw.shape[1:])!=(4,256) or raw.shape[0]<1: raise ValueError('raw[B,4,256] required')
    active=torch.tensor(layout.active_slots,device=raw.device)
    u=raw.index_select(-1,active)
    power=u.abs().square().sum(-1,keepdim=True)
    if not torch.isfinite(power).all() or (power<=1e-12).any(): raise FloatingPointError('Degenerate active codeword')
    normalized=8*u/power.sqrt()
    canonical=torch.zeros_like(raw).scatter(-1,active[None,None].expand(raw.shape[0],4,-1),normalized)
    support=torch.tensor(layout.supports,device=raw.device)
    physical=torch.zeros_like(raw).scatter(-1,support[None].expand(raw.shape[0],-1,-1),normalized)
    energy=canonical.abs().square().sum(-1)
    if not torch.allclose(energy,torch.full_like(energy,64),rtol=0,atol=1e-4): raise FloatingPointError('Energy64 violation')
    return physical,canonical,energy


def transmit(H_ul,X,layout,*,N0,standard_noise):
    import torch
    _complex(X,'physical symbols')
    if X.ndim!=3 or tuple(X.shape[1:])!=(4,256): raise ValueError('Physical symbols must have shape[B,4,256]')
    _complex(H_ul,'UL',(X.shape[0],4,32,256))
    _complex(standard_noise,'physical noise',(X.shape[0],32,256))
    if X.device!=H_ul.device or standard_noise.device!=H_ul.device: raise ValueError('Physical devices differ')
    if standard_noise.requires_grad or not math.isfinite(N0) or N0<=0: raise ValueError('Nonlearned positive noise required')
    if torch.count_nonzero(X.masked_select(~tensor_geometry(layout,X.device)['physical_mask'][None])):
        raise ValueError('Transmission outside actual physical support')
    return (H_ul*X[:,:,None]).sum(1)+math.sqrt(N0/2)*standard_noise


def recover(Y,H_ul,layout,*,N0,prior_power):
    import torch
    _complex(Y,'Y')
    batch=Y.shape[0];_complex(Y,'Y',(batch,32,256));_complex(H_ul,'UL',(batch,4,32,256))
    if not math.isfinite(N0) or N0<=0 or H_ul.device!=Y.device or prior_power.device!=Y.device:
        raise ValueError('Invalid receiver N0/device')
    if prior_power.shape!=(4,256) or prior_power.dtype!=torch.float32 or prior_power.requires_grad:
        raise ValueError('Committed detached float32 per-user/coordinate prior required')
    geometry=tensor_geometry(layout,Y.device)
    if not torch.isfinite(prior_power).all() or (prior_power[geometry['active_mask']]<=0).any(): raise ValueError('Invalid positive prior')
    if torch.count_nonzero(prior_power[~geometry['active_mask']]): raise ValueError('Inactive prior must be zero')
    z=torch.zeros(batch,4,256,dtype=torch.complex64,device=Y.device)
    error=torch.zeros(batch,4,256,dtype=torch.float32,device=Y.device)
    inverse=layout.physical_to_canonical
    for users,resources in layout.receive_groups:
        ui=torch.tensor(users,device=Y.device);qi=torch.tensor(resources,device=Y.device)
        G=H_ul.index_select(1,ui).index_select(-1,qi).permute(0,3,2,1)
        slots=torch.tensor([[inverse[k][q] for q in resources] for k in users],device=Y.device)
        pi=torch.stack([prior_power[k,slots[j]] for j,k in enumerate(users)],-1).clamp_min(PI_FLOOR)
        gh=G.mH;eye=torch.eye(len(users),device=Y.device,dtype=torch.complex64)
        A=gh@G+torch.diag_embed(N0/pi).to(torch.complex64)[None]+RIDGE*eye
        L=torch.linalg.solve(A,gh)
        received=Y.index_select(-1,qi).transpose(1,2)[...,None]
        estimate=(L@received).squeeze(-1)
        mismatch=L@G-eye
        diagonal=(mismatch.abs().square()*pi[None,:,None,:]).sum(-1)+N0*L.abs().square().sum(-1)
        for j,k in enumerate(users):
            z[:,k,slots[j]]=estimate[...,j]
            error[:,k,slots[j]]=diagonal[...,j]
    if not torch.isfinite(z).all() or not torch.isfinite(error).all() or (error<0).any():
        raise FloatingPointError('Invalid linear recovery/error proxy')
    feature=torch.log1p(error/prior_power.clamp_min(PI_FLOOR)[None]).masked_fill(~geometry['active_mask'][None],0)
    return dict(z_hat=z,error_proxy=error,error_feature=feature,prior_power=prior_power)


def anchor_context(soft,error_feature,query_fixed,exclusive_mask,input_layer,output_layer):
    """Exact small learned anchor path; callable in CPU tests without model imports."""
    import torch
    if soft.ndim!=3 or tuple(soft.shape[1:])!=(4,256) or error_feature.shape!=soft.shape:
        raise ValueError('Anchor requires per-user canonical soft/error tensors')
    if tuple(query_fixed.shape)!=(4,256,12) or tuple(exclusive_mask.shape)!=(4,256):
        raise ValueError('Anchor geometry mismatch')
    count=exclusive_mask.sum(-1)
    if (count<=0).any(): raise ValueError('Each user needs genuine exclusive anchor observations')
    coordinates=query_fixed[...,:6][None].expand(soft.shape[0],-1,-1,-1)
    features=torch.cat((torch.view_as_real(soft.contiguous()),error_feature[...,None],coordinates),-1)
    features=features.masked_fill(~exclusive_mask[None,:,:,None],0)
    encoded=torch.nn.functional.gelu(input_layer(features))
    pooled=(encoded*exclusive_mask[None,:,:,None]).sum(-2)/count[None,:,None]
    return output_layer(pooled)


def make_power_prior(layout):
    """Small real nn.Module, importable/testable without importing any model/Sionna."""
    import torch
    from torch import nn
    class HistoricalPowerPrior(nn.Module):
        def __init__(self):
            super().__init__()
            self.layout=layout
            uniform=torch.zeros(4,256,dtype=torch.float32)
            uniform[:,list(layout.active_slots)]=64/layout.active_per_user
            self.register_buffer('pi_nodes',uniform[None].repeat(6,1,1))
            self.register_buffer('update_count',torch.tensor(0,dtype=torch.int64))
            self.register_buffer('node_update_mass',torch.zeros(6,dtype=torch.float64))
            self.register_buffer('uniform',uniform,persistent=False)
        def validate_state(self,expected_step=None):
            count=int(self.update_count.item())
            if self.update_count.dtype!=torch.int64 or self.update_count.ndim!=0 or count<0 or (expected_step is not None and count!=expected_step):
                raise ValueError('Historical prior update count mismatch')
            if self.pi_nodes.shape!=(6,4,256) or self.pi_nodes.dtype!=torch.float32 or not torch.isfinite(self.pi_nodes).all():
                raise ValueError('Historical prior shape/dtype/finite mismatch')
            active=self.uniform>0
            if (self.pi_nodes[:,active]<=0).any() or torch.count_nonzero(self.pi_nodes[:,~active]): raise ValueError('Historical prior active support invalid')
            if not torch.allclose(self.pi_nodes.sum(-1),torch.full_like(self.pi_nodes.sum(-1),64),rtol=0,atol=1e-4): raise ValueError('Historical prior must sum64 per UE/node')
            if (self.node_update_mass.shape!=(6,) or self.node_update_mass.dtype!=torch.float64
                    or not torch.isfinite(self.node_update_mass).all() or (self.node_update_mass<0).any()
                    or not math.isclose(self.node_update_mass.sum().item(),count,rel_tol=0,abs_tol=1e-7)):
                raise ValueError('Historical node update mass/count mismatch')
        def read(self,gamma):
            self.validate_state()
            return sum(self.pi_nodes[node]*weight for node,weight in interpolation(gamma)).detach().clone()
        def pending(self,canonical,gamma,step):
            _complex(canonical,'normalized canonical')
            if canonical.ndim!=3 or tuple(canonical.shape[1:])!=(4,256): raise ValueError('Invalid canonical statistics shape')
            if type(step) is not int or step!=int(self.update_count.item())+1: raise ValueError('Statistics must refer to next absolute update')
            return dict(schema=NAMESPACE,layout=self.layout.to_dict(),step=step,gamma_db=float(gamma),
                read_count=int(self.update_count.item()),power_mean=canonical.detach().abs().square().mean(0).clone())
        @torch.no_grad()
        def commit(self,stats):
            self.validate_state()
            if not self.training: raise ValueError('Evaluation cannot commit training power statistics')
            if (not isinstance(stats,dict) or stats.get('schema')!=NAMESPACE or stats.get('layout')!=self.layout.to_dict()
                    or type(stats.get('step')) is not int or stats['step']!=int(self.update_count.item())+1
                    or stats.get('read_count')!=int(self.update_count.item())): raise ValueError('Stale/duplicate/cross-layout power statistics')
            power=stats.get('power_mean')
            if (not torch.is_tensor(power) or power.requires_grad or power.shape!=(4,256)
                    or power.dtype!=torch.float32 or power.device!=self.pi_nodes.device
                    or not torch.isfinite(power).all() or (power<0).any()): raise ValueError('Invalid detached normalized-codeword statistic')
            active=self.uniform>0
            if torch.count_nonzero(power[~active]) or not torch.allclose(power.sum(-1),torch.full_like(power.sum(-1),64),rtol=0,atol=1e-4):
                raise ValueError('Statistics must use actual jointly normalized symbols, not raw or summed users')
            target=(1-UNIFORM_SHRINK)*power+UNIFORM_SHRINK*self.uniform
            nodes=self.pi_nodes.clone();mass=self.node_update_mass.clone()
            for node,weight in interpolation(stats['gamma_db']):
                rate=EMA_BETA*weight
                updated=(1-rate)*nodes[node]+rate*target
                nodes[node]=updated*(64/updated.sum(-1,keepdim=True))
                mass[node]+=weight
            if not torch.isfinite(nodes).all(): raise FloatingPointError('Invalid pending prior update')
            self.pi_nodes.copy_(nodes);self.node_update_mass.copy_(mass);self.update_count.add_(1)
            self.validate_state(stats['step'])
    return HistoricalPowerPrior()
