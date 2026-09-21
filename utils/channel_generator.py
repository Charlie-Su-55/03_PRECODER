# import torch
# import torch.nn as nn
# from sionna.phy.channel.tr38901 import CDL, AntennaArray
# from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

# class CDLChannelGenerator_Feedback:
#     """
#     Sionna 2.0 (Pure PyTorch) 隐式反馈专用信道生成器
    
#     核心逻辑：
#     1. 彻底去 TF 化，全流程 GPU 运算。
#     2. 双实例采样：H_dl 和 H_ul 独立生成，消除符号间的欺骗性相关性。
#     3. 维度对齐：[Batch, K, Nt, Nsc]，完美适配 Transformer 输入。
#     """
#     def __init__(self, Nt=32, Nsc=256, carrier_freq=3.5e9, speed=1.0, num_users=4):
#         self.Nt = Nt
#         self.Nsc = Nsc
#         self.num_users = num_users
        
#         # 1. 配置基站天线阵列 (4x8 单极化 = 32天线)
#         self.bs_array = AntennaArray(
#             num_rows=4, num_cols=8,
#             polarization="single", polarization_type="V",
#             antenna_pattern="38.901", carrier_frequency=carrier_freq
#         )
        
#         # 2. 配置用户天线阵列 (1x1 单天线)
#         self.ue_array = AntennaArray(
#             num_rows=1, num_cols=1,
#             polarization="single", polarization_type="V",
#             antenna_pattern="38.901", carrier_frequency=carrier_freq
#         )
        
#         # 3. 初始化 CDL-C 模型
#         self.cdl = CDL(
#             model="C", delay_spread=300e-9, carrier_frequency=carrier_freq,
#             ut_array=self.ue_array, bs_array=self.bs_array,
#             direction="downlink", min_speed=speed
#         )
        
#         # 4. 子载波频率配置
#         self.frequencies = subcarrier_frequencies(Nsc, 30e3)
#         # 每秒采样频率 (基于 30kHz 子载波间隔的符号周期)
#         self.sampling_freq = 1.0 / ((1.0 + 0.07) / 30e3)

#     def to(self, device):
#         """ 将组件移动到 5090 显存 """
#         self.cdl.to(device)
#         self.frequencies = self.frequencies.to(device)
#         return self

#     def _generate_single_snapshot(self, batch_size):
#         """ 内部函数：生成一次全用户的信道快照 """
#         # [total_samples, 1, 1, 1, 32, path, 1]
#         a, tau = self.cdl(
#             batch_size=batch_size * self.num_users,
#             num_time_steps=1,
#             sampling_frequency=self.sampling_freq
#         )
#         # 转换到频域 [batch*K, 1, 1, 1, 32, 1, 256]
#         h_freq = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)
#         # 压缩并重塑 [Batch, K, Nt, Nsc]
#         h = h_freq.squeeze(dim=(1, 2, 3, 5))
#         h = h.reshape(batch_size, self.num_users, self.Nt, self.Nsc)
        
#         # 功率归一化
#         pwr = torch.mean(torch.abs(h)**2, dim=(-2, -1), keepdim=True)
#         return h / torch.sqrt(pwr + 1e-12)

#     def generate_batch_data(self, batch_size, device='cuda'):
#         """
#         核心动作：调用两次 cdl()，产生统计一致但瞬时独立的上下行信道
#         """
#         # 生成下行信道 (Ground Truth)
#         H_dl = self._generate_single_snapshot(batch_size)
        
#         # 生成上行信道 (Feedback Carrier)
#         H_ul = self._generate_single_snapshot(batch_size)
        
#         return H_dl.to(device), H_ul.to(device)


import torch
from sionna.phy.channel.tr38901 import CDL, AntennaArray
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel


class CDLChannelGenerator_Feedback:
    """
    Sionna 2.0 (Pure PyTorch) CSI-feedback channel generator.

    H_dl and H_ul are independently sampled from the same CDL profile.
    Output shape:
        H_dl, H_ul: [Batch, K, Nt, Nsc]

    Parameters
    ----------
    cdl_model:
        3GPP CDL profile, one of {"A", "B", "C", "D", "E"}.
        Default "C" preserves all legacy experiments.
    """

    def __init__(self, Nt=32, Nsc=256, carrier_freq=3.5e9, speed=1.0, num_users=4, cdl_model="C"):
        self.Nt = Nt
        self.Nsc = Nsc
        self.num_users = num_users

        cdl_model = str(cdl_model).upper()
        if cdl_model not in {"A", "B", "C", "D", "E"}:
            raise ValueError(
                f"Unsupported CDL model '{cdl_model}'. "
                "Expected one of {'A', 'B', 'C', 'D', 'E'}."
            )

        self.cdl_model = cdl_model

        # BS antenna array: 4 x 8 single-polarized antennas = 32 antennas.
        self.bs_array = AntennaArray(
            num_rows=4,
            num_cols=8,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=carrier_freq,
        )

        # Single-antenna UE.
        self.ue_array = AntennaArray(
            num_rows=1,
            num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=carrier_freq,
        )

        # Configurable CDL profile.
        # Default CDL-C exactly preserves the legacy setup.
        self.cdl = CDL(
            model=self.cdl_model,
            delay_spread=300e-9,
            carrier_frequency=carrier_freq,
            ut_array=self.ue_array,
            bs_array=self.bs_array,
            direction="downlink",
            min_speed=speed,
        )

        # 30-kHz subcarrier spacing.
        self.frequencies = subcarrier_frequencies(Nsc, 30e3)

        # Sampling frequency corresponding to the OFDM symbol duration.
        self.sampling_freq = 1.0 / ((1.0 + 0.07) / 30e3)

    def to(self, device):
        """Move channel-model components to the selected device."""
        self.cdl.to(device)
        self.frequencies = self.frequencies.to(device)
        return self

    def _generate_single_snapshot(self, batch_size):
        """Generate one independent multi-user channel snapshot."""
        a, tau = self.cdl(
            batch_size=batch_size * self.num_users,
            num_time_steps=1,
            sampling_frequency=self.sampling_freq,
        )

        h_freq = cir_to_ofdm_channel(
            self.frequencies,
            a,
            tau,
            normalize=True,
        )

        # [Batch*K, 1, 1, 1, Nt, 1, Nsc]
        # -> [Batch, K, Nt, Nsc]
        h = h_freq.squeeze(dim=(1, 2, 3, 5))
        h = h.reshape(
            batch_size,
            self.num_users,
            self.Nt,
            self.Nsc,
        )

        # Per-user average channel-power normalization.
        pwr = torch.mean(
            torch.abs(h).square(),
            dim=(-2, -1),
            keepdim=True,
        )

        return h / torch.sqrt(pwr + 1e-12)

    def generate_batch_data(self, batch_size, device="cuda"):
        """
        Generate statistically matched but instantaneously independent
        downlink and feedback-link channels.
        """
        H_dl = self._generate_single_snapshot(batch_size)
        H_ul = self._generate_single_snapshot(batch_size)

        return H_dl.to(device), H_ul.to(device)