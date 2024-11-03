# Copyright (c) 2024 Javad Komijani

import torch
import numpy as np


class SinosoidalTimeEmbedder(torch.nn.Module):
    """Following "Attension is all you need", this class yields a sinosoidal
    time (position) embedding. Unlike the mentioned papar, in which position is
    integer, we are interested in the case that time is not integer and in fact
    it is not larger than one; therefore, we use different settings, that can
    be controlled by changing `max_freq` and `base`.
    """
    
    def __init__(self,
                 embed_len: int,
                 max_freq: float = 10,
                 base: float = 1000,
                 rand_flag: bool = False
                ):
        super().__init__()

        assert embed_len % 2 == 0
        self.embed_len = embed_len
        
        if rand_flag:
            power = torch.rand(embed_len // 2)
        else:
            power = torch.linspace(0, 1, embed_len // 2)

        self.freq = max_freq / base**power
        
    def forward(self, time):
        phase = 2 * np.pi * time[:, None] * self.freq[None, :]
        time_embedded = torch.zeros(
            [len(time), self.embed_len], device=time.device
        )
        time_embedded[:, 0::2] = torch.sin(phase)
        time_embedded[:, 1::2] = torch.cos(phase)
        return time_embedded
