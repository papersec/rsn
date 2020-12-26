
from io import BytesIO
from tempfile import NamedTemporaryFile

import numpy as np
import torch
import skvideo.io
import skvideo
skvideo.setFFmpegPath("/usr/bin/")

from rsn.util import ARHS
from rsn.util.make_1darray import make_1darray
from rsn.hyper_parameter import N_STEP_BOOTSTRAPING

class Experience:
    """
    Saves sequence of ARHS using compress method
    """

    def __init__(self, burnin, sequence, bootstrap):
        """
        burnin, sequence, bootstrap - ndarray(ARHS)
        """
        self.seq_length = {"burnin":len(burnin), "sequence":len(sequence), "bootstrap":len(bootstrap)}

        a, r, h, s = zip(*(np.concatenate((burnin, sequence, bootstrap))))

        a = np.array(a, dtype=np.int8)
        r = np.array(r, dtype=np.float32)
        h, c = zip(*h)
        h = h[0].numpy()
        c = c[0].numpy()

        s = np.stack([obs.numpy() for obs in s])
        self.stack = s.shape[1] # Frame stack 개수 (=4)
        
        # Compress
        self.a = BytesIO()
        np.savez_compressed(self.a, a)
        self.r = BytesIO()
        np.savez_compressed(self.r, r)
        self.h = BytesIO()
        np.savez_compressed(self.h, h)
        self.c = BytesIO()
        np.savez_compressed(self.c, c)

        # Encode s to video using H.264
        fp = NamedTemporaryFile(suffix=".mp4")
        writer = skvideo.io.FFmpegWriter(fp.name, outputdict={'-vcodec': 'libx264', '-crf': '0'})
        for i in range(len(s)):
            writer.writeFrame(s[i,3])
        writer.close()
        fp.seek(0)

        self.s = BytesIO()
        self.s.write(fp.read())
        fp.close()


    def decompress(self):
        data = (self.a, self.r, self.h, self.c)
        for e in data:
            e.seek(0)
        
        a, r, h, c = (np.load(e)['arr_0'] for e in data)

        l_burnin, l_sequence, l_bootstrap = (self.seq_length[s] for s in self.seq_length.keys())

        s_burnin = slice(l_burnin)
        s_sequence = slice(l_burnin, l_burnin+l_sequence)
        s_bootsrap = slice(l_burnin+l_sequence, None)

        h = [(torch.from_numpy(h).float(), torch.from_numpy(c).float())] + [None] * (l_burnin+l_sequence+l_bootstrap-1)

        # Read from Video
        fp = NamedTemporaryFile(suffix=".mp4")
        self.s.seek(0)
        fp.write(self.s.read())
        fp.seek(0)

        s = skvideo.io.vread(fp.name, as_grey=True)
        fp.close()
        s = s.reshape(s.shape[:3]) # Drop color channel

        s_stack = np.empty((s.shape[0], self.stack, *s.shape[1:]), dtype=np.float32)

        for i in range(self.stack):
            s_stack[:i,self.stack-1-i] = s[0]
            s_stack[i:,self.stack-1-i] = s[:len(s)-i]
        
        s_stack = torch.from_numpy(s_stack).float()

        arhs_arr = make_1darray([ARHS(*arhs) for arhs in zip(a, r, h, s_stack)])

        burnin = arhs_arr[s_burnin]
        sequence = arhs_arr[s_sequence]
        bootstrap = arhs_arr[s_bootsrap]

        return burnin, sequence, bootstrap
