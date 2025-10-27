r"基于产生的 histogram.py , 计算 probe 的 R(r, \theta, t) 与 \lambda(r, \theta) 函数"

import h5py as h5
import numpy as np
from coefficient import ProbeBase

# probe函数积分上限(ns)
T_MAX = 1000


class Probe(ProbeBase):
    "Probe 类的一个实现, 使用 histogram.h5 数据计算 probe 函数"

    probe : h5.Dataset = None

    def load_data(self):
        "读取 histogram.h5 中的数据"
        if self.probe is None:
            with h5.File('./histogram.h5', 'r') as h5file_r:
                probe_data = h5file_r["Probe"]
                # r格子数目
                self.rbins = probe_data.attrs.get('R_Bins')
                # theta格子数目 = self.thetabins
                self.thetabins = probe_data.attrs.get('Theta_Bins')
                # t格子数目
                self.tbins = probe_data.attrs.get('T_Bins')
                self.probe = probe_data[:]

    def get_mu(self, rs, thetas):
        self.load_data()

        # 计算 lambda 函数(数组形式)
        lambda_func = np.sum(self.probe, axis=2) * (T_MAX / self.tbins)

        # 计算 (r, theta) 所在网格
        r_grid = np.clip(np.floor(rs * self.rbins), 0, self.rbins - 1).astype(int)
        theta_grid = np.clip(np.floor(thetas / np.pi * self.thetabins), 0, self.thetabins-1).astype(int)

        # 直接读取沿 t 轴的和作为 mu 值
        mu = lambda_func[r_grid, theta_grid]
        return mu

    def get_lc(self, rs, thetas, ts):
        self.load_data()

        # 计算 (r, theta, t) 所在网格
        r_grid = np.clip(np.floor(rs * self.rbins), 0, self.rbins - 1).astype(int)
        theta_grid = np.clip(np.floor(thetas / np.pi * self.thetabins), 0, self.thetabins-1).astype(int)
        t_grid = np.clip(np.floor(ts / T_MAX * self.tbins), 0, self.tbins - 1).astype(int)

        # 直接读取网格值作为 lc 值
        lc = self.probe[r_grid, theta_grid, t_grid]
        return lc
