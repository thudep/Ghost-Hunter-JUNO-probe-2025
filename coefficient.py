"ConcatInfo 类与 ProbeBase 类的定义"
# 此文件不可更改

from abc import ABCMeta, abstractmethod
import time
import h5py as h5
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.integrate import quad

PMT_NUM = 17612  # PMT数量
T_MAX = 1000  # probe函数积分上限(ns)

NV = 1000  # 一致性检测取点数
PROCESS_MAX = cpu_count() // 4  # 最大进程数

class ConcatInfo:
    "存储 concat.h5 的数据集"

    def __init__(self, filename: str):
        """
        读取 H5 文件, 载入 PE 事件数据及对应的顶点信息

        Parameters
        ----------
        filename : str
            concat 文件的文件名
        """
        with h5.File(filename, "r", swmr=True) as file:
            concat = file["Concat"][()]
            self.pe_rs = concat["r"]
            self.pe_thetas = concat["theta"]
            self.pe_ts = concat["t"]

            vertices = file["Vertices"][()]
            self.v_rs = vertices["r"]
            self.v_thetas = vertices["theta"]
            self.vertices_num = vertices.shape[0] // PMT_NUM


class ProbeBase(metaclass=ABCMeta):
    "各种类型的 Probe 类的基类, 不应更改"

    @abstractmethod
    def get_mu(self, rs, thetas) -> np.ndarray:
        """
        计算不含时的 probe 函数 \\lambda(r, \\theta)

        Parameters
        ----------
        rs, thetas : numpy.ndarray
            顶点坐标, r 范围为 [0, 1], theta 范围为 [0, pi]

        Returns
        -------
         : numpy.ndarray
            与 rs 形状相同
        """
        raise NotImplementedError

    @abstractmethod
    def get_lc(self, rs, thetas, ts) -> np.ndarray:
        """
        计算含时的 probe 函数 R(r, \\theta, t)

        Parameters
        ----------
        rs, thetas, ts : numpy.ndarray
            事件坐标(含时坐标), r 范围为 [0, 1], theta 范围为 [0, pi]

        Returns
        -------
         : numpy.ndarray
            与 rs 形状相同
        """
        raise NotImplementedError

    def _integrate_point(self, r, theta):
        "数值积分计算不含时的 probe 函数"
        def lc(t):
            return self.get_lc(np.array([r]), np.array([theta]), np.array([t]))[0]

        integral, _ = quad(lc, 0, T_MAX, limit=10000, epsabs=1e-6, epsrel=1e-4)
        return integral

    def is_consistent(self) -> bool:
        "含时与不含时的 probe 函数的一致性检测"
        np.random.seed(time.time_ns() % 65536)
        rs = np.random.rand(NV)
        thetas = np.random.rand(NV) * np.pi

        marginal = self.get_mu(rs, thetas)

        with Pool(processes=min(PROCESS_MAX, NV)) as pool:
            integral = pool.starmap(
                self._integrate_point, zip(rs, thetas)
            )

        # 较为严格的一致性检测
        if np.allclose(marginal, integral, rtol=5e-4, atol=1e-5):
            return True
        # 较为宽松的一致性检测
        elif np.allclose(marginal, integral, rtol=1e-2, atol=1e-3):
            print("\033[33m[Warning] Your get_mu() function implementation may have some issues, "\
                  "but it is basically consistent with get_lc(), so the grading will continue.\033[0m")
            return True
        # 未通过一致性检测
        else:
            print("\033[31m[Error] Your get_mu() function is not consistent with get_lc().\033[0m")
            return False

    def validate(self, v_rs, v_thetas, pe_rs, pe_thetas, pe_ts):
        """
        计算该 Probe 的似然函数的对数值

        Parameters
        ----------
        v_rs, v_thetas : numpy.ndarray
            顶点坐标, r 范围为 [0, 1], theta 范围为 [0, pi]

        pe_rs, pe_thetas, pe_ts : numpy.ndarray
            事件坐标(含时坐标), r 范围为 [0, 1], theta 范围为 [0, pi]

        Returns
        -------
         : float
            似然函数的对数值
        """
        assert self.is_consistent()

        mu = self.get_mu(v_rs, v_thetas)
        if mu.shape != v_rs.shape:
            raise ValueError("Arrays v_rs and mu have different shapes.")
        nonhit = np.sum(mu)

        lc = self.get_lc(pe_rs, pe_thetas, pe_ts)
        if lc.shape != pe_rs.shape:
            raise ValueError("Arrays pe_rs and lc have different shapes.")
        hit = np.sum(np.log(lc))

        return hit - nonhit
