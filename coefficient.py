"ConcatInfo 类与 ProbeBase 类的定义"
# 此文件不可更改

from abc import ABCMeta, abstractmethod
from multiprocessing import Pool, cpu_count
import time
import h5py as h5
import numpy as np
from scipy.integrate import quad, simpson, trapezoid

PMT_NUM = 17612  # PMT数量
T_MAX = 1000  # probe函数积分上限(ns)

NV = 1000  # 检测取点数
NT = 100000  # 积分时间取点数
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

        integral, _ = quad(lc, 0, T_MAX, limit=NT, epsabs=1e-6, epsrel=1e-4)
        return integral

    def is_consistent(self) -> bool:
        "含时与不含时的 probe 函数的一致性检测"
        np.random.seed(time.time_ns() % 65536)
        rs = np.random.rand(NV)
        thetas = np.random.rand(NV) * np.pi

        marginal = self.get_mu(rs, thetas)

        # 较为严格的一致性检测
        atol = 1e-6
        rtol = 1e-3
        # 使用 quad 积分
        with Pool(processes=min(PROCESS_MAX, NV)) as pool:
            integral = np.array(pool.starmap(
                self._integrate_point, zip(rs, thetas))
            )
        error_quad = marginal - integral
        error_limit_quad = atol + rtol * integral
        if np.sum(error_quad < error_limit_quad) >= int(0.99 * NV):
            return True
        # 使用 simpson 积分
        ts = np.linspace(0, T_MAX, NT+1)
        lc = self.get_lc(rs[:,None], thetas[:,None], ts[None,:])
        integral = simpson(lc, ts, axis=1)
        error_simpson = marginal - integral
        error_limit_simpson = atol + rtol * integral
        if np.sum(error_simpson < error_limit_simpson) >= int(0.99 * NV):
            return True
        # 使用 trapz 积分
        integral = trapezoid(lc, ts, axis=1)
        error_trapezoid = marginal - integral
        error_limit_trapezoid = atol + rtol * integral
        if np.sum(error_trapezoid < error_limit_trapezoid) >= int(0.99 * NV):
            return True

        # 较为宽松的一致性检测
        def warning():
            print("\033[33m[Warning] Your get_mu() function implementation may have some issues, "\
                  "but it is basically consistent with get_lc(), so the grading will continue.\033[0m")
        if np.sum(error_quad < error_limit_quad) >= int(0.9 * NV):
            warning()
            return True
        if np.sum(error_simpson < error_limit_simpson) >= int(0.9 * NV):
            warning()
            return True
        if np.sum(error_trapezoid < error_limit_trapezoid) >= int(0.9 * NV):
            warning()
            return True
        if np.sum(error_quad < 10 * error_limit_quad) >= int(0.99 * NV):
            warning()
            return True
        if np.sum(error_simpson < 10 * error_limit_simpson) >= int(0.99 * NV):
            warning()
            return True
        if np.sum(error_trapezoid < 10 * error_limit_trapezoid) >= int(0.99 * NV):
            warning()
            return True

        # 未通过一致性检测
        print("\033[31m[Error] Your get_mu() function is not consistent with get_lc().\033[0m")
        return False

    def is_nondelta(self, pe_rs, pe_thetas, pe_ts) -> bool:
        "无 delta 峰检测"
        np.random.seed(time.time_ns() % 65536)
        index = np.random.choice(len(pe_rs), size=NV)
        rs = pe_rs[index]
        thetas = pe_thetas[index]
        ts = pe_ts[index]

        mu = self.get_mu(rs, thetas)
        lc = self.get_lc(rs, thetas, ts)

        if np.all(lc < 0.1 * mu):
            return True
        elif np.all(lc < mu):
            print("\033[33m[Warning] Your get_lc() function has some significantly larger values, "\
                  "but they are still within an acceptable range, so the grading will continue.\033[0m")
            return True
        else:
            print("\033[31m[Error] Your get_lc() function has some unreasonable extremely large values.\033[0m")
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
        assert self.is_nondelta(pe_rs, pe_thetas, pe_ts)

        mu = self.get_mu(v_rs, v_thetas)
        if mu.shape != v_rs.shape:
            raise ValueError("Arrays v_rs and mu have different shapes.")
        nonhit = np.sum(mu)

        lc = self.get_lc(pe_rs, pe_thetas, pe_ts)
        if lc.shape != pe_rs.shape:
            raise ValueError("Arrays pe_rs and lc have different shapes.")
        hit = np.sum(np.log(lc))

        return hit - nonhit
