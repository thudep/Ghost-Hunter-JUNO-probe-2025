"基于数据文件与几何文件, 生成直方图数据文件 histogram.h5"

import argparse
from multiprocessing import Pool, cpu_count
import h5py as h5
import numpy as np

# PMT数量
PMT_NUM = 17612
# 液闪区域半径(mm)
R0 = 17700
# 数据文件个数与编号
N0 = 1
Seq = np.arange(16001, 16001+N0)
# probe函数积分上限(ns)
T_MAX = 1000
# 0
EPSILON = 1e-6
# 最大进程数
PROCESS_MAX = cpu_count() // 4

# 统计函数
def count(arguments):
    data_file, bins, geo = arguments
    i, data = data_file
    rbins, thetabins, tbins = bins
    geo_theta, geo_phi = geo

    # 数据集读取
    with h5.File(f"{data}/{i}.h5", 'r') as h5file_r:
        particle_truth = h5file_r["ParticleTruth"][...]
        pe_truth = h5file_r["PETruth"][...]

    # 把顶点坐标归一化
    vertex = (np.stack((particle_truth['x'], particle_truth['y'], particle_truth['z']), axis=-1) / R0)[..., None]
    # 数据量统计
    vertices_num = vertex.shape[0]

    # 得到所有顶点相对于所有 PMT 的 (r, theta) 坐标
    vertices_r = np.linalg.norm(vertex, axis=-2) * np.ones((vertices_num, PMT_NUM))
    vertices_theta = np.arccos(np.clip((vertex[..., 0, :] * np.sin(geo_theta) * np.cos(geo_phi) +
                                        vertex[..., 1, :] * np.sin(geo_theta) * np.sin(geo_phi) +
                                        vertex[..., 2, :] * np.cos(geo_theta)) / vertices_r, -1.0, 1.0))

    # 整理所有事件与顶点相对于所有 PMT 的 (r, theta (,t)) 坐标
    pe_r = vertices_r[pe_truth["EventID"], pe_truth["ChannelID"]]
    pe_theta = vertices_theta[pe_truth["EventID"], pe_truth["ChannelID"]]
    pe_t = np.array(pe_truth["PETime"])
    vertices_r = vertices_r.flatten()
    vertices_theta = vertices_theta.flatten()

    # 找到每个事件与顶点的网格位置
    pe_r_bin = np.clip(np.floor(pe_r * rbins), 0, rbins - 1).astype(int)
    pe_theta_bin = np.clip(np.floor(pe_theta / np.pi * thetabins), 0, thetabins - 1).astype(int)
    pe_t_bin = np.clip(np.floor(pe_t / T_MAX * tbins), 0, tbins - 1).astype(int)
    vertices_r_bin = np.clip(np.floor(vertices_r * rbins), 0, rbins - 1).astype(int)
    vertices_theta_bin = np.clip(np.floor(vertices_theta / np.pi * thetabins), 0, thetabins - 1).astype(int)

    # 统计每个 (r,θ,t) 的事件数目 pe_num 与每个 (r,θ) 的顶点数目 vertex_num
    pe_bin = (pe_r_bin * thetabins + pe_theta_bin) * tbins + pe_t_bin
    vertices_bin = vertices_r_bin * thetabins + vertices_theta_bin
    pe_num = np.bincount(pe_bin, minlength=rbins * thetabins * tbins).reshape(rbins, thetabins, tbins)
    vertex_num = np.bincount(vertices_bin, minlength=rbins * thetabins).reshape(rbins, thetabins)

    return pe_num, vertex_num

def main():
    # 设置 args 参数
    psr = argparse.ArgumentParser()
    psr.add_argument("-g", "--geo", dest="geo", type=str, help="geometry file")
    psr.add_argument("--data", dest="data", type=str, help="training data")
    psr.add_argument("-o", "--output", dest="opt", type=str, help="output file")
    psr.add_argument("-r", "--rbins", dest="rbins", type=str, help="output file")
    psr.add_argument("-theta", "--thetabins", dest="thetabins", type=str, help="output file")
    psr.add_argument("-t", "--tbins", dest="tbins", type=str, help="output file")
    args = psr.parse_args()

    rbins = int(args.rbins)
    thetabins = int(args.thetabins)
    tbins = int(args.tbins)

    # 读取探测器数据
    with h5.File(args.geo, 'r') as h5file_r:
        geo = h5file_r["Geometry"][...]
    geo_theta = np.deg2rad(geo["theta"][None, :PMT_NUM])
    geo_phi = np.deg2rad(geo["phi"][None, :PMT_NUM])

    # 使用 PROCESS 个进程分别处理 N0 个数据集
    params = [((i, args.data), (rbins, thetabins, tbins), (geo_theta, geo_phi)) for i in Seq]
    with Pool(processes=min(PROCESS_MAX, N0)) as pool:
        results = pool.map(count, params)

    # 拆出处理结果 pe_num 与 vertex_num
    pe_num = np.sum([r[0] for r in results], axis=0)
    vertex_num = (np.sum([r[1] for r in results], axis=0))[:,:,None]

    # 计算Probe函数
    with np.errstate(invalid='ignore'): #忽略掉 / 0 的警告, 实际上由于 np.where 这一错误不会发生
        # 把顶点数为 0 的位置的 probe 记为 0 (实际上这是一种不正确的近似)
        probe = np.where(vertex_num > 0, pe_num / vertex_num, 0) + EPSILON

    with h5.File(args.opt, 'w') as h5file_w:
        dataset = h5file_w.create_dataset('Probe', data=probe)
        dataset.attrs['R_Bins'] = rbins
        dataset.attrs['Theta_Bins'] = thetabins
        dataset.attrs['T_Bins'] = tbins

if __name__ == "__main__":
    main()
