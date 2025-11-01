# 2025 Ghost Hunter 的赛题: JUNO probe

## 赛题简介

**probe 函数**用来描述探测器对顶点的响应, 我们通常将含时的 probe 函数记为 $R(r,\theta,t)$, 该函数表示在相对于 PMT 坐标 $(r, \theta)$ 下随时间 $t$ 变化的 PMT 上接收到的期望 PE 数. 将时间维度积分后, 得到不含时间的 probe 函数 $\lambda(r, \theta)$. 即: 这两个函数之间的关系是: $\lambda(r, \theta)$ = $\int_{0}^{T} R(r, \theta, t) \mathrm{d}t$, 这里 $T$ 是一个足够大的时间窗口, 我们取为 1000ns.

我们已经有了足够好的模拟数据作为训练集, 你需要用这些训练集, 得到一个 probe 函数.

## 数据集介绍

注: 所有数据集文件均无需手动下载, 使用 [`Makefile`](./Makefile) 运行训练或评分时会自动下载.

### 训练集

训练集文件名如 `16001.h5` , 下载后位于 `data/` 文件夹下. 每个 `h5` 文件包含 **`ParticleTruth`** 以及 **`PETruth`** 两个 **dataset**, 其格式为:

**`ParticleTruth`**

包含 10000 个顶点.

| 名称      | 说明               |
| --------- | ----------------- |
| `EventID` | 事件编号           |
| `x`       | 顶点坐标 $x$ / mm  |
| `y`       | 顶点坐标 $y$ / mm  |
| `z`       | 顶点坐标 $z$ / mm  |
| `Ek`      | 顶点动能 / MeV     |
| `Evis`    | 顶点可见能量 / MeV |

注: `Ek` 与 `Evis` 数据在该比赛中无需被用到.

**`PETruth`**

| 名称        | 说明                 |
| ----------- | ------------------- |
| `EventID`   | 事件编号             |
| `PETime`    | PE 击中时间 / ns     |
| `LightTime` | 光子被发出的时间 / ns |
| `ChannelID` | PMT 编号             |

注: 两个文件中的 `PETruth` 是对应的; `LightTime` 数据在该比赛中无需被用到.

### 几何文件

几何文件名为 `geo.h5` , 其格式为:

**`Geometry`**

| 名称        | 说明                       |
| ----------- | ------------------------- |
| `ChannelID` | PMT 编号                  |
| `theta`     | 球坐标 $\theta$ (**角度**) |
| `phi`       | 球坐标 $\phi$ (**角度**)   |

### 测试集

测试集文件名为 `concat.h5` , 包含 **`Concat`** 以及 **`Vertices`** 两个 **dataset**, 其格式为:

**`Concat`**

所有探测到的事件的数据.

| 名称    | 说明                              |
| ------- | -------------------------------- |
| `r`     | 相对 PMT 坐标 $r$ (已归一化)      |
| `theta` | 相对 PMT 坐标 $\theta$ (**弧度**) |
| `t`     | PE 击中时间 / ns                  |

**`Vertices`**

所有相对于 PMT 的顶点数据.

| 名称    | 说明                              |
| ------- | -------------------------------- |
| `r`     | 相对 PMT 坐标 $r$ (已归一化)      |
| `theta` | 相对 PMT 坐标 $\theta$ (**弧度**) |

注: 你可以不用关心 `concat.h5` 文件, 它的处理已经写好, 且原则上你不能更改. (当然, 你可以使用 `concat.h5` 来训练, 不过由于我们最终使用隐藏测例进行测试, 所以使用 `concat.h5` 训练与使用提供的训练集训练并没有实质区别.)

## 任务说明

任务是修改 [`probe.py`](./probe.py) 中的 `get_mu()` 与 `get_lc()` 函数, 构造出含时与不含时的 probe 函数, 尽量使分数更高. 当前的 [`probe.py`](./probe.py) 是依赖于 [`histogram.py`](./histogram.py) 生成的直方图 [`histogram.h5`](./histogram.h5) 来构造 probe 函数, 如果你采取不同的方案, 也可以删除 [`histogram.py`](./histogram.py) .

请**不要更改** [`coefficient.py`](./coefficient.py) 与 [`grade.py`](./grade.py) , 如果发现更改则被视为作弊, 成绩无效. 其它文件均可修改, 也可以添加额外的文件, 但请确保 [`Makefile`](./Makefile) 中的 `make score` 可以正常运行; 如果你发现你引用的包在我们的环境中没有提供, 你也可以与组委会联系, 征得同意后可以修改 [ci](./.gitea/workflows/score.yaml) 以下载包.

我们将提供一个测试集供大家在本地评分作为参考. 但是, **为了防止大家对着测试集过拟合, 本地评测分数及CI排行仅供参考, 最终的排名将由隐藏测例来决定!**

## 附: 评分说明

给定顶点和 PMT 的相对坐标 $(r,\theta)$, 我们可以得到一个 PE 时间序列 $\vec{z} = (t_1,t_2,...t_k)$, 这里 $k$ 是序列长度. 这也是一个非齐次泊松过程, 其似然函数为:

$$
P(\vec{z} | r, \theta) = \mathrm{e}^{- \lambda(r, \theta) \mathrm{d}t} \prod_{j=1}^{k} R(r, \theta, t_j)
$$

如果我们有 $N$ 次采样, 那么似然函数就是:

$$
\mathcal{L} = \prod_{i=1}^{N} P(\vec{z_i} | r_i, \theta_i) = \prod_{i=1}^{N} \mathrm{e}^{- \lambda(r_i, \theta_i) \mathrm{d}t} \prod_{j=1}^{k_i} R(r_i, \theta_i, t_{ij})
$$

取对数得到:

$$
\log \mathcal{L} = \sum_{i=1}^{N} \left[- \lambda(r_i, \theta_i) + \sum_{j=1}^{k_i} \log R(r_i, \theta_i, t_{ij})\right]
$$

评分时, 会将你的 probe 函数 $R(r,\theta,t)$ 与 $\lambda(r, \theta)$ 输入到 $\log \mathcal{L}$ 中, 得到的值越大越好.

为了便于评分, 首先把似然函数的对数值除以评分集顶点数进行归一化 ( 这里的顶点数是在原来的空间中的顶点数, 而不是相对于 PMT 的顶点数, 后者是前者的 `PMT_NUM = 17612` 倍 ), 然后使用如下函数把分数映射到 $0 \sim 100$ :

$$
\text{score}(x) = \begin{cases}
10 \ \exp(x + 17000) \ , & x < -17000 \\
(x + 17000) / 100 + 10 \ , & -17000 \le x < -9000 \\
(x + 9000) / 500 \times 9 + 90 \ , & -9000 \le x < -8500 \\
100 - \exp[- (x + 8500)] \ , & x \ge -8500
\end{cases}
$$
