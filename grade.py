"对 probe 函数进行评分"
# 此文件不可更改

import argparse
import numpy as np
from coefficient import ConcatInfo, ProbeBase
from probe import Probe

def validate(probe: ProbeBase, c: ConcatInfo) -> float:
    "计算似然函数对数值"
    return probe.validate(c.v_rs, c.v_thetas, c.pe_rs, c.pe_thetas, c.pe_ts)

def get_score(lln: float) -> float:
    "根据似然函数的对数的归一化结果, 进行评分"
    if lln < -17000:
        return 10 * np.exp(lln + 17000)
    elif lln < -9000:
        return (lln + 17000) / 100 + 10
    elif lln < -8500:
        return (lln + 9000) / 500 * 9 + 90
    else:
        return 100 - np.exp(- (lln + 8500))


def main():
    psr = argparse.ArgumentParser()
    psr.add_argument("--concat", dest="concat", type=str, help="concat file")
    psr.add_argument("-o", "--output", dest="opt", type=str, help="output file")
    args = psr.parse_args()

    concat = ConcatInfo(args.concat)
    probe = Probe()
    log_l = validate(probe, concat)
    print(f"the logarithm of the likelihood function: \033[32m{log_l}\033[0m")
    log_l_normalization = log_l / concat.vertices_num
    print(f"normalization of the logarithm of the likelihood function: \033[32m{log_l_normalization}\033[0m")
    score = get_score(log_l_normalization)
    print(f"score: \033[32m{score}\033[0m")
    print(f"upload score: {score}")

if __name__ == "__main__":
    main()
