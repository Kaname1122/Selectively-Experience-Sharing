

import glob
import os
import numpy as np
import torch
import torch.nn as nn

# NNの重みとバイアスを初期化する関数
# gainは初期化の際に乗算される係数
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# 指定したディレクトリ（log_dir）内のモニタリングファイルを削除
def cleanup_log_dir(log_dir):
    try:
        # ディレクトリが存在しない場合は作成
        os.makedirs(log_dir)

    except OSError:
        # 既に存在する場合は、モニタリングファイルを削除
        files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)
