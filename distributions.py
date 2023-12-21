import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init

"""
Modify standard PyTorch distributions so they are compatible with this code.
pytorchの標準的な分布を変更、このコードと互換性があるようにする。
"""

# 修正したカテゴリカル分布
# Categorical
class FixedCategorical(torch.distributions.Categorical):
    # 分布からサンプリングされた値を返す
    def sample(self):
        # unsqueeze(-1)でテンソルの最後に次元を一つ追加する
        return super().sample().unsqueeze(-1)

    # 各actionの対数尤度を計算、2次元tensorとして返す
    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))  # squeeze(-1)で最後にサイズが1の次元があれば削除
            .view(actions.size(0), -1)  # tensorの要素数を変えずに形状を変更、各actionが1行に対応するように自動変更
            .sum(-1)    # 各行（各action）の合計
            .unsqueeze(-1)  # 次元の追加
        )

    # 確率が最大となるactionのインデックスを取得
    # 最頻値
    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# カテゴリカル分布を表すNNモデル
class Categorical(nn.Module):
    # 初期化、ネットワークの重みを初期化する
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)
