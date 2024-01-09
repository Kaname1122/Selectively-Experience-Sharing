import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical
from utils import init

# for VAE
import wandb
import logging
import tempfile
from cpprb import ReplayBuffer
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class FCNetwork(nn.Module):
    def __init__(self, dims, out_layer=None):
        """
        Creates a network using ReLUs between layers and no activation at the end
        :param dims: tuple in the form of (100, 100, ..., 5). for dim sizes
        """
        super().__init__()
        input_size = dims[0]
        h_sizes = dims[1:]

        mods = [nn.Linear(input_size, h_sizes[0])]
        for i in range(len(h_sizes) - 1):
            mods.append(nn.ReLU())
            mods.append(nn.Linear(h_sizes[i], h_sizes[i + 1]))

        if out_layer:
            mods.append(out_layer)

        self.layers = nn.Sequential(*mods)

    def forward(self, x):
        # Feedforward
        return self.layers(x)

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


class Policy(nn.Module):
    def __init__(self, obs_space, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()

        obs_shape = obs_space.shape

        if base_kwargs is None:
            base_kwargs = {}

        self.base = MLPBase(obs_shape[0], **base_kwargs)

        num_outputs = action_space.n
        self.dist = Categorical(self.base.output_size, num_outputs)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


# ------------------------------for clustering------------------------------
# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self, features, input_size, extra_decoder_input, reconstruct_size):
        super(LinearVAE, self).__init__()
        HIDDEN=64
        self.features = features
        # encoder
        self.gru = nn.GRU(input_size=input_size, hidden_size=HIDDEN, batch_first=True) # not used for now

        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=2*features)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=features + extra_decoder_input, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=reconstruct_size),
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def encode(self, x):
#         x, _ = self.gru(x)
        x = self.encoder(x)
        mu = x[:, :self.features]
        log_var = x[:, self.features:]
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        return mu

    def forward(self, x, xp):
        # encoding
#         x, _ = self.gru(x)
        x = self.encoder(x)

        mu = x[: , :self.features]
        log_var = x[:, self.features:]

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        dec_input = torch.cat([z, xp], axis=-1)
        # decoding
        reconstruction = self.decoder(dec_input)
        return reconstruction, mu, log_var


clustering_config = {
    "timestep": 100,

    "delay": 0,
    "pretraining_steps": 5000,
    "pretraining_times": 1,

    "batch_size": 128,
    "clusters": None,
    "lr": 3e-4,
    "epochs": 10,
    "z_features": 10,

    "kl_weight": 0.0001,
    "delay_training": False,

    "human_selected_idx": False,

    "encoder_in": ["agent"],
    "decoder_in": ["obs", "act"],
    "reconstruct": ["next_obs", "rew"]
}

# リプレイバッファからエンコーダ入力、デコーダ入力、再構築のためのデータを取得
class rbDataSet(Dataset):
    def __init__(
        self,
        rb,
        encoder_in=clustering_config["encoder_in"],
        decoder_in=clustering_config["decoder_in"],
        reconstruct=clustering_config["reconstruct"]
    ):
        self.rb = rb
        self.data = []

        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in encoder_in], dim=1))
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in decoder_in], dim=1))
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in reconstruct], dim=1))

        print([x.shape for x in self.data])

    def __len__(self):
        return self.data[0].shape[0]
    def __getitem__(self, idx):
        return [x[idx, :] for x in self.data]


# クラスタリングを実行するための処理
def compute_clusters(
    rb,
    agent_count,
    batch_size=clustering_config["batch_size"],
    clusters=clustering_config["clusters"],
    lr=clustering_config["lr"],
    epochs=clustering_config["epochs"],
    z_features=clustering_config["z_features"],
    kl_weight=clustering_config["kl_weight"],
):
    device = "cpu"

    dataset = rbDataSet(rb)

    input_size = dataset.data[0].shape[-1]
    extra_decoder_input = dataset.data[1].shape[-1]
    reconstruct_size = dataset.data[2].shape[-1]

    # VAEモデルのインスタンスを作成、表示
    model = LinearVAE(z_features, input_size, extra_decoder_input, reconstruct_size)
    print(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # criterion = nn.BCELoss(reduction='sum')
    criterion = nn.MSELoss(reduction="sum")

    # 損失関数の定義（reconstruction loss + KL-Divergence）
    def final_loss(bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + kl_weight*KLD

    def fit(model, dataloader):
        model.train()
        running_loss = 0.0
        for i, (encoder_in, decoder_in, y) in enumerate(dataloader):
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(encoder_in, decoder_in)
            bce_loss = criterion(reconstruction, y)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = running_loss/len(dataloader.dataset)
        return train_loss

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loss = []
    for epoch in tqdm(range(epochs)):
        train_epoch_loss = fit(model, dataloader)
        train_loss.append(train_epoch_loss)

    print(f"Train Loss: {train_epoch_loss:.6f}")
    x = torch.eye(agent_count)

    with torch.no_grad():
        z = model.encode(x)
    z = z.to("cpu")
    z = z[:, :]

    if clusters is None:
        clusters = find_optimal_cluster_number(z)
    logging.info(f"Creating {clusters} clusters.")
    # run k-means from scikit-learn
    kmeans = KMeans(
        n_clusters=clusters, init='k-means++',
        n_init=10
    )
    cluster_ids_x = kmeans.fit_predict(z) # predict labels
    if z_features == 2:
        plot_clusters(kmeans.cluster_centers_, z)
    return torch.from_numpy(cluster_ids_x).long()

# クラスタリングの結果を可視化するためのプロットを作成
def plot_clusters(cluster_centers, z, human_selected_idx=clustering_config["human_selected_idx"]):

    # 人が明示的にIDを設定していない場合
    if human_selected_idx is None:
        plt.plot(z[:, 0], z[:, 1], 'o')
        plt.plot(cluster_centers[:, 0], cluster_centers[:, 1], 'x')

        for i in range(z.shape[0]):
            plt.annotate(str(i), xy=(z[i, 0], z[i, 1]))

    else:
        colors = 'bgrcmykw'
        for i in range(len(human_selected_idx)):
            plt.plot(z[i, 0], z[i, 1], 'o' + colors[human_selected_idx[i]])

        plt.plot(cluster_centers[:, 0], cluster_centers[:, 1], 'x')
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile, format="png") # File position is at the end of the file.
        fig = plt.figure()
        wandb.log({"cluster_image": wandb.Image(fig)})
    # plt.savefig("cluster.png")

# 与えられたデータセットに対して最適なクラスタ数を見つけるための関数
def find_optimal_cluster_number(X):

    # 探索するクラスタ数の範囲
    range_n_clusters = list(range(2, X.shape[0]))
    scores = {}

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        scores[n_clusters] = davies_bouldin_score(X, cluster_labels)

    # スコアが最小となるクラスタ数を選択
    max_key = min(scores, key=lambda k: scores[k])
    return max_key
