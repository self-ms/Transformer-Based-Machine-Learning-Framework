import torch.nn as nn

class TransformerModel(nn.Module):

    def __init__(self, d_model, nhead, num_enc, d_feed, dropout, activation, num_cls):
        super().__init__()
        self.encoder = nn.Transformer(d_model = d_model,
                                      nhead = nhead,
                                      num_encoder_layers = num_enc,
                                      num_decoder_layers=0,
                                      dim_feedforward = d_feed,
                                      dropout = dropout,
                                      activation = activation,
                                      batch_first = True).encoder

        self.fc_cls = nn.LazyLinear(num_cls)
        self.input_extend = nn.LazyLinear(d_model)
        self.bn0 = nn.LazyBatchNorm1d()


    def forward(self, x):
        x = self.bn0(self.input_extend(x)).sigmoid()
        y = self.encoder(x).sigmoid()
        # y = y.mean(dim=1)
        # y = y[:,-1,:]
        # y = self.fc_cls(y[:,-1])
        y = self.fc_cls(y)
        return y


    def get_num_params(self, k = 1e6):
        nums = sum(p.numel() for p in self.parameters())/k
        return nums
