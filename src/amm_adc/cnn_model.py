import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryAccuracy, AUROC


class ConvFeatureNet(nn.Module):
    def __init__(self, input_size, hidden_channels, kernel_size, num_layers, dropout_rate):
        super().__init__()
        layers = []
        c_in = 1
        for _ in range(num_layers):
            layers.append(nn.Conv1d(c_in, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.Dropout(dropout_rate))
            c_in = hidden_channels
        self.conv_net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.float().unsqueeze(1)
        x = self.conv_net(x)
        return x.mean(dim=2)


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        scores = self.attention(x.float())
        weights = torch.softmax(scores, dim=1)
        return x * weights


def boundary_weighted_bce_with_logits(logits, y_true, ic50_nM, boundary_nM=10.0,
                                      tau_log10=0.25, w_min=0.20, normalize=True):
    per_sample = F.binary_cross_entropy_with_logits(logits, y_true.float(), reduction='none')
    ic50_nM = torch.clamp(ic50_nM, min=1e-6, max=1e9)
    b = torch.as_tensor(boundary_nM, dtype=ic50_nM.dtype, device=ic50_nM.device)
    d = torch.abs(torch.log10(ic50_nM) - torch.log10(b))
    w = w_min + (1.0 - w_min) * (1.0 - torch.exp(-d / tau_log10))
    if normalize:
        w = w / (w.mean().detach() + 1e-8)
    return (w * per_sample).mean()


class ADCModel(pl.LightningModule):
    def __init__(self, feature_params, classifier_params, lr, boundary, optimizer_type='adam', loss='boundary_weighted_bce'):
        super().__init__()
        self.save_hyperparameters()
        self.boundary = boundary
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.loss_name = loss
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_true_labels = []
        self.test_pred_probs = []
        self.train_losses = []
        self.val_losses = []

        self.linear_group1_feature1 = nn.Linear(feature_params['group1_feature1_input_dim'], feature_params['group1_feature1_linear_dim'])
        self.linear_group1_feature3 = nn.Linear(feature_params['group1_feature3_input_dim'], feature_params['group1_feature3_linear_dim'])

        self.linear_group2_feature2 = nn.Linear(feature_params['group2_feature2_input_dim'], feature_params['group2_feature2_linear_dim'])
        self.linear_group2_feature4 = nn.Linear(feature_params['group2_feature4_input_dim'], feature_params['group2_feature4_linear_dim'])
        self.linear_group2_feature5 = nn.Linear(feature_params['group2_feature5_input_dim'], feature_params['group2_feature5_linear_dim'])
        self.linear_group2_feature9 = nn.Linear(feature_params['group2_feature9_input_dim'], feature_params['group2_feature9_linear_dim'])
        self.linear_group2_feature10 = nn.Linear(feature_params['group2_feature10_input_dim'], feature_params['group2_feature10_linear_dim'])

        self.linear_group3_feature6 = nn.Linear(feature_params['group3_feature6_input_dim'], feature_params['group3_feature6_linear_dim'])
        self.linear_group3_feature7 = nn.Linear(feature_params['group3_feature7_input_dim'], feature_params['group3_feature7_linear_dim'])

        g1_dim = feature_params['group1_feature1_linear_dim'] + feature_params['group1_feature3_linear_dim']
        self.convNet1 = ConvFeatureNet(g1_dim, feature_params['group1_hidden_channels'], feature_params['group1_kernel_size'], feature_params['group1_num_layers'], feature_params['group1_dropout'])

        g2_dim = (feature_params['group2_feature2_linear_dim'] + feature_params['group2_feature4_linear_dim'] +
                  feature_params['group2_feature5_linear_dim'] + feature_params['group2_feature9_linear_dim'] +
                  feature_params['group2_feature10_linear_dim'])
        self.convNet2 = ConvFeatureNet(g2_dim, feature_params['group2_hidden_channels'], feature_params['group2_kernel_size'], feature_params['group2_num_layers'], feature_params['group2_dropout'])

        g3_dim = feature_params['group3_feature6_linear_dim'] + feature_params['group3_feature7_linear_dim']
        self.convNet3 = ConvFeatureNet(g3_dim, feature_params['group3_hidden_channels'], feature_params['group3_kernel_size'], feature_params['group3_num_layers'], feature_params['group3_dropout'])

        hidden = feature_params['group1_hidden_channels'] + feature_params['group2_hidden_channels'] + feature_params['group3_hidden_channels']
        self.attention = Attention(hidden, hidden_dim=128)

        layers = []
        in_size = hidden
        for _ in range(classifier_params['num_layers']):
            layers.append(nn.Linear(in_size, classifier_params['hidden_dim']))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(classifier_params['hidden_dim']))
            layers.append(nn.Dropout(classifier_params['dropout']))
            in_size = classifier_params['hidden_dim']
        layers.append(nn.Linear(classifier_params['hidden_dim'], 1))
        self.classifier = nn.Sequential(*layers)

        self.accuracy = BinaryAccuracy()
        self.auroc = AUROC(task='binary')

    def _encode(self, batch):
        f1 = self.linear_group1_feature1(batch['feature_1'].float())
        f3 = self.linear_group1_feature3(batch['feature_3'].float())
        o1 = self.convNet1(torch.cat([f1, f3], dim=1))

        f2 = self.linear_group2_feature2(batch['feature_2'].float())
        f4 = self.linear_group2_feature4(batch['feature_4'].float())
        f5 = self.linear_group2_feature5(batch['feature_5'].float())
        f9 = self.linear_group2_feature9(batch['feature_9'].float())
        f10 = self.linear_group2_feature10(batch['feature_10'].float())
        o2 = self.convNet2(torch.cat([f2, f4, f5, f9, f10], dim=1))

        f6 = self.linear_group3_feature6(batch['feature_6'].float())
        f7 = self.linear_group3_feature7(batch['feature_7'].float())
        o3 = self.convNet3(torch.cat([f6, f7], dim=1))

        return self.attention(torch.cat([o1, o2, o3], dim=1))

    def forward(self, batch):
        h = self._encode(batch)
        logits = self.classifier(h).squeeze(1)
        return {'logits': logits, 'probs': torch.sigmoid(logits)}

    def _shared_step(self, batch):
        out = self(batch)
        loss = boundary_weighted_bce_with_logits(out['logits'], batch['flag'], batch['ic50'], boundary_nM=self.boundary)
        y = batch['flag'].float().int()
        acc = self.accuracy(out['probs'], y)
        auc = self.auroc(out['probs'], y)
        return loss, acc, auc, y.float(), out['probs']

    def training_step(self, batch, batch_idx):
        loss, acc, auc, _, _ = self._shared_step(batch)
        self.log('train_step_loss', loss, on_step=True, on_epoch=False, logger=False)
        self.train_step_outputs.append({'loss': loss.detach(), 'acc': acc.detach(), 'auc': auc.detach()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, auc, y, probs = self._shared_step(batch)
        self.val_step_outputs.append({'loss': loss.detach(), 'acc': acc.detach(), 'auc': auc.detach(), 'y': y.detach(), 'p': probs.detach()})
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc, auc, y, probs = self._shared_step(batch)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_auc', auc, prog_bar=True)
        self.test_true_labels.extend(y.detach().cpu().numpy())
        self.test_pred_probs.extend(probs.detach().cpu().numpy())
        return loss

    def on_train_epoch_end(self):
        if not self.train_step_outputs:
            return
        loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
        acc = torch.stack([x['acc'] for x in self.train_step_outputs]).mean()
        auc = torch.stack([x['auc'] for x in self.train_step_outputs]).mean()
        self.train_losses.append(float(loss.detach().cpu()))
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_auc', auc, prog_bar=True)
        self.train_step_outputs.clear()
        self.accuracy.reset()
        self.auroc.reset()

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return
        loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
        acc = torch.stack([x['acc'] for x in self.val_step_outputs]).mean()
        auc = torch.stack([x['auc'] for x in self.val_step_outputs]).mean()
        self.val_losses.append(float(loss.detach().cpu()))
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_auc', auc, prog_bar=True)
        self.val_step_outputs.clear()
        self.accuracy.reset()
        self.auroc.reset()

    def configure_optimizers(self):
        if self.optimizer_type == 'sgd':
            opt = SGD(self.parameters(), lr=self.lr, momentum=0.9)
        else:
            opt = Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=8, min_lr=1e-6)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'monitor': 'val_loss'}}
