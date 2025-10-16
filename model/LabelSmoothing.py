import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):
    def __init__(self, num_classes=1000, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.k = num_classes

    def forward(self, target, pred):
        """
        pred (FloatTensor): [batch_size,n_classes]
        target (LongTensor): [batch_size]
        Ex - for batch_size = 2
        target = tensor([[1], [2]])
        pred = tensor([[0.0200, 0.0200, 0.0200, 0.0200, 0.0200],
                      [0.0200, 0.0200, 0.0200, 0.0200, 0.0200]])
        output:
        tensor([[0.0200, 0.9200, 0.0200, 0.0200, 0.0200],
                [0.0200, 0.0200, 0.9200, 0.0200, 0.0200]])
        """
        batch_size = target.shape[0]
        confidence = torch.as_tensor(batch_size * [(1.0 - self.smoothing)]).unsqueeze(1)
        q = torch.zeros_like(pred).fill_((self.smoothing / self.k)).scatter_(dim=1, index=target.unsqueeze(1),
                                                                             src=confidence, reduce='add')

        return q