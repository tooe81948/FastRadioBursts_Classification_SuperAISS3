import torch
from torchmetrics import Metric


class PulseRecall(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("pulse_class_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        preds = torch.sigmoid(preds) > 0.5
        
        mask = target[:, 0].bool()  # Only consider instances where the first class is present in the ground truth
        masked_preds = preds[mask]
        masked_target = target[mask]

        true_positives = (masked_preds[:, 0].bool() & masked_target[:, 0].bool()).sum()
        pulse_class_total = masked_target[:, 0].sum().long()

        self.true_positives = self.true_positives + true_positives.item()
        self.pulse_class_total = self.pulse_class_total + pulse_class_total.item()

    def compute(self):
        pulse_class_recall = self.true_positives.float() / self.pulse_class_total.float()
        return pulse_class_recall


class FRBAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("pulse_class_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        preds = torch.sigmoid(preds) > 0.5
        correct = (preds == target).sum()
        total_elements = target.numel()
        true_positives = (preds[:, 0].bool() & target[:, 0].bool()).sum()
        pulse_class_total = target[:, 0].sum().long()

        self.correct = self.correct + correct.item()
        self.total = self.total + total_elements
        self.true_positives = self.true_positives + true_positives.item()
        self.pulse_class_total = self.pulse_class_total + pulse_class_total.item()

    def compute(self):
        all_class_accuracy = self.correct.float() / self.total.float()
        pulse_class_recall = self.true_positives.float() / self.pulse_class_total.float()
        return pulse_class_recall * all_class_accuracy