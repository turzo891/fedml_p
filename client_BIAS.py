# Tracks per‑client bias statistics during local training.
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

class BiasTracker:
    """
    Collects statistics needed for BIAS evaluation.
    BIAS = |AUC_{group0} - AUC_{group1}|
    """

    def __init__(self, group_label: int = 0):
        """
        Parameters
        ----------
        group_label: int
            Which sensitive attribute value to treat as the *reference* group
            (usually 0). The other group is treated as “disadvantaged”.
        """
        self.group_label = group_label

    def compute(self, model, dataloader, device):
        """
        Computes group‑specific AUCs for a binary classifier.
        Returns
        -------
        bias: float
            Absolute difference between AUCs of the two groups.
        """
        model.eval()
        preds, trues, groups = [], [], []

        with torch.no_grad():
            for X, y, g in dataloader:
                X = X.to(device)
                y = y.to(device)
                g = g.to(device)

                logits = model(X).squeeze()
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
                trues.append(y.cpu().numpy())
                groups.append(g.cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        groups = np.concatenate(groups)

        # Group 0 (reference)
        idx0 = groups == self.group_label
        # Group 1 (disadvantaged)
        idx1 = groups != self.group_label

        def safe_auc(y_true, y_score):
            try:
                # Need at least one positive and one negative
                if len(np.unique(y_true)) < 2:
                    return 0.5
                return roc_auc_score(y_true, y_score)
            except Exception:
                return 0.5

        auc0 = safe_auc(trues[idx0], preds[idx0])
        auc1 = safe_auc(trues[idx1], preds[idx1])

        bias = abs(auc0 - auc1)
        return bias
