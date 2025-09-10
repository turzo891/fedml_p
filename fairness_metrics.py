import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

def compute_bias(truth, scores, sensitive):
    """
    Return |AUC0 - AUC1|
    """
    g0 = sensitive == 0
    g1 = sensitive != 0
    auc0 = roc_auc_score(truth[g0], scores[g0])
    auc1 = roc_auc_score(truth[g1], scores[g1])
    return abs(auc0 - auc1)

def balanced_accuracy(truth, preds):
    return balanced_accuracy_score(truth, preds)

# Example usage
if __name__ == "__main__":
    # dummy data
    y_true = np.random.randint(0, 2, 200)
    y_scores = np.random.rand(200)
    sensitive = np.random.randint(0, 2, 200)

    bias = compute_bias(y_true, y_scores, sensitive)
    print("Bias:", bias)

# >  **Integrate into simulation**
# > You can extend `FedServer`’s `aggregate_fit` to compute these metrics locally for each client by temporarily
# loading the client’s model and running it on its held‑out validation set.
# > Alternatively, let each client send its validation AUCs along with the model weights; the server can aggregate
# them and log a per‑client fairness report.
