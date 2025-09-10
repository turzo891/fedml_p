# A lightweight wrapper around SHAP or LIME to generate local explanations.For brevity we use SHAP’s *GradientExplainer*.
import shap
import torch

class LocalExplainer:
    """
    Generates feature importance for a single local sample.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        # SHAP gradient explainer for a differentiable model
        self.explainer = shap.GradientExplainer(
            self.model,
            torch.zeros(1, *model.input_shape).to(device)  # dummy background
        )

    def explain(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor   (shape: [1, ...])
            Input sample

        Returns
        -------
        shap_values: np.ndarray
            Feature importance scores
        """
        x = x.to(self.device)
        shap_vals = self.explainer.shap_values(x.unsqueeze(0))[0][0]
        return shap_vals.cpu().numpy()
