import numpy as np

from fusion_methods.FusionMethod import FusionMethod


class FusionMethodExclusive(FusionMethod):
    def fusion(self, transformed_views):
        """
        Simple exclusive fusion.

        Parameters
        ----------
        transformed_views : list of ndarrays
            transformed input views

        Returns
        -------
        ndarray
            Fusion of input views
        """
        if len(transformed_views) > 1:
            weights = np.zeros(transformed_views.shape, dtype=bool)
            mask0 = transformed_views[0] > 0
            mask1 = transformed_views[1] > 0
            # weigh smallest (specialized) shape highest
            if np.count_nonzero(mask0) < np.count_nonzero(mask1):
                weights[0][mask0] = True
                weights[1][~mask0] = True
            else:
                weights[0][~mask1] = True
                weights[1][mask1] = True
            transformed_views *= weights

        return np.nansum(transformed_views, axis=0).astype(transformed_views[0].dtype)
