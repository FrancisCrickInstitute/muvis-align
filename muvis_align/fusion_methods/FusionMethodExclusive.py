import numpy as np

from muvis_align.fusion_methods.FusionMethod import FusionMethod


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
            mask = ~np.isnan(transformed_views)
            # weigh smallest (specialized) shape highest
            if np.count_nonzero(mask[0]) < np.count_nonzero(mask[1]):
                weights[0][mask[0]] = True
                weights[1][~mask[0]] = True
            else:
                weights[0][~mask[1]] = True
                weights[1][mask[1]] = True
            transformed_views *= weights

        return np.nansum(transformed_views, axis=0).astype(transformed_views[0].dtype)
