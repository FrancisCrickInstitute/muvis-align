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
        needs_cleanup = False
        if len(transformed_views) > 1:
            needs_cleanup = True
            weights = np.zeros(transformed_views.shape, dtype=bool)
            mask = ~np.isnan(transformed_views)
            rem_mask = np.ones(transformed_views.shape[1:], dtype=bool)
            n = np.count_nonzero(mask, axis=(1,2))
            indices = np.argsort(n)
            # prioritise smallest shapes
            for index in indices:
                mask1 = mask[index]
                weights[index][mask1 * rem_mask] = True
                rem_mask *= ~mask1
            transformed_views *= weights

        fused = np.nansum(transformed_views, axis=0).astype(transformed_views[0].dtype)
        if needs_cleanup:
            del weights
            del mask
            del rem_mask
        return fused
