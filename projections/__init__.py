
class BaseProjector(object):
    """Base class for all projections.
    """
    def __init__(self, maxIter, minIter, stopping_measure='kl', **kwargs):
        self._kwargs = kwargs
        self.maxIter = maxIter
        self.minIter = minIter
        self.sm = stopping_measure

    def project(self, P_0, r, c, eps, u, v):
        """Project the matrix P_0 onto U(r, c) with error epsilon given projection params initialized to u, v."""
        raise NotImplementedError

    def normalize(self, P, u, v):
        """ Cheap reduction in the dual objective, but may slow down convergence in some cases. Use with caution."""
        P_norm = P.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        P /= P_norm
        u -= P_norm.log().squeeze()

        return P, u, v
