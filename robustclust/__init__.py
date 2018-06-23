"""robustclust is a package for constrained clustering 
under uncertainty.

"""
from .active_learning import FFQS, \
                             MMFFQS, \
                             ActiveClassDiscovery

from .constrained import E2CP, \
                         SpectralLearning

from .robust import ImperfectOracles, \
                    ConstraintsToLabels

from .utils import plot_constraints, \
                   plot_labels 
