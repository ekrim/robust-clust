"""robustclust is a package for constrained clustering 
under uncertainty.

"""
from .utils import plot_constraints, \
                   plot_labels 

from .active_learning import get_constraints, \
                             FFQS, \
                             MMFFQS, \
                             ActiveClassDiscovery

from .constrained import E2CP, \
                         SpectralLearning

from .robust import ImperfectOracles, \
                    ConstraintsToLabels

