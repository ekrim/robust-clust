"""robustclust is a package for constrained clustering 
under uncertainty.

"""
from .utils import plot_constraints

from .active_learning import get_constraints, \
                             FFQS, \
                             MMFFQS, \
                             active_class_discovery 

from .constrained import E2CP, \
                         SpectralLearning

from .robust import ImperfectOracles, \
                    ConstraintsToLabels

