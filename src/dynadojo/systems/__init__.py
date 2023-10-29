from . import opinion
from . import lv
from . import epidemic
from . import fbsnn_pde

from .lds import LDSystem
from .ctln import CTLNSystem
from .ca import CASystem
from .snn import SNNSystem
from .lorenz import LorenzSystem
from .kuramoto import KuramotoSystem
from .heat import HeatEquation
''' 
must run first: pip install dynadojo[nbody] 
NBody uses packages only supported on Linux/Mac OSX
'''
from .santi import NBodySystem


ALL_SYSTEMS = [
    NBodySystem, CTLNSystem, CASystem, SNNSystem, LorenzSystem, KuramotoSystem, HeatEquation,
    opinion.MediaBiasSystem, opinion.DeffuantSystem, opinion.WHKSystem, opinion.ARWHKSystem, opinion.HKSystem,
    lv.PreyPredatorSystem, lv.CompetitiveLVSystem,
    epidemic.SIRSystem, epidemic.SISSystem, epidemic.SEISSystem,
    fbsnn_pde.BSBSystem, fbsnn_pde.HJBSystem
]