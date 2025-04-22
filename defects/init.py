# Este archivo permite importar el módulo como un paquete
from .diagonal_crack.processor import DiagonalCrackProcessor
from .midway_crack.processor import MidwayCrackProcessor
from .corner_crack.processor import CornerCrackProcessor
from .nucleo_esponjoso.processor import NucleoEsponjosoProcessor
from .inclusion_no_metalica.processor import InclusionNoMetalicaProcessor
from .rechupe.processor import RechupeProcessor
from .estrella.processor import EstrellaProcessor
from .sopladura.processor import SopladuraProcessor
from .romboidad.processor import RomboidadProcessor
from .abombamiento.processor import AbombamientoProcessor

__all__ = [
    'DiagonalCrackProcessor',
    'MidwayCrackProcessor',
    'CornerCrackProcessor',
    'NucleoEsponjosoProcessor',
    'InclusionNoMetalicaProcessor',
    'RechupeProcessor',
    'EstrellaProcessor',
    'SopladuraProcessor',
    'RomboidadProcessor',
    'AbombamientoProcessor'
]