# Este archivo permite importar el módulo como un paquete
from .diagonal_crack.processor import DiagonalCrackProcessor
from .midway_crack.processor import MidwayCrackProcessor
from .corner_crack.processor import CornerCrackProcessor
from .nucleo_esponjoso.processor import NucleoEsponjosoProcessor

__all__ = [
    'DiagonalCrackProcessor',
    'MidwayCrackProcessor',
    'CornerCrackProcessor',
    'NucleoEsponjosoProcessor'
]