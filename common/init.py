# Este archivo permite importar el módulo como un paquete
from .zone_generator import generate_zone_masks, visualize_zones
from .detector import analizar_direccion_grieta_mask, analizar_direccion_grieta
from .defect_classifier import classify_defects_with_masks, visualize_results_with_masks

__all__ = [
    'generate_zone_masks', 
    'visualize_zones',
    'analizar_direccion_grieta_mask',
    'analizar_direccion_grieta',
    'classify_defects_with_masks',
    'visualize_results_with_masks'
]