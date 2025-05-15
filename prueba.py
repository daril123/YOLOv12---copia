import cv2
import numpy as np
import os
import pandas as pd

class InclusionNoMetalicaProcessor:
    """
    Procesador para inclusiones no metálicas con análisis dentro de un área fija de 500x500 píxeles
    """
    
    def __init__(self):
        """
        Inicializa el procesador de inclusiones no metálicas
        """
        self.name = "inclusion_no_metalica"
        self.square_size = 500  # Tamaño fijo del cuadrado de análisis (500x500 píxeles)
        

    def process(self):
        print(self.square_size)