import os
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

class DefectDetector:
    """
    Detector de defectos utilizando YOLO que mantiene todas las categorías originales 
    sin hacer mapeo (a diferencia del PalanquillaYoloProcessor original)
    """
    
    def __init__(self, model_path=None):
        """
        Inicializa el detector de defectos con YOLO
        
        Args:
            model_path: Ruta al modelo YOLO entrenado
        """
        # Configuración de CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        
        # Ruta al modelo YOLO
        self.model_path = model_path if model_path else r"D:\Trabajo modelos\PACC\YOLOv12 - copia\Models\modelo_1.pt"
        
        # Obtener el nombre base del modelo (sin extensión) para buscar el YAML con el mismo nombre
        model_base_name = os.path.splitext(os.path.basename(self.model_path))[0]
        model_dir = os.path.dirname(self.model_path)
        yaml_path = os.path.join(model_dir, f"{model_base_name}.yaml")
        
        print(f"Buscando archivo YAML en: {yaml_path}")
        
        self.class_names = []
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)
                    if "names" in yaml_data:
                        # Manejar tanto si names es una lista como si es un diccionario
                        if isinstance(yaml_data["names"], list):
                            self.class_names = yaml_data["names"]
                        else:
                            self.class_names = [yaml_data["names"][i] for i in sorted(yaml_data["names"].keys())]
                        print(f"Clases cargadas desde {yaml_path}: {self.class_names}")
            except Exception as e:
                print(f"Error al cargar archivo YAML: {e}")
        else:
            print(f"Archivo YAML no encontrado: {yaml_path}")
        
        # Umbrales para detección
        self.conf_threshold = 0.20
        self.iou_threshold = 0.60
        
        # Cargar el modelo YOLO
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"Modelo YOLO cargado desde: {self.model_path}")
            
            # Si no se pudieron cargar las clases desde YAML, usarlas del modelo
            if not self.class_names and hasattr(self.model, "names"):
                # Convertir el diccionario de nombres a una lista
                if isinstance(self.model.names, dict):
                    self.class_names = [self.model.names[i] for i in sorted(self.model.names.keys())]
                else:
                    self.class_names = self.model.names
                print(f"Clases cargadas desde modelo: {self.class_names}")
        except Exception as e:
            print(f"Error al cargar el modelo YOLO: {e}")
            self.model = None
    
    def detect_defects(self, image):
        """
        Detecta defectos en la imagen usando el modelo YOLO preservando todas las categorías originales
        
        Args:
            image: Imagen original o ruta a la imagen
        
        Returns:
            detections: Diccionario con detecciones por clase
            result: Resultado original de YOLO
        """
        if self.model is None:
            print("Error: Modelo YOLO no cargado")
            return {}, None
        
        # Si se proporciona una ruta en lugar de una imagen, cargar la imagen
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                print(f"Error: No se pudo cargar la imagen.")
                return {}, None
        
        # Realizar inferencia con YOLO
        results = self.model.predict(
            image, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold, 
            device=self.device,
            retina_masks=True  # Asegurar que devuelva máscaras de segmentación
        )
        
        # Inicializar diccionario de detecciones
        detections = {}
        
        # Procesar resultados
        if len(results) > 0:
            result = results[0]
            
            # Obtener cajas y clases
            boxes = result.boxes
            
            # Guardar las máscaras si están disponibles
            masks = None
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks
            
            # Asegurarse de que estamos usando los nombres de clase correctos
            # Si no tenemos class_names, usar los del resultado
            if not self.class_names and hasattr(result, "names"):
                if isinstance(result.names, dict):
                    self.class_names = [result.names[i] for i in sorted(result.names.keys())]
                else:
                    self.class_names = result.names
            
            for i, box in enumerate(boxes):
                # Obtener coordenadas
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Obtener clase y confianza
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # Obtener nombre de la clase
                if 0 <= cls_id < len(self.class_names):
                    cls_name = self.class_names[cls_id]
                else:
                    cls_name = f"clase_{cls_id}"  # Nombre genérico si no lo encontramos
                
                # Añadir a detecciones - SIN MAPEO para preservar todas las categorías
                if cls_name not in detections:
                    detections[cls_name] = []
                
                detection_info = {
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'class': cls_name,
                    'cls_id': cls_id,
                    'index': i
                }
                
                # Agregar máscara si está disponible
                if masks is not None and i < len(masks):
                    detection_info['mask_index'] = i
                
                detections[cls_name].append(detection_info)
            
            print(f"Detectados: {', '.join([f'{k}: {len(v)}' for k, v in detections.items()])}")
            
            return detections, result
        
        return {}, None