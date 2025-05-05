import os
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image
from utils.utils import order_points
from utils.contorno import obtener_contorno_imagen

class VertexDetector:
    def __init__(self, model_path=None):
        """
        Inicializa el detector de vértices
        
        Args:
            model_path: Ruta al modelo YOLO entrenado para la detección de vértices
        """
        # Ruta al modelo YOLO
        self.model_path = model_path if model_path else r"D:\Trabajo modelos\PACC\YOLOv12 - copia\Models\modelo_1.pt"
        
        # Obtener el nombre base del modelo (sin extensión) para buscar el YAML con el mismo nombre
        model_base_name = os.path.splitext(os.path.basename(self.model_path))[0]
        model_dir = os.path.dirname(self.model_path)
        yaml_path = os.path.join(model_dir, f"{model_base_name}.yaml")
        
        print(f"Buscando archivo YAML en: {yaml_path}")
        
        self.class_names = []
        self.target_class = "palanquilla"  # Clase por defecto
        
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
                        
                        # CORRECCIÓN: Buscar específicamente la clase "palanquilla"
                        palanquilla_idx = None
                        for idx, name in enumerate(self.class_names):
                            if name.lower() == "palanquilla":
                                palanquilla_idx = idx
                                break
                        
                        if palanquilla_idx is not None:
                            self.target_class = self.class_names[palanquilla_idx]
                            print(f"Clase objetivo 'palanquilla' encontrada en índice {palanquilla_idx}")
                        else:
                            # Si no se encuentra "palanquilla", usar la primera clase
                            self.target_class = self.class_names[0]
                            print(f"Clase 'palanquilla' no encontrada, usando clase por defecto: {self.target_class}")
            except Exception as e:
                print(f"Error al cargar archivo YAML: {e}")
        else:
            print(f"Archivo YAML no encontrado: {yaml_path}")
        
        # Configuración de CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        
        # Umbrales para detección (reducidos para mejorar la detección)
        self.conf_threshold = 0.35
        self.padding = 20  # Margen para la imagen
        self.color = (255, 255, 255)  # Color blanco
        
        # Cargar el modelo YOLO
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"Modelo YOLO para detección de vértices cargado desde: {self.model_path}")
            print(f"Clase objetivo configurada como: {self.target_class}")  # Diagnóstico adicional
            
            # Si no se pudieron cargar las clases desde YAML, usarlas del modelo
            if not self.class_names and hasattr(self.model, "names"):
                # Convertir el diccionario de nombres a una lista
                if isinstance(self.model.names, dict):
                    self.class_names = [self.model.names[i] for i in sorted(self.model.names.keys())]
                else:
                    self.class_names = self.model.names
                print(f"Clases cargadas desde modelo: {self.class_names}")
                
                # CORRECCIÓN: Buscar específicamente la clase "palanquilla"
                palanquilla_idx = None
                for idx, name in enumerate(self.class_names):
                    if name.lower() == "palanquilla":
                        palanquilla_idx = idx
                        break
                
                if palanquilla_idx is not None:
                    self.target_class = self.class_names[palanquilla_idx]
                    print(f"Clase objetivo 'palanquilla' encontrada en índice {palanquilla_idx}")
                else:
                    # Si no se encuentra "palanquilla", usar la primera clase
                    self.target_class = self.class_names[0]
                    print(f"Clase 'palanquilla' no encontrada, usando clase por defecto: {self.target_class}")
        except Exception as e:
            print(f"Error al cargar el modelo YOLO para vértices: {e}")
            self.model = None
    
    def detect_vertices(self, image_path):
        """
        Detecta los vértices de la palanquilla en la imagen
        
        Args:
            image_path: Ruta a la imagen
        
        Returns:
            vertices: Array con las coordenadas de los 4 vértices ordenados
            success: True si se detectó correctamente, False en caso contrario
            mask: Máscara de segmentación de la palanquilla
        """
        if self.model is None:
            print("Error: Modelo YOLO de vértices no cargado")
            return None, False, None
        
        try:
            # Cargar la imagen
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error al cargar la imagen: {image_path}")
                return None, False, None
            
            # Usar la función mejorada para obtener el contorno, pasando la clase objetivo
            vertices, contorno_principal, mask_img = obtener_contorno_imagen(
                image, self.model, self.conf_threshold, target_class=self.target_class)
            
            if vertices is None or len(vertices) != 4:
                print("No se pudo obtener un contorno válido. Intentando método alternativo...")
                return self.detect_vertices_alternative(image)
            
            # Verificar que las coordenadas de los vértices son razonables
            h, w = image.shape[:2]
            for vertex in vertices:
                if vertex[0] < 0 or vertex[0] >= w or vertex[1] < 0 or vertex[1] >= h:
                    print(f"Advertencia: Vértice fuera de los límites de la imagen: {vertex}")
                    return self.detect_vertices_alternative(image)
            
            return vertices, True, mask_img
            
        except Exception as e:
            print(f"Error en detect_vertices: {e}")
            import traceback
            traceback.print_exc()
            
            # Intentar un método alternativo en caso de error
            return self.detect_vertices_alternative(image_path)
    
    def detect_vertices_alternative(self, image):
        """
        Método alternativo para detectar vértices cuando el principal falla
        
        Args:
            image: Imagen o ruta a la imagen
        
        Returns:
            vertices: Array con las coordenadas de los 4 vértices ordenados
            success: True si se detectó correctamente, False en caso contrario
            mask: Máscara de segmentación de la palanquilla
        """
        try:
            # Si se proporciona una ruta en lugar de una imagen, cargar la imagen
            if isinstance(image, str):
                image = cv2.imread(image)
                if image is None:
                    print(f"Error: No se pudo cargar la imagen alternativa.")
                    return None, False, None
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Método 1: Umbral adaptativo
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Método 2: Umbral de Otsu
            _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Combinar ambos métodos
            thresh_combined = cv2.bitwise_or(thresh, thresh_otsu)
            
            # Operaciones morfológicas para limpiar la imagen
            kernel = np.ones((5, 5), np.uint8)
            thresh_cleaned = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel)
            thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_OPEN, kernel)
            
            # Método 3: Detección de bordes con Canny
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Intentar encontrar contornos en cada resultado
            methods = [
                ("Umbral combinado", thresh_cleaned),
                ("Canny", edges)
            ]
            
            for method_name, method_result in methods:
                try:
                    contornos, _ = cv2.findContours(method_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if not contornos:
                        print(f"No se encontraron contornos con método: {method_name}")
                        continue
                    
                    # Filtrar contornos pequeños (ruido)
                    contornos = [c for c in contornos if cv2.contourArea(c) > 1000]
                    
                    if not contornos:
                        print(f"No hay contornos significativos con método: {method_name}")
                        continue
                    
                    # Seleccionar el contorno principal (el más grande)
                    contorno_principal = max(contornos, key=cv2.contourArea)
                    
                    # Probar varios valores de epsilon para la aproximación poligonal
                    epsilon_factors = [0.02, 0.01, 0.015, 0.025, 0.03, 0.035, 0.04, 0.05]
                    
                    for eps in epsilon_factors:
                        perimetro = cv2.arcLength(contorno_principal, True)
                        epsilon = eps * perimetro
                        approx = cv2.approxPolyDP(contorno_principal, epsilon, True)
                        
                        if len(approx) == 4:
                            print(f"Método alternativo exitoso ({method_name}) con epsilon={eps}")
                            vertices = approx.reshape(-1, 2)
                            vertices = order_points(vertices)
                            
                            # Verificar que las coordenadas son razonables
                            valid = True
                            h, w = image.shape[:2]
                            for vertex in vertices:
                                if vertex[0] < 0 or vertex[0] >= w or vertex[1] < 0 or vertex[1] >= h:
                                    valid = False
                                    break
                            
                            if valid:
                                return vertices, True, method_result
                    
                    # Si no se encontró una aproximación de 4 vértices, usar rectángulo mínimo
                    print(f"Usando rectángulo mínimo con método: {method_name}")
                    rect = cv2.minAreaRect(contorno_principal)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    vertices = order_points(box)
                    
                    return vertices, True, method_result
                    
                except Exception as method_error:
                    print(f"Error con método {method_name}: {method_error}")
                    continue
            
            # Si todo falla, crear un rectángulo con los bordes de la imagen
            print("Todos los métodos alternativos fallaron. Usando bordes de la imagen.")
            h, w = image.shape[:2]
            vertices = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.int32)
            
            # Crear una máscara con toda la imagen
            mask = np.ones((h, w), dtype=np.uint8) * 255
            
            return vertices, False, mask
            
        except Exception as e:
            print(f"Error en método alternativo: {e}")
            import traceback
            traceback.print_exc()
            
            # Si todo falla, usar los bordes de la imagen
            h, w = image.shape[:2]
            vertices = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.int32)
            
            # Crear una máscara con toda la imagen
            mask = np.ones((h, w), dtype=np.uint8) * 255
            
            return vertices, False, mask
    
    def visualize_vertices(self, image_path, vertices, output_path=None):
        """
        Visualiza los vértices detectados en la imagen
        
        Args:
            image_path: Ruta a la imagen original
            vertices: Array con las coordenadas de los vértices
            output_path: Ruta donde guardar la imagen con vértices visualizados
        
        Returns:
            True si se visualizó correctamente, False en caso contrario
        """
        try:
            # Cargar la imagen
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error al cargar la imagen: {image_path}")
                return False
            
            # Crear una copia para dibujar
            imagen_con_vertices = image.copy()
            
            # Dibujar el polígono formado por los vértices
            cv2.polylines(imagen_con_vertices, [vertices], True, (0, 255, 0), 3)
            
            # Dibujar y numerar los vértices
            for i, vertice in enumerate(vertices):
                cv2.circle(imagen_con_vertices, tuple(vertice), 10, (0, 0, 255), -1)
                cv2.putText(imagen_con_vertices, str(i+1), tuple(vertice), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Añadir texto informativo
            cv2.putText(imagen_con_vertices, "Vértices detectados", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Si se especificó una ruta de salida, guardar la imagen
            if output_path:
                # Asegurar que el directorio exista
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, imagen_con_vertices)
                print(f"Imagen con vértices guardada en: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Error en visualize_vertices: {e}")
            return False
            
    def preprocess_image(self, image_path):
        """
        Preprocesa la imagen para mejorar la detección de vértices
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Imagen preprocesada
        """
        try:
            # Cargar la imagen
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error al cargar la imagen: {image_path}")
                return None
            
            # Ajustar el contraste y brillo
            alpha = 1.2  # Factor de contraste
            beta = 10    # Factor de brillo
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            # Aplicar filtro de reducción de ruido
            denoised = cv2.fastNlMeansDenoisingColored(adjusted, None, 10, 10, 7, 21)
            
            # Añadir borde blanco para ayudar en la detección
            h, w = denoised.shape[:2]
            border_size = 20
            with_border = cv2.copyMakeBorder(denoised, border_size, border_size, border_size, border_size,
                                          cv2.BORDER_CONSTANT, value=(255, 255, 255))
            
            return with_border
            
        except Exception as e:
            print(f"Error en preprocess_image: {e}")
            # Devolver la imagen original en caso de error
            return cv2.imread(image_path)