import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image
from utils.utils import order_points

class VertexDetector:
    def __init__(self, model_path=None):
        """
        Inicializa el detector de vértices
        
        Args:
            model_path: Ruta al modelo YOLO entrenado para la detección de vértices
        """
        # Ruta al modelo YOLO
        self.model_path = model_path if model_path else r"D:\Trabajo modelos\PACC\YOLOv12 - copia\vertices\best.pt"
        
        # Configuración de CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        
        # Umbrales para detección
        self.conf_threshold = 0.7
        self.padding = 20  # Margen para la imagen
        self.color = (255, 255, 255)  # Color blanco
        
        # Cargar el modelo YOLO
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"Modelo YOLO para detección de vértices cargado desde: {self.model_path}")
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
            
            # Se utiliza copyMakeBorder para ampliar la imagen con un borde con color
            image_padded = cv2.copyMakeBorder(image, self.padding, self.padding, self.padding, self.padding, 
                                             cv2.BORDER_CONSTANT, value=self.color)
            
            # Prediciendo
            resultado = self.model.predict(image_padded, conf=self.conf_threshold, device=self.device)[0]
            
            # Verificar si se detectaron máscaras
            if resultado.masks is None:
                print("No se detectaron máscaras en la imagen.")
                return None, False, None
            
            # Obtener las máscaras binarias
            mask_img = resultado.masks.data.cpu().numpy()
            mask_img = np.moveaxis(mask_img, 0, -1)  # (H, W, N)
            # Escalar las máscaras al tamaño de la imagen original
            mask_img = scale_image(mask_img, image_padded.shape)
            mask_img = np.moveaxis(mask_img, -1, 0)  # (N, H, W)
            # Convertir la máscara en un array de NumPy
            mask_img = mask_img.astype(np.uint8).reshape(mask_img.shape[1], mask_img.shape[2])
            mask_img = mask_img * 255
            
            # Limpiar la máscara (eliminar ruido y pequeñas regiones)
            # 1. Encontrar contornos en la máscara original
            contornos, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contornos:
                print("No se encontraron contornos en la máscara.")
                return None, False, None
            
            # Seleccionar el contorno principal
            contorno_principal = max(contornos, key=cv2.contourArea)
            
            # 2. Crear una máscara solo con el contorno principal (elimina pequeñas regiones desconectadas)
            mask_contorno = np.zeros_like(mask_img)
            cv2.drawContours(mask_contorno, [contorno_principal], 0, 255, -1)
            
            # Opcional: Aplicar un suavizado suave para mejorar los bordes
            kernel_suave = np.ones((3, 3), np.uint8)
            mask_final = cv2.morphologyEx(mask_contorno, cv2.MORPH_CLOSE, kernel_suave)
            
            # Detectar vértices
            # Encontrar contornos en la máscara final
            contornos_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contornos_final:
                print("No se encontraron contornos en la máscara final.")
                return None, False, None
            
            # Seleccionar el contorno principal
            contorno_principal = max(contornos_final, key=cv2.contourArea)
            
            # Aproximar el contorno para encontrar los vértices
            perimetro = cv2.arcLength(contorno_principal, True)
            epsilon = 0.02 * perimetro  # Ajustar según sea necesario
            approx = cv2.approxPolyDP(contorno_principal, epsilon, True)
            
            # Intentar obtener exactamente 4 vértices
            if len(approx) != 4:
                print(f"La aproximación produjo {len(approx)} vértices, ajustando para obtener 4...")
                
                # Probar diferentes valores de epsilon
                epsilons = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
                
                for eps in epsilons:
                    test_epsilon = eps * perimetro
                    test_approx = cv2.approxPolyDP(contorno_principal, test_epsilon, True)
                    print(f"Epsilon {eps}: {len(test_approx)} vértices")
                    
                    if len(test_approx) == 4:
                        approx = test_approx
                        print(f"Se encontró una buena aproximación con epsilon = {eps}")
                        break
                
                # Si aún no tenemos 4 vértices, usar el rectángulo mínimo
                if len(approx) != 4:
                    rect = cv2.minAreaRect(contorno_principal)
                    box = cv2.boxPoints(rect)
                    approx = np.int0(box).reshape(-1, 1, 2)
                    print("Se usó minAreaRect para obtener 4 vértices")
            
            # Extraer los vértices
            vertices = approx.reshape(-1, 2)
            
            # Ordenar los vértices [top-left, top-right, bottom-right, bottom-left]
            vertices = order_points(vertices)
            
            # Ajustar las coordenadas para compensar el padding
            vertices = vertices - self.padding
            
            # Recortar la máscara final para eliminar el padding y ajustarla a la imagen original
            mask_final_original = mask_final[self.padding:-self.padding, self.padding:-self.padding]
            
            return vertices, True, mask_final_original
            
        except Exception as e:
            print(f"Error en detect_vertices: {e}")
            import traceback
            traceback.print_exc()
            return None, False, None
    
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