import cv2
import numpy as np
import os
import pandas as pd

from utils.utils import draw_arrow, find_extreme_points

class EstrellaProcessor:
    """
    Procesador para estrellas
    """
    
    def __init__(self):
        """
        Inicializa el procesador de estrellas
        """
        self.name = "estrella"
    
    def measure_estrella(self, estrella_mask, corners=None, estrella_img=None, bbox=None, original_image=None):
        """
        Mide el diámetro de una estrella en píxeles
        
        Args:
            estrella_mask: Máscara binaria de la estrella (ROI recortado)
            corners: Esquinas de la palanquilla [top-left, top-right, bottom-right, bottom-left] en coordenadas globales
            estrella_img: Imagen recortada de la estrella (opcional, para visualizaciones)
            bbox: Bounding box de la estrella en coordenadas globales (x1, y1, x2, y2)
            original_image: Imagen original completa
                
        Returns:
            metrics: Diccionario con la métrica diámetro en píxeles
        """
        # Encontrar contornos de la estrella
        contours, _ = cv2.findContours(estrella_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'diametro': 0
            }
        
        # Obtener el contorno principal (el más grande)
        contour = max(contours, key=cv2.contourArea)
        
        # CALCULAR DIÁMETRO: distancia máxima entre dos puntos del contorno
        local_pt1, local_pt2, diametro = find_extreme_points(contour)
        
        # Para visualización, necesitamos las coordenadas globales
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            global_pt1 = (local_pt1[0] + x1, local_pt1[1] + y1)
            global_pt2 = (local_pt2[0] + x1, local_pt2[1] + y1)
            
            # Convertir el contorno a coordenadas globales
            global_contour = contour.copy()
            global_contour[:, :, 0] += x1
            global_contour[:, :, 1] += y1
        else:
            global_pt1 = local_pt1
            global_pt2 = local_pt2
            global_contour = contour
        
        # Visualización si se proporciona imagen
        if estrella_img is not None:
            viz_img = estrella_img.copy() if len(estrella_img.shape) == 3 else cv2.cvtColor(estrella_img, cv2.COLOR_GRAY2BGR)
            
            # Dibujar el contorno de la estrella en el ROI
            cv2.drawContours(viz_img, [contour], -1, (0, 255, 0), 2)
            
            # Dibujar los puntos extremos (para el diámetro)
            if local_pt1 is not None and local_pt2 is not None:
                cv2.circle(viz_img, local_pt1, 5, (255, 0, 0), -1)
                cv2.circle(viz_img, local_pt2, 5, (255, 0, 0), -1)
                
                # Dibujar flecha para el diámetro
                draw_arrow(viz_img, local_pt1, local_pt2, (0, 255, 255), 2, 10, f"D={diametro:.1f}px", (5, -10))
            
            # Si tenemos suficiente información para crear una visualización global
            full_img = None
            
            if corners is not None and bbox is not None and original_image is not None:
                try:
                    # Usar directamente la imagen original
                    full_img = original_image.copy()
                    
                    # Dibujar el contorno de la palanquilla
                    cv2.polylines(full_img, [np.array(corners)], True, (0, 255, 0), 2)
                    
                    # Dibujar los vértices numerados
                    for i, corner in enumerate(corners):
                        cv2.circle(full_img, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), -1)
                        cv2.putText(full_img, str(i+1), (int(corner[0])-4, int(corner[1])+4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Dibujar la posición de la estrella
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(full_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    
                    # Dibujar el contorno de la estrella
                    cv2.drawContours(full_img, [global_contour], -1, (0, 255, 0), 2)
                    
                    # Dibujar los puntos extremos de la estrella en la vista global
                    cv2.circle(full_img, (int(global_pt1[0]), int(global_pt1[1])), 5, (255, 0, 0), -1)
                    cv2.circle(full_img, (int(global_pt2[0]), int(global_pt2[1])), 5, (255, 0, 0), -1)
                    
                    # Dibujar la flecha para el diámetro
                    draw_arrow(full_img, 
                            (int(global_pt1[0]), int(global_pt1[1])), 
                            (int(global_pt2[0]), int(global_pt2[1])), 
                            (0, 255, 255), 2, 15, f"D={diametro:.1f}px")
                    
                except Exception as e:
                    print(f"Error al crear visualización global para estrella: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Retornar información adicional para el visualizador de resultados
            return {
                'diametro': round(diametro, 2),
                'local_visualization': viz_img,
                'global_visualization': full_img,
                'local_pt1': local_pt1,
                'local_pt2': local_pt2,
                'global_pt1': global_pt1,
                'global_pt2': global_pt2
            }
        
        return {
            'diametro': round(diametro, 2)
        }
    
    def generate_report(self, image_name, estrellas_data, output_dir):
        """
        Genera un informe simple de las estrellas
        
        Args:
            image_name: Nombre de la imagen original
            estrellas_data: Lista de diccionarios con los datos de las estrellas
            output_dir: Directorio donde guardar el informe
        
        Returns:
            report_path: Ruta al archivo de informe generado
        """
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear un DataFrame con los datos, excluyendo objetos complejos
        df = pd.DataFrame([{k: v for k, v in estrella.items() if k not in ['visualization', 'rect', 'box', 'local_visualization', 
                                                                        'global_visualization', 'local_pt1', 'local_pt2', 
                                                                        'global_pt1', 'global_pt2']} 
                           for estrella in estrellas_data])
        
        # Formato del informe
        report_path = os.path.join(output_dir, f"{image_name}_estrella_report.csv")
        
        # Guardar como CSV
        df.to_csv(report_path, index=False)
        
        # También generar una versión en formato de texto para fácil visualización
        text_report_path = os.path.join(output_dir, f"{image_name}_estrella_report.txt")
        
        with open(text_report_path, 'w') as f:
            f.write(f"REPORTE DE ESTRELLAS - {image_name}\n")
            f.write("="*50 + "\n\n")
            
            for i, estrella in enumerate(estrellas_data):
                f.write(f"ESTRELLA #{i+1}\n")
                f.write(f"  Diámetro: {estrella['diametro']} píxeles\n")
                f.write(f"  Confianza: {estrella['conf']:.2f}\n\n")
        
        print(f"Reporte generado en: {report_path}")
        print(f"Reporte de texto generado en: {text_report_path}")
        
        return report_path, text_report_path
    
    def process(self, detections, image, corners, zone_masks, image_name=None, output_dir=None):
        """
        Procesa todas las estrellas detectadas
        
        Args:
            detections: Lista de detecciones de estrellas
            image: Imagen original
            corners: Esquinas de la palanquilla
            zone_masks: Máscaras de zonas
                
        Returns:
            processed_data: Diccionario con los resultados del procesamiento
        """
        results = []
        visualizations = {}
        
        # Procesar cada detección
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            conf = detection.get('conf', 0)
            
            # Extraer la máscara
            mask = detection.get('mask', None)
            if mask is None:
                # Si no hay máscara, crear una a partir de la ROI
                roi = image[y1:y2, x1:x2].copy()
                if len(roi.shape) == 3:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    roi_gray = roi
                
                # Binarizar para obtener la máscara
                _, roi_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Usar sólo el ROI para el análisis
                estrella_mask = roi_mask
            else:
                # Si hay máscara, recortar al ROI
                estrella_mask = mask[y1:y2, x1:x2]
            
            # Medir la estrella
            metrics = self.measure_estrella(
                estrella_mask, 
                corners, 
                image[y1:y2, x1:x2].copy(), 
                (x1, y1, x2, y2),
                image  # Pasar la imagen original
            )
            
            # Combinar datos
            estrella_data = {
                'id': i+1,
                'diametro': metrics['diametro'],
                'conf': conf,
                'bbox': (x1, y1, x2, y2)
            }
            
            # Guardar las visualizaciones si existen
            if 'local_visualization' in metrics:
                visualization_key = f"estrella_{i+1}_local"
                visualizations[visualization_key] = metrics['local_visualization']
            
            if 'global_visualization' in metrics:
                visualization_key = f"estrella_{i+1}_global"
                visualizations[visualization_key] = metrics['global_visualization']
            
            results.append(estrella_data)
        
        # Si hay resultados, generar un reporte
        report_paths = None
        if results and image_name and output_dir:
            # Crear directorio para este tipo de defecto
            defect_dir = os.path.join(output_dir, image_name, self.name)
            os.makedirs(defect_dir, exist_ok=True)
            
            # Generar el reporte en la carpeta específica
            report_paths = self.generate_report(image_name, results, defect_dir)
        
        return {
            'processed_data': results,
            'visualizations': visualizations,
            'report_paths': report_paths
        }