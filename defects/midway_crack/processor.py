import cv2
import numpy as np
import os
import pandas as pd

from utils.utils import draw_arrow, find_extreme_points, find_closest_edge_point

class MidwayCrackProcessor:
    """
    Procesador para las grietas de medio camino
    """
    
    def __init__(self):
        """
        Inicializa el procesador de grietas de medio camino
        """
        self.name = "grietas_medio_camino"
    
    def measure_crack(self, crack_mask, corners=None, crack_img=None, bbox=None):
        """
        Mide las propiedades de una grieta de medio camino en píxeles, 
        calculando la distancia al borde más cercano de la palanquilla.
        
        Args:
            crack_mask: Máscara binaria de la grieta (ROI recortado)
            corners: Esquinas de la palanquilla [top-left, top-right, bottom-right, bottom-left] en coordenadas globales
            crack_img: Imagen recortada de la grieta (opcional, para visualizaciones)
            bbox: Bounding box de la grieta en coordenadas globales (x1, y1, x2, y2)
            
        Returns:
            metrics: Diccionario con las métricas L, e, D en píxeles
        """
        # Encontrar contornos de la grieta
        contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'L': 0,
                'e': 0,
                'D': 0
            }
        
        # Obtener el contorno principal (el más grande)
        contour = max(contours, key=cv2.contourArea)
        
        # 1. CALCULAR L: longitud real de la grieta (distancia entre los puntos más extremos)
        local_pt1, local_pt2, L = find_extreme_points(contour)
        
        # Rectángulo rotado mínimo para la grieta (usado para calcular el espesor e)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 2. CALCULAR e: espesor de la grieta (el lado más corto del rectángulo)
        (cx, cy), (width, height), angle = rect
        e = min(width, height)
        
        # 3. CALCULAR D: distancia desde el borde de la grieta hasta el borde más cercano de la palanquilla
        D = 0
        contour_point = None
        edge_point = None
        closest_edge = None
        
        # Para calcular D necesitamos convertir a coordenadas globales
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            
            # Convertir el contorno a coordenadas globales
            global_contour = contour.copy()
            global_contour[:, :, 0] += x1
            global_contour[:, :, 1] += y1
            
            # Convertir puntos extremos a coordenadas globales
            global_pt1 = (local_pt1[0] + x1, local_pt1[1] + y1)
            global_pt2 = (local_pt2[0] + x1, local_pt2[1] + y1)
        else:
            global_contour = contour
            global_pt1 = local_pt1
            global_pt2 = local_pt2
        
        # Si tenemos las esquinas de la palanquilla, calcular la distancia desde el contorno
        if corners is not None and len(corners) == 4:
            # Definir los cuatro bordes de la palanquilla (pares de puntos)
            edges = [
                (corners[0], corners[1]),  # Superior (TL a TR)
                (corners[1], corners[2]),  # Derecho (TR a BR)
                (corners[2], corners[3]),  # Inferior (BR a BL)
                (corners[3], corners[0])   # Izquierdo (BL a TL)
            ]
            
            # Encontrar el punto del contorno más cercano a cualquier borde de la palanquilla
            contour_point, edge_point, D, closest_edge = find_closest_edge_point(global_contour, edges)
        
        # Visualización si se proporciona imagen
        if crack_img is not None:
            viz_img = crack_img.copy() if len(crack_img.shape) == 3 else cv2.cvtColor(crack_img, cv2.COLOR_GRAY2BGR)
            
            # Dibujar el contorno de la grieta en el ROI
            cv2.drawContours(viz_img, [contour], -1, (0, 255, 0), 2)
            
            # Dibujar el rectángulo rotado en el ROI
            cv2.drawContours(viz_img, [box], 0, (0, 0, 255), 2)
            
            # Dibujar los puntos extremos (para L)
            if local_pt1 is not None and local_pt2 is not None:
                cv2.circle(viz_img, local_pt1, 5, (255, 0, 0), -1)
                cv2.circle(viz_img, local_pt2, 5, (255, 0, 0), -1)
                
                # Dibujar flecha para L (longitud real)
                draw_arrow(viz_img, local_pt1, local_pt2, (0, 255, 255), 2, 10, f"L={L:.1f}px", (5, -10))
            
            # Dibujar el centro del rectángulo
            center = (int(cx), int(cy))
            cv2.circle(viz_img, center, 5, (255, 0, 255), -1)
            
            # Dibujar flecha para e (espesor)
            if width > height:  # La grieta es horizontal, e es vertical
                arrow_start = (int(cx), int(cy - e/2))
                arrow_end = (int(cx), int(cy + e/2))
            else:  # La grieta es vertical, e es horizontal
                arrow_start = (int(cx - e/2), int(cy))
                arrow_end = (int(cx + e/2), int(cy))
            
            draw_arrow(viz_img, arrow_start, arrow_end, (255, 0, 255), 2, 10, f"e={e:.1f}px", (5, 5))
            
            # Anotar el valor de D en la imagen ROI
            cv2.putText(viz_img, f"D={D:.1f}px", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Guardar la visualización del ROI
            cv2.imwrite("temp_midway_crack_analysis_roi.jpg", viz_img)
            
            # Si tenemos suficiente información para crear una visualización global
            full_img = None
            
            if corners is not None and bbox is not None and contour_point is not None and edge_point is not None:
                try:
                    # Estimar un tamaño razonable para la imagen completa
                    x_values = [p[0] for p in corners]
                    y_values = [p[1] for p in corners]
                    min_x, max_x = min(x_values), max(x_values)
                    min_y, max_y = min(y_values), max(y_values)
                    
                    # Añadir un poco de margen
                    margin = 50
                    img_width = max_x - min_x + 2*margin
                    img_height = max_y - min_y + 2*margin
                    
                    # Asegurar tamaño mínimo
                    img_width = max(img_width, 800)
                    img_height = max(img_height, 800)
                    
                    # Crear imagen en blanco
                    full_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                    
                    # Ajustar las coordenadas para que estén dentro de la imagen
                    adjusted_corners = [(x - min_x + margin, y - min_y + margin) for x, y in corners]
                    adjusted_contour_point = (contour_point[0] - min_x + margin, contour_point[1] - min_y + margin)
                    adjusted_edge_point = (edge_point[0] - min_x + margin, edge_point[1] - min_y + margin)
                    
                    # Convertir los puntos extremos a coordenadas ajustadas
                    adjusted_pt1 = (global_pt1[0] - min_x + margin, global_pt1[1] - min_y + margin)
                    adjusted_pt2 = (global_pt2[0] - min_x + margin, global_pt2[1] - min_y + margin)
                    
                    # Dibujar el contorno de la palanquilla
                    cv2.polylines(full_img, [np.array(adjusted_corners)], True, (0, 255, 0), 2)
                    
                    # Dibujar los vértices numerados
                    for i, corner in enumerate(adjusted_corners):
                        cv2.circle(full_img, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), -1)
                        cv2.putText(full_img, str(i+1), (int(corner[0])-4, int(corner[1])+4), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Dibujar la posición de la grieta
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        adj_x1, adj_y1 = x1 - min_x + margin, y1 - min_y + margin
                        adj_x2, adj_y2 = x2 - min_x + margin, y2 - min_y + margin
                        cv2.rectangle(full_img, (int(adj_x1), int(adj_y1)), (int(adj_x2), int(adj_y2)), (0, 0, 255), 2)
                    
                    # Dibujar el contorno de la grieta
                    adjusted_contour = global_contour.copy()
                    adjusted_contour[:,:,0] = adjusted_contour[:,:,0] - min_x + margin
                    adjusted_contour[:,:,1] = adjusted_contour[:,:,1] - min_y + margin
                    cv2.drawContours(full_img, [adjusted_contour], -1, (0, 255, 0), 2)
                    
                    # Dibujar los puntos extremos de la grieta en la vista global
                    cv2.circle(full_img, (int(adjusted_pt1[0]), int(adjusted_pt1[1])), 5, (255, 0, 0), -1)
                    cv2.circle(full_img, (int(adjusted_pt2[0]), int(adjusted_pt2[1])), 5, (255, 0, 0), -1)
                    
                    # Dibujar la flecha L para la longitud total
                    draw_arrow(full_img, 
                              (int(adjusted_pt1[0]), int(adjusted_pt1[1])), 
                              (int(adjusted_pt2[0]), int(adjusted_pt2[1])), 
                              (0, 255, 255), 2, 15, f"L={L:.1f}px")
                    
                    # Marcar con un rectángulo rojo el punto del contorno más cercano al borde
                    cv2.rectangle(full_img, 
                                 (int(adjusted_contour_point[0])-5, int(adjusted_contour_point[1])-5), 
                                 (int(adjusted_contour_point[0])+5, int(adjusted_contour_point[1])+5), 
                                 (0, 0, 255), 2)
                    
                    # Dibujar el punto en el borde
                    cv2.circle(full_img, (int(adjusted_edge_point[0]), int(adjusted_edge_point[1])), 5, (255, 255, 0), -1)
                    
                    # Dibujar la flecha para D desde el punto del contorno al punto en el borde
                    draw_arrow(full_img, 
                              (int(adjusted_contour_point[0]), int(adjusted_contour_point[1])), 
                              (int(adjusted_edge_point[0]), int(adjusted_edge_point[1])), 
                              (0, 165, 255), 2, 15, f"D={D:.1f}px")
                    
                    # Guardar la visualización completa
                    cv2.imwrite("temp_midway_crack_analysis_full.jpg", full_img)
                except Exception as e:
                    print(f"Error al crear visualización global: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Retornar información adicional para el visualizador de resultados
            return {
                'L': round(L, 2),
                'e': round(e, 2),
                'D': round(D, 2),
                'local_visualization': viz_img,
                'global_visualization': full_img,
                'rect': rect,
                'box': box,
                'local_pt1': local_pt1,
                'local_pt2': local_pt2,
                'global_pt1': global_pt1,
                'global_pt2': global_pt2,
                'contour_point': contour_point,
                'edge_point': edge_point
            }
        
        return {
            'L': round(L, 2),
            'e': round(e, 2),
            'D': round(D, 2)
        }
    
    def generate_report(self, image_name, cracks_data, output_dir):
        """
        Genera un informe simple de las grietas de medio camino
        
        Args:
            image_name: Nombre de la imagen original
            cracks_data: Lista de diccionarios con los datos de las grietas
            output_dir: Directorio donde guardar el informe
        
        Returns:
            report_path: Ruta al archivo de informe generado
        """
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear un DataFrame con los datos, excluyendo objetos complejos
        df = pd.DataFrame([{k: v for k, v in crack.items() if k not in ['visualization', 'rect', 'box', 'bottom_center', 'closest_corner', 
                                                                       'local_visualization', 'global_visualization', 'local_center', 'global_center',
                                                                       'contour_point', 'edge_point', 'closest_edge', 'extreme_points',
                                                                       'local_pt1', 'local_pt2', 'global_pt1', 'global_pt2']} 
                           for crack in cracks_data])
        
        # Formato del informe
        report_path = os.path.join(output_dir, f"{image_name}_grietas_medio_camino_report.csv")
        
        # Guardar como CSV
        df.to_csv(report_path, index=False)
        
        # También generar una versión en formato de texto para fácil visualización
        text_report_path = os.path.join(output_dir, f"{image_name}_grietas_medio_camino_report.txt")
        
        with open(text_report_path, 'w') as f:
            f.write(f"REPORTE DE GRIETAS DE MEDIO CAMINO - {image_name}\n")
            f.write("="*50 + "\n\n")
            
            for i, crack in enumerate(cracks_data):
                f.write(f"GRIETA #{i+1}\n")
                f.write(f"  Longitud (L): {crack['L']} píxeles\n")
                f.write(f"  Espesor (e): {crack['e']} píxeles\n")
                f.write(f"  Distancia a superficie (D): {crack['D']} píxeles\n")
                f.write(f"  Ángulo: {crack['angulo']:.1f}°\n")
                f.write(f"  Dirección: {crack['direccion']}\n")
                f.write(f"  Confianza: {crack['conf']:.2f}\n\n")
        
        print(f"Reporte generado en: {report_path}")
        print(f"Reporte de texto generado en: {text_report_path}")
        
        return report_path, text_report_path
    
    def process(self, detections, image, corners, zone_masks, image_name=None, output_dir=None):
        """
        Procesa todas las grietas de medio camino detectadas
        
        Args:
            detections: Lista de detecciones de grietas de medio camino
            image: Imagen original
            corners: Esquinas de la palanquilla
            zone_masks: Máscaras de zonas
            image_name: Nombre de la imagen (sin extensión)
            output_dir: Directorio de salida para guardar reportes
            
        Returns:
            processed_data: Diccionario con los resultados del procesamiento
        """
        results = []
        visualizations = {}
        
        # Procesar cada detección
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            angulo = detection.get('angulo', 0)
            direccion = detection.get('direccion', 'indeterminada')
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
                crack_mask = roi_mask
            else:
                # Si hay máscara, recortar al ROI
                crack_mask = mask[y1:y2, x1:x2]
            
            # Medir la grieta
            metrics = self.measure_crack(crack_mask, corners, image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2))
            
            # Combinar datos
            crack_data = {
                'id': i+1,
                'L': metrics['L'],
                'e': metrics['e'],
                'D': metrics['D'],
                'angulo': angulo,
                'direccion': direccion,
                'conf': conf,
                'bbox': (x1, y1, x2, y2)
            }
            
            # Guardar las visualizaciones si existen
            if 'local_visualization' in metrics:
                visualization_key = f"grieta_{i+1}_local"
                visualizations[visualization_key] = metrics['local_visualization']
            
            if 'global_visualization' in metrics:
                visualization_key = f"grieta_{i+1}_global"
                visualizations[visualization_key] = metrics['global_visualization']
            
            results.append(crack_data)
        
        # Si hay resultados y tenemos un nombre de imagen y un directorio de salida, generar un reporte
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