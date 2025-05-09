import cv2
import numpy as np
import os
import pandas as pd

from utils.utils import draw_arrow, find_extreme_points

class NucleoEsponjosoProcessor:
    """
    Procesador para núcleos esponjosos
    """
    
    def __init__(self):
        """
        Inicializa el procesador de núcleos esponjosos
        """
        self.name = "nucleo_esponjoso"
    
    def measure_nucleo(self, nucleo_mask, corners=None, nucleo_img=None, bbox=None, full_image_shape=None):
        """
        Mide las propiedades de un núcleo esponjoso en píxeles
        y calcula el porcentaje de área que ocupa respecto a la palanquilla
        
        Args:
            nucleo_mask: Máscara binaria del núcleo esponjoso (ROI recortado)
            corners: Esquinas de la palanquilla [top-left, top-right, bottom-right, bottom-left] en coordenadas globales
            nucleo_img: Imagen recortada del núcleo (opcional, para visualizaciones)
            bbox: Bounding box del núcleo en coordenadas globales (x1, y1, x2, y2)
            full_image_shape: Forma de la imagen completa (altura, ancho)
            
        Returns:
            metrics: Diccionario con las métricas de diámetro, área y porcentaje
        """
        # Encontrar contornos del núcleo
        contours, _ = cv2.findContours(nucleo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'diametro': 0,
                'area_nucleo': 0,
                'area_palanquilla': 0,
                'porcentaje_area': 0
            }
        
        # Obtener el contorno principal (el más grande)
        contour = max(contours, key=cv2.contourArea)
        
        # Calcular área del núcleo esponjoso en píxeles
        area_nucleo = cv2.contourArea(contour)
        
        # CALCULAR DIÁMETRO: distancia máxima entre dos puntos del contorno
        local_pt1, local_pt2, diametro = find_extreme_points(contour)
        
        # Calcular área de la palanquilla
        area_palanquilla = 0
        if corners is not None and len(corners) == 4:
            # Crear máscara del tamaño de la imagen completa si tenemos la información
            if full_image_shape is not None:
                h, w = full_image_shape[:2]
                palanquilla_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(palanquilla_mask, [np.array(corners)], 255)
                area_palanquilla = cv2.countNonZero(palanquilla_mask)
            else:
                # Si no tenemos la forma completa, aproximar el área con el polígono
                area_palanquilla = cv2.contourArea(np.array(corners))
        
        # Calcular porcentaje del área
        porcentaje_area = 0
        if area_palanquilla > 0:
            porcentaje_area = (area_nucleo / area_palanquilla) * 100
        
        # Para visualización, necesitamos las coordenadas globales
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            global_pt1 = (local_pt1[0] + x1, local_pt1[1] + y1)
            global_pt2 = (local_pt2[0] + x1, local_pt2[1] + y1)
        else:
            global_pt1 = local_pt1
            global_pt2 = local_pt2
        
        # Visualización si se proporciona imagen
        if nucleo_img is not None:
            viz_img = nucleo_img.copy() if len(nucleo_img.shape) == 3 else cv2.cvtColor(nucleo_img, cv2.COLOR_GRAY2BGR)
            
            # Dibujar el contorno del núcleo en el ROI
            cv2.drawContours(viz_img, [contour], -1, (0, 255, 0), 2)
            
            # Dibujar los puntos extremos (para el diámetro)
            if local_pt1 is not None and local_pt2 is not None:
                cv2.circle(viz_img, local_pt1, 5, (255, 0, 0), -1)
                cv2.circle(viz_img, local_pt2, 5, (255, 0, 0), -1)
                
                # Dibujar flecha para el diámetro
                draw_arrow(viz_img, local_pt1, local_pt2, (0, 255, 255), 2, 10, f"D={diametro:.1f}px", (5, -10))
            
            # Añadir información del área y porcentaje
            cv2.putText(viz_img, f"Área: {area_nucleo:.0f} px²", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(viz_img, f"% Área: {porcentaje_area:.2f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Guardar la visualización del ROI
            #cv2.imwrite("temp_nucleo_esponjoso_analysis_roi.jpg", viz_img)
            
            # Si tenemos suficiente información para crear una visualización global
            full_img = None
            
            if corners is not None and bbox is not None:
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
                    
                    # Convertir los puntos extremos a coordenadas ajustadas
                    adjusted_pt1 = (global_pt1[0] - min_x + margin, global_pt1[1] - min_y + margin)
                    adjusted_pt2 = (global_pt2[0] - min_x + margin, global_pt2[1] - min_y + margin)
                    
                    # Dibujar el contorno de la palanquilla
                    cv2.polylines(full_img, [np.array(adjusted_corners)], True, (0, 255, 0), 2)
                    cv2.fillPoly(full_img, [np.array(adjusted_corners)], (20, 20, 20))  # Fondo oscuro para la palanquilla
                    
                    # Dibujar los vértices numerados
                    for i, corner in enumerate(adjusted_corners):
                        cv2.circle(full_img, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), -1)
                        cv2.putText(full_img, str(i+1), (int(corner[0])-4, int(corner[1])+4), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Dibujar la posición del núcleo esponjoso
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        adj_x1, adj_y1 = x1 - min_x + margin, y1 - min_y + margin
                        adj_x2, adj_y2 = x2 - min_x + margin, y2 - min_y + margin
                        cv2.rectangle(full_img, (int(adj_x1), int(adj_y1)), (int(adj_x2), int(adj_y2)), (0, 0, 255), 2)
                    
                    # Dibujar el contorno del núcleo
                    adjusted_contour = contour.copy()
                    adjusted_contour[:,:,0] = adjusted_contour[:,:,0] + int(adj_x1)
                    adjusted_contour[:,:,1] = adjusted_contour[:,:,1] + int(adj_y1)
                    cv2.drawContours(full_img, [adjusted_contour], -1, (0, 255, 0), 2)
                    
                    # Calcular el centro aproximado del contorno
                    M = cv2.moments(adjusted_contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        center = (cx, cy)
                        
                        # Dibujar un círculo aproximado para visualizar mejor el núcleo esponjoso
                        cv2.circle(full_img, center, int(diametro/2), (0, 255, 255), 2)
                    
                    # Dibujar los puntos extremos del núcleo en la vista global
                    cv2.circle(full_img, (int(adjusted_pt1[0]), int(adjusted_pt1[1])), 5, (255, 0, 0), -1)
                    cv2.circle(full_img, (int(adjusted_pt2[0]), int(adjusted_pt2[1])), 5, (255, 0, 0), -1)
                    
                    # Dibujar la flecha para el diámetro
                    draw_arrow(full_img, 
                              (int(adjusted_pt1[0]), int(adjusted_pt1[1])), 
                              (int(adjusted_pt2[0]), int(adjusted_pt2[1])), 
                              (0, 255, 255), 2, 15, f"D={diametro:.1f}px")
                    
                    # Añadir información de área y porcentaje en la esquina superior
                    cv2.putText(full_img, f"Área núcleo: {area_nucleo:.0f} px²", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(full_img, f"Área palanquilla: {area_palanquilla:.0f} px²", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(full_img, f"Porcentaje de área: {porcentaje_area:.2f}%", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Guardar la visualización completa
                    #cv2.imwrite("temp_nucleo_esponjoso_analysis_full.jpg", full_img)
                except Exception as e:
                    print(f"Error al crear visualización global para núcleo esponjoso: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Retornar información adicional para el visualizador de resultados
            return {
                'diametro': round(diametro, 2),
                'area_nucleo': round(area_nucleo, 2),
                'area_palanquilla': round(area_palanquilla, 2),
                'porcentaje_area': round(porcentaje_area, 2),
                'local_visualization': viz_img,
                'global_visualization': full_img,
                'local_pt1': local_pt1,
                'local_pt2': local_pt2,
                'global_pt1': global_pt1,
                'global_pt2': global_pt2
            }
        
        return {
            'diametro': round(diametro, 2),
            'area_nucleo': round(area_nucleo, 2),
            'area_palanquilla': round(area_palanquilla, 2),
            'porcentaje_area': round(porcentaje_area, 2)
        }
    
    def generate_report(self, image_name, nucleos_data, output_dir):
        """
        Genera un informe simple de los núcleos esponjosos
        
        Args:
            image_name: Nombre de la imagen original
            nucleos_data: Lista de diccionarios con los datos de los núcleos esponjosos
            output_dir: Directorio donde guardar el informe
        
        Returns:
            report_path: Ruta al archivo de informe generado
        """
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear un DataFrame con los datos, excluyendo objetos complejos
        df = pd.DataFrame([{k: v for k, v in nucleo.items() if k not in ['visualization', 'rect', 'box', 'local_visualization', 
                                                                         'global_visualization', 'local_pt1', 'local_pt2', 
                                                                         'global_pt1', 'global_pt2']} 
                           for nucleo in nucleos_data])
        
        # Formato del informe
        report_path = os.path.join(output_dir, f"{image_name}_nucleo_esponjoso_report.csv")
        
        # Guardar como CSV
        df.to_csv(report_path, index=False)
        
        # También generar una versión en formato de texto para fácil visualización
        text_report_path = os.path.join(output_dir, f"{image_name}_nucleo_esponjoso_report.txt")
        
        with open(text_report_path, 'w') as f:
            f.write(f"REPORTE DE NÚCLEOS ESPONJOSOS - {image_name}\n")
            f.write("="*50 + "\n\n")
            
            for i, nucleo in enumerate(nucleos_data):
                f.write(f"NÚCLEO ESPONJOSO #{i+1}\n")
                f.write(f"  Diámetro: {nucleo['diametro']} píxeles\n")
                f.write(f"  Área del núcleo: {nucleo['area_nucleo']} píxeles²\n")
                f.write(f"  Área de la palanquilla: {nucleo['area_palanquilla']} píxeles²\n")
                f.write(f"  Porcentaje de área: {nucleo['porcentaje_area']:.2f}%\n")
                f.write(f"  Confianza: {nucleo['conf']:.2f}\n\n")
        
        print(f"Reporte generado en: {report_path}")
        print(f"Reporte de texto generado en: {text_report_path}")
        
        return report_path, text_report_path
    
    def process(self, detections, image, corners, zone_masks, image_name=None, output_dir=None):
        """
        Procesa todos los núcleos esponjosos detectados
        
        Args:
            detections: Lista de detecciones de núcleos esponjosos
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
                nucleo_mask = roi_mask
            else:
                # Si hay máscara, recortar al ROI
                nucleo_mask = mask[y1:y2, x1:x2]
            
            # Medir el núcleo esponjoso
            metrics = self.measure_nucleo(nucleo_mask, corners, image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2), image.shape)
            
            # Combinar datos
            nucleo_data = {
                'id': i+1,
                'diametro': metrics['diametro'],
                'area_nucleo': metrics['area_nucleo'],
                'area_palanquilla': metrics['area_palanquilla'],
                'porcentaje_area': metrics['porcentaje_area'],
                'conf': conf,
                'bbox': (x1, y1, x2, y2)
            }
            
            # Guardar las visualizaciones si existen
            if 'local_visualization' in metrics:
                visualization_key = f"nucleo_{i+1}_local"
                visualizations[visualization_key] = metrics['local_visualization']
            
            if 'global_visualization' in metrics:
                visualization_key = f"nucleo_{i+1}_global"
                visualizations[visualization_key] = metrics['global_visualization']
            
            results.append(nucleo_data)
        
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