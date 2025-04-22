import cv2
import numpy as np
import os
import pandas as pd

class InclusionNoMetalicaProcessor:
    """
    Procesador para inclusiones no metálicas
    """
    
    def __init__(self):
        """
        Inicializa el procesador de inclusiones no metálicas
        """
        self.name = "inclusion_no_metalica"
    
    def measure_inclusion(self, inclusion_mask, corners=None, inclusion_img=None, bbox=None, sensitivity=0.5):
        """
        Analiza inclusiones no metálicas contando el número de puntos/inclusiones 
        SOLAMENTE dentro del área segmentada (máscara) con sensibilidad ajustable
        
        Args:
            inclusion_mask: Máscara binaria de la región con inclusiones (ROI recortado)
            corners: Esquinas de la palanquilla [top-left, top-right, bottom-right, bottom-left] en coordenadas globales
            inclusion_img: Imagen recortada de la región (opcional, para visualizaciones)
            bbox: Bounding box de la región en coordenadas globales (x1, y1, x2, y2)
            sensitivity: Valor entre 0 y 1 que controla la sensibilidad de detección (0: menos sensible, 1: más sensible)
            
        Returns:
            metrics: Diccionario con métricas como número de inclusiones, área y concentración
        """
        if inclusion_mask is None or inclusion_mask.size == 0:
            return {
                'num_inclusiones': 0,
                'area_pixeles': 0, 
                'concentracion': 0
            }
        
        try:
            # Preparar la máscara - asegurarnos de que sea binaria
            if np.max(inclusion_mask) > 1:
                _, binary_mask = cv2.threshold(inclusion_mask, 127, 255, cv2.THRESH_BINARY)
            else:
                binary_mask = inclusion_mask.astype(np.uint8) * 255
            
            # Asegurarnos de que la máscara esté en formato uint8
            binary_mask = binary_mask.astype(np.uint8)
            
            # Calcular el área de la máscara (solo región segmentada)
            area_pixeles = cv2.countNonZero(binary_mask)
            
            # Procesar la imagen para detectar inclusiones
            num_inclusiones = 0
            valid_contours = []
            
            if inclusion_img is not None:
                # Convertir la imagen a escala de grises si es necesario
                if len(inclusion_img.shape) > 2:
                    gray_img = cv2.cvtColor(inclusion_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray_img = inclusion_img.copy()
                
                # Asegurarnos de que la máscara tenga el mismo tamaño que la imagen
                if gray_img.shape != binary_mask.shape:
                    binary_mask = cv2.resize(binary_mask, (gray_img.shape[1], gray_img.shape[0]))
                
                # Aplicar la máscara sobre la imagen en escala de grises 
                masked_gray = cv2.bitwise_and(gray_img, gray_img, mask=binary_mask)
                
                # Aplicar filtro gaussiano para reducir ruido
                blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)
                
                # AJUSTE DE SENSIBILIDAD: Calibrar parámetros según la sensibilidad
                # BlockSize: Tamaño de la ventana de adaptación (mayor = menos sensible)
                # Para sensibilidad 0.5, usar 16
                block_size = int(21 - 12 * sensitivity)  # Rango: 9-21 (menor = más sensible)
                block_size = max(9, block_size)  # Mínimo 9
                block_size = block_size if block_size % 2 == 1 else block_size + 1  # Debe ser impar
                
                # Valor C: Constante de umbral (mayor = menos sensible)
                # Para sensibilidad 0.5, usar 4
                c_value = int(8 - 6 * sensitivity)  # Rango: 2-8
                c_value = max(2, c_value)  # Mínimo 2
                
                # Usar un umbral adaptativo ajustado
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, block_size, c_value)
                
                # Aplicar nuevamente la máscara
                thresh_masked = cv2.bitwise_and(thresh, thresh, mask=binary_mask)
                
                # Filtrado morfológico moderado
                kernel = np.ones((2, 2), np.uint8)  # Kernel más pequeño para mantener más detalles
                opening = cv2.morphologyEx(thresh_masked, cv2.MORPH_OPEN, kernel, iterations=1)
                
                # Encontrar contornos
                # Encontrar contornos
                contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # AJUSTE DE SENSIBILIDAD: Área mínima según sensibilidad
                # Para sensibilidad 0.5, usar área mínima 8
                min_area = int(12 - 8 * sensitivity)  # Rango: 4-12 píxeles
                min_area = max(4, min_area)  # Mínimo 4
                valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
                
                # FILTRO DE CONTRASTE: Menos estricto que antes
                filtered_contours = []
                for contour in valid_contours:
                    # Calcular el valor medio dentro del contorno
                    mask = np.zeros_like(gray_img)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    mean_value = cv2.mean(gray_img, mask=mask)[0]
                    
                    # Calcular el valor medio del entorno
                    dilated_contour = cv2.dilate(mask, kernel, iterations=2)
                    env_mask = cv2.subtract(dilated_contour, mask)
                    env_mean = cv2.mean(gray_img, mask=env_mask)[0]
                    
                    # Calcular el contraste relativo
                    if env_mean > 0:
                        contrast = (env_mean - mean_value) / env_mean
                        # Umbral de contraste ajustado según sensibilidad
                        # Para sensibilidad 0.5, usar 0.08
                        contrast_threshold = 0.04 + (0.08 * (1 - sensitivity))  # Rango: 0.04-0.12
                        if contrast > contrast_threshold:
                            filtered_contours.append(contour)
                    else:
                        filtered_contours.append(contour)
                
                # Usar los contornos filtrados
                valid_contours = filtered_contours
                
                # Contar el número final de inclusiones
                num_inclusiones = len(valid_contours)
            
            # Calcular la concentración
            concentracion = 0
            if area_pixeles > 0:
                concentracion = num_inclusiones / area_pixeles
            
            # Visualización
            if inclusion_img is not None:
                viz_img = inclusion_img.copy() if len(inclusion_img.shape) == 3 else cv2.cvtColor(inclusion_img, cv2.COLOR_GRAY2BGR)
                
                # Encontrar contornos de la máscara para visualización
                mask_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Dibujar la región segmentada con un overlay transparente verde
                overlay = viz_img.copy()
                cv2.drawContours(overlay, mask_contours, -1, (0, 255, 0), 2)  # Contorno verde
                cv2.fillPoly(overlay, mask_contours, (0, 255, 0, 128))  # Relleno verde semi-transparente
                alpha = 0.3  # Factor de transparencia
                cv2.addWeighted(overlay, alpha, viz_img, 1-alpha, 0, viz_img)
                
                # Dibujar las inclusiones detectadas con círculos rojos
                for i, contour in enumerate(valid_contours):
                    # Obtener el centro del contorno
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Dibujar un círculo rojo para la inclusión
                        cv2.circle(viz_img, (cx, cy), 5, (0, 0, 255), -1)
                        
                        # Etiquetar la inclusión con un número
                        cv2.putText(viz_img, str(i+1), (cx+7, cy+7), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Añadir información sobre el número de inclusiones y el área
                cv2.putText(viz_img, f"Inclusiones: {num_inclusiones}", (10, 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(viz_img, f"Área ROI: {area_pixeles} px²", (10, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(viz_img, f"Concentración: {concentracion:.6f}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Guardar la visualización del ROI
                cv2.imwrite("temp_inclusion_no_metalica_analysis_roi.jpg", viz_img)
                
                # Visualización global (código igual que antes)
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
                        
                        # Dibujar el contorno de la palanquilla
                        cv2.polylines(full_img, [np.array(adjusted_corners)], True, (0, 255, 0), 2)
                        cv2.fillPoly(full_img, [np.array(adjusted_corners)], (20, 20, 20))
                        
                        # Dibujar los vértices numerados
                        for i, corner in enumerate(adjusted_corners):
                            cv2.circle(full_img, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), -1)
                            cv2.putText(full_img, str(i+1), (int(corner[0])-4, int(corner[1])+4), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Dibujar la posición de la región con inclusiones
                        if bbox is not None:
                            x1, y1, x2, y2 = bbox
                            adj_x1, adj_y1 = x1 - min_x + margin, y1 - min_y + margin
                            adj_x2, adj_y2 = x2 - min_x + margin, y2 - min_y + margin
                            
                            # Dibujar un rectángulo para resaltar la zona de inclusiones
                            cv2.rectangle(full_img, (int(adj_x1), int(adj_y1)), (int(adj_x2), int(adj_y2)), (0, 255, 255), 2)
                            
                            # Calcular el centro del rectángulo
                            center_x = int((adj_x1 + adj_x2) / 2)
                            center_y = int((adj_y1 + adj_y2) / 2)
                            
                            # Añadir texto informativo sobre inclusiones
                            cv2.putText(full_img, f"Inclusiones: {num_inclusiones}", (center_x - 80, adj_y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Añadir información general en la parte superior de la imagen
                        cv2.putText(full_img, f"Inclusiones no metálicas: {num_inclusiones}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(full_img, f"Área analizada: {area_pixeles} px²", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(full_img, f"Concentración: {concentracion:.6f} inclusiones/px²", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Guardar la visualización completa
                        cv2.imwrite("temp_inclusion_no_metalica_analysis_full.jpg", full_img)
                    except Exception as e:
                        print(f"Error al crear visualización global para inclusiones no metálicas: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Retornar información completa
                return {
                    'num_inclusiones': num_inclusiones,
                    'area_pixeles': area_pixeles,
                    'concentracion': concentracion,
                    'local_visualization': viz_img,
                    'global_visualization': full_img,
                    'contours': valid_contours
                }
            
            # Retornar solo métricas básicas
            return {
                'num_inclusiones': num_inclusiones,
                'area_pixeles': area_pixeles,
                'concentracion': concentracion
            }
            
        except Exception as e:
            print(f"Error en measure_inclusion_no_metalica: {e}")
            import traceback
            traceback.print_exc()
            
            # Retornar valores por defecto en caso de error
            return {
                'num_inclusiones': 0,
                'area_pixeles': 0,
                'concentracion': 0
            }
    
    def generate_report(self, image_name, inclusiones_data, output_dir):
        """
        Genera un informe simple de las inclusiones no metálicas
        
        Args:
            image_name: Nombre de la imagen original
            inclusiones_data: Lista de diccionarios con los datos de las inclusiones
            output_dir: Directorio donde guardar el informe
        
        Returns:
            report_path: Ruta al archivo de informe generado
        """
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear un DataFrame con los datos, excluyendo objetos complejos
        df = pd.DataFrame([{k: v for k, v in inclusion.items() if k not in ['local_visualization', 
                                                                          'global_visualization', 
                                                                          'contours']} 
                         for inclusion in inclusiones_data])
        
        # Formato del informe
        report_path = os.path.join(output_dir, f"{image_name}_inclusion_no_metalica_report.csv")
        
        # Guardar como CSV
        df.to_csv(report_path, index=False)
        
        # También generar una versión en formato de texto para fácil visualización
        text_report_path = os.path.join(output_dir, f"{image_name}_inclusion_no_metalica_report.txt")
        
        with open(text_report_path, 'w') as f:
            f.write(f"REPORTE DE INCLUSIONES NO METÁLICAS - {image_name}\n")
            f.write("="*50 + "\n\n")
            
            for i, inclusion in enumerate(inclusiones_data):
                f.write(f"REGIÓN DE INCLUSIONES #{i+1}\n")
                f.write(f"  Número de inclusiones: {inclusion['num_inclusiones']}\n")
                f.write(f"  Área analizada: {inclusion['area_pixeles']} píxeles²\n")
                f.write(f"  Concentración: {inclusion['concentracion']:.6f} inclusiones/píxel²\n")
                if 'conf' in inclusion:
                    f.write(f"  Confianza: {inclusion['conf']:.2f}\n")
                f.write("\n")
        
        print(f"Reporte generado en: {report_path}")
        print(f"Reporte de texto generado en: {text_report_path}")
        
        return report_path, text_report_path
    
    def process(self, detections, image, corners, zone_masks, image_name=None, output_dir=None):
        """
        Procesa todas las inclusiones no metálicas detectadas
        
        Args:
            detections: Lista de detecciones de inclusiones
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
                inclusion_mask = roi_mask
            else:
                # Si hay máscara, recortar al ROI
                inclusion_mask = mask[y1:y2, x1:x2]
            
            # Usar una sensibilidad alta (0.7) para detectar un número razonable de inclusiones
            metrics = self.measure_inclusion(inclusion_mask, corners, image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2), sensitivity=0.7)
            
            # Combinar datos
            inclusion_data = {
                'id': i+1,
                'num_inclusiones': metrics['num_inclusiones'],
                'area_pixeles': metrics['area_pixeles'],
                'concentracion': metrics['concentracion'],
                'conf': conf,
                'bbox': (x1, y1, x2, y2)
            }
            
            # Guardar las visualizaciones si existen
            if 'local_visualization' in metrics:
                visualization_key = f"inclusion_{i+1}_local"
                visualizations[visualization_key] = metrics['local_visualization']
            
            if 'global_visualization' in metrics:
                visualization_key = f"inclusion_{i+1}_global"
                visualizations[visualization_key] = metrics['global_visualization']
            
            results.append(inclusion_data)
        
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