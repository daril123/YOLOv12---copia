import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import math

class IMFProcessor:
    """
    Procesador para análisis IMF (Inicial-Medio-Final) de palanquillas
    """
    
    def __init__(self):
        """
        Inicializa el procesador de IMF
        """
        self.name = "imf"
    
    def calcular_IMF(self, contorno):
        """
        Calcula las posiciones y longitudes de las líneas I, M y F tanto horizontal como verticalmente.
        
        Args:
            contorno: Contorno de la palanquilla (numpy array)
        
        Returns:
            Un diccionario con los valores I, M y F para ambas direcciones
        """
        # Asegurarse de que el contorno tenga el formato correcto (n,2)
        if len(contorno.shape) == 3 and contorno.shape[1] == 1:
            contorno = contorno.reshape(contorno.shape[0], 2)
        
        # Convertir a numpy array si no lo es ya
        polygon_points = np.array(contorno, dtype=np.int32)
        
        # Calcular los rangos del polígono
        y_coords = polygon_points[:, 1]
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)
        y_range = max_y - min_y
        
        x_coords = polygon_points[:, 0]
        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        x_range = max_x - min_x
        
        # Diccionario para almacenar resultados
        resultados = {
            "horizontal": {
                "I": {"posicion": None, "longitud": None, "puntos": None},
                "M": {"posicion": None, "longitud": None, "puntos": None},
                "F": {"posicion": None, "longitud": None, "puntos": None}
            },
            "vertical": {
                "I": {"posicion": None, "longitud": None, "puntos": None},
                "M": {"posicion": None, "longitud": None, "puntos": None},
                "F": {"posicion": None, "longitud": None, "puntos": None}
            },
            "dimensiones": {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "ancho": x_range,
                "alto": y_range
            }
        }
        
        # CALCULAR LÍNEAS HORIZONTALES (I, M, F)
        h_line_positions = [1/6, 3/6, 5/6]
        h_line_names = ['I', 'M', 'F']
        
        for idx, pos in enumerate(h_line_positions):
            # Calcular la posición Y para esta línea horizontal
            y_pos = min_y + (pos * y_range)
            
            # Encontrar las intersecciones de esta línea horizontal con el contorno
            intersections = []
            for i in range(len(polygon_points)):
                j = (i + 1) % len(polygon_points)
                p1 = polygon_points[i]
                p2 = polygon_points[j]
                
                # Si la línea horizontal cruza el segmento
                if (p1[1] <= y_pos and p2[1] >= y_pos) or (p1[1] >= y_pos and p2[1] <= y_pos):
                    # Calcular el punto de intersección
                    if p2[1] - p1[1] != 0:  # Evitar división por cero
                        x_intersect = p1[0] + (p2[0] - p1[0]) * (y_pos - p1[1]) / (p2[1] - p1[1])
                        intersections.append((x_intersect, y_pos))
            
            # Ordenar las intersecciones por coordenada X
            intersections.sort(key=lambda p: p[0])
            
            if len(intersections) >= 2:
                # Tomar el punto más a la izquierda y más a la derecha
                left_point = intersections[0]
                right_point = intersections[-1]
                
                # Calcular longitud de la línea
                line_length = math.sqrt((right_point[0] - left_point[0])**2 + 
                                      (right_point[1] - left_point[1])**2)
                
                # Guardar resultados
                resultados["horizontal"][h_line_names[idx]]["posicion"] = y_pos
                resultados["horizontal"][h_line_names[idx]]["longitud"] = line_length
                resultados["horizontal"][h_line_names[idx]]["puntos"] = [left_point, right_point]
        
        # CALCULAR LÍNEAS VERTICALES (I, M, F)
        v_line_positions = [1/6, 3/6, 5/6]
        v_line_names = ['I', 'M', 'F']
        
        for idx, pos in enumerate(v_line_positions):
            # Calcular la posición X para esta línea vertical
            x_pos = min_x + (pos * x_range)
            
            # Encontrar las intersecciones de esta línea vertical con el contorno
            intersections = []
            for i in range(len(polygon_points)):
                j = (i + 1) % len(polygon_points)
                p1 = polygon_points[i]
                p2 = polygon_points[j]
                
                # Si la línea vertical cruza el segmento
                if (p1[0] <= x_pos and p2[0] >= x_pos) or (p1[0] >= x_pos and p2[0] <= x_pos):
                    # Calcular el punto de intersección
                    if p2[0] - p1[0] != 0:  # Evitar división por cero
                        y_intersect = p1[1] + (p2[1] - p1[1]) * (x_pos - p1[0]) / (p2[0] - p1[0])
                        intersections.append((x_pos, y_intersect))
            
            # Ordenar las intersecciones por coordenada Y
            intersections.sort(key=lambda p: p[1])
            
            if len(intersections) >= 2:
                # Tomar el punto más arriba y más abajo
                top_point = intersections[0]
                bottom_point = intersections[-1]
                
                # Calcular longitud de la línea
                line_length = math.sqrt((bottom_point[0] - top_point[0])**2 + 
                                      (bottom_point[1] - top_point[1])**2)
                
                # Guardar resultados
                resultados["vertical"][v_line_names[idx]]["posicion"] = x_pos
                resultados["vertical"][v_line_names[idx]]["longitud"] = line_length
                resultados["vertical"][v_line_names[idx]]["puntos"] = [top_point, bottom_point]
        
        # Calcular el rectángulo mínimo para referencia
        rect = cv2.minAreaRect(polygon_points)
        width = rect[1][0]
        height = rect[1][1]
        
        # Calcular la relación pixel/mm (asumiendo palanquilla de 160mm)
        px_per_mm_width = width / 160
        px_per_mm_height = height / 160
        
        # Agregar dimensiones del rectángulo mínimo
        resultados["rectangulo_minimo"] = {
            "centro": rect[0],
            "tamano": (width, height),
            "angulo": rect[2],
            "px_por_mm": (px_per_mm_width, px_per_mm_height)
        }
        
        return resultados
    
    def visualizar_IMF(self, image, contorno, resultados_IMF=None, mask_img=None):
        """
        Visualiza una palanquilla con las líneas I, M y F tanto horizontales como verticales.
        
        Args:
            image: Imagen original
            contorno: Contorno de la palanquilla
            resultados_IMF: Diccionario con los resultados del cálculo IMF (opcional)
            mask_img: Máscara binaria de la palanquilla (opcional)
        
        Returns:
            Imagen con la visualización de las medidas
        """
        # Si no se proporcionan resultados, calcularlos
        if resultados_IMF is None:
            resultados_IMF = self.calcular_IMF(contorno)
        
        # Crear una copia de la imagen para dibujar
        img_visualizacion = image.copy()
        
        # Asegurarse de que el contorno tenga formato correcto
        if len(contorno.shape) == 3 and contorno.shape[1] == 1:
            contorno = contorno.reshape(contorno.shape[0], 2)
        
        polygon_points = np.array(contorno, dtype=np.int32)
        
        # Extraer dimensiones
        min_x = resultados_IMF["dimensiones"]["min_x"]
        max_x = resultados_IMF["dimensiones"]["max_x"]
        min_y = resultados_IMF["dimensiones"]["min_y"]
        max_y = resultados_IMF["dimensiones"]["max_y"]
        
        # Extraer relación píxel/mm
        px_per_mm_width = resultados_IMF["rectangulo_minimo"]["px_por_mm"][0]
        px_per_mm_height = resultados_IMF["rectangulo_minimo"]["px_por_mm"][1]
        width = resultados_IMF["rectangulo_minimo"]["tamano"][0]
        height = resultados_IMF["rectangulo_minimo"]["tamano"][1]
        
        # Si tenemos la máscara, usarla como transparencia
        if mask_img is not None:
            # Crear una copia de la imagen original
            mask_overlay = image.copy()
            
            # Crear una versión coloreada de la máscara
            colored_mask = np.zeros_like(image)
            colored_mask[mask_img > 0] = [0, 255, 0]  # Verde
            
            # Aplicar la máscara con transparencia
            alpha = 0.3
            cv2.addWeighted(colored_mask, alpha, mask_overlay, 1 - alpha, 0, mask_overlay)
            
            # Usar esta imagen como base
            img_visualizacion = mask_overlay
        
        # Dibujar el contorno de la palanquilla
        cv2.drawContours(img_visualizacion, [polygon_points], -1, (0, 0, 255), 2)
        
        # DIBUJAR LÍNEAS HORIZONTALES
        h_line_names = ['I', 'M', 'F']
        for name in h_line_names:
            datos = resultados_IMF["horizontal"][name]
            if datos["puntos"] is not None:
                left_point, right_point = datos["puntos"]
                
                # Convertir a enteros
                left_point = (int(left_point[0]), int(left_point[1]))
                right_point = (int(right_point[0]), int(right_point[1]))
                
                # Dibujar la línea horizontal
                cv2.line(img_visualizacion, left_point, right_point, (0, 0, 255), 2)
                
                # Agregar etiqueta
                mid_x = int((left_point[0] + right_point[0]) / 2)
                mid_y = int(left_point[1])  # La Y es constante para líneas horizontales
                
                # Texto con dimensión en píxeles
                texto = f"H-{name}: {datos['longitud']:.1f} px"
                
                # Fondo para mejor visibilidad
                text_size, _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img_visualizacion, 
                            (mid_x - text_size[0] // 2 - 5, mid_y - 25 - text_size[1]), 
                            (mid_x + text_size[0] // 2 + 5, mid_y - 25 + 5), 
                            (255, 255, 255), -1)
                
                # Texto centrado
                cv2.putText(img_visualizacion, texto, 
                          (mid_x - text_size[0] // 2, mid_y - 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # DIBUJAR LÍNEAS VERTICALES
        v_line_names = ['I', 'M', 'F']
        for name in v_line_names:
            datos = resultados_IMF["vertical"][name]
            if datos["puntos"] is not None:
                top_point, bottom_point = datos["puntos"]
                
                # Convertir a enteros
                top_point = (int(top_point[0]), int(top_point[1]))
                bottom_point = (int(bottom_point[0]), int(bottom_point[1]))
                
                # Dibujar la línea vertical
                cv2.line(img_visualizacion, top_point, bottom_point, (255, 0, 0), 2)
                
                # Agregar etiqueta
                mid_x = int(top_point[0])  # La X es constante para líneas verticales
                mid_y = int((top_point[1] + bottom_point[1]) / 2)
                
                # Texto con dimensión en píxeles
                texto = f"V-{name}: {datos['longitud']:.1f} px"
                
                # Fondo para mejor visibilidad
                text_size, _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img_visualizacion, 
                            (mid_x + 10, mid_y - text_size[1] // 2 - 5), 
                            (mid_x + 10 + text_size[0], mid_y + text_size[1] // 2 + 5), 
                            (255, 255, 255), -1)
                
                # Texto
                cv2.putText(img_visualizacion, texto, 
                          (mid_x + 10, mid_y + 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Etiquetas para las líneas verticales en la parte inferior
        for name in v_line_names:
            datos = resultados_IMF["vertical"][name]
            if datos["posicion"] is not None:
                # Texto
                x_pos = int(datos["posicion"])
                texto = name
                
                # Fondo para mejor visibilidad
                text_size, _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(img_visualizacion, 
                            (x_pos - text_size[0] // 2 - 5, max_y + 30 - text_size[1]), 
                            (x_pos + text_size[0] // 2 + 5, max_y + 30 + 5), 
                            (255, 255, 255), -1)
                
                # Texto centrado
                cv2.putText(img_visualizacion, texto, 
                          (x_pos - text_size[0] // 2, max_y + 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Título con dimensiones solo en pixeles
        titulo = f"Analisis IMF - Dimensiones: {width:.1f} x {height:.1f} px"
        
        # Añadir título a la imagen
        cv2.putText(img_visualizacion, titulo, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        return img_visualizacion
    
    def generate_report(self, image_name, imf_data, output_dir):
        """
        Genera un informe con las medidas IMF
        
        Args:
            image_name: Nombre de la imagen original
            imf_data: Datos del análisis IMF
            output_dir: Directorio donde guardar el informe
        
        Returns:
            report_paths: Rutas a los archivos de informe generados
        """
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Extraer datos para el informe
        horizontal_data = {}
        for pos in ['I', 'M', 'F']:
            h_data = imf_data["horizontal"][pos]
            if h_data["longitud"] is not None:
                horizontal_data[f"{pos}_horizontal_px"] = h_data["longitud"]
                # Eliminar la conversión a mm
        
        vertical_data = {}
        for pos in ['I', 'M', 'F']:
            v_data = imf_data["vertical"][pos]
            if v_data["longitud"] is not None:
                vertical_data[f"{pos}_vertical_px"] = v_data["longitud"]
                # Eliminar la conversión a mm
        
        # Añadir dimensiones del rectángulo mínimo (solo en píxeles)
        dimensiones = {
            "ancho_px": imf_data["rectangulo_minimo"]["tamano"][0],
            "alto_px": imf_data["rectangulo_minimo"]["tamano"][1],
            # Eliminar dimensiones en mm
        }
        
        # Combinar todos los datos
        report_data = {**horizontal_data, **vertical_data, **dimensiones}
        
        # Crear DataFrame para guardar como CSV
        df = pd.DataFrame([report_data])
        
        # Formato del informe
        report_path = os.path.join(output_dir, f"{image_name}_imf_report.csv")
        
        # Guardar como CSV
        df.to_csv(report_path, index=False)
        
        # También generar una versión en formato de texto para fácil visualización
        text_report_path = os.path.join(output_dir, f"{image_name}_imf_report.txt")
        
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write(f"INFORME DE ANÁLISIS IMF (INICIAL-MEDIO-FINAL) - {image_name}\n")
            f.write("="*60 + "\n\n")
            
            f.write("DIMENSIONES DE LA PALANQUILLA:\n")
            f.write(f"  Ancho: {dimensiones['ancho_px']:.1f} px\n")
            f.write(f"  Alto: {dimensiones['alto_px']:.1f} px\n\n")
            
            f.write("MEDIDAS HORIZONTALES:\n")
            for pos in ['I', 'M', 'F']:
                key_px = f"{pos}_horizontal_px"
                if key_px in horizontal_data:
                    f.write(f"  Línea H-{pos}: {horizontal_data[key_px]:.1f} px\n")
            f.write("\n")
            
            f.write("MEDIDAS VERTICALES:\n")
            for pos in ['I', 'M', 'F']:
                key_px = f"{pos}_vertical_px"
                if key_px in vertical_data:
                    f.write(f"  Línea V-{pos}: {vertical_data[key_px]:.1f} px\n")
            f.write("\n")
            
            f.write("RELACIÓN PÍXEL/MM:\n")
            f.write(f"  Horizontal: {imf_data['rectangulo_minimo']['px_por_mm'][0]:.4f} px/mm\n")
            f.write(f"  Vertical: {imf_data['rectangulo_minimo']['px_por_mm'][1]:.4f} px/mm\n")
        
        print(f"Informe IMF generado en: {report_path}")
        print(f"Informe de texto IMF generado en: {text_report_path}")
        
        return report_path, text_report_path
    
    def process(self, image, contorno, mask=None, image_name=None, output_dir=None, pixel_per_mm=None):
        """
        Procesa la imagen para realizar análisis IMF
        
        Args:
            image: Imagen original
            contorno: Esquinas de la palanquilla [top-left, top-right, bottom-right, bottom-left]
            mask: Máscara binaria de la palanquilla (opcional)
            image_name: Nombre de la imagen (sin extensión)
            output_dir: Directorio de salida para guardar reportes
            pixel_per_mm: Relación píxel/mm (opcional)
                
        Returns:
            processed_data: Diccionario con los resultados del procesamiento
        """
        try:
            # Verificar que los contornos sean válidos
            if contorno is None or len(contorno) < 4:
                print("Error: Se requiere un contorno válido para el análisis IMF")
                return None
            
            # Asegurarnos de tener un contorno_principal
            contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                contorno_principal = max(contornos, key=cv2.contourArea)
            else:
                # Si no hay contornos en la máscara, crear uno desde las esquinas
                contorno_principal = np.array(contorno).reshape(-1, 1, 2)
            
            # Calcular IMF
            resultados_IMF = self.calcular_IMF(contorno_principal)
            
            # Crear visualización
            img_IMF = self.visualizar_IMF(image, contorno_principal, resultados_IMF, mask)
            
            # Obtener resultados para el informe
            processed_data = {
                'resultados_IMF': resultados_IMF,
                'img_IMF': img_IMF
            }
            
            visualizations = {
                'imf_global': img_IMF
            }
            
            # Si hay un nombre de imagen y un directorio de salida, generar un reporte
            report_paths = None
            if image_name and output_dir:
                # Crear directorio para este tipo de análisis
                defect_dir = os.path.join(output_dir, image_name, self.name)
                os.makedirs(defect_dir, exist_ok=True)
                
                # Guardar la visualización
                viz_path = os.path.join(defect_dir, f"{image_name}_imf_visualization.jpg")
                cv2.imwrite(viz_path, img_IMF)
                
                # Generar el reporte
                report_paths = self.generate_report(image_name, resultados_IMF, defect_dir)
            
            return {
                'processed_data': processed_data,
                'visualizations': visualizations,
                'report_paths': report_paths
            }
            
        except Exception as e:
            import traceback
            print(f"Error en el procesamiento IMF: {e}")
            traceback.print_exc()
            return None