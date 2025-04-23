import cv2
import numpy as np
import os
import pandas as pd

class AbombamientoProcessor:
    """
    Procesador para medir el abombamiento (concavidad/convexidad) de las palanquillas
    """
    
    def __init__(self):
        """
        Inicializa el procesador de abombamiento
        """
        self.name = "abombamiento"
    
    def measure_abombamiento(self, image, corners, visualize=True, mask=None):
        """
        Mide el abombamiento (concavidad/convexidad) de la palanquilla calculando
        la distancia perpendicular máxima desde el contorno real hasta la línea nominal.
        
        Args:
            image: Imagen original
            corners: Esquinas de la palanquilla [top-left, top-right, bottom-right, bottom-left]
            visualize: Si es True, genera una visualización
            mask: Máscara de segmentación de la palanquilla
                
        Returns:
            metrics: Diccionario con las métricas de abombamiento para cada lado
        """
        try:
            # Hacer una copia de la imagen para visualización
            viz_img = image.copy() if visualize else None
            
            # Definir colores para visualización (BGR)
            colors = [(0, 0, 255),   # Rojo - Superior
                    (0, 255, 255),  # Amarillo - Derecho
                    (0, 255, 0),    # Verde - Inferior
                    (255, 0, 0)]    # Azul - Izquierdo
            
            # Nombres de los lados
            side_names = ["Superior", "Derecho", "Inferior", "Izquierdo"]
            
            # Valores para cada lado (C%)
            results = {}
            
            # Convertir a arreglo numpy para facilitar operaciones
            corners_np = np.array(corners)
            
            # Verificar que corners_np tenga la forma correcta
            if corners_np.shape != (4, 2):
                print(f"Advertencia: corners_np tiene una forma inesperada: {corners_np.shape}")
                # Intentar corregir
                if len(corners_np.shape) > 2:
                    corners_np = corners_np.reshape(4, 2)
                elif corners_np.shape[0] != 4:
                    # Si no tenemos 4 puntos, crear un rectángulo predeterminado
                    h, w = image.shape[:2]
                    corners_np = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="int")
            
            # Crear una máscara binaria si no se proporciona una
            if mask is None:
                print("Creando nueva máscara a partir de los vértices")
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [corners_np], 255)
            elif mask.dtype != np.uint8:
                # Asegurarse de que la máscara sea uint8
                print(f"Convirtiendo máscara de tipo {mask.dtype} a uint8")
                mask = mask.astype(np.uint8)
            
            # Si la máscara tiene valores distintos a 0 y 255, binarizarla
            if np.max(mask) == 0:
                print("¡Advertencia! Máscara completamente negra (todos los valores son 0)")
                # Crear una máscara desde los vértices en este caso
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [corners_np], 255)
            elif np.max(mask) != 255 or (np.min(mask[mask > 0]) != 255):
                print(f"Binarizando máscara. Valores actuales: min={np.min(mask)}, max={np.max(mask)}")
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Guardar la máscara para diagnóstico
            cv2.imwrite("debug_mask.png", mask)
            
            # Suavizar ligeramente la máscara para eliminar ruido
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Extraer el contorno detallado de la palanquilla usando la máscara
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            print(f"Número de contornos encontrados: {len(contours)}")
            
            # Debe haber al menos un contorno
            if not contours:
                print("No se encontraron contornos en la máscara de la palanquilla")
                return {
                    'resultados': {},
                    'visualization': viz_img if visualize else None
                }
            
            # Tomar el contorno más grande (debe ser la palanquilla)
            contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(contour)
            print(f"Área del contorno principal: {contour_area}")
            
            if contour_area < 1000:  # Valor arbitrario para un contorno muy pequeño
                print("¡Advertencia! El contorno detectado es muy pequeño.")
            
            # Para visualización, dibujar el contorno real de la palanquilla
            if visualize:
                # Dibujar el contorno completo detectado
                cv2.drawContours(viz_img, [contour], 0, (0, 255, 0), 2)
                
                # Dibujar y numerar las esquinas
                for i, corner in enumerate(corners):
                    cv2.circle(viz_img, tuple(corner), 8, (0, 0, 255), -1)
                    cv2.putText(viz_img, str(i+1), (corner[0]-4, corner[1]+4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Inicializar resultados por defecto en caso de errores
            default_results = {}
            for i, side_name in enumerate(side_names):
                default_results[side_name] = {
                    'C': 0.0,
                    'X': 0.0,
                    'nominal': 0.0,
                    'max_dev_point': None,
                    'proj_point': None
                }
            
            # Convertir el contorno a puntos
            contour_points = contour.reshape(-1, 2)
            
            # Procesar cada lado de la palanquilla
            for i in range(4):
                try:
                    print(f"Procesando lado: {side_names[i]}")
                    # Obtener puntos para este lado (vértices de las esquinas)
                    p1 = tuple(corners[i].astype(int))
                    p2 = tuple(corners[(i+1) % 4].astype(int))
                    
                    print(f"  Vértices del lado: p1={p1}, p2={p2}")
                    
                    # Calcular la longitud nominal del lado (distancia entre esquinas)
                    nominal_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    
                    if nominal_length < 1:
                        print(f"  ¡Advertencia! Longitud nominal muy pequeña: {nominal_length}")
                        results[side_names[i]] = default_results[side_names[i]]
                        continue
                    
                    # Calcular los coeficientes de la ecuación de la línea ax + by + c = 0
                    a = p2[1] - p1[1]
                    b = p1[0] - p2[0]
                    c = p2[0] * p1[1] - p1[0] * p2[1]
                    
                    # Normalizar 
                    line_length = np.sqrt(a**2 + b**2)
                    if line_length < 0.001:  # Evitar división por cero
                        print(f"  ¡Advertencia! Longitud de línea casi cero: {line_length}")
                        results[side_names[i]] = default_results[side_names[i]]
                        continue
                    
                    # Vector normal unitario a la línea (perpendicular)
                    nx = a / line_length
                    ny = b / line_length
                    
                    # Verificar si el vector normal apunta hacia dentro de la palanquilla
                    center = np.mean(corners_np, axis=0).astype(int)
                    mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    
                    # Vector desde el punto medio del lado hacia el centro
                    mid_to_center = np.array([center[0] - mid_point[0], center[1] - mid_point[1]])
                    
                    # Producto punto con el vector normal
                    dot_product = nx * mid_to_center[0] + ny * mid_to_center[1]
                    
                    # Si el vector normal apunta hacia fuera, invertirlo
                    if dot_product < 0:
                        nx = -nx
                        ny = -ny
                    
                    # Para cada punto del contorno, calcular la distancia a la línea
                    max_distance = 0
                    max_point = None
                    max_proj_point = None
                    
                    for point in contour_points:
                        # Filtrar puntos que están muy lejos de este lado
                        # Calcular distancia entre el punto y cada vértice del lado
                        dist_to_p1 = np.sqrt((point[0] - p1[0])**2 + (point[1] - p1[1])**2)
                        dist_to_p2 = np.sqrt((point[0] - p2[0])**2 + (point[1] - p2[1])**2)
                        
                        # Si el punto está muy lejos de ambos vértices, probablemente no pertenece a este lado
                        # Usamos un umbral basado en la longitud nominal
                        if dist_to_p1 > nominal_length * 1.5 and dist_to_p2 > nominal_length * 1.5:
                            continue
                        
                        # Calcular la distancia perpendicular (con signo)
                        # d = (ax + by + c) / √(a² + b²)
                        px, py = point
                        distance = (a * px + b * py + c) / line_length
                        
                        # El signo de la distancia indica si está dentro o fuera
                        # Queremos tomar el valor absoluto para encontrar el punto más lejano
                        abs_distance = abs(distance)
                        
                        # Calcular el punto proyectado en la línea
                        proj_x = px - nx * distance
                        proj_y = py - ny * distance
                        
                        # Verificar si el punto proyectado está en el segmento
                        t = 0
                        if abs(p2[0] - p1[0]) > abs(p2[1] - p1[1]):
                            # La línea es más horizontal
                            t = (proj_x - p1[0]) / (p2[0] - p1[0])
                        else:
                            # La línea es más vertical
                            t = (proj_y - p1[1]) / (p2[1] - p1[1])
                        
                        # Solo considerar puntos cuya proyección está dentro del segmento
                        if 0 <= t <= 1:
                            # Actualizar si encontramos una distancia mayor
                            if abs_distance > max_distance:
                                max_distance = abs_distance
                                max_point = (int(px), int(py))
                                max_proj_point = (int(proj_x), int(proj_y))
                    
                    # Si no encontramos ningún punto válido, usar valores por defecto
                    if max_point is None:
                        print("  No se encontró un punto de desviación máxima válido")
                        max_distance = 0
                        max_point = mid_point
                        max_proj_point = mid_point
                    
                    # Calcular el porcentaje de abombamiento (C)
                    # C = (X / nominal) * 100%
                    C = (max_distance / nominal_length) * 100.0
                    
                    print(f"  Resultados: C={C:.2f}%, X={max_distance:.2f}px, Nominal={nominal_length:.2f}px")
                    
                    # Almacenar resultados
                    results[side_names[i]] = {
                        'C': round(C, 2),
                        'X': round(max_distance, 2),
                        'nominal': round(nominal_length, 2),
                        'max_dev_point': max_point,
                        'proj_point': max_proj_point
                    }
                    
                    # Visualizar resultados para este lado
                    if visualize and max_point and max_proj_point:
                        # Dibujar la línea recta del lado (nominal)
                        cv2.line(viz_img, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)  # Línea blanca
                        
                        # Dibujar el punto de máxima desviación
                        cv2.circle(viz_img, max_point, 8, colors[i], -1)
                        
                        # Dibujar la línea que muestra la desviación X (perpendicular)
                        cv2.line(viz_img, max_point, max_proj_point, (0, 255, 255), 2)  # Línea amarilla
                        
                        # Dibujar flechas bidireccionales para la distancia X
                        arrow_size = 10
                        midpoint = ((max_point[0] + max_proj_point[0]) // 2, 
                                    (max_point[1] + max_proj_point[1]) // 2)
                        
                        # Vector unitario desde el punto proyectado hacia el punto del contorno
                        dx = max_point[0] - max_proj_point[0]
                        dy = max_point[1] - max_proj_point[1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist > 0:
                            dx = dx / dist
                            dy = dy / dist
                        else:
                            dx, dy = 0, 0
                        
                        # Puntas de flecha
                        cv2.line(viz_img, midpoint, 
                                (int(midpoint[0] + arrow_size * dx), int(midpoint[1] + arrow_size * dy)), 
                                (0, 255, 255), 2)  # Flecha hacia el punto del contorno
                        cv2.line(viz_img, midpoint, 
                                (int(midpoint[0] - arrow_size * dx), int(midpoint[1] - arrow_size * dy)), 
                                (0, 255, 255), 2)  # Flecha hacia el punto proyectado
                        
                        # Anotar el valor de X cerca del punto de máxima desviación
                        label_x = midpoint[0] + int(20 * dx)
                        label_y = midpoint[1] + int(20 * dy)
                        
                        # Texto de las etiquetas
                        x_text = f"X={max_distance:.1f}px"
                        
                        # Añadir borde negro para mejor visibilidad
                        cv2.putText(viz_img, x_text, (label_x, label_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                        cv2.putText(viz_img, x_text, (label_x, label_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Texto amarillo
                except Exception as e:
                    print(f"  Error procesando lado {side_names[i]}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Usar valores predeterminados para este lado
                    results[side_names[i]] = default_results[side_names[i]]
            
            # Crear visualización global
            if visualize:
                # Título principal
                cv2.putText(viz_img, "Análisis de Abombamiento (Convexidad/Concavidad)", 
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Leyenda con los valores en colores correspondientes
                y_offset = 70
                for i, side in enumerate(side_names):
                    if side in results:
                        result = results[side]
                        # Mostrar valores con signo para indicar convexidad/concavidad
                        text = f"{side}: C={result['C']}% (X={result['X']}px, Nominal={result['nominal']}px)"
                        cv2.putText(viz_img, text, (20, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
                        y_offset += 30
                
                # Guardar visualización
                cv2.imwrite("temp_abombamiento_analysis.jpg", viz_img)
            
            return {
                'resultados': results,
                'visualization': viz_img if visualize else None
            }
        
        except Exception as e:
            print(f"Error general en measure_abombamiento: {e}")
            import traceback
            traceback.print_exc()
            
            # Crear resultados por defecto
            default_results = {}
            for side_name in ["Superior", "Derecho", "Inferior", "Izquierdo"]:
                default_results[side_name] = {
                    'C': 0.0,
                    'X': 0.0,
                    'nominal': 0.0
                }
            
            return {
                'resultados': default_results,
                'visualization': None
            }
    
    def generate_report(self, image_name, abombamiento_data, output_dir):
        """
        Genera un informe del abombamiento
        
        Args:
            image_name: Nombre de la imagen original
            abombamiento_data: Datos del abombamiento
            output_dir: Directorio donde guardar el informe
        
        Returns:
            report_paths: Rutas a los archivos de informe generados
        """
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Extraer los resultados
        resultados = abombamiento_data['resultados']
        
        # Crear DataFrame para el informe
        df = pd.DataFrame({
            'Lado': list(resultados.keys()),
            'Abombamiento_C(%)': [res['C'] for res in resultados.values()],
            'Desvio_X(px)': [res['X'] for res in resultados.values()],
            'Longitud_Nominal(px)': [res['nominal'] for res in resultados.values()]
        })
        
        # Formato del informe
        report_path = os.path.join(output_dir, f"{image_name}_abombamiento_report.csv")
        
        # Guardar como CSV
        df.to_csv(report_path, index=False)
        
        # También generar una versión en formato de texto para fácil visualización
        text_report_path = os.path.join(output_dir, f"{image_name}_abombamiento_report.txt")
        
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write(f"REPORTE DE ABOMBAMIENTO - {image_name}\n")
            f.write("="*50 + "\n\n")
            
            for lado, datos in resultados.items():
                f.write(f"LADO: {lado}\n")
                f.write(f"  Abombamiento (C): {datos['C']}%\n")
                f.write(f"  Desviación (X): {datos['X']} píxeles\n")
                f.write(f"  Longitud nominal: {datos['nominal']} píxeles\n")
                f.write("\n")
                
            # Añadir información sobre la garantía
            f.write("\nNOTA: El valor de garantía para el abombamiento (C) no debe ser mayor a ±1.5%\n")
        
        print(f"Reporte generado en: {report_path}")
        print(f"Reporte de texto generado en: {text_report_path}")
        
        # Guardar la visualización si existe
        if 'visualization' in abombamiento_data and abombamiento_data['visualization'] is not None:
            viz_path = os.path.join(output_dir, f"{image_name}_abombamiento_visualization.jpg")
            cv2.imwrite(viz_path, abombamiento_data['visualization'])
            print(f"Visualización guardada en: {viz_path}")
            return report_path, text_report_path, viz_path
        
        return report_path, text_report_path
    
    def process(self, image, corners, image_name=None, output_dir=None, mask=None):
        """
        Procesa la imagen para detectar abombamiento
        
        Args:
            image: Imagen original
            corners: Esquinas de la palanquilla
            image_name: Nombre de la imagen (sin extensión)
            output_dir: Directorio de salida para guardar reportes
            mask: Máscara de segmentación de la palanquilla (opcional)
            
        Returns:
            processed_data: Diccionario con los resultados del procesamiento
        """
        try:
            # Código de diagnóstico para verificar si la máscara está llegando correctamente
            if mask is not None:
                print(f"Recibida máscara de forma: {mask.shape}, tipo: {mask.dtype}")
                # Guardar la máscara para verificación visual
                if image_name and output_dir:
                    mask_dir = os.path.join(output_dir, image_name, self.name)
                    os.makedirs(mask_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(mask_dir, f"{image_name}_mask.jpg"), mask)
            else:
                print("No se recibió máscara, se creará una a partir de los vértices")
            
            # Verificamos que corners sea válido
            if corners is None or len(corners) != 4:
                print(f"Error: Vértices inválidos. Usando bordes de la imagen en su lugar.")
                h, w = image.shape[:2]
                corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="int")
            
            # Asegurarse de que los vértices estén en el formato correcto
            corners = np.array(corners, dtype=np.int32)
                
            # Medir el abombamiento con manejo de errores
            try:
                print("Iniciando medición de abombamiento...")
                results = self.measure_abombamiento(image, corners, mask=mask)
                print("Medición de abombamiento completada")
            except Exception as e:
                print(f"Error durante la medición de abombamiento: {e}")
                import traceback
                traceback.print_exc()
                
                # Crear un resultado vacío para continuar con el procesamiento
                results = {
                    'resultados': {
                        'Superior': {'C': 0.0, 'X': 0.0, 'nominal': 0.0},
                        'Derecho': {'C': 0.0, 'X': 0.0, 'nominal': 0.0},
                        'Inferior': {'C': 0.0, 'X': 0.0, 'nominal': 0.0},
                        'Izquierdo': {'C': 0.0, 'X': 0.0, 'nominal': 0.0}
                    },
                    'visualization': None
                }
            
            visualizations = {}
            
            # Agregar visualización si existe
            if 'visualization' in results and results['visualization'] is not None:
                visualizations['abombamiento_global'] = results['visualization']
            
            # Si hay un nombre de imagen y un directorio de salida, generar un reporte
            report_paths = None
            if image_name and output_dir:
                # Crear directorio para este tipo de defecto
                defect_dir = os.path.join(output_dir, image_name, self.name)
                os.makedirs(defect_dir, exist_ok=True)
                
                # Generar el reporte en la carpeta específica
                try:
                    report_paths = self.generate_report(image_name, results, defect_dir)
                except Exception as e:
                    print(f"Error al generar reporte de abombamiento: {e}")
                    import traceback
                    traceback.print_exc()
                    report_paths = None
            
            return {
                'processed_data': results['resultados'],
                'visualizations': visualizations,
                'report_paths': report_paths
            }
        
        except Exception as e:
            print(f"Error general en el procesamiento de abombamiento: {e}")
            import traceback
            traceback.print_exc()
            
            # Devolver un resultado mínimo para evitar que se detenga todo el pipeline
            return {
                'processed_data': {
                    'Superior': {'C': 0.0, 'X': 0.0, 'nominal': 0.0},
                    'Derecho': {'C': 0.0, 'X': 0.0, 'nominal': 0.0},
                    'Inferior': {'C': 0.0, 'X': 0.0, 'nominal': 0.0},
                    'Izquierdo': {'C': 0.0, 'X': 0.0, 'nominal': 0.0}
                },
                'visualizations': {},
                'report_paths': None
            }