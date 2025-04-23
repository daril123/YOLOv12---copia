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
        Mide el abombamiento (concavidad/convexidad) de la palanquilla utilizando
        los bordes de la palanquilla como referencia, no los bordes de la imagen.
        
        Args:
            image: Imagen original
            corners: Esquinas de la palanquilla [top-left, top-right, bottom-right, bottom-left]
            visualize: Si es True, genera una visualización
            mask: Máscara de segmentación de la palanquilla (opcional)
            
        Returns:
            metrics: Diccionario con las métricas de abombamiento para cada lado
        """
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
        
        # Calcular el centro de la palanquilla
        center = np.mean(corners_np, axis=0).astype(int)
        
        if visualize:
            # Dibujar el contorno de la palanquilla
            cv2.polylines(viz_img, [corners_np], True, (0, 255, 0), 2)
            
            # Dibujar y numerar las esquinas
            for i, corner in enumerate(corners):
                cv2.circle(viz_img, tuple(corner), 8, (0, 0, 255), -1)
                cv2.putText(viz_img, str(i+1), (corner[0]-4, corner[1]+4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Dibujar el centro
            cv2.circle(viz_img, tuple(center), 8, (255, 255, 0), -1)
            cv2.putText(viz_img, "C", (center[0]-4, center[1]+4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Crear una máscara binaria de la palanquilla
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [corners_np], 255)
        
        # Extraer el contorno de la palanquilla usando la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Debe haber al menos un contorno
        if not contours:
            print("No se encontraron contornos en la máscara de la palanquilla")
            return {
                'resultados': {},
                'visualization': viz_img if visualize else None
            }
        
        # Tomar el contorno más grande (debe ser la palanquilla)
        contour = max(contours, key=cv2.contourArea)
        
        # Procesar cada lado de la palanquilla
        for i in range(4):
            # Obtener puntos para este lado
            p1 = corners[i]
            p2 = corners[(i+1) % 4]
            
            # Calcular el punto medio del lado
            mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            
            # Calcular la longitud nominal del lado (distancia entre esquinas)
            nominal_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Vector unitario perpendicular al lado
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx**2 + dy**2)
            
            # Vector perpendicular (rotación de 90 grados hacia adentro)
            # Asegurar que el vector apunte hacia el interior de la palanquilla
            nx, ny = -dy/length, dx/length
            
            # Verificar que el vector perpendicular apunte hacia el centro
            mid_to_center = np.array([center[0] - mid_point[0], center[1] - mid_point[1]])
            dot_product = nx * mid_to_center[0] + ny * mid_to_center[1]
            
            # Si el producto punto es negativo, el vector apunta hacia afuera, invertirlo
            if dot_product < 0:
                nx, ny = -nx, -ny
            
            # Encontrar el punto de máxima desviación para este lado
            max_deviation = 0
            max_dev_point = None
            proj_point = None
            
            # Comprobar cada punto del contorno para este lado
            for point in contour[:, 0, :]:
                # Calcular distancia del punto al segmento
                # Utilizamos la fórmula de la distancia de un punto a una línea
                px, py = point
                x1, y1 = p1
                x2, y2 = p2
                
                # Proyección del punto en la línea (parámetro t de 0 a 1)
                t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (nominal_length ** 2)))
                
                # Punto proyectado en el segmento
                proj_x = x1 + t * (x2 - x1)
                proj_y = y1 + t * (y2 - y1)
                
                # Distancia entre el punto y su proyección
                dev = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
                
                # Determinar si la desviación es hacia adentro o hacia afuera
                # Vector desde proyección al punto contorno
                vx, vy = px - proj_x, py - proj_y
                
                # Producto punto con el vector normal
                dot = vx * nx + vy * ny
                
                # Si el producto punto es positivo, la desviación es hacia el interior
                # Si es negativo, es hacia el exterior
                signed_dev = dev if dot >= 0 else -dev
                
                # Buscar puntos cerca del punto medio (t cercano a 0.5)
                # y actualizar si encontramos una desviación mayor
                if abs(t - 0.5) < 0.1 and abs(signed_dev) > abs(max_deviation):
                    max_deviation = signed_dev
                    max_dev_point = (int(px), int(py))
                    proj_point = (int(proj_x), int(proj_y))
            
            # Calcular el porcentaje de abombamiento (C)
            C = (max_deviation / nominal_length) * 100.0
            
            # Almacenar resultados
            results[side_names[i]] = {
                'C': round(C, 2),
                'X': round(max_deviation, 2),
                'nominal': round(nominal_length, 2),
                'mid_point': mid_point,
                'max_dev_point': max_dev_point,
                'proj_point': proj_point
            }
            
            if visualize and max_dev_point and proj_point:
                # Dibujar el punto medio del lado
                cv2.circle(viz_img, mid_point, 6, colors[i], -1)
                
                # Dibujar el punto de máxima desviación y la línea al punto proyectado
                cv2.circle(viz_img, max_dev_point, 6, colors[i], -1)
                
                # Dibujar la línea que muestra la desviación (X)
                cv2.line(viz_img, max_dev_point, proj_point, colors[i], 2)
                
                # Anotar el valor del abombamiento
                label_point = ((max_dev_point[0] + proj_point[0]) // 2, 
                              (max_dev_point[1] + proj_point[1]) // 2)
                
                cv2.putText(viz_img, f"X={abs(max_deviation):.1f}px", 
                           (label_point[0] + 10, label_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
                
                cv2.putText(viz_img, f"C={C:.2f}%", 
                           (label_point[0] + 10, label_point[1] + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
        
        # Crear visualización global
        if visualize:
            # Título principal
            cv2.putText(viz_img, "Análisis de Abombamiento (Convexidad/Concavidad)", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Leyenda con los valores en colores correspondientes
            y_offset = 70
            for i, side in enumerate(side_names):
                result = results[side]
                # Mostrar valores con signo para indicar convexidad/concavidad
                text = f"{side}: C={result['C']}% (X={result['X']}px, Nominal={result['nominal']}px)"
                cv2.putText(viz_img, text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
                y_offset += 30
            
            # Guardar visualización temporalmente
            cv2.imwrite("temp_abombamiento_analysis.jpg", viz_img)
        
        return {
            'resultados': results,
            'visualization': viz_img if visualize else None
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
        # Medir el abombamiento
        results = self.measure_abombamiento(image, corners, mask=mask)
        
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
            report_paths = self.generate_report(image_name, results, defect_dir)
        
        return {
            'processed_data': results['resultados'],
            'visualizations': visualizations,
            'report_paths': report_paths
        }