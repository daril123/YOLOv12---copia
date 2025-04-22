import cv2
import numpy as np
import os
import pandas as pd

class RomboidadProcessor:
    """
    Procesador para medir la romboidad (diferencia entre diagonales) de las palanquillas
    """
    
    def __init__(self):
        """
        Inicializa el procesador de romboidad
        """
        self.name = "romboidad"
    
    def measure_romboidad(self, image, corners, visualize=True):
        """
        Mide la romboidad de la palanquilla calculando la diferencia entre las diagonales
        
        Args:
            image: Imagen original
            corners: Esquinas de la palanquilla [top-left, top-right, bottom-right, bottom-left]
            visualize: Si es True, genera una visualización
            
        Returns:
            metrics: Diccionario con las métricas de romboidad
        """
        # Hacer una copia de la imagen para visualización
        viz_img = image.copy() if visualize else None
        
        # Convertir a arreglo numpy para facilitar operaciones
        corners_np = np.array(corners)
        
        # Calcular el centro de la palanquilla
        center = np.mean(corners_np, axis=0).astype(int)
        
        # Calcular las diagonales
        diagonal1 = np.sqrt((corners[0][0] - corners[2][0])**2 + (corners[0][1] - corners[2][1])**2)  # top-left a bottom-right
        diagonal2 = np.sqrt((corners[1][0] - corners[3][0])**2 + (corners[1][1] - corners[3][1])**2)  # top-right a bottom-left
        
        # Determinar cuál es la diagonal mayor y cuál la menor
        if diagonal1 >= diagonal2:
            d_mayor = diagonal1
            d_menor = diagonal2
            corners_diagonal_mayor = (corners[0], corners[2])
            corners_diagonal_menor = (corners[1], corners[3])
        else:
            d_mayor = diagonal2
            d_menor = diagonal1
            corners_diagonal_mayor = (corners[1], corners[3])
            corners_diagonal_menor = (corners[0], corners[2])
        
        # Calcular la diferencia entre diagonales
        diferencia = d_mayor - d_menor
        
        # Preparar resultados (sin clasificación por niveles)
        results = {
            'diagonal_mayor': round(d_mayor, 2),
            'diagonal_menor': round(d_menor, 2),
            'diferencia': round(diferencia, 2),
            'corners_diagonal_mayor': corners_diagonal_mayor,
            'corners_diagonal_menor': corners_diagonal_menor
        }
        
        # Visualización
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
            
            # Dibujar la diagonal mayor (D) en blanco
            cv2.line(viz_img, 
                    corners_diagonal_mayor[0], 
                    corners_diagonal_mayor[1], 
                    (255, 255, 255), 2)
            
            # Dibujar la diagonal menor (d) en amarillo
            cv2.line(viz_img, 
                    corners_diagonal_menor[0], 
                    corners_diagonal_menor[1], 
                    (0, 255, 255), 2)
            
            # Calcular puntos para las etiquetas de las diagonales (cerca del centro pero no sobrepuesto)
            label_offset = 30
            d_mayor_label_x = center[0] + label_offset
            d_mayor_label_y = center[1] + label_offset
            d_menor_label_x = center[0] - label_offset
            d_menor_label_y = center[1] - label_offset
            
            # Etiquetas para las diagonales
            cv2.putText(viz_img, f"D={d_mayor:.2f}px", 
                       (d_mayor_label_x, d_mayor_label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(viz_img, f"d={d_menor:.2f}px", 
                       (d_menor_label_x, d_menor_label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Título principal
            cv2.putText(viz_img, "Análisis de Romboidad", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Información sobre la diferencia
            cv2.putText(viz_img, f"Diferencia (D-d): {diferencia:.2f}px", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Guardar visualización temporalmente
            cv2.imwrite("temp_romboidad_analysis.jpg", viz_img)
        
        return {
            'resultados': results,
            'visualization': viz_img if visualize else None
        }
    
    def generate_report(self, image_name, romboidad_data, output_dir):
        """
        Genera un informe de la romboidad
        
        Args:
            image_name: Nombre de la imagen original
            romboidad_data: Datos de la romboidad
            output_dir: Directorio donde guardar el informe
        
        Returns:
            report_paths: Rutas a los archivos de informe generados
        """
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Extraer los resultados
        resultados = romboidad_data['resultados']
        
        # Crear DataFrame para el informe
        df = pd.DataFrame({
            'Diagonal_Mayor_D(px)': [resultados['diagonal_mayor']],
            'Diagonal_Menor_d(px)': [resultados['diagonal_menor']],
            'Diferencia(px)': [resultados['diferencia']]
        })
        
        # Formato del informe
        report_path = os.path.join(output_dir, f"{image_name}_romboidad_report.csv")
        
        # Guardar como CSV
        df.to_csv(report_path, index=False)
        
        # También generar una versión en formato de texto para fácil visualización
        text_report_path = os.path.join(output_dir, f"{image_name}_romboidad_report.txt")
        
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write(f"REPORTE DE ROMBOIDAD - {image_name}\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Diagonal Mayor (D): {resultados['diagonal_mayor']} píxeles\n")
            f.write(f"Diagonal Menor (d): {resultados['diagonal_menor']} píxeles\n")
            f.write(f"Diferencia (D-d): {resultados['diferencia']} píxeles\n\n")
        
        print(f"Reporte generado en: {report_path}")
        print(f"Reporte de texto generado en: {text_report_path}")
        
        # Guardar la visualización si existe
        if 'visualization' in romboidad_data and romboidad_data['visualization'] is not None:
            viz_path = os.path.join(output_dir, f"{image_name}_romboidad_visualization.jpg")
            cv2.imwrite(viz_path, romboidad_data['visualization'])
            print(f"Visualización guardada en: {viz_path}")
            return report_path, text_report_path, viz_path
        
        return report_path, text_report_path
    
    def process(self, image, corners, image_name=None, output_dir=None):
        """
        Procesa la imagen para detectar romboidad
        
        Args:
            image: Imagen original
            corners: Esquinas de la palanquilla
            image_name: Nombre de la imagen (sin extensión)
            output_dir: Directorio de salida para guardar reportes
            
        Returns:
            processed_data: Diccionario con los resultados del procesamiento
        """
        # Medir la romboidad
        results = self.measure_romboidad(image, corners)
        
        visualizations = {}
        
        # Agregar visualización si existe
        if 'visualization' in results and results['visualization'] is not None:
            visualizations['romboidad_global'] = results['visualization']
        
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