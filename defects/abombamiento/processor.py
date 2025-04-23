import cv2
import numpy as np
import os
import pandas as pd

from utils.utils import draw_arrow
from utils.contorno import obtener_contorno_imagen, rotar_imagen_lado
from .abombamiento import (
    obtener_abombamiento,
    obtener_lado_rotacion_abombamiento,
    visualizar_abombamiento,
    calcular_abombamiento
)

class AbombamientoProcessor:
    """
    Procesador para análisis de abombamiento en palanquillas
    """
    
    def __init__(self):
        """
        Inicializa el procesador de abombamiento
        """
        self.name = "abombamiento"
    
    def measure_abombamiento(self, image, corners, contorno_principal, mask=None, factor_mm_px=1.0):
        """
        Mide el abombamiento en todos los lados de la palanquilla
        
        Args:
            image: Imagen original
            corners: Esquinas de la palanquilla [top-left, top-right, bottom-right, bottom-left]
            contorno_principal: Contorno completo de la palanquilla
            mask: Máscara de la palanquilla (opcional)
            factor_mm_px: Factor de conversión de píxeles a milímetros
            
        Returns:
            metrics: Diccionario con las métricas de abombamiento
        """
        # Convertir corners a array numpy si es necesario
        corners_np = np.array(corners, dtype=np.float32)
        
        try:
            # Obtener los resultados de abombamiento para cada lado
            max_porcentaje, max_abombamiento_pix, lado_max, punto_max, abombamientos_por_lado = obtener_abombamiento(corners_np, contorno_principal)
            
            # Generar resultados completos por lado con el formato C = X / L * 100
            resultados_abombamiento = calcular_abombamiento(image, corners_np, contorno_principal, factor_mm_px)
            
            # Crear visualizaciones
            imagen_resultado = visualizar_abombamiento(image, corners_np, contorno_principal)
            
            # Preparar resultados
            results = {
                'lado_max_abombamiento': lado_max,
                'max_abombamiento_porcentaje': round(max_porcentaje, 2),
                'max_abombamiento_pixeles': round(max_abombamiento_pix, 2),
                'punto_max_abombamiento': punto_max,
                'resultados_por_lado': resultados_abombamiento
            }
            
            # Si hay visualización, añadirla a los resultados
            if imagen_resultado is not None:
                results['visualization'] = imagen_resultado
            
            return results
            
        except Exception as e:
            print(f"Error en measure_abombamiento: {e}")
            import traceback
            traceback.print_exc()
            # Crear resultados por defecto en caso de error
            default_results = {
                'lado_max_abombamiento': "Lado 1 (Top)",
                'max_abombamiento_porcentaje': 0.0,
                'max_abombamiento_pixeles': 0.0,
                'punto_max_abombamiento': None,
                'resultados_por_lado': {
                    "Lado 1 (Top)": {'X_mm': 0.0, 'L_mm': 1.0, 'C_porcentaje': 0.0, 'X_px': 0.0, 'L_px': 1.0},
                    "Lado 2 (Right)": {'X_mm': 0.0, 'L_mm': 1.0, 'C_porcentaje': 0.0, 'X_px': 0.0, 'L_px': 1.0},
                    "Lado 3 (Bottom)": {'X_mm': 0.0, 'L_mm': 1.0, 'C_porcentaje': 0.0, 'X_px': 0.0, 'L_px': 1.0},
                    "Lado 4 (Left)": {'X_mm': 0.0, 'L_mm': 1.0, 'C_porcentaje': 0.0, 'X_px': 0.0, 'L_px': 1.0}
                }
            }
            
            # Intentar crear una visualización básica en caso de error
            if image is not None:
                default_results['visualization'] = image.copy()
            
            return default_results
    
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
        
        try:
            # Extraer los resultados
            lado_max = abombamiento_data.get('lado_max_abombamiento', "Lado 1 (Top)")
            max_abombamiento = abombamiento_data.get('max_abombamiento_porcentaje', 0.0)
            resultados_por_lado = abombamiento_data.get('resultados_por_lado', {})
            
            # Crear DataFrame para el informe
            datos = []
            for lado, valores in resultados_por_lado.items():
                # Verificar que los valores tienen las claves esperadas
                X_mm = valores.get('X_mm', 0.0)
                L_mm = valores.get('L_mm', 1.0)
                C_porcentaje = valores.get('C_porcentaje', 0.0)
                
                datos.append({
                    'Lado': lado,
                    'X_mm': X_mm,
                    'L_mm': L_mm,
                    'Abombamiento_Porcentaje': C_porcentaje
                })
            
            df = pd.DataFrame(datos)
            
            # Formato del informe
            report_path = os.path.join(output_dir, f"{image_name}_abombamiento_report.csv")
            
            # Guardar como CSV
            df.to_csv(report_path, index=False)
            
            # También generar una versión en formato de texto para fácil visualización
            text_report_path = os.path.join(output_dir, f"{image_name}_abombamiento_report.txt")
            
            with open(text_report_path, 'w', encoding='utf-8') as f:
                f.write(f"REPORTE DE ABOMBAMIENTO - {image_name}\n")
                f.write("="*50 + "\n\n")
                
                for lado, valores in resultados_por_lado.items():
                    f.write(f"{lado}:\n")
                    f.write(f"  Distancia de anormalidad (X): {valores.get('X_mm', 0.0):.2f} mm\n")
                    f.write(f"  Longitud nominal (L): {valores.get('L_mm', 1.0):.2f} mm\n")
                    f.write(f"  Abombamiento (C = X/L*100): {valores.get('C_porcentaje', 0.0):.2f}%\n\n")
                
                f.write("RESUMEN:\n")
                f.write(f"  Lado con mayor abombamiento: {lado_max}\n")
                f.write(f"  Valor de abombamiento: {max_abombamiento:.2f}%\n")
            
            print(f"Reporte generado en: {report_path}")
            print(f"Reporte de texto generado en: {text_report_path}")
            
            # Guardar la visualización si existe
            viz_path = None
            if 'visualization' in abombamiento_data:
                viz_path = os.path.join(output_dir, f"{image_name}_abombamiento_visualization.jpg")
                cv2.imwrite(viz_path, abombamiento_data['visualization'])
                print(f"Visualización guardada en: {viz_path}")
            
            return report_path, text_report_path, viz_path if viz_path else None
            
        except Exception as e:
            print(f"Error al generar reporte de abombamiento: {e}")
            import traceback
            traceback.print_exc()
            # Crear archivos de reporte básicos en caso de error
            report_path = os.path.join(output_dir, f"{image_name}_abombamiento_report.csv")
            text_report_path = os.path.join(output_dir, f"{image_name}_abombamiento_report.txt")
            
            # Crear un CSV básico
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("Lado,X_mm,L_mm,Abombamiento_Porcentaje\n")
                for lado in ["Lado 1 (Top)", "Lado 2 (Right)", "Lado 3 (Bottom)", "Lado 4 (Left)"]:
                    f.write(f"{lado},0.00,1.00,0.00\n")
            
            # Crear un informe de texto básico
            with open(text_report_path, 'w', encoding='utf-8') as f:
                f.write(f"REPORTE DE ABOMBAMIENTO - {image_name}\n")
                f.write("="*50 + "\n\n")
                f.write("Error en el análisis de abombamiento.\n")
            
            return report_path, text_report_path, None
    
    def process(self, image, corners, image_name=None, output_dir=None, model=None, conf_threshold=0.5, mask=None):
        """
        Procesa la imagen para detectar abombamiento
        
        Args:
            image: Imagen original
            corners: Esquinas de la palanquilla
            image_name: Nombre de la imagen (sin extensión)
            output_dir: Directorio de salida para guardar reportes
            model: Modelo para obtener contornos (debe ser el modelo de vértices)
            conf_threshold: Umbral de confianza para detección
            mask: Máscara de la palanquilla (opcional)
            
        Returns:
            processed_data: Diccionario con los resultados del procesamiento
        """
        try:
            # Si no se proporcionan las esquinas correctamente, intentar detectarlas
            if corners is None or len(corners) != 4:
                if model is not None:
                    print("Detectando contornos de la palanquilla para análisis de abombamiento usando modelo de vértices...")
                    corners, contorno_principal, mask = obtener_contorno_imagen(image, model, conf_threshold)
                    
                    if corners is None:
                        print("Error: No se pudo detectar el contorno de la palanquilla. Utilizando valores predeterminados.")
                        # Crear valores predeterminados para el análisis
                        h, w = image.shape[:2]
                        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
                        contorno_principal = corners.reshape(-1, 1, 2)
                else:
                    print("Error: Se requieren las esquinas de la palanquilla o un modelo para detectarlas")
                    # Crear valores predeterminados para el análisis
                    h, w = image.shape[:2]
                    corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
                    contorno_principal = corners.reshape(-1, 1, 2)
            else:
                # Si se proporcionan las esquinas pero no el contorno principal, crearlo
                contorno_principal = np.array(corners).reshape(-1, 1, 2)
            
            
            
            # Verificar que tanto corners como contorno_principal son válidos
            if corners is None or len(corners) != 4 or contorno_principal is None:
                print("Error: Datos de contorno no válidos. Usando contorno predeterminado.")
                h, w = image.shape[:2]
                corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
                contorno_principal = corners.reshape(-1, 1, 2)

            # Determinar el lado más recto para la rotación
            try:
                lado_mas_recto = obtener_lado_rotacion_abombamiento(corners, contorno_principal)
            except Exception as e:
                print(f"Error al determinar el lado más recto: {e}")
                lado_mas_recto = "Lado 1 (Top)"  # Valor predeterminado
            
            # Medir abombamiento
            results = self.measure_abombamiento(image, corners, contorno_principal)
            
            visualizations = {}
            
            # Agregar visualización si existe
            if 'visualization' in results:
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
                'processed_data': results,
                'visualizations': visualizations,
                'report_paths': report_paths
            }
            
        except Exception as e:
            print(f"Error general en el procesamiento de abombamiento: {e}")
            import traceback
            traceback.print_exc()
            
            # Crear resultados por defecto
            default_results = {
                'lado_max_abombamiento': "Lado 1 (Top)",
                'max_abombamiento_porcentaje': 0.0,
                'max_abombamiento_pixeles': 0.0,
                'punto_max_abombamiento': None,
                'resultados_por_lado': {
                    "Lado 1 (Top)": {'X_mm': 0.0, 'L_mm': 1.0, 'C_porcentaje': 0.0, 'X_px': 0.0, 'L_px': 1.0},
                    "Lado 2 (Right)": {'X_mm': 0.0, 'L_mm': 1.0, 'C_porcentaje': 0.0, 'X_px': 0.0, 'L_px': 1.0},
                    "Lado 3 (Bottom)": {'X_mm': 0.0, 'L_mm': 1.0, 'C_porcentaje': 0.0, 'X_px': 0.0, 'L_px': 1.0},
                    "Lado 4 (Left)": {'X_mm': 0.0, 'L_mm': 1.0, 'C_porcentaje': 0.0, 'X_px': 0.0, 'L_px': 1.0}
                }
            }
            
            # Crear visualización básica
            visualizations = {}
            if image is not None:
                default_vis = image.copy()
                h, w = default_vis.shape[:2]
                cv2.putText(default_vis, "Error en análisis de abombamiento", (w//4, h//2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                visualizations['abombamiento_global'] = default_vis
            
            return {
                'processed_data': default_results,
                'visualizations': visualizations,
                'report_paths': None
            }