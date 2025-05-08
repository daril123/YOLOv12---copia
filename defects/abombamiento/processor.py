import cv2
import numpy as np
import os
import pandas as pd

from utils.utils import draw_arrow
from utils.contorno import obtener_contorno_imagen, rotar_imagen_lado
from .abombamiento import (
    obtener_lado_rotacion_abombamiento,
    calcular_abombamiento
)
from .enhanced_visualization import visualizar_abombamiento_enhanced, generar_reporte_enhanced

class AbombamientoProcessor:
    """
    Procesador para análisis de abombamiento en palanquillas
    """
    
    def __init__(self):
        """
        Inicializa el procesador de abombamiento
        """
        self.name = "abombamiento"
    
    def measure_abombamiento(self, image, corners, contorno_principal=None, mask=None, factor_mm_px=1.0):
        """
        Mide el abombamiento en todos los lados de la palanquilla usando la visualización mejorada
        con vectores perpendiculares y etiquetas X1, X2, X3, X4 y D1, D2, D3, D4
        
        Args:
            image: Imagen original
            corners: Esquinas de la palanquilla [top-left, top-right, bottom-right, bottom-left]
            contorno_principal: Contorno completo de la palanquilla (opcional)
            mask: Máscara de la palanquilla (opcional)
            factor_mm_px: Factor de conversión de píxeles a milímetros
            
        Returns:
            metrics: Diccionario con las métricas de abombamiento
        """
        # Convertir corners a array numpy si es necesario
        corners_np = np.array(corners, dtype=np.float32)
        
        try:
            # Si no se proporciona contorno_principal, intentar crearlo desde la máscara o corners
            if contorno_principal is None:
                if mask is not None:
                    # Encontrar contornos en la máscara
                    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contornos:
                        contorno_principal = max(contornos, key=cv2.contourArea)
                    else:
                        # Si no hay contornos en la máscara, crear uno simple desde las esquinas
                        contorno_principal = corners_np.reshape(-1, 1, 2).astype(np.int32)
                else:
                    # Crear un contorno simple desde las esquinas
                    contorno_principal = corners_np.reshape(-1, 1, 2).astype(np.int32)
            
            # Usar la visualización mejorada
            imagen_resultado, resultados_por_lado = visualizar_abombamiento_enhanced(
                image, corners_np, contorno_principal)
            
            # Determinar el lado con mayor abombamiento
            if resultados_por_lado:
                # Encontrar el lado con valor máximo de C
                lado_max = max(resultados_por_lado.items(), key=lambda x: x[1]['C_porcentaje'])[0]
                max_abombamiento_pix = resultados_por_lado[lado_max]['X_px']
                max_porcentaje = resultados_por_lado[lado_max]['C_porcentaje']
            else:
                lado_max = "Lado 1 (Top)"
                max_abombamiento_pix = 0
                max_porcentaje = 0
            
            # Preparar resultados
            results = {
                'lado_max_abombamiento': lado_max,
                'max_abombamiento_porcentaje': round(max_porcentaje, 2),
                'max_abombamiento_pixeles': round(max_abombamiento_pix, 2),
                'resultados_por_lado': resultados_por_lado
            }
            
            # Añadir la visualización mejorada
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
                'resultados_por_lado': {
                    "Lado 1 (Top)": {'X_px': 0.0, 'D_px': 1.0, 'C_porcentaje': 0.0},
                    "Lado 2 (Right)": {'X_px': 0.0, 'D_px': 1.0, 'C_porcentaje': 0.0},
                    "Lado 3 (Bottom)": {'X_px': 0.0, 'D_px': 1.0, 'C_porcentaje': 0.0},
                    "Lado 4 (Left)": {'X_px': 0.0, 'D_px': 1.0, 'C_porcentaje': 0.0}
                }
            }
            
            # Intentar crear una visualización básica en caso de error
            if image is not None:
                default_results['visualization'] = image.copy()
            
            return default_results
    
    def generate_report(self, image_name, abombamiento_data, output_dir):
        """
        Genera un informe del abombamiento con la notación mejorada (X1, X2, X3, X4 y D1, D2, D3, D4)
        
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
            
            # Definir índices para cada lado
            indices = {
                "Lado 1 (Top)": "1",
                "Lado 2 (Right)": "2",
                "Lado 3 (Bottom)": "3",
                "Lado 4 (Left)": "4"
            }
            
            # Crear DataFrame para el informe
            datos = []
            for lado, valores in resultados_por_lado.items():
                # Verificar que los valores tienen las claves esperadas
                indice = indices.get(lado, "")
                X_px = valores.get('X_px', 0.0)
                D_px = valores.get('D_px', 1.0)
                C_porcentaje = valores.get('C_porcentaje', 0.0)
                
                datos.append({
                    'Lado': lado,
                    f'X{indice}_px': X_px,
                    f'D{indice}_px': D_px,
                    'Abombamiento_Porcentaje': C_porcentaje
                })
            
            df = pd.DataFrame(datos)
            
            # Formato del informe
            report_path = os.path.join(output_dir, f"{image_name}_abombamiento_report.csv")
            
            # Guardar como CSV
            df.to_csv(report_path, index=False)
            
            # Generar reporte mejorado en formato de texto
            text_report_path = os.path.join(output_dir, f"{image_name}_abombamiento_report_enhanced.txt")
            generar_reporte_enhanced(resultados_por_lado, text_report_path)
            
            print(f"Reporte generado en: {report_path}")
            print(f"Reporte mejorado generado en: {text_report_path}")
            
            # Guardar la visualización si existe
            viz_path = None
            if 'visualization' in abombamiento_data:
                viz_path = os.path.join(output_dir, f"{image_name}_abombamiento_visualization_enhanced.jpg")
                cv2.imwrite(viz_path, abombamiento_data['visualization'])
                print(f"Visualización mejorada guardada en: {viz_path}")
            
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
                f.write("Lado,X_px,D_px,Abombamiento_Porcentaje\n")
                for lado in ["Lado 1 (Top)", "Lado 2 (Right)", "Lado 3 (Bottom)", "Lado 4 (Left)"]:
                    f.write(f"{lado},0.00,1.00,0.00\n")
            
            # Crear un informe de texto básico
            with open(text_report_path, 'w', encoding='utf-8') as f:
                f.write(f"REPORTE DE ABOMBAMIENTO - {image_name}\n")
                f.write("="*50 + "\n\n")
                f.write("Error en el análisis de abombamiento.\n")
            
            return report_path, text_report_path, None
    
    def process(self, image, corners, image_name=None, output_dir=None, model=None, conf_threshold=0.35, mask=None, rotacion_info=None):
        """
        Procesa la imagen para detectar abombamiento con la visualización mejorada
        
        Args:
            image: Imagen original
            corners: Esquinas de la palanquilla
            image_name: Nombre de la imagen (sin extensión)
            output_dir: Directorio de salida para guardar reportes
            model: Modelo para obtener contornos (opcional, si no se proporcionan las esquinas)
            conf_threshold: Umbral de confianza para detección
            mask: Máscara de la palanquilla (opcional)
            rotacion_info: Información sobre la rotación aplicada a la imagen
                
        Returns:
            processed_data: Diccionario con los resultados del procesamiento
        """
        try:
            # Si no se proporcionan las esquinas correctamente, intentar detectarlas
            if corners is None or len(corners) != 4:
                if model is not None:
                    print("Detectando contornos de la palanquilla...")
                    corners, contorno_principal, mask = obtener_contorno_imagen(image, model, conf_threshold)
                    
                    if corners is None:
                        print("Error: No se pudo detectar el contorno de la palanquilla")
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
                # Si se proporcionan las esquinas pero no el contorno principal
                if mask is not None:
                    # Intentar extraer el contorno principal de la máscara
                    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contornos:
                        contorno_principal = max(contornos, key=cv2.contourArea)
                    else:
                        # Si no hay contornos en la máscara, crear uno simple desde las esquinas
                        contorno_principal = np.array(corners).reshape(-1, 1, 2)
                else:
                    # Crear un contorno simple desde las esquinas
                    contorno_principal = np.array(corners).reshape(-1, 1, 2)
            
            # Verificar que tanto corners como contorno_principal son válidos
            if corners is None or len(corners) != 4 or contorno_principal is None:
                print("Error: Datos de contorno no válidos. Usando contorno predeterminado.")
                h, w = image.shape[:2]
                corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
                contorno_principal = corners.reshape(-1, 1, 2)

            # PASO 1: Medir abombamiento con visualización mejorada en la imagen rotada
            results_rotated = self.measure_abombamiento(image, corners, contorno_principal, mask)
            
            # PASO 2: Ajustar los resultados según la rotación aplicada
            if rotacion_info is not None and 'angulo' in rotacion_info:
                angulo_rotacion = rotacion_info['angulo']
                print(f"Ajustando resultados de abombamiento para rotación de {angulo_rotacion}°...")
                
                # Crear una copia de los resultados para ajustar
                adjusted_results = results_rotated.copy()
                
                if 'resultados_por_lado' in adjusted_results and isinstance(adjusted_results['resultados_por_lado'], dict):
                    resultados_ajustados = {}
                    
                    # Para cada lado en los resultados de la imagen rotada
                    for lado_rotado, valores in adjusted_results['resultados_por_lado'].items():
                        # Mapear al lado original
                        lado_original = self.mapear_lados_originales(lado_rotado, angulo_rotacion)
                        print(f"Mapeando medidas: {lado_rotado} -> {lado_original}")
                        
                        # Transferir los valores al lado original
                        resultados_ajustados[lado_original] = valores.copy()
                        
                        # IMPORTANTE: Actualizar las etiquetas de visualización con el lado correcto
                        # (esto es para que la visualización muestre X1, X2, etc. en los lados correctos)
                        if 'indice' in valores:
                            # Mantener el índice original (1, 2, 3, 4)
                            indice_original = valores['indice']
                            resultados_ajustados[lado_original]['indice'] = indice_original
                    
                    # Reemplazar los resultados por lado con los ajustados
                    adjusted_results['resultados_por_lado'] = resultados_ajustados
                    
                    # Ajustar también el lado de máximo abombamiento
                    if 'lado_max_abombamiento' in adjusted_results:
                        lado_max_rotado = adjusted_results['lado_max_abombamiento']
                        lado_max_original = self.mapear_lados_originales(lado_max_rotado, angulo_rotacion)
                        adjusted_results['lado_max_abombamiento'] = lado_max_original
                        print(f"Lado con máximo abombamiento ajustado: {lado_max_rotado} -> {lado_max_original}")
                
                # Usar los resultados ajustados
                results = adjusted_results
            else:
                # Sin rotación, usar los resultados tal cual
                results = results_rotated
                
            # PASO 3: Recrear la visualización con los resultados ajustados
            if 'resultados_por_lado' in results:
                try:
                    # Crear visualización mejorada con los resultados ajustados
                    imagen_resultado, _ = visualizar_abombamiento_enhanced(
                        image, 
                        corners, 
                        contorno_principal,
                        resultados_ajustados=results['resultados_por_lado']  # Pasar los resultados ajustados
                    )
                    # Actualizar la visualización
                    results['visualization'] = imagen_resultado
                except Exception as e:
                    print(f"Error al crear visualización ajustada: {e}")
                    # Mantener la visualización original si falla
            
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
                'resultados_por_lado': {
                    "Lado 1 (Top)": {'X_px': 0.0, 'D_px': 1.0, 'C_porcentaje': 0.0},
                    "Lado 2 (Right)": {'X_px': 0.0, 'D_px': 1.0, 'C_porcentaje': 0.0},
                    "Lado 3 (Bottom)": {'X_px': 0.0, 'D_px': 1.0, 'C_porcentaje': 0.0},
                    "Lado 4 (Left)": {'X_px': 0.0, 'D_px': 1.0, 'C_porcentaje': 0.0}
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
    def mapear_lados_originales(self, lado, angulo_rotacion):
        """
        Maps a side in the rotated image back to its original orientation
        
        Args:
            lado: The side in the rotated image ("Lado 1 (Top)", "Lado 2 (Right)", etc.)
            angulo_rotacion: The rotation angle (0, 90, 180, -90)
                
        Returns:
            The side in the original orientation
        """
        # Normalizar el ángulo a uno de los ángulos estándar (0, 90, 180, 270, -90)
        angulo_normalizado = round(angulo_rotacion / 90) * 90
        if angulo_normalizado == 360 or angulo_normalizado == -360:
            angulo_normalizado = 0
        elif angulo_normalizado == -90:
            angulo_normalizado = 270
        elif angulo_normalizado == -180:
            angulo_normalizado = 180
        elif angulo_normalizado == -270:
            angulo_normalizado = 90
        
        print(f"Ángulo normalizado para mapeo: {angulo_normalizado}°")
        
        # No rotation, sides remain the same
        if angulo_normalizado == 0:
            return lado
        
        # Mapping for 90 degrees rotation (clockwise)
        if angulo_normalizado == 90:
            mapping = {
                "Lado 1 (Top)": "Lado 4 (Left)",     # Top becomes Left
                "Lado 2 (Right)": "Lado 1 (Top)",    # Right becomes Top
                "Lado 3 (Bottom)": "Lado 2 (Right)", # Bottom becomes Right
                "Lado 4 (Left)": "Lado 3 (Bottom)"   # Left becomes Bottom
            }
            return mapping.get(lado, lado)
        
        # Mapping for 180 degrees rotation
        if angulo_normalizado == 180:
            mapping = {
                "Lado 1 (Top)": "Lado 3 (Bottom)",   # Top becomes Bottom
                "Lado 2 (Right)": "Lado 4 (Left)",   # Right becomes Left
                "Lado 3 (Bottom)": "Lado 1 (Top)",   # Bottom becomes Top
                "Lado 4 (Left)": "Lado 2 (Right)"    # Left becomes Right
            }
            return mapping.get(lado, lado)
        
        # Mapping for 270 degrees rotation (clockwise) = -90 degrees (counter-clockwise)
        if angulo_normalizado == 270 or angulo_normalizado == -90:
            mapping = {
                "Lado 1 (Top)": "Lado 2 (Right)",    # Top becomes Right
                "Lado 2 (Right)": "Lado 3 (Bottom)", # Right becomes Bottom
                "Lado 3 (Bottom)": "Lado 4 (Left)",  # Bottom becomes Left
                "Lado 4 (Left)": "Lado 1 (Top)"      # Left becomes Top
            }
            return mapping.get(lado, lado)
        
        # For any other angle, warn and return the original side
        print(f"ADVERTENCIA: Ángulo de rotación no estándar: {angulo_rotacion}° (normalizado a {angulo_normalizado}°)")
        return lado