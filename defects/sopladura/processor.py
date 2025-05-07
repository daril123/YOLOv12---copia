import cv2
import numpy as np
import os
import pandas as pd
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

class SopladuraProcessor:
    """
    Procesador para sopladuras que usa el código proporcionado
    """
    
    def __init__(self):
        """
        Inicializa el procesador de sopladuras
        """
        self.name = "sopladura"
    
    def determinar_lado_sopladura(self, center_point, corners):
        """
        Determina en qué lado de la palanquilla se encuentra la sopladura
        
        Args:
            center_point: Centro de la sopladura (x, y)
            corners: Esquinas de la palanquilla [top-left, top-right, bottom-right, bottom-left]
            
        Returns:
            lado: 'arriba', 'abajo', 'izquierda' o 'derecha'
        """
        # Calcular centro de la palanquilla
        center_x = sum(corner[0] for corner in corners) / 4
        center_y = sum(corner[1] for corner in corners) / 4
        
        # Calcular la distancia a cada borde
        top_edge_dist = abs(center_point[1] - min(corner[1] for corner in corners))
        bottom_edge_dist = abs(center_point[1] - max(corner[1] for corner in corners))
        left_edge_dist = abs(center_point[0] - min(corner[0] for corner in corners))
        right_edge_dist = abs(center_point[0] - max(corner[0] for corner in corners))
        
        # Encontrar la distancia mínima
        min_dist = min(top_edge_dist, bottom_edge_dist, left_edge_dist, right_edge_dist)
        
        # Devolver el lado correspondiente
        if min_dist == top_edge_dist:
            return "arriba"
        elif min_dist == bottom_edge_dist:
            return "abajo"
        elif min_dist == left_edge_dist:
            return "izquierda"
        else:
            return "derecha"
    
    def analisis_completo_sopladuras(self, imagen_roi, direccion, umbral_sopladura=35):
        """
        Analisis completo de sopladuras con contraste mejorado en lugar de mapa HOT
        
        Args:
            imagen_roi: ROI de la sopladura
            direccion: Dirección para medir ('arriba', 'abajo', 'izquierda', 'derecha')
            umbral_sopladura: Valor umbral para detectar manchas negras (default: 20)
            
        Returns:
            Diccionario con resultados
        """
        try:
            # Verificar que la entrada sea válida
            if imagen_roi is None or imagen_roi.size == 0:
                return {
                    'L': 0,
                    'D': 0,
                    'direccion': direccion,
                    'area': 0,
                    'visualization': np.zeros((100, 100, 3), dtype=np.uint8),  # Imagen vacía como fallback
                    'binaria': np.zeros((100, 100), dtype=np.uint8)
                }
            
            # Convertir a escala de grises si es necesario
            if len(imagen_roi.shape) == 3:
                gris = cv2.cvtColor(imagen_roi, cv2.COLOR_BGR2GRAY)
            else:
                gris = imagen_roi.copy()
            
            # 2. MEJORAR CONTRASTE
            # Ecualizar histograma para mejorar contraste
            ecualizada = cv2.equalizeHist(gris)
            
            # Mejora adicional de contraste mediante CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            mejorada = clahe.apply(ecualizada)
            
            # Aplicar suavizado bilateral para reducir ruido pero preservar bordes
            suavizada = cv2.bilateralFilter(mejorada, 9, 75, 75)
            
            # Convertir a imagen a color para visualización
            resultado_color = cv2.cvtColor(suavizada, cv2.COLOR_GRAY2BGR)
            
            # 3. DETECTAR MANCHAS NEGRAS (SOPLADURAS)
            # Umbralizar para detectar áreas muy oscuras (negras)
            _, binaria = cv2.threshold(gris, umbral_sopladura, 255, cv2.THRESH_BINARY_INV)
            
            # Operaciones morfológicas para limpiar ruido
            kernel = np.ones((3,3), np.uint8)
            binaria_limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
            binaria_limpia = cv2.morphologyEx(binaria_limpia, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 4. ENCONTRAR CONTORNOS DE LAS MANCHAS NEGRAS
            contornos, _ = cv2.findContours(binaria_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contornos:
                return {
                    'L': 0,
                    'D': 0,
                    'direccion': direccion,
                    'area': 0,
                    'visualization': resultado_color,
                    'binaria': binaria_limpia
                }
            
            # 5. IDENTIFICAR LA MANCHA NEGRA CON MAYOR ÁREA
            areas = [cv2.contourArea(contorno) for contorno in contornos]
            if not areas:
                return {
                    'L': 0,
                    'D': 0,
                    'direccion': direccion,
                    'area': 0,
                    'visualization': resultado_color,
                    'binaria': binaria_limpia
                }
                
            # Usar sorted con zip para manejar correctamente el ordenamiento
            pares_ordenados = sorted(zip(areas, contornos), key=lambda pair: pair[0], reverse=True)
            areas_ordenadas = [area for area, _ in pares_ordenados]
            contornos_ordenados = [contorno for _, contorno in pares_ordenados]
            
            contorno_principal = contornos_ordenados[0]
            area_principal = areas_ordenadas[0]
            
            # 6. MEDIR LONGITUD MÁXIMA DE LA MANCHA PRINCIPAL
            puntos = contorno_principal.reshape(-1, 2)
            if len(puntos) < 2:
                return {
                    'L': 0,
                    'D': 0,
                    'direccion': direccion,
                    'area': area_principal,
                    'visualization': resultado_color,
                    'binaria': binaria_limpia
                }
            
            # Calcular el diámetro de Feret (distancia máxima entre dos puntos)
            from scipy.spatial import distance as dist
            D = dist.cdist(puntos, puntos)
            i, j = np.unravel_index(D.argmax(), D.shape)
            
            # Puntos extremos
            punto1 = tuple(puntos[i])
            punto2 = tuple(puntos[j])
            
            # Longitud máxima
            longitud_maxima = D.max()
            
            # 7. MEDIR DISTANCIAS A LOS BORDES
            altura, ancho = gris.shape
            
            # Puntos extremos en cada dirección
            punto_superior = min(puntos, key=lambda p: p[1])
            punto_inferior = max(puntos, key=lambda p: p[1])
            punto_izquierdo = min(puntos, key=lambda p: p[0])
            punto_derecho = max(puntos, key=lambda p: p[0])
            
            # Calcular distancias a los bordes
            distancia_arriba = punto_superior[1]
            distancia_abajo = altura - punto_inferior[1]
            distancia_izquierda = punto_izquierdo[0]
            distancia_derecha = ancho - punto_derecho[0]
            
            # Diccionario de distancias y puntos
            distancias = {
                'arriba': (distancia_arriba, punto_superior),
                'abajo': (distancia_abajo, punto_inferior),
                'izquierda': (distancia_izquierda, punto_izquierdo),
                'derecha': (distancia_derecha, punto_derecho)
            }
            
            # Usar la distancia en la dirección especificada
            if direccion in distancias:
                distancia_valor, punto = distancias[direccion]
            else:
                # Si no se especificó una dirección válida, usar la mínima
                distancia_valor = min(distancia_arriba, distancia_abajo, distancia_izquierda, distancia_derecha)
                # Encontrar el lado correspondiente a la distancia mínima
                for dir_name, (dist_val, pt) in distancias.items():
                    if dist_val == distancia_valor:
                        direccion = dir_name
                        punto = pt
                        break
            
            # 8. VISUALIZAR RESULTADOS CON CONTRASTE MEJORADO
            imagen_resultado = resultado_color.copy()
            
            # Dibujar contorno de la mancha negra principal con grosor 2
            cv2.drawContours(imagen_resultado, [contorno_principal], -1, (0, 255, 0), 2)
            
            # Dibujar línea de longitud máxima más gruesa
            cv2.line(imagen_resultado, punto1, punto2, (255, 255, 255), 2)
            
            # Dibujar la línea de distancia según la dirección
            if direccion == 'arriba':
                cv2.line(imagen_resultado, punto, (punto[0], 0), (0, 255, 255), 2)
            elif direccion == 'abajo':
                cv2.line(imagen_resultado, punto, (punto[0], altura), (0, 255, 255), 2)
            elif direccion == 'izquierda':
                cv2.line(imagen_resultado, punto, (0, punto[1]), (0, 255, 255), 2)
            elif direccion == 'derecha':
                cv2.line(imagen_resultado, punto, (ancho, punto[1]), (0, 255, 255), 2)
            
            # Añadir texto con mediciones
            cv2.putText(imagen_resultado, f"Area: {area_principal:.2f} px²", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(imagen_resultado, f"L: {longitud_maxima:.2f} px", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(imagen_resultado, f"Dist {direccion}: {distancia_valor:.2f} px", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Dibujar contornos secundarios más claros
            if len(contornos) > 1:
                for i in range(1, min(5, len(contornos))):
                    cv2.drawContours(imagen_resultado, [contornos_ordenados[i]], -1, (255, 0, 255), 1)
            
            return {
                'L': round(longitud_maxima, 2),
                'D': round(distancia_valor, 2),
                'direccion': direccion,
                'area': round(area_principal, 2),
                'visualization': imagen_resultado,
                'binaria': binaria_limpia
            }
        
        except Exception as e:
            # En caso de error, devolver valores por defecto y una imagen vacía
            print(f"Error en analisis_completo_sopladuras: {e}")
            import traceback
            traceback.print_exc()
            
            # Crear una imagen de fallback con mensaje de error
            fallback_img = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(fallback_img, "Error en analisis", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(fallback_img, str(e), (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
            fallback_binaria = np.zeros((200, 400), dtype=np.uint8)
            
            return {
                'L': 0,
                'D': 0,
                'direccion': direccion,
                'area': 0,
                'visualization': fallback_img,
                'binaria': fallback_binaria
            }
    def generate_report(self, image_name, sopladuras_data, output_dir):
        """
        Genera un informe simple de las sopladuras
        
        Args:
            image_name: Nombre de la imagen original
            sopladuras_data: Lista de diccionarios con los datos de las sopladuras
            output_dir: Directorio donde guardar el informe
        
        Returns:
            report_path: Ruta al archivo de informe generado
        """
        try:
            # Crear directorio si no existe
            os.makedirs(output_dir, exist_ok=True)
            
            # Crear un DataFrame con los datos (excluyendo campos no serializables)
            df_data = []
            for sopladura in sopladuras_data:
                # Crear una copia filtrada de los datos sin imágenes ni objetos complejos
                sopladura_filtrada = {
                    'id': sopladura.get('id', 0),
                    'lado': sopladura.get('lado', 'desconocido'),
                    'L': sopladura.get('L', 0),
                    'D': sopladura.get('D', 0),
                    'area': sopladura.get('area', 0),
                    'conf': sopladura.get('conf', 0),
                    'direccion': sopladura.get('direccion', 'desconocido')
                }
                df_data.append(sopladura_filtrada)
            
            # Crear DataFrame
            df = pd.DataFrame(df_data)
            
            # Formato del informe
            report_path = os.path.join(output_dir, f"{image_name}_sopladura_report.csv")
            
            # Guardar como CSV
            df.to_csv(report_path, index=False)
            
            # También generar una versión en formato de texto
            text_report_path = os.path.join(output_dir, f"{image_name}_sopladura_report.txt")
            
            with open(text_report_path, 'w', encoding='utf-8') as f:
                f.write(f"REPORTE DE SOPLADURAS - {image_name}\n")
                f.write("="*50 + "\n\n")
                
                for sopladura in df_data:
                    f.write(f"SOPLADURA #{sopladura['id']} ({sopladura['lado']})\n")
                    f.write(f"  Longitud (L): {sopladura['L']} píxeles\n")
                    f.write(f"  Distancia a borde (D): {sopladura['D']} píxeles\n")
                    f.write(f"  Dirección del borde: {sopladura['direccion']}\n")
                    f.write(f"  Área: {sopladura['area']} píxeles²\n")
                    f.write(f"  Confianza: {sopladura['conf']:.2f}\n\n")
            
            print(f"Reporte generado en: {report_path}")
            print(f"Reporte de texto generado en: {text_report_path}")
            
            return report_path, text_report_path
        
        except Exception as e:
            print(f"Error generando reporte de sopladuras: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def process(self, detections, image, corners, zone_masks, image_name=None, output_dir=None):
        """
        Procesa las sopladuras detectadas:
        1. Genera 4 imágenes, una para cada lado (arriba, abajo, izquierda, derecha)
        2. Selecciona sopladuras basadas en longitud máxima en la dirección correspondiente
        """
        results = []
        visualizations = {}
        
        # Diccionario para agrupar sopladuras por lado
        sopladuras_por_lado = {
            'arriba': [],
            'abajo': [],
            'izquierda': [],
            'derecha': []
        }
        
        # 1. FILTRAR SOPLADURAS EN ZONA VERDE
        print(f"Procesando {len(detections)} posibles sopladuras...")
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            conf = detection.get('conf', 0)
            
            # Extraer la máscara
            mask = detection.get('mask', None)
            if mask is None:
                # Si no hay máscara, crear una a partir del ROI
                roi = image[y1:y2, x1:x2].copy()
                if len(roi.shape) == 3:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    roi_gray = roi
                
                # Binarizar para obtener la máscara
                _, roi_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Crear máscara global
                full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = roi_mask
            else:
                full_mask = mask
            
            # Verificar si la sopladura está en la zona verde
            overlap = cv2.bitwise_and(full_mask, zone_masks['verde'])
            overlap_area = cv2.countNonZero(overlap)
            
            # Si hay solapamiento significativo con la zona verde
            if overlap_area > 0:
                # Obtener la ROI para análisis
                roi = image[y1:y2, x1:x2].copy()
                
                # Calcular centro de la sopladura
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Determinar a qué lado pertenece esta sopladura
                lado = self.determinar_lado_sopladura((center_x, center_y), corners)
                
                # Calcular contorno para mediciones más precisas
                if len(roi.shape) == 3:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    roi_gray = roi
                
                _, threshold = cv2.threshold(roi_gray, 20, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Valores por defecto
                longitud_vertical = y2 - y1
                longitud_horizontal = x2 - x1
                
                # Si hay contornos, calcular dimensiones más precisas
                if contours and len(contours) > 0:
                    main_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(main_contour)
                    longitud_vertical = h
                    longitud_horizontal = w
                
                # Guardar información de la sopladura
                sopladura_info = {
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'id': i+1,
                    'roi': roi,
                    'mask': full_mask,
                    'longitud_vertical': longitud_vertical,
                    'longitud_horizontal': longitud_horizontal
                }
                
                # Agregar a la lista correspondiente según su lado
                sopladuras_por_lado[lado].append(sopladura_info)
        
        # 2. SELECCIONAR LA MEJOR SOPLADURA PARA CADA LADO
        seleccionadas = []
        
        # Para arriba y abajo, seleccionar por longitud vertical
        for lado in ['arriba', 'abajo']:
            if sopladuras_por_lado[lado]:
                # Ordenar por longitud vertical (mayor a menor)
                sorted_sopladuras = sorted(sopladuras_por_lado[lado], 
                                        key=lambda x: x['longitud_vertical'], 
                                        reverse=True)
                # Seleccionar la de mayor longitud vertical
                mejor_sopladura = sorted_sopladuras[0]
                mejor_sopladura['lado'] = lado
                seleccionadas.append(mejor_sopladura)
                print(f"Sopladura seleccionada para {lado}: longitud vertical = {mejor_sopladura['longitud_vertical']}")
        
        # Para izquierda y derecha, seleccionar por longitud horizontal
        for lado in ['izquierda', 'derecha']:
            if sopladuras_por_lado[lado]:
                # Ordenar por longitud horizontal (mayor a menor)
                sorted_sopladuras = sorted(sopladuras_por_lado[lado], 
                                        key=lambda x: x['longitud_horizontal'], 
                                        reverse=True)
                # Seleccionar la de mayor longitud horizontal
                mejor_sopladura = sorted_sopladuras[0]
                mejor_sopladura['lado'] = lado
                seleccionadas.append(mejor_sopladura)
                print(f"Sopladura seleccionada para {lado}: longitud horizontal = {mejor_sopladura['longitud_horizontal']}")
        
        # Crear imagen para visualización de depuración - copia de la imagen original
        debug_img = image.copy()
        
        # 3. PROCESAR CADA SOPLADURA SELECCIONADA
        for i, sopladura in enumerate(seleccionadas):
            bbox = sopladura['bbox']
            x1, y1, x2, y2 = bbox
            roi = sopladura['roi']
            lado = sopladura['lado']
            
            # Aplicar el análisis completo de sopladuras con el lado correspondiente
            umbral_sopladura = 21  # Umbral para mejor detección
            resultado = self.analisis_completo_sopladuras(roi, lado, umbral_sopladura)
            
            # Agregar datos adicionales para el reporte
            resultado['conf'] = sopladura['conf']
            resultado['bbox'] = bbox
            resultado['id'] = i+1
            resultado['lado'] = lado
            
            # Guardar visualización específica para este lado
            visualization_key = f"sopladura_{lado}"
            if 'visualization' in resultado:
                visualizations[visualization_key] = resultado['visualization']
            
            # Guardar también la imagen binaria (como máscara) para depuración
            if 'binaria' in resultado:
                mask_key = f"sopladura_{lado}_mask"
                # Guardar como imagen en escala de grises (no convertir a BGR)
                visualizations[mask_key] = resultado['binaria']
            
            # Marcar esta sopladura en la imagen de depuración
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(debug_img, f"{lado}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Agregar a resultados finales
            results.append(resultado)
        
        # Añadir la imagen de debug a las visualizaciones
        visualizations['sopladuras_overview'] = debug_img
        
        # Generar reporte solo si hay resultados
        report_paths = None
        if results and image_name and output_dir:
            # Crear directorio para este tipo de defecto
            defect_dir = os.path.join(output_dir, image_name, self.name)
            os.makedirs(defect_dir, exist_ok=True)
            
            # Generar el reporte en la carpeta específica
            try:
                report_paths = self.generate_report(image_name, results, defect_dir)
                print(f"Reporte de sopladuras generado exitosamente en: {report_paths[0]}")
            except Exception as e:
                print(f"Error al generar reporte de sopladuras: {e}")
                import traceback
                traceback.print_exc()
        
        return {
            'processed_data': results,
            'visualizations': visualizations,
            'report_paths': report_paths
        }