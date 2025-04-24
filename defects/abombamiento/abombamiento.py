import cv2
import numpy as np
import matplotlib.pyplot as plt

def distancia_a_segmento(p1, p2, punto):
    """
    Calcula la distancia mínima de un punto al segmento definido por p1 y p2.
    """
    try:
        p1, p2, punto = np.array(p1, dtype=float), np.array(p2, dtype=float), np.array(punto, dtype=float)
        v = p2 - p1
        u = punto - p1
        t = np.dot(u, v) / np.dot(v, v)
        if t < 0:
            return np.linalg.norm(punto - p1)
        elif t > 1:
            return np.linalg.norm(punto - p2)
        else:
            return np.abs(np.cross(v, u)) / np.linalg.norm(v)
    except Exception as e:
        print(f"Error en distancia_a_segmento: {e}")
        return float('inf')  # Valor por defecto en caso de error

def distancia_punto_a_linea(p1, p2, punto):
    """
    Calcula la distancia perpendicular de un punto a la línea infinita
    definida por p1 y p2 (para el cálculo del abombamiento).
    
    Args:
        p1, p2: Dos puntos que definen la línea
        punto: Punto del que calcular la distancia
        
    Returns:
        La distancia perpendicular a la línea
    """
    try:
        p1, p2, punto = np.array(p1, dtype=float), np.array(p2, dtype=float), np.array(punto, dtype=float)
        # Vector de la línea
        linea = p2 - p1
        # Magnitud del vector de la línea
        mag_linea = np.linalg.norm(linea)
        # Evitar división por cero
        if mag_linea == 0:
            return np.linalg.norm(punto - p1)
        # Cálculo de la distancia perpendicular
        return np.abs(np.cross(linea, punto - p1)) / mag_linea
    except Exception as e:
        print(f"Error al calcular distancia de punto a línea: {e}")
        return 0.0

def proyectar_punto_en_linea(p1, p2, p):
    """
    Devuelve el punto de proyección de 'p' sobre la línea infinita definida por p1 y p2.
    
    Args:
        p1, p2: Dos puntos que definen la línea
        p: Punto a proyectar
        
    Returns:
        El punto proyectado sobre la línea
    """
    if p is None:
        return None
        
    try:
        p = np.array(p, dtype=float)
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        
        # Vector de dirección de la línea
        line_vec = p2 - p1
        
        # Si la línea es un punto, retornar ese punto
        if np.all(line_vec == 0):
            return p1
            
        # Proyección escalar del vector (p - p1) sobre line_vec
        t = np.dot(p - p1, line_vec) / np.dot(line_vec, line_vec)
        
        # Punto proyectado
        proj = p1 + t * line_vec
        
        return tuple(int(x) for x in proj)
    except Exception as e:
        print(f"Error al proyectar punto en línea: {e}")
        return None

def calcular_orientacion_punto(p1, p2, punto):
    """
    Determina si un punto está a la izquierda, derecha o sobre la línea definida por p1 y p2.
    
    Args:
        p1, p2: Dos puntos que definen la línea
        punto: Punto a evaluar
    
    Returns:
        1 si el punto está a la derecha de la línea (convexo)
        -1 si el punto está a la izquierda de la línea (cóncavo)
        0 si el punto está sobre la línea
    """
    try:
        p1, p2, punto = np.array(p1, dtype=float), np.array(p2, dtype=float), np.array(punto, dtype=float)
        # Cálculo del producto cruz
        val = (p2[1] - p1[1]) * (punto[0] - p2[0]) - (p2[0] - p1[0]) * (punto[1] - p2[1])
        
        if val > 0:
            return 1  # Convexo (punto a la derecha de la línea)
        elif val < 0:
            return -1  # Cóncavo (punto a la izquierda de la línea)
        else:
            return 0  # Punto sobre la línea
    except Exception as e:
        print(f"Error al calcular orientación del punto: {e}")
        return 0

def obtener_lado_rotacion_abombamiento(contorno, contorno_principal):
    """
    Determina el lado más recto de la palanquilla para usar como referencia en la rotación.
    
    Args:
        contorno: Vértices de la palanquilla [top-left, top-right, bottom-right, bottom-left]
        contorno_principal: Contorno completo de la palanquilla
    
    Returns:
        El nombre del lado más recto ("Lado 1 (Top)", "Lado 2 (Right)", etc.)
    """
    try:
        # Verificar si los datos de entrada son válidos
        if contorno is None or contorno_principal is None:
            print("Error: Contornos no válidos para determinar el lado más recto.")
            return "Lado 1 (Top)"  # Valor predeterminado
        
        # Asegurar que contorno es un array numpy
        contorno = np.array(contorno, dtype=np.float32)
            
        # Definir los lados de la caja (pares de vértices consecutivos)
        lados = [
            (contorno[0], contorno[1], "Lado 1 (Top)"),
            (contorno[1], contorno[2], "Lado 2 (Right)"),
            (contorno[2], contorno[3], "Lado 3 (Bottom)"),
            (contorno[3], contorno[0], "Lado 4 (Left)"),
        ]

        # Extraer los puntos del contorno
        try:
            contorno_pts = contorno_principal.reshape(-1, 2)
        except Exception as e:
            print(f"Error: El contorno principal no es un array válido: {e}")
            return "Lado 1 (Top)"  # Valor predeterminado
            
        puntos_por_lado = [[] for _ in range(4)]

        # Asignar cada punto del contorno al lado más cercano
        for punto in contorno_pts:
            try:
                distancias = [distancia_a_segmento(p1, p2, punto) for p1, p2, _ in lados]
                indice_min = int(np.argmin(distancias))
                puntos_por_lado[indice_min].append(punto)
            except Exception as e:
                print(f"Error al procesar punto del contorno: {e}")
                continue
            
        # Calcular el abombamiento (distancia perpendicular máxima) para cada lado
        abombamientos = []
        punto_max_por_lado = []  # Punto que produce el máximo abombamiento en cada lado
        for i, (p1, p2, nombre) in enumerate(lados):
            try:
                puntos_lado = np.array(puntos_por_lado[i]) if puntos_por_lado[i] else np.array([])
                if len(puntos_lado) > 0:
                    distancias = np.array([distancia_punto_a_linea(p1, p2, pt) for pt in puntos_lado])
                    abombamiento = np.max(distancias) if len(distancias) > 0 else 0
                    punto_max = puntos_lado[int(np.argmax(distancias))] if len(distancias) > 0 else None
                else:
                    abombamiento = 0
                    punto_max = None
                abombamientos.append(abombamiento)
                punto_max_por_lado.append(punto_max)
                print(f"{nombre}: abombamiento = {abombamiento:.2f}")
            except Exception as e:
                print(f"Error al procesar lado {nombre}: {e}")
                abombamientos.append(float('inf'))  # Valor arbitrariamente alto
                punto_max_por_lado.append(None)

        # Se determina el lado más recto (el de menor abombamiento)
        if len(abombamientos) > 0:
            indice_recto = int(np.argmin(abombamientos))
            lado_recto_info = lados[indice_recto]
            print("El lado más recto es:", lado_recto_info[2])
            return lado_recto_info[2]
        else:
            return "Lado 1 (Top)"  # Valor predeterminado si no hay datos
    except Exception as e:
        print(f"Error general en obtener_lado_rotacion_abombamiento: {e}")
        import traceback
        traceback.print_exc()
        return "Lado 1 (Top)"  # Valor predeterminado

def obtener_abombamiento(contorno, contorno_principal, mask=None):
    """
    Calcula el abombamiento (desviación) para cada lado del cuadrilátero.
    Método corregido de acuerdo al criterio específico de medición:
    - X: distancia desde el punto rojo al lado correspondiente (líneas de colores)
    - D: longitud del lado correspondiente (líneas de colores)
    
    Args:
        contorno: Vértices de la palanquilla [top-left, top-right, bottom-right, bottom-left]
        contorno_principal: Contorno completo de la palanquilla
        mask: Máscara binaria de la palanquilla (opcional, para verificación adicional)
    
    Returns:
        Una tupla con:
        - max_porcentaje: Porcentaje máximo de abombamiento
        - max_abombamiento_pix: Valor máximo de abombamiento en píxeles
        - lado_max: Lado donde se encuentra el máximo abombamiento
        - punto_max: Punto donde se encuentra el máximo abombamiento
        - abombamientos_por_lado: Diccionario con los valores de abombamiento por lado
    """
    # Verificar que los contornos no son None
    if contorno is None or contorno_principal is None:
        print("Error: Contornos inválidos para calcular abombamiento")
        # Retornar valores por defecto
        return 0.0, 0.0, "Lado 1 (Top)", None, {}
    
    # Convertir contorno a array numpy si no lo es ya
    contorno = np.array(contorno, dtype=np.float32)
        
    try:
        # Definir los lados (cada lado es un par de vértices consecutivos)
        lados = {
            "Lado 1 (Top)": (contorno[0], contorno[1]),
            "Lado 2 (Right)": (contorno[1], contorno[2]),
            "Lado 3 (Bottom)": (contorno[2], contorno[3]),
            "Lado 4 (Left)": (contorno[3], contorno[0])
        }
        
        # Extraer todos los puntos del contorno principal
        try:
            contorno_pts = contorno_principal.reshape(-1, 2)
        except Exception as e:
            print(f"Error al procesar contorno principal: {e}")
            contorno_pts = np.array([])
            
        puntos_max_por_lado = {}
        distancias_max_por_lado = {}
        proyecciones_por_lado = {}
        orientaciones_por_lado = {}
        
        # Para cada lado, encontrar el punto más lejano
        for nombre, (p1, p2) in lados.items():
            # Calcular longitudes de los lados (D)
            longitud_nominal = np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float))
            
            # Inicializar valores
            max_distancia = 0
            punto_max = None
            proyeccion_max = None
            orientacion = 0
            
            # Analizar todos los puntos del contorno para encontrar el más lejano a este lado
            for punto in contorno_pts:
                # Calcular la distancia perpendicular a la línea
                dist_perp = distancia_punto_a_linea(p1, p2, punto)
                
                # Si es la distancia máxima hasta ahora, actualizar
                if dist_perp > max_distancia:
                    max_distancia = dist_perp
                    punto_max = punto
                    
                    # Calcular la proyección del punto sobre la línea
                    proyeccion_max = proyectar_punto_en_linea(p1, p2, punto)
                    
                    # Calcular la orientación (si es convexo o cóncavo)
                    orientacion = calcular_orientacion_punto(p1, p2, punto)
            
            # Guardar el resultado para este lado
            puntos_max_por_lado[nombre] = punto_max
            distancias_max_por_lado[nombre] = max_distancia
            proyecciones_por_lado[nombre] = proyeccion_max
            orientaciones_por_lado[nombre] = orientacion
        
        # Calcular porcentajes de abombamiento (C = X/D*100)
        abombamientos_pixeles = {}
        abombamientos_porcentuales = {}
        
        for nombre, (p1, p2) in lados.items():
            # X = distancia máxima (en píxeles)
            X = distancias_max_por_lado.get(nombre, 0)
            
            # D = longitud nominal del lado (en píxeles)
            D = np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float))
            
            # C = X/D*100 (porcentaje de abombamiento)
            C = (X / D * 100) if D > 0 else 0
            
            abombamientos_pixeles[nombre] = X
            abombamientos_porcentuales[nombre] = C
            
            tipo = "convexo" if orientaciones_por_lado.get(nombre, 0) >= 0 else "cóncavo"
            print(f"{nombre}: X={X:.2f}px, D={D:.2f}px, C={C:.2f}%, {tipo}")
        
        # Identificar el lado con mayor abombamiento (según porcentaje)
        if abombamientos_porcentuales:
            # Encontrar el lado con valor máximo en el diccionario
            lado_max = max(abombamientos_porcentuales.items(), key=lambda x: x[1])[0]
            max_abombamiento_pix = abombamientos_pixeles[lado_max]
            max_porcentaje = abombamientos_porcentuales[lado_max]
        else:
            lado_max = "Lado 1 (Top)"
            max_abombamiento_pix = 0
            max_porcentaje = 0
        
        print("El mayor abombamiento se encuentra en", lado_max, "con un valor de", 
              max_abombamiento_pix, "px, lo que equivale a", max_porcentaje, "%",
              ", tipo:", "convexo" if orientaciones_por_lado.get(lado_max, 0) >= 0 else "cóncavo")
        
        # Crear diccionario con todos los resultados por lado
        abombamientos_por_lado = {
            lado: {
                'X_px': abombamientos_pixeles.get(lado, 0),
                'D_px': np.linalg.norm(np.array(lados[lado][0], dtype=float) - np.array(lados[lado][1], dtype=float)),
                'C_porcentaje': abombamientos_porcentuales.get(lado, 0),
                'punto_max': puntos_max_por_lado.get(lado, None),
                'proyeccion': proyecciones_por_lado.get(lado, None),
                'vertices': lados.get(lado, (None, None)),
                'orientacion': orientaciones_por_lado.get(lado, 0)
            }
            for lado in lados
        }
        
    except Exception as e:
        print(f"Error general en obtener_abombamiento: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, "Lado 1 (Top)", None, {}
    
    return max_porcentaje, max_abombamiento_pix, lado_max, puntos_max_por_lado.get(lado_max), abombamientos_por_lado

def visualizar_abombamiento(image, contorno, contorno_principal, guardar_path=None):
    """
    Visualiza el abombamiento en todos los lados de la palanquilla.
    Usa el esquema de colores especificado:
    - Lado 1 (Top): azul
    - Lado 2 (Right): verde
    - Lado 3 (Bottom): rojo
    - Lado 4 (Left): amarillo
    
    Args:
        image: Imagen original
        contorno: Vértices de la palanquilla [top-left, top-right, bottom-right, bottom-left]
        contorno_principal: Contorno completo de la palanquilla
        guardar_path: Ruta donde guardar la imagen resultante (opcional)
    
    Returns:
        Imagen con visualización del abombamiento
    """
    try:
        # Verificar entradas
        if image is None or contorno is None or contorno_principal is None:
            print("Error: Entrada inválida para visualizar abombamiento")
            if image is not None:
                return image.copy()
            return None
            
        # Convertir contorno a array numpy si no lo es ya
        contorno = np.array(contorno, dtype=np.float32)
            
        # Obtener los datos de abombamiento
        _, _, _, _, abombamientos_por_lado = obtener_abombamiento(contorno, contorno_principal)
        
        # Crear una copia de la imagen para no modificar la original
        img_resultado = image.copy()
        
        # Dibujar el contorno de la palanquilla
        pts = contorno.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img_resultado, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Colores para cada lado (BGR)
        colores = {
            "Lado 1 (Top)": (255, 0, 0),      # Azul
            "Lado 2 (Right)": (0, 255, 0),    # Verde
            "Lado 3 (Bottom)": (0, 0, 255),   # Rojo
            "Lado 4 (Left)": (0, 255, 255)    # Amarillo
        }
        
        # Título principal
        cv2.putText(img_resultado, "Análisis de Abombamiento", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Para cada lado, dibujar la línea nominal con su color específico y el punto de máximo abombamiento
        y_offset = 70  # Posición inicial para texto
        for lado, datos in abombamientos_por_lado.items():
            p1, p2 = datos['vertices']
            
            # Verificar que ambos puntos sean válidos
            if p1 is None or p2 is None:
                continue
                
            p1 = tuple(np.array(p1).astype(int))
            p2 = tuple(np.array(p2).astype(int))
            
            # Dibujar la línea nominal con su color específico (borde de la zona de eventos)
            color_lado = colores[lado]
            cv2.line(img_resultado, p1, p2, color_lado, 2)
            
            # Si hay punto de máximo abombamiento, dibujarlo
            if datos['punto_max'] is not None and datos['proyeccion'] is not None:
                punto_max = tuple(np.array(datos['punto_max']).astype(int))
                proyeccion = tuple(np.array(datos['proyeccion']).astype(int))
                
                # Dibujar el punto rojo (punto máximo)
                cv2.circle(img_resultado, punto_max, 5, (0, 0, 255), -1)  # Rojo para punto máximo
                
                # Dibujar una línea que conecte el punto de máximo abombamiento con su proyección
                cv2.line(img_resultado, punto_max, proyeccion, (255, 255, 255), 1, cv2.LINE_AA)  # Línea blanca
                
                # Añadir etiqueta con valores
                X_px = datos['X_px']
                D_px = datos['D_px']
                C_porcentaje = datos['C_porcentaje']
                tipo = "convexo" if datos['orientacion'] >= 0 else "cóncavo"
                
                # Crear texto para este lado
                texto_lado = f"{lado}: X={X_px:.1f}px, D={D_px:.1f}px, C={C_porcentaje:.1f}%, {tipo}"
                
                # Añadir fondo negro para mejor visibilidad
                (text_width, text_height), _ = cv2.getTextSize(texto_lado, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img_resultado, 
                             (20, y_offset - text_height - 5), 
                             (20 + text_width + 10, y_offset + 5), 
                             (0, 0, 0), -1)
                
                # Añadir texto con el color correspondiente al lado
                cv2.putText(img_resultado, texto_lado, (25, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_lado, 1)
                
                y_offset += 25  # Incrementar offset para el siguiente texto
        
        # Guardar la imagen si se proporciona una ruta
        if guardar_path:
            cv2.imwrite(guardar_path, img_resultado)
            print(f"Imagen guardada en: {guardar_path}")
        
        return img_resultado
    except Exception as e:
        print(f"Error al visualizar abombamiento: {e}")
        # Devolver la imagen original en caso de error
        return image.copy() if image is not None else None

def calcular_abombamiento(image, contorno, contorno_principal, factor_mm_px=1.0):
    """
    Calcula el abombamiento en milímetros según la fórmula C = X / D * 100.
    
    Args:
        image: Imagen original
        contorno: Vértices de la palanquilla [top-left, top-right, bottom-right, bottom-left]
        contorno_principal: Contorno completo de la palanquilla
        factor_mm_px: Factor de conversión de píxeles a milímetros
    
    Returns:
        Un diccionario con los resultados del abombamiento para cada lado
    """
    try:
        # Verificar entradas
        if contorno is None or contorno_principal is None:
            print("Error: Entrada inválida para calcular abombamiento")
            return {}
            
        # Convertir contorno a array numpy si no lo es ya
        contorno = np.array(contorno, dtype=np.float32)
            
        # Obtener datos de abombamiento
        _, _, _, _, abombamientos_por_lado = obtener_abombamiento(contorno, contorno_principal)
        
        # Crear diccionario de resultados
        resultados = {}
        
        for lado, datos in abombamientos_por_lado.items():
            # X es el valor de abombamiento en píxeles
            X_px = datos['X_px']
            X_mm = X_px * factor_mm_px
            
            # D es la longitud nominal del lado
            D_px = datos['D_px']
            D_mm = D_px * factor_mm_px
            
            # C es el porcentaje de abombamiento según la fórmula C = X/D*100
            C = X_mm / D_mm * 100 if D_mm > 0 else 0
            
            resultados[lado] = {
                'X_mm': X_mm,
                'D_mm': D_mm,
                'C_porcentaje': C,
                'X_px': X_px,
                'D_px': D_px,
                'orientacion': datos['orientacion']
            }
        
        return resultados
    except Exception as e:
        print(f"Error al calcular abombamiento: {e}")
        return {}