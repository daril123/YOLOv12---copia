import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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
    Método corregido que asegura que las proyecciones se realizan perpendiculares 
    a cada lado y en la dirección correcta hacia afuera.
    
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
        
        # Definir direcciones esperadas para cada lado
        direcciones_esperadas = {
            "Lado 1 (Top)": np.array([0, -1]),     # Hacia arriba
            "Lado 2 (Right)": np.array([1, 0]),    # Hacia la derecha
            "Lado 3 (Bottom)": np.array([0, 1]),   # Hacia abajo
            "Lado 4 (Left)": np.array([-1, 0])     # Hacia la izquierda
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
        
        # Para cada lado, encontrar el punto más lejano en la dirección correcta
        for nombre, (p1, p2) in lados.items():
            # Calcular longitudes de los lados (D)
            longitud_nominal = np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float))
            
            # Calcular vector del lado
            lado_vec = np.array(p2, dtype=float) - np.array(p1, dtype=float)
            lado_len = np.linalg.norm(lado_vec)
            
            if lado_len > 0:
                # Vector unitario del lado
                lado_unit = lado_vec / lado_len
                
                # Vector perpendicular al lado (rotación de 90 grados)
                perp_unit_initial = np.array([-lado_unit[1], lado_unit[0]])
                
                # Verificar y ajustar la dirección del vector perpendicular
                direccion_esperada = direcciones_esperadas[nombre]
                if np.dot(perp_unit_initial, direccion_esperada) > 0:
                    perp_unit = perp_unit_initial
                else:
                    perp_unit = -perp_unit_initial
                
                # Inicializar valores
                max_distancia = 0
                punto_max = None
                proyeccion_max = None
                
                # Analizar todos los puntos del contorno para encontrar el más lejano en la dirección correcta
                for punto in contorno_pts:
                    # Vector desde p1 al punto
                    vec_p1_to_punto = punto - np.array(p1, dtype=float)
                    
                    # Proyección escalar sobre el vector del lado
                    proj_lado = np.dot(vec_p1_to_punto, lado_unit)
                    
                    # Si la proyección está dentro del segmento de línea
                    if 0 <= proj_lado <= lado_len:
                        # Calcular punto proyectado en el lado
                        punto_proj = np.array(p1, dtype=float) + proj_lado * lado_unit
                        
                        # Vector desde punto proyectado al punto
                        vec_proj_to_punto = punto - punto_proj
                        
                        # Distancia perpendicular
                        dist_perp = np.linalg.norm(vec_proj_to_punto)
                        
                        # Solo considerar el punto si está en la dirección correcta
                        if dist_perp > max_distancia and np.dot(vec_proj_to_punto, direccion_esperada) > 0:
                            max_distancia = dist_perp
                            punto_max = punto
                            proyeccion_max = punto_proj
            
            # Guardar el resultado para este lado
            puntos_max_por_lado[nombre] = punto_max
            distancias_max_por_lado[nombre] = max_distancia
            proyecciones_por_lado[nombre] = proyeccion_max
        
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
            
            print(f"{nombre}: X={X:.2f}px, D={D:.2f}px, C={C:.2f}%")
        
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
              max_abombamiento_pix, "px, lo que equivale a", max_porcentaje, "%")
        
        # Crear diccionario con todos los resultados por lado
        abombamientos_por_lado = {
            lado: {
                'X_px': abombamientos_pixeles.get(lado, 0),
                'D_px': np.linalg.norm(np.array(lados[lado][0], dtype=float) - np.array(lados[lado][1], dtype=float)),
                'C_porcentaje': abombamientos_porcentuales.get(lado, 0),
                'punto_max': puntos_max_por_lado.get(lado, None),
                'proyeccion': proyecciones_por_lado.get(lado, None),
                'vertices': lados.get(lado, (None, None))
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
                
                # Crear texto para este lado
                texto_lado = f"{lado}: X={X_px:.1f}px, D={D_px:.1f}px, C={C_porcentaje:.1f}%"
                
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

def visualizar_abombamiento_enhanced(image, contorno, contorno_principal):
    """
    Visualización mejorada con vectores perpendiculares para X y etiquetas claras para D
    con corrección para asegurar que las proyecciones se realicen en la dirección correcta.
    
    Args:
        image: Imagen original
        contorno: Vértices de la palanquilla [top-left, top-right, bottom-right, bottom-left]
        contorno_principal: Contorno completo de la palanquilla
        
    Returns:
        Imagen con visualización mejorada y diccionario con resultados por lado
    """
    try:
        # Verificar entradas
        if image is None or contorno is None:
            print("Error: Entrada inválida para visualizar abombamiento")
            if image is not None:
                return image.copy()
            return None, {}
            
        # Convertir contorno a array numpy si no lo es ya
        contorno = np.array(contorno, dtype=np.float32)
        
        # Crear una copia de la imagen para no modificar la original
        img_resultado = image.copy()
        
        # Definir los lados como pares de vértices
        lados = {
            "Lado 1 (Top)": (contorno[0], contorno[1]),
            "Lado 2 (Right)": (contorno[1], contorno[2]),
            "Lado 3 (Bottom)": (contorno[2], contorno[3]),
            "Lado 4 (Left)": (contorno[3], contorno[0])
        }
        
        # Colores específicos para cada lado (BGR)
        colores = {
            "Lado 1 (Top)": (255, 0, 0),      # Azul
            "Lado 2 (Right)": (0, 255, 0),    # Verde
            "Lado 3 (Bottom)": (0, 0, 255),   # Rojo
            "Lado 4 (Left)": (0, 255, 255)    # Amarillo
        }
        
        # Índices numerados para X y D
        indices = {
            "Lado 1 (Top)": "1",
            "Lado 2 (Right)": "2",
            "Lado 3 (Bottom)": "3",
            "Lado 4 (Left)": "4"
        }
        
        # Título principal
        cv2.putText(img_resultado, "Análisis de Abombamiento", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Extraer todos los puntos del contorno principal
        try:
            contorno_pts = contorno_principal.reshape(-1, 2)
        except Exception as e:
            print(f"Error al procesar contorno principal: {e}")
            contorno_pts = np.array([])
        
        # Guardar los resultados de cada lado
        resultados_por_lado = {}
        
        # Definir vectores de dirección esperados para cada lado
        direcciones_esperadas = {
            "Lado 1 (Top)": np.array([0, -1]),     # Hacia arriba
            "Lado 2 (Right)": np.array([1, 0]),    # Hacia la derecha
            "Lado 3 (Bottom)": np.array([0, 1]),   # Hacia abajo
            "Lado 4 (Left)": np.array([-1, 0])     # Hacia la izquierda
        }
        
        # Para cada lado, encontrar el punto más lejano y dibujar las mediciones
        y_offset = 70  # Posición inicial para texto
        for nombre_lado, (p1, p2) in lados.items():
            # Convertir a tupla de enteros para dibujar
            p1 = tuple(np.array(p1).astype(int))
            p2 = tuple(np.array(p2).astype(int))
            
            # Obtener color específico e índice
            color_lado = colores[nombre_lado]
            indice = indices[nombre_lado]
            
            # Dibujar la línea del lado con su color específico
            cv2.line(img_resultado, p1, p2, color_lado, 2)
            
            # Calcular punto medio del lado para etiqueta D
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2)
            
            # Calcular longitud del lado (D)
            D = np.linalg.norm(np.array(p1) - np.array(p2))
            
            # Etiqueta D con su índice
            etiqueta_D = f"D{indice}={D:.1f}px"
            
            # Colocar etiqueta D cerca del punto medio del lado
            # Determinar posición relativa según el lado
            if nombre_lado == "Lado 1 (Top)":
                offset_D = (0, -20)  # Arriba
            elif nombre_lado == "Lado 2 (Right)":
                offset_D = (20, 0)   # Derecha
            elif nombre_lado == "Lado 3 (Bottom)":
                offset_D = (0, 20)   # Abajo
            else:  # Lado 4 (Left)
                offset_D = (-20, 0)  # Izquierda
                
            pos_D = (mid_x + offset_D[0], mid_y + offset_D[1])
            
            # Agregar fondo negro para mejor visibilidad
            (text_width, text_height), _ = cv2.getTextSize(etiqueta_D, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_resultado, 
                         (pos_D[0] - 5, pos_D[1] - text_height - 5), 
                         (pos_D[0] + text_width + 5, pos_D[1] + 5), 
                         (0, 0, 0), -1)
            
            # Dibujar etiqueta D
            cv2.putText(img_resultado, etiqueta_D, pos_D, 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_lado, 2)
            
            # Calcular vector del lado (para encontrar perpendicular)
            lado_vec = np.array(p2) - np.array(p1)
            lado_len = np.linalg.norm(lado_vec)
            
            if lado_len > 0:
                # Vector unitario del lado
                lado_unit = lado_vec / lado_len
                
                # Vector perpendicular (rotación de 90 grados)
                # CORRECCIÓN: Asegurar que el vector perpendicular apunta en la dirección correcta
                # El vector perpendicular debe apuntar hacia afuera de la palanquilla
                direccion_esperada = direcciones_esperadas[nombre_lado]
                
                # Calcular el vector perpendicular inicial
                perp_unit_initial = np.array([-lado_unit[1], lado_unit[0]])
                
                # Verificar si el vector perpendicular inicial apunta en la dirección esperada
                # Si el producto escalar es positivo, están en la misma dirección
                if np.dot(perp_unit_initial, direccion_esperada) > 0:
                    perp_unit = perp_unit_initial
                else:
                    # Si no, invertir la dirección
                    perp_unit = -perp_unit_initial
                
                # Inicializar para encontrar el punto más lejano
                max_distancia = 0
                punto_max = None
                punto_proyectado = None
                
                # Buscar el punto más lejano en dirección perpendicular
                for punto in contorno_pts:
                    # Vector desde p1 al punto
                    vec_p1_to_punto = punto - np.array(p1)
                    
                    # Proyección escalar sobre el vector del lado
                    proj_lado = np.dot(vec_p1_to_punto, lado_unit)
                    
                    # Si la proyección está dentro del segmento de línea
                    if 0 <= proj_lado <= lado_len:
                        # Calcular punto proyectado en el lado
                        punto_proj = np.array(p1) + proj_lado * lado_unit
                        
                        # Vector desde punto proyectado al punto
                        vec_proj_to_punto = punto - punto_proj
                        
                        # Distancia perpendicular
                        dist_perp = np.linalg.norm(vec_proj_to_punto)
                        
                        # CORRECCIÓN: Verificar que el punto está en la dirección correcta
                        # Calculamos el producto escalar del vector punto-proyección con la dirección esperada
                        # Solo consideramos el punto si está en la dirección esperada para ese lado
                        if dist_perp > max_distancia and np.dot(vec_proj_to_punto, direccion_esperada) > 0:
                            max_distancia = dist_perp
                            punto_max = punto
                            punto_proyectado = punto_proj
                
                # Si encontramos un punto máximo
                if punto_max is not None and punto_proyectado is not None:
                    # Convertir a enteros para dibujar
                    punto_max = tuple(punto_max.astype(int))
                    punto_proyectado = tuple(punto_proyectado.astype(int))
                    
                    # Calcular X (distancia perpendicular)
                    X = max_distancia
                    
                    # Calcular C (porcentaje)
                    C = (X / D * 100) if D > 0 else 0
                    
                    # Etiqueta X con su índice
                    etiqueta_X = f"X{indice}={X:.1f}px"
                    
                    # Dibujar punto rojo (punto de máximo abombamiento)
                    cv2.circle(img_resultado, punto_max, 5, (0, 0, 255), -1)
                    
                    # Dibujar punto proyectado
                    cv2.circle(img_resultado, punto_proyectado, 3, (255, 255, 255), -1)
                    
                    # Dibujar línea desde el punto proyectado hasta el punto máximo
                    cv2.line(img_resultado, punto_proyectado, punto_max, (255, 255, 255), 1)
                    
                    # Dibujar punta de flecha
                    arrow_size = 10
                    angle_rad = math.atan2(punto_max[1] - punto_proyectado[1], punto_max[0] - punto_proyectado[0])
                    pt_arrow1 = (int(punto_max[0] - arrow_size * math.cos(angle_rad + math.pi/6)),
                                int(punto_max[1] - arrow_size * math.sin(angle_rad + math.pi/6)))
                    pt_arrow2 = (int(punto_max[0] - arrow_size * math.cos(angle_rad - math.pi/6)),
                                int(punto_max[1] - arrow_size * math.sin(angle_rad - math.pi/6)))
                    
                    cv2.line(img_resultado, punto_max, pt_arrow1, (255, 255, 255), 1)
                    cv2.line(img_resultado, punto_max, pt_arrow2, (255, 255, 255), 1)
                    
                    # Posición para la etiqueta X (cerca del punto medio del vector X)
                    pos_X = (int((punto_proyectado[0] + punto_max[0])/2), 
                             int((punto_proyectado[1] + punto_max[1])/2))
                    
                    # Agregar fondo negro para mejor visibilidad
                    (text_width, text_height), _ = cv2.getTextSize(etiqueta_X, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img_resultado, 
                                 (pos_X[0] - 5, pos_X[1] - text_height - 5), 
                                 (pos_X[0] + text_width + 5, pos_X[1] + 5), 
                                 (0, 0, 0), -1)
                    
                    # Dibujar etiqueta X
                    cv2.putText(img_resultado, etiqueta_X, pos_X, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Guardar los resultados
                    resultados_por_lado[nombre_lado] = {
                        'X_px': X,
                        'D_px': D,
                        'C_porcentaje': C,
                        'punto_max': punto_max,
                        'punto_proyectado': punto_proyectado
                    }
                    
                    # Formar texto para resumen en la parte superior
                    texto_lado = f"{nombre_lado}: X{indice}={X:.1f}px, D{indice}={D:.1f}px, C={C:.1f}%"
                    
                    # Añadir fondo negro para mejor visibilidad
                    (text_width, text_height), _ = cv2.getTextSize(texto_lado, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(img_resultado, 
                                 (20, y_offset - text_height - 5), 
                                 (20 + text_width + 10, y_offset + 5), 
                                 (0, 0, 0), -1)
                    
                    # Añadir texto con el color correspondiente al lado
                    cv2.putText(img_resultado, texto_lado, (25, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_lado, 1)
                    
                    # Incrementar el offset para el siguiente texto
                    y_offset += 25
        
        return img_resultado, resultados_por_lado
        
    except Exception as e:
        print(f"Error en visualizar_abombamiento_enhanced: {e}")
        import traceback
        traceback.print_exc()
        # Devolver la imagen original en caso de error
        return image.copy() if image is not None else None, {}

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
                'D_px': D_px
            }
        
        return resultados
    except Exception as e:
        print(f"Error al calcular abombamiento: {e}")
        return {}

def generar_reporte_enhanced(resultados_por_lado, file_path):
    """
    Genera un reporte mejorado con etiquetas X1, X2, X3, X4 y D1, D2, D3, D4
    
    Args:
        resultados_por_lado: Diccionario con resultados por lado
        file_path: Ruta donde guardar el reporte
        
    Returns:
        True si se generó correctamente, False en caso contrario
    """
    try:
        # Índices para cada lado
        indices = {
            "Lado 1 (Top)": "1",
            "Lado 2 (Right)": "2",
            "Lado 3 (Bottom)": "3",
            "Lado 4 (Left)": "4"
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE ABOMBAMIENTO MEJORADO\n")
            f.write("="*50 + "\n\n")
            
            for lado, datos in resultados_por_lado.items():
                indice = indices[lado]
                f.write(f"{lado}:\n")
                f.write(f"  X{indice} (Distancia perpendicular): {datos.get('X_px', 0.0):.2f} px\n")
                f.write(f"  D{indice} (Longitud del lado): {datos.get('D_px', 1.0):.2f} px\n")
                f.write(f"  C = X{indice}/D{indice}*100: {datos.get('C_porcentaje', 0.0):.2f}%\n\n")
            
            # Determinar el lado con mayor abombamiento
            if resultados_por_lado:
                lado_max = max(resultados_por_lado.items(), key=lambda x: x[1]['C_porcentaje'])[0]
                max_abombamiento = resultados_por_lado[lado_max]['C_porcentaje']
                indice_max = indices[lado_max]
                
                f.write("RESUMEN:\n")
                f.write(f"  Lado con mayor abombamiento: {lado_max} (X{indice_max}/D{indice_max})\n")
                f.write(f"  Valor de abombamiento: {max_abombamiento:.2f}%\n\n")
            
            f.write("CRITERIO DE MEDICIÓN:\n")
            f.write("  X es la distancia perpendicular desde el lado hasta el punto más lejano\n")
            f.write("  D es la longitud del lado correspondiente\n")
            f.write("  C es el abombamiento en % calculado como C = X/D*100\n")
        
        return True
    except Exception as e:
        print(f"Error al generar reporte mejorado: {e}")
        return False