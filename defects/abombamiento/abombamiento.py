import cv2
import numpy as np
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
            
        # Definir direcciones esperadas para cada lado
        direcciones_esperadas = {
            "Lado 1 (Top)": np.array([0, -1]),     # Hacia arriba
            "Lado 2 (Right)": np.array([1, 0]),    # Hacia la derecha
            "Lado 3 (Bottom)": np.array([0, 1]),   # Hacia abajo
            "Lado 4 (Left)": np.array([-1, 0])     # Hacia la izquierda
        }
        
        abombamientos = []
        
        # Calcular el abombamiento para cada lado
        for i, (p1, p2, nombre) in enumerate(lados):
            # Vector del lado
            lado_vec = np.array(p2) - np.array(p1)
            lado_len = np.linalg.norm(lado_vec)
            
            if lado_len > 0:
                # Vector unitario del lado
                lado_unit = lado_vec / lado_len
                
                # Vector perpendicular
                perp_unit_initial = np.array([-lado_unit[1], lado_unit[0]])
                
                # Verificar y ajustar la dirección del vector perpendicular
                direccion_esperada = direcciones_esperadas[nombre]
                if np.dot(perp_unit_initial, direccion_esperada) > 0:
                    perp_unit = perp_unit_initial
                else:
                    perp_unit = -perp_unit_initial
                
                # Inicializar para encontrar el punto más lejano
                max_distancia = 0
                
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
                        
                        # Solo considerar el punto si está en la dirección correcta
                        if dist_perp > max_distancia and np.dot(vec_proj_to_punto, direccion_esperada) > 0:
                            max_distancia = dist_perp
            
            # Guardar el abombamiento de este lado
            abombamientos.append(max_distancia)
            print(f"{nombre}: abombamiento = {max_distancia:.2f}")

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

def calcular_abombamiento(contorno, contorno_principal, mask=None):
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
        - abombamientos_por_lado: Diccionario con los valores de abombamiento por lado
    """
    # Verificar que los contornos no son None
    if contorno is None or contorno_principal is None:
        print("Error: Contornos inválidos para calcular abombamiento")
        # Retornar valores por defecto
        return 0.0, 0.0, "Lado 1 (Top)", {}
    
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
            
        abombamientos_por_lado = {}
        
        # Para cada lado, encontrar el punto más lejano en la dirección correcta
        for nombre, (p1, p2) in lados.items():
            # Calcular vector del lado
            lado_vec = np.array(p2, dtype=float) - np.array(p1, dtype=float)
            lado_len = np.linalg.norm(lado_vec)
            
            # Calcular longitud del lado (D)
            D = lado_len
            
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
                
                # Inicializar para encontrar el punto más lejano
                max_distancia = 0
                punto_max = None
                proyeccion_max = None
                
                # Buscar el punto más lejano en dirección perpendicular
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
            
                # Calcular X (distancia máxima)
                X = max_distancia
                
                # Calcular C (porcentaje de abombamiento)
                C = (X / D * 100) if D > 0 else 0
            else:
                # Si el lado tiene longitud cero
                X = 0
                C = 0
                punto_max = None
                proyeccion_max = None
            
            # Guardar los resultados para este lado
            abombamientos_por_lado[nombre] = {
                'X_px': X,
                'D_px': D,
                'C_porcentaje': C,
                'punto_max': punto_max,
                'proyeccion': proyeccion_max
            }
            
            print(f"{nombre}: X={X:.2f}px, D={D:.2f}px, C={C:.2f}%")
        
        # Identificar el lado con mayor abombamiento (según porcentaje)
        if abombamientos_por_lado:
            # Encontrar el lado con valor máximo en el diccionario
            lado_max = max(abombamientos_por_lado.items(), key=lambda x: x[1]['C_porcentaje'])[0]
            max_abombamiento_pix = abombamientos_por_lado[lado_max]['X_px']
            max_porcentaje = abombamientos_por_lado[lado_max]['C_porcentaje']
        else:
            lado_max = "Lado 1 (Top)"
            max_abombamiento_pix = 0
            max_porcentaje = 0
        
        print("El mayor abombamiento se encuentra en", lado_max, "con un valor de", 
              max_abombamiento_pix, "px, lo que equivale a", max_porcentaje, "%")
        
    except Exception as e:
        print(f"Error general en calcular_abombamiento: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, "Lado 1 (Top)", {}
    
    return max_porcentaje, max_abombamiento_pix, lado_max, abombamientos_por_lado