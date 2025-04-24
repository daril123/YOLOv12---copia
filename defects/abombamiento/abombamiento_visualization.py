import cv2
import numpy as np

def visualizar_abombamiento_especifico(image, contorno, puntos_max, proyecciones):
    """
    Visualiza el abombamiento exactamente como se requiere:
    - Lado 1 (Top): azul
    - Lado 2 (Right): verde
    - Lado 3 (Bottom): rojo
    - Lado 4 (Left): amarillo
    
    Con X medido desde puntos rojos a las líneas de colores
    y D siendo la longitud de cada línea coloreada.
    
    Args:
        image: Imagen original
        contorno: Vértices de la palanquilla [top-left, top-right, bottom-right, bottom-left]
        puntos_max: Diccionario con los puntos de máximo abombamiento por lado
        proyecciones: Diccionario con las proyecciones de los puntos máximos sobre cada lado
        
    Returns:
        Imagen con la visualización específica del abombamiento
    """
    try:
        # Verificar entradas
        if image is None or contorno is None:
            print("Error: Entrada inválida para visualizar abombamiento")
            if image is not None:
                return image.copy()
            return None
            
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
        
        # Título principal
        cv2.putText(img_resultado, "Análisis de Abombamiento", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Para cada lado, dibujar la línea con su color específico
        y_offset = 70  # Posición inicial para texto
        for nombre_lado, (p1, p2) in lados.items():
            p1 = tuple(np.array(p1).astype(int))
            p2 = tuple(np.array(p2).astype(int))
            
            # Dibujar la línea del lado con su color específico
            color_lado = colores[nombre_lado]
            cv2.line(img_resultado, p1, p2, color_lado, 2)
            
            # Dibujar los vértices como puntos rojos pequeños
            cv2.circle(img_resultado, p1, 5, (0, 0, 255), -1)
            cv2.circle(img_resultado, p2, 5, (0, 0, 255), -1)
            
            # Si tenemos el punto de máximo abombamiento y su proyección
            if nombre_lado in puntos_max and puntos_max[nombre_lado] is not None:
                punto_max = tuple(np.array(puntos_max[nombre_lado]).astype(int))
                
                # Dibujar el punto de máximo abombamiento como punto rojo
                cv2.circle(img_resultado, punto_max, 5, (0, 0, 255), -1)
                
                # Si tenemos la proyección, dibujar la línea X (distancia)
                if nombre_lado in proyecciones and proyecciones[nombre_lado] is not None:
                    proyeccion = tuple(np.array(proyecciones[nombre_lado]).astype(int))
                    
                    # Dibujar línea blanca desde el punto máximo a su proyección (X)
                    cv2.line(img_resultado, punto_max, proyeccion, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Calcular X (distancia) y D (longitud nominal)
                    X = np.linalg.norm(np.array(punto_max) - np.array(proyeccion))
                    D = np.linalg.norm(np.array(p1) - np.array(p2))
                    
                    # Calcular C (porcentaje)
                    C = (X / D * 100) if D > 0 else 0
                    
                    # Formar texto para mostrar
                    texto_lado = f"{nombre_lado}: X={X:.1f}px, D={D:.1f}px, C={C:.1f}%"
                    
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
        
        return img_resultado
        
    except Exception as e:
        print(f"Error en visualizar_abombamiento_especifico: {e}")
        import traceback
        traceback.print_exc()
        # Devolver la imagen original en caso de error
        return image.copy() if image is not None else None


def reanalizar_abombamiento(image, contorno, contorno_principal):
    """
    Realiza un análisis completo de abombamiento según el criterio especificado:
    - X = distancia desde punto rojo al lado coloreado
    - D = longitud del lado coloreado
    - C = X/D*100
    
    Args:
        image: Imagen original
        contorno: Vértices de la palanquilla [top-left, top-right, bottom-right, bottom-left]
        contorno_principal: Contorno completo de la palanquilla
        
    Returns:
        Tuple conteniendo:
        - Imagen visualizada
        - Diccionario con resultados por lado
    """
    try:
        # Verificar entradas
        if image is None or contorno is None or contorno_principal is None:
            print("Error: Entrada inválida para analizar abombamiento")
            return None, {}
            
        # Convertir contorno a array numpy si no lo es ya
        contorno = np.array(contorno, dtype=np.float32)
        
        # Definir los lados como pares de vértices
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
        
        # Encontrar los puntos de máximo abombamiento para cada lado
        puntos_max = {}
        proyecciones = {}
        resultados = {}
        
        for nombre_lado, (p1, p2) in lados.items():
            # Inicializar valores
            max_distancia = 0
            punto_max = None
            proyeccion_max = None
            
            # Calcular la longitud nominal del lado (D)
            D = np.linalg.norm(np.array(p1) - np.array(p2))
            
            # Buscar el punto más lejano de este lado
            for punto in contorno_pts:
                # Vector de la línea
                linea = np.array(p2) - np.array(p1)
                # Magnitud de la línea
                mag_linea = np.linalg.norm(linea)
                
                if mag_linea > 0:
                    # Calcular la distancia perpendicular
                    dist = np.abs(np.cross(linea, punto - np.array(p1))) / mag_linea
                    
                    # Si es la mayor distancia encontrada hasta ahora
                    if dist > max_distancia:
                        max_distancia = dist
                        punto_max = punto
                        
                        # Calcular la proyección del punto sobre la línea
                        linea_norm = linea / mag_linea
                        v = punto - np.array(p1)
                        proj_length = np.dot(v, linea_norm)
                        proyeccion_max = np.array(p1) + proj_length * linea_norm
                        proyeccion_max = tuple(int(x) for x in proyeccion_max)
            
            # Guardar resultados
            puntos_max[nombre_lado] = punto_max
            proyecciones[nombre_lado] = proyeccion_max
            
            # Calcular X (distancia entre punto max y su proyección)
            if punto_max is not None and proyeccion_max is not None:
                X = np.linalg.norm(np.array(punto_max) - np.array(proyeccion_max))
                
                # Calcular C (porcentaje)
                C = (X / D * 100) if D > 0 else 0
                
                # Guardar resultados
                resultados[nombre_lado] = {
                    'X_px': X,
                    'D_px': D,
                    'C_porcentaje': C
                }
                
                print(f"{nombre_lado}: X={X:.2f}px, D={D:.2f}px, C={C:.2f}%")
        
        # Visualizar los resultados
        vis_img = visualizar_abombamiento_especifico(image, contorno, puntos_max, proyecciones)
        
        return vis_img, resultados
    
    except Exception as e:
        print(f"Error en reanalizar_abombamiento: {e}")
        import traceback
        traceback.print_exc()
        return image.copy() if image is not None else None, {}