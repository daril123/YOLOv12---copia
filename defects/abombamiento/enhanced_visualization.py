import cv2
import numpy as np
import math

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
                return image.copy(), {}
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
        
        # Definir vectores de dirección correctos para cada lado (hacia adentro)
        direcciones_correctas = {
            "Lado 1 (Top)": np.array([0, 1]),      # Hacia abajo (interior)
            "Lado 2 (Right)": np.array([-1, 0]),   # Hacia la izquierda (interior)
            "Lado 3 (Bottom)": np.array([0, -1]),  # Hacia arriba (interior)
            "Lado 4 (Left)": np.array([1, 0])      # Hacia la derecha (interior)
        }
        
        # Guardar los resultados de cada lado
        resultados_por_lado = {}
        
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
            pos_D = (mid_x, mid_y)
            # Determinar ajuste de posición según el lado
            if nombre_lado == "Lado 1 (Top)":
                pos_D = (mid_x, mid_y - 20)  # Arriba
            elif nombre_lado == "Lado 2 (Right)":
                pos_D = (mid_x + 20, mid_y)   # Derecha
            elif nombre_lado == "Lado 3 (Bottom)":
                pos_D = (mid_x, mid_y + 20)   # Abajo
            else:  # Lado 4 (Left)
                pos_D = (mid_x - 20, mid_y)   # Izquierda
            
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
                
                # Vector perpendicular apuntando hacia el interior (CORREGIDO)
                # Usar directamente el vector de dirección correcta predefinido
                perp_unit = direcciones_correctas[nombre_lado]
                
                # Inicializar para encontrar el punto más lejano
                max_distancia = 0
                punto_max = None
                punto_proyectado = None
                
                # Buscar el punto más lejano en dirección perpendicular (hacia adentro)
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
                        
                        # CORRECCIÓN: Verificar que el punto está en la dirección CORRECTA (hacia adentro)
                        # El producto escalar debe ser positivo con nuestra dirección predefinida
                        if dist_perp > max_distancia and np.dot(vec_proj_to_punto, perp_unit) > 0:
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
                    
                    # CORRECCIÓN: Dibujar línea desde el punto proyectado hasta el punto máximo
                    # (ahora correctamente desde el borde hacia el interior)
                    cv2.line(img_resultado, punto_proyectado, punto_max, (255, 255, 255), 1)
                    
                    # Dibujar punta de flecha
                    arrow_size = 10
                    angle_rad = math.atan2(punto_max[1] - punto_proyectado[1], 
                                          punto_max[0] - punto_proyectado[0])
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