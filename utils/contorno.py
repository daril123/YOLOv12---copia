import cv2
import numpy as np
from ultralytics.utils.ops import scale_image

def obtener_contorno_imagen(image, model, conf_threshold=0.35, f_epsilon=0.02):
    """
    Obtiene el contorno de la palanquilla en la imagen usando un modelo YOLO
    
    Args:
        image: Imagen original
        model: Modelo YOLO cargado
        conf_threshold: Umbral de confianza para detección (reducido para mejorar detección)
        f_epsilon: Factor para la aproximación de polígonos
    
    Returns:
        contorno: Puntos del contorno ordenados [top-left, top-right, bottom-right, bottom-left]
        contorno_principal: Contorno completo de la palanquilla
        mask_img: Máscara binaria de la palanquilla
    """
    try:
        # Intentar varias configuraciones si la primera falla
        conf_thresholds = [conf_threshold, 0.25, 0.15]  # Probar con umbrales más bajos si es necesario
        
        for threshold in conf_thresholds:
            try:
                # Realizar predicción con el modelo YOLO
                resultado = model.predict(image, conf=threshold, iou=0.5, device=model.device)[0]
                
                # Verificar si se detectaron máscaras
                if resultado.masks is None:
                    print(f"No se detectaron máscaras con umbral {threshold}. Probando con umbral más bajo.")
                    continue
                
                # Obtener las máscaras binarias
                mask_img = resultado.masks.data.cpu().numpy()
                mask_img = np.moveaxis(mask_img, 0, -1)  # (H, W, N)
                
                # Verificar si hay máscaras disponibles
                if mask_img.shape[-1] == 0:
                    print(f"No hay máscaras disponibles con umbral {threshold}. Probando con umbral más bajo.")
                    continue
                
                # Escalar las máscaras al tamaño de la imagen original
                mask_img = scale_image(mask_img, image.shape)
                mask_img = np.moveaxis(mask_img, -1, 0)  # (N, H, W)
                
                # Tomar la primera máscara (la de mayor confianza)
                mask_img = mask_img[0]
                
                # Convertir la máscara en un array de NumPy
                mask_img = mask_img.astype(np.uint8)
                mask_img = mask_img * 255
                
                # Preprocesamiento de la máscara para mejorar la detección
                kernel = np.ones((5, 5), np.uint8)
                mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
                mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)

                # Encontrar los contornos en la máscara
                contornos, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contornos:
                    print(f"No se encontraron contornos con umbral {threshold}. Probando con umbral más bajo.")
                    continue
                
                # Filtrar contornos pequeños (ruido)
                contornos = [c for c in contornos if cv2.contourArea(c) > 1000]
                
                if not contornos:
                    print(f"No hay contornos significativos con umbral {threshold}. Probando con umbral más bajo.")
                    continue
                
                # Seleccionar el contorno de mayor área (la palanquilla principal)
                contorno_principal = max(contornos, key=cv2.contourArea)
                
                # Intentar diferentes valores de epsilon para la aproximación poligonal
                epsilon_factors = [f_epsilon, 0.01, 0.015, 0.025, 0.03, 0.035, 0.04, 0.05]
                
                for eps_factor in epsilon_factors:
                    perimetro = cv2.arcLength(contorno_principal, True)
                    epsilon = eps_factor * perimetro
                    approx = cv2.approxPolyDP(contorno_principal, epsilon, True)
                    
                    # Si encontramos una aproximación con 4 vértices, la usamos
                    if len(approx) == 4:
                        print(f"Se encontró aproximación poligonal con 4 vértices usando factor epsilon = {eps_factor}")
                        break
                
                # Si no encontramos una aproximación con 4 vértices, usar el rectángulo mínimo
                if len(approx) != 4:
                    print(f"No se pudo aproximar a 4 vértices (se obtuvieron {len(approx)}). Usando rectángulo mínimo.")
                    rect = cv2.minAreaRect(contorno_principal)
                    box = cv2.boxPoints(rect)
                    approx = np.int0(box).reshape(-1, 1, 2)
                
                # Convertir la forma (4,1,2) a (4,2)
                vertices = approx.reshape(4, 2)
                
                # Reordenar los vértices para mantener orden consistente
                # La suma mínima corresponde a la esquina superior izquierda
                # La suma máxima corresponde a la esquina inferior derecha
                # La diferencia mínima corresponde a la esquina superior derecha
                # La diferencia máxima corresponde a la esquina inferior izquierda
                s = np.sum(vertices, axis=1)
                diff = np.diff(vertices, axis=1).reshape(-1)
                
                ordered = np.zeros((4, 2), dtype=np.int32)
                min_sum_idx = int(np.argmin(s))
                max_sum_idx = int(np.argmax(s))
                min_diff_idx = int(np.argmin(diff))
                max_diff_idx = int(np.argmax(diff))
                
                ordered[0] = vertices[min_sum_idx]    # top-left
                ordered[2] = vertices[max_sum_idx]    # bottom-right
                ordered[1] = vertices[min_diff_idx]   # top-right
                ordered[3] = vertices[max_diff_idx]   # bottom-left
                
                # Verificar que los vértices no estén demasiado juntos o colineales
                valid = True
                min_distance = float('inf')
                
                for i in range(4):
                    for j in range(i+1, 4):
                        dist = np.sqrt(np.sum((ordered[i] - ordered[j])**2))
                        min_distance = min(min_distance, dist)
                
                # Si los vértices están demasiado juntos, puede indicar una mala detección
                if min_distance < 20:
                    print(f"Advertencia: Vértices demasiado cercanos (distancia mínima = {min_distance}).")
                    valid = False
                
                # Verificar que el cuadrilátero tenga un área razonable
                area = cv2.contourArea(ordered)
                if area < 10000:  # Valor umbral arbitrario, ajustar según el tamaño de las imágenes
                    print(f"Advertencia: Área del cuadrilátero demasiado pequeña ({area}).")
                    valid = False
                
                if valid:
                    return ordered, contorno_principal, mask_img
                
                print("La detección no cumple los criterios de validación. Probando con umbral más bajo.")
            
            except Exception as e:
                print(f"Error en la detección con umbral {threshold}: {e}")
                continue
        
        # Si todos los intentos fallan, recurrir a un método alternativo
        print("Todos los intentos de detección fallaron. Recurriendo a método alternativo.")
        
        # Método alternativo: detección basada en umbralización
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Aplicar umbral adaptativo
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Aplicar umbral de Otsu
            _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Combinar ambos métodos
            thresh_combined = cv2.bitwise_or(thresh, thresh_otsu)
            
            # Operaciones morfológicas para limpiar
            kernel = np.ones((5, 5), np.uint8)
            thresh_cleaned = cv2.morphologyEx(thresh_combined, cv2.MORPH_CLOSE, kernel)
            thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_OPEN, kernel)
            
            # Encontrar contornos
            contornos, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contornos:
                # Seleccionar el contorno de mayor área
                contorno_principal = max(contornos, key=cv2.contourArea)
                
                # Rectángulo mínimo
                rect = cv2.minAreaRect(contorno_principal)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Ordenar los vértices
                s = np.sum(box, axis=1)
                diff = np.diff(box, axis=1).reshape(-1)
                
                ordered = np.zeros((4, 2), dtype=np.int32)
                min_sum_idx = int(np.argmin(s))
                max_sum_idx = int(np.argmax(s))
                min_diff_idx = int(np.argmin(diff))
                max_diff_idx = int(np.argmax(diff))
                
                ordered[0] = box[min_sum_idx]    # top-left
                ordered[2] = box[max_sum_idx]    # bottom-right
                ordered[1] = box[min_diff_idx]   # top-right
                ordered[3] = box[max_diff_idx]   # bottom-left
                
                return ordered, contorno_principal, thresh_cleaned
            
        except Exception as e:
            print(f"Error en el método alternativo: {e}")
        
        # Si todo falla, usar los bordes de la imagen
        h, w = image.shape[:2]
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        contour = np.array(corners).reshape(-1, 1, 2)
        
        # Crear una máscara simple que cubra toda la imagen
        simple_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        print("Usando bordes de la imagen como último recurso.")
        return corners, contour, simple_mask
        
    except Exception as e:
        print(f"Error general en obtener_contorno_imagen: {e}")
        import traceback
        traceback.print_exc()
        
        # Si ocurre un error fatal, devolver los bordes de la imagen
        h, w = image.shape[:2]
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        contour = np.array(corners).reshape(-1, 1, 2)
        
        # Crear una máscara simple que cubra toda la imagen
        simple_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        print("Error en detección de contorno. Usando bordes de la imagen como último recurso.")
        return corners, contour, simple_mask

def rotar_imagen_lado(image, lado_rotar, contorno):
    """
    Rota la imagen para que el lado especificado quede recto (horizontal o vertical)
    
    Args:
        image: Imagen original
        lado_rotar: Lado a rotar ("Lado 1 (Top)", "Lado 2 (Right)", etc.)
        contorno: Vértices de la palanquilla [top-left, top-right, bottom-right, bottom-left]
    
    Returns:
        Imagen rotada
    """
    try:
        # Verificar que tenemos 4 vértices válidos
        if contorno is None or len(contorno) != 4:
            print("Error: Contorno inválido para rotación de imagen")
            return image
            
        # Calcular los ángulos según cada lado
        angulos_rotacion_base = {"Lado 1 (Top)": 180, "Lado 2 (Right)": 270, "Lado 3 (Bottom)": 180, "Lado 4 (Left)": 270}
        angles = {}  # Diccionario para almacenar los ángulos con su etiqueta
        
        # Asegurar que contorno sea un array numpy
        contorno = np.array(contorno, dtype=np.float32)
        
        # Lado 1 (Top): ángulo entre el lado (punto0 -> punto1) y la horizontal (eje X).
        vec_top = contorno[1] - contorno[0]
        if np.all(vec_top == 0):
            angle_top = 0
        else:
            angle_top = np.degrees(np.arctan2(float(vec_top[1]), float(vec_top[0])))
        angles["Lado 1 (Top)"] = angle_top
        
        # Lado 2 (Right): ángulo entre el lado (punto1 -> punto2) y la vertical (eje Y).
        vec_right = contorno[2] - contorno[1]
        if np.all(vec_right == 0):
            angle_right = 0
        else:
            # Se invierte el orden de los componentes para comparar con la vertical:
            angle_right = np.degrees(np.arctan2(float(vec_right[0]), float(vec_right[1])))
        angles["Lado 2 (Right)"] = angle_right
        
        # Lado 3 (Bottom): ángulo entre el lado (punto2 -> punto3) y la horizontal.
        vec_bottom = contorno[3] - contorno[2]
        if np.all(vec_bottom == 0):
            angle_bottom = 0
        else:
            angle_bottom = np.degrees(np.arctan2(float(vec_bottom[1]), float(vec_bottom[0])))
        angles["Lado 3 (Bottom)"] = angle_bottom
        
        # Lado 4 (Left): vector desde bottom-left hasta top-left.
        vec_left = contorno[0] - contorno[3]
        if np.all(vec_left == 0):
            angle_left = 0
        else:
            angle_left = np.degrees(np.arctan2(float(vec_left[0]), float(vec_left[1])))
        angles["Lado 4 (Left)"] = angle_left
        
        # Mostrar los ángulos calculados en consola
        for lado, ang in angles.items():
            print(f"{lado}: {ang:.2f} grados")
        
        # Rotar en base al ángulo del lado especificado
        angulo_lado = angles[lado_rotar]
        print("Lado a rotar: ", lado_rotar)
        print("Ángulo del lado a rotar:", angulo_lado)
        rotation_angle = -angulo_lado + angulos_rotacion_base[lado_rotar]
        print("Ángulo de rotación necesario:", rotation_angle)
        
        # Rotar la imagen usando OpenCV
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        # Se obtiene la matriz de rotación:
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        # Se aplica la transformación a la imagen
        image_rotada = cv2.warpAffine(image, M, (w, h))
        
        return image_rotada
    except Exception as e:
        print(f"Error al rotar la imagen: {e}")
        import traceback
        traceback.print_exc()
        return image  # Devolver la imagen original en caso de error

def redimensionar_imagen(imagen, alto_deseado=1000, padding=20):
    """
    Redimensiona una imagen manteniendo su proporción
    
    Args:
        imagen: Imagen original
        alto_deseado: Altura deseada en píxeles
        padding: Margen a añadir alrededor de la imagen
    
    Returns:
        Imagen redimensionada
    """
    try:
        # Obtener dimensiones originales
        original_height, original_width = imagen.shape[:2]
        
        # Calcular nuevo ancho manteniendo proporción
        aspect_ratio = original_width / original_height
        nuevo_ancho = int(alto_deseado * aspect_ratio)
        
        # Redimensionar la imagen
        imagen = cv2.resize(imagen, (nuevo_ancho, alto_deseado), interpolation=cv2.INTER_AREA)
        
        # Añadir margen si se especifica
        if padding > 0:
            color = (255, 255, 255)  # Color blanco
            imagen = cv2.copyMakeBorder(imagen, padding, padding, padding, padding, 
                                      cv2.BORDER_CONSTANT, value=color)
        
        return imagen
    except Exception as e:
        print(f"Error al redimensionar imagen: {e}")
        return imagen

def recortar_palanquilla(image, model, conf_threshold=0.35, padding=100):
    """
    Recorta la imagen para mantener solo la palanquilla con un margen
    
    Args:
        image: Imagen original
        model: Modelo YOLO cargado
        conf_threshold: Umbral de confianza para detección
        padding: Margen a añadir alrededor de la palanquilla
    
    Returns:
        Imagen recortada
    """
    try:
        # Intentar obtener el contorno de la palanquilla
        vertices, _, _ = obtener_contorno_imagen(image, model, conf_threshold)
        
        if vertices is None or len(vertices) != 4:
            print("No se pudo obtener un contorno válido para recortar la palanquilla.")
            return image
        
        # Calcular los límites del contorno con margen
        x_values = [p[0] for p in vertices]
        y_values = [p[1] for p in vertices]
        
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        
        # Añadir margen y asegurar que los límites están dentro de la imagen
        h, w = image.shape[:2]
        x_min = max(x_min - padding, 0)
        y_min = max(y_min - padding, 0)
        x_max = min(x_max + padding, w - 1)
        y_max = min(y_max + padding, h - 1)
        
        # Recortar la imagen
        image_recortada = image[y_min:y_max, x_min:x_max]
        
        return image_recortada
    except Exception as e:
        print(f"Error al recortar palanquilla: {e}")
        return image

def redimensionar_recortar_palanquilla(image, model, conf_threshold=0.35, 
                                       alto_deseado=1000, 
                                       padding_ampliacion=20, 
                                       padding_recorte=100):
    """
    Combina el redimensionamiento y recorte de la palanquilla en un solo paso
    
    Args:
        image: Imagen original
        model: Modelo YOLO cargado
        conf_threshold: Umbral de confianza para detección
        alto_deseado: Altura deseada en píxeles
        padding_ampliacion: Margen a añadir alrededor de la imagen redimensionada
        padding_recorte: Margen a añadir alrededor de la palanquilla recortada
    
    Returns:
        Imagen procesada
    """
    try:
        # Primero redimensionar
        image = redimensionar_imagen(image, alto_deseado, padding_ampliacion)
        
        # Luego recortar
        image = recortar_palanquilla(image, model, conf_threshold, padding_recorte)
        
        return image
    except Exception as e:
        print(f"Error en redimensionar_recortar_palanquilla: {e}")
        return image