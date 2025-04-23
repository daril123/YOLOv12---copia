import cv2
import numpy as np
from ultralytics.utils.ops import scale_image

def obtener_contorno_imagen(image, model, conf_threshold, f_epsilon=0.02):
    """
    Obtiene el contorno de la palanquilla en la imagen usando un modelo YOLO
    
    Args:
        image: Imagen original
        model: Modelo YOLO cargado
        conf_threshold: Umbral de confianza para detección
        f_epsilon: Factor para la aproximación de polígonos
    
    Returns:
        contorno: Puntos del contorno ordenados [top-left, top-right, bottom-right, bottom-left]
        contorno_principal: Contorno completo de la palanquilla
        mask_img: Máscara binaria de la palanquilla
    """
    # Realizar predicción con el modelo YOLO
    resultado = model.predict(image, conf=conf_threshold, device=model.device)[0]
    
    # Verificar si se detectaron máscaras
    if resultado.masks is None:
        print("No se detectaron máscaras en la imagen.")
        return None, None, None
    
    # Obtener las máscaras binarias y las confianzas de las detecciones en la imagen
    conf_mask = resultado.boxes.conf.cpu().numpy()
    print("Confianza de la máscara detectada: ", conf_mask)
    
    # Procesar la máscara
    mask_img = resultado.masks.data.cpu().numpy()
    mask_img = np.moveaxis(mask_img, 0, -1)  # (H, W, N)
    
    # Escalar las máscaras al tamaño de la imagen original
    mask_img = scale_image(mask_img, image.shape)
    mask_img = np.moveaxis(mask_img, -1, 0)  # (N, H, W)
    
    # Tomar la primera máscara
    if mask_img.shape[0] > 0:
        mask_img = mask_img[0]
    else:
        print("No se encontraron máscaras")
        return None, None, None
    
    # Convertir la máscara en un array de NumPy
    mask_img = mask_img.astype(np.uint8)
    mask_img = mask_img * 255

    # Encontrar los contornos en la máscara
    contornos, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        print("No se encontraron contornos en la máscara.")
        return None, None, None
    
    # Seleccionar el contorno de mayor área (la caja principal)
    contorno_principal = max(contornos, key=cv2.contourArea)
    
    # Aproximar el contorno a un polígono con 4 vértices
    perimetro = cv2.arcLength(contorno_principal, True)
    epsilon = f_epsilon * perimetro  # Ajusta este factor según convenga
    approx = cv2.approxPolyDP(contorno_principal, epsilon, True)

    # Si no obtenemos exactamente 4 vértices, intentar con otros valores de epsilon
    if len(approx) != 4:
        print(f"La aproximación no arrojó 4 vértices, se obtuvieron: {len(approx)}")
        
        if len(approx) < 4:
            print("Se requieren al menos 4 vértices para formar un rectángulo.")
            
            # Intentar con diferentes valores de epsilon
            for eps in [0.01, 0.015, 0.025, 0.03, 0.035, 0.04]:
                test_epsilon = eps * perimetro
                test_approx = cv2.approxPolyDP(contorno_principal, test_epsilon, True)
                print(f"Epsilon {eps}: {len(test_approx)} vértices")
                
                if len(test_approx) == 4:
                    approx = test_approx
                    print(f"Se encontró una buena aproximación con epsilon = {eps}")
                    break
        
        # Si aún no tenemos 4 vértices, usar el rectángulo mínimo
        if len(approx) != 4:
            rect = cv2.minAreaRect(contorno_principal)
            box = cv2.boxPoints(rect)
            approx = np.int0(box).reshape(-1, 1, 2)
            print("Se usó minAreaRect para obtener 4 vértices")
    
    # Convertir la forma (4,1,2) a (4,2)
    vertices = approx.reshape(4, 2)
    
    # Reordenar los vértices para que tengan un orden consistente:
    # - La suma mínima corresponde a la esquina superior izquierda.
    # - La suma máxima a la esquina inferior derecha.
    # - La diferencia mínima a la esquina superior derecha.
    # - La diferencia máxima a la esquina inferior izquierda.
    s = vertices.sum(axis=1)
    diff = np.diff(vertices, axis=1).reshape(4)
    ordered = np.zeros((4, 2), dtype=np.int32)
    ordered[0] = vertices[np.argmin(s)]       # top-left
    ordered[2] = vertices[np.argmax(s)]       # bottom-right
    ordered[1] = vertices[np.argmin(diff)]    # top-right
    ordered[3] = vertices[np.argmax(diff)]    # bottom-left
    
    return ordered, contorno_principal, mask_img

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
    # Calcular los ángulos según cada lado
    # Para cada lado se calcula la diferencia entre dos puntos y se obtiene
    # el ángulo entre el vector resultante y el eje de referencia.
    # Los ángulos se calcularán en grados y se toman valores absolutos para ver
    # la desviación con respecto a la horizontal (Top/Bottom) o vertical (Left/Right).
    angulos_rotacion_base = {"Lado 1 (Top)": 180, "Lado 2 (Right)": 270, "Lado 3 (Bottom)": 180, "Lado 4 (Left)": 270}
    angles = {}  # Diccionario para almacenar los ángulos con su etiqueta
    
    # Lado 1 (Top): ángulo entre el lado (punto0 -> punto1) y la horizontal (eje X).
    vec_top = contorno[1] - contorno[0]
    angle_top = np.degrees(np.arctan2(vec_top[1], vec_top[0]))
    angles["Lado 1 (Top)"] = angle_top
    
    # Lado 2 (Right): ángulo entre el lado (punto1 -> punto2) y la vertical (eje Y).
    vec_right = contorno[2] - contorno[1]
    # Se invierte el orden de los componentes para comparar con la vertical:
    angle_right = np.degrees(np.arctan2(vec_right[0], vec_right[1]))
    angles["Lado 2 (Right)"] = angle_right
    
    # Lado 3 (Bottom): ángulo entre el lado (punto2 -> punto3) y la horizontal.
    vec_bottom = contorno[3] - contorno[2]
    angle_bottom = np.degrees(np.arctan2(vec_bottom[1], vec_bottom[0]))
    angles["Lado 3 (Bottom)"] = angle_bottom
    
    # Lado 4 (Left): se calcula usando el vector desde bottom-left hasta top-left:
    # es decir, vector = top-left - bottom-left.
    # En nuestro orden, top-left es contorno[0] y bottom-left es contorno[3].
    vec_left = contorno[0] - contorno[3]
    angle_left = np.degrees(np.arctan2(vec_left[0], vec_left[1]))
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
    # Se rota la imagen alrededor de su centro.
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # Se obtiene la matriz de rotación:
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    # Se aplica la transformación a la imagen
    image_rotada = cv2.warpAffine(image, M, (w, h))
    
    return image_rotada

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

def recortar_palanquilla(image, model, conf_threshold, padding=100):
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
    # Predicción con el modelo
    resultado = model.predict(image, conf=conf_threshold, device=model.device)[0]
    
    # Verificar si se detectaron máscaras
    if resultado.masks is None:
        print("No se detectaron máscaras en la imagen.")
        return image
    
    # Procesar la máscara
    mask_img = resultado.masks.data.cpu().numpy()
    mask_img = np.moveaxis(mask_img, 0, -1)  # (H, W, N)
    
    # Escalar las máscaras al tamaño de la imagen original
    mask_img = scale_image(mask_img, image.shape)
    mask_img = np.moveaxis(mask_img, -1, 0)  # (N, H, W)
    
    # Tomar la primera máscara
    if mask_img.shape[0] > 0:
        mask_img = mask_img[0]
    else:
        print("No se encontraron máscaras")
        return image
    
    # Convertir la máscara en un array binario
    mask_img = mask_img.astype(np.uint8)
    
    # Encontrar los píxeles donde la máscara es 1
    ys, xs = np.where(mask_img > 0)
    
    # Si no hay píxeles en la máscara, devolver la imagen original
    if len(ys) == 0 or len(xs) == 0:
        return image
    
    # Bounding box de esos píxeles
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    
    # Agregar padding (y asegurar que no se salga de los límites)
    height, width = mask_img.shape
    y_min = max(y_min - padding, 0)
    y_max = min(y_max + padding, height)
    x_min = max(x_min - padding, 0)
    x_max = min(x_max + padding, width)
    
    # Recortar la imagen
    image_recortada = image[y_min:y_max, x_min:x_max]
    
    return image_recortada

def redimensionar_recortar_palanquilla(image, model, conf_threshold, 
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
    # Primero redimensionar
    image = redimensionar_imagen(image, alto_deseado, padding_ampliacion)
    
    # Luego recortar
    image = recortar_palanquilla(image, model, conf_threshold, padding_recorte)
    
    return image