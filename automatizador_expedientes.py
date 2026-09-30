import time
import os
import re

import cv2
import numpy as np
import pyautogui


DEBUG_DIR = os.path.join(os.getcwd(), "debug_automatizador")
CONFIG_CORTES = os.path.join(os.getcwd(), "configuracion_corte.txt")
os.makedirs(DEBUG_DIR, exist_ok=True)


def leer_porcentajes_celdas(ruta=CONFIG_CORTES, fila=None):
    """Lee porcentajes por celda desde un TXT externo.

    Soporta porcentajes globales:
      celda_0=10
      celda_1=15
      celda_2=35
      celda_3=40

    Y un override por fila opcional:
      fila_06=10,25,30,35
    """
    if not os.path.exists(ruta):
        return None

    valores = {}
    extra_fila = {}
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea or linea.startswith("#"):
                    continue
                if "=" not in linea:
                    continue
                clave, valor = [p.strip() for p in linea.split("=", 1)]
                if re.match(r"^celda_[0-3]$", clave):
                    try:
                        valores[clave] = float(valor)
                    except ValueError:
                        continue
                elif re.match(r"^fila_[0-9]+$", clave):
                    try:
                        partes = [float(x.strip()) for x in valor.split(",") if x.strip()]
                        if len(partes) == 4:
                            extra_fila[clave] = partes
                    except ValueError:
                        continue
    except Exception:
        return None

    if fila is not None:
        clave_fila = f"fila_{fila}"
        if clave_fila in extra_fila:
            total = sum(extra_fila[clave_fila])
            if total > 0:
                return [v / total for v in extra_fila[clave_fila]]

    orden = [f"celda_{i}" for i in range(4)]
    if any(clave not in valores for clave in orden):
        return None

    total = sum(valores[clave] for clave in orden)
    if total <= 0:
        return None

    return [valores[clave] / total for clave in orden]


def leer_pulsaciones_derecha(ruta=CONFIG_CORTES):
    """Lee la cantidad exacta de pulsaciones a la derecha desde el archivo de config."""
    if not os.path.exists(ruta):
        return 8

    try:
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea or linea.startswith("#"):
                    continue
                if "=" not in linea:
                    continue
                clave, valor = [p.strip() for p in linea.split("=", 1)]
                if clave in {"pulsaciones_derecha", "derecha_pulsaciones"}:
                    try:
                        valor_int = int(float(valor))
                        if valor_int > 0:
                            return valor_int
                    except ValueError:
                        continue
    except Exception:
        return 8

    return 8


def leer_roi_config(ruta=CONFIG_CORTES):
    """Lee el porcentaje y píxeles mínimos para el recorte ROI."""
    porcentaje = 15
    min_px = 450

    if not os.path.exists(ruta):
        return porcentaje, min_px

    try:
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea or linea.startswith("#"):
                    continue
                if "=" not in linea:
                    continue
                clave, valor = [p.strip() for p in linea.split("=", 1)]
                if clave in {"roi_porcentaje", "roi_percent"}:
                    try:
                        porcentaje = int(float(valor))
                        if porcentaje <= 0:
                            porcentaje = 15
                    except ValueError:
                        pass
                elif clave in {"roi_min_px", "roi_minimo_px"}:
                    try:
                        min_px = int(float(valor))
                        if min_px <= 0:
                            min_px = 450
                    except ValueError:
                        pass
    except Exception:
        pass

    return porcentaje, min_px


def leer_ancho_base_pantalla(ruta=CONFIG_CORTES):
    """Lee el ancho de referencia de la pantalla para ajustar por resolucion."""
    if not os.path.exists(ruta):
        return 1920

    try:
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea or linea.startswith("#"):
                    continue
                if "=" not in linea:
                    continue
                clave, valor = [p.strip() for p in linea.split("=", 1)]
                if clave in {"ancho_pantalla_base", "screen_width_base", "base_width"}:
                    try:
                        ancho = int(float(valor))
                        if ancho > 0:
                            return ancho
                    except ValueError:
                        continue
    except Exception:
        return 1920

    return 1920


def calcular_pulsaciones_derecha(ruta=CONFIG_CORTES):
    """Ajusta el valor base segun el ancho real de la pantalla del equipo."""
    base = leer_pulsaciones_derecha(ruta)
    ancho_base = leer_ancho_base_pantalla(ruta)
    ancho_actual = pyautogui.size().width
    if ancho_actual <= 0 or ancho_base <= 0:
        return base
    ratio = ancho_actual / ancho_base
    pulsaciones = int(round(base * ratio))
    return max(1, pulsaciones)


def mover_hasta_ultima_columna(pulsaciones=None):
    if pulsaciones is None:
        pulsaciones = leer_pulsaciones_derecha()
    for _ in range(pulsaciones):
        pyautogui.press("right")
        time.sleep(0.03)
    time.sleep(0.08)


def detectar_fila_azul(img_bgr):
    """Busca el bloque azul de la fila activa dentro de la captura completa."""
    h, w = img_bgr.shape[:2]
    b = img_bgr[:, :, 0].astype(np.int16)
    g = img_bgr[:, :, 1].astype(np.int16)
    r = img_bgr[:, :, 2].astype(np.int16)

    mascara_azul = (b > 110) & (b > g + 25) & (b > r + 25)
    num_componentes, _, stats, _ = cv2.connectedComponentsWithStats(
        mascara_azul.astype(np.uint8), 8
    )

    candidatos = []
    for i in range(1, num_componentes):
        x, y, ancho, alto, area = stats[i]
        if ancho > 80 and alto > 8 and area > 300:
            candidatos.append((area, x, y, ancho, alto))

    if not candidatos:
        return None

    _, x, y, ancho, alto = max(candidatos, key=lambda t: t[0])
    y0 = max(0, y - 5)
    y1 = min(h, y + alto + 10)
    x0 = max(0, x - 10)
    x1 = min(w, x + ancho + 10)
    return (x0, y0, x1, y1)


def recortar_ultima_parte_fila(img_bgr, coords):
    x0, y0, x1, y1 = coords
    ancho_total = x1 - x0
    # Lee configuración del ROI desde el archivo de config
    roi_porcentaje, roi_min_px = leer_roi_config()
    # Calcula el ancho del recorte usando el porcentaje y el mínimo configurado
    ancho_ultimos_datos = max(roi_min_px, int(ancho_total * roi_porcentaje / 100))
    x_recorte_izq = max(0, x1 - ancho_ultimos_datos)
    # Mantenemos el margen superior y recortamos un poco más la parte inferior.
    margen_superior = max(0, (y1 - y0) // 8)
    margen_inferior = max(0, (y1 - y0) // 12)
    y0_ajustado = y0 + margen_superior
    y1_ajustado = y1 - margen_inferior
    print(f"    [DEBUG] Recorte: ancho_total={ancho_total}, ancho_recorte={ancho_ultimos_datos} (config: {roi_porcentaje}% / min {roi_min_px}px), x={x_recorte_izq}:{x1}, y={y0_ajustado}:{y1_ajustado}")
    return img_bgr[y0_ajustado:y1_ajustado, x_recorte_izq:x1]


def _parece_digito_cero(c):
    """Identifica el glifo 0 por su contorno (tolerante a distintos tamaños de fuente)."""
    alto, ancho = c.shape
    if not (3 <= ancho <= 14 and 6 <= alto <= 24):
        return False
    if np.count_nonzero(c[0]) < 2 or np.count_nonzero(c[-1]) < 2:
        return False
    if np.count_nonzero(c[:, 0]) / alto < 0.5 or np.count_nonzero(c[:, -1]) / alto < 0.5:
        return False
    if any(np.count_nonzero(f) == ancho for f in c[1:-1]):
        return False
    return True


def _mascara_texto(roi_bgr):
    """Máscara del texto claro dentro de la banda azul, sin líneas de la grilla ni zona blanca inferior."""
    b = roi_bgr[:, :, 0].astype(np.int16)
    r = roi_bgr[:, :, 2].astype(np.int16)
    azul = (b > 150) & (b > r + 60)
    filas = np.where(azul.mean(axis=1) > 0.5)[0]
    if len(filas) == 0:
        return None
    y0, y1 = filas.min(), filas.max()
    cols = np.where(azul[y0:y1 + 1].mean(axis=0) > 0.3)[0]
    x_max = cols.max()
    luz = roi_bgr.min(axis=2) > 130
    banda = luz[y0:y1 + 1, :x_max + 1].copy()
    lineas = banda.mean(axis=0) > 0.9          # líneas verticales de la grilla
    banda[:, lineas] = False
    banda = cv2.morphologyEx(banda.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    banda[:, lineas] = False
    return banda


def extraer_ultimos_4_valores(fila_bgr, paso=0):
    """Lee los 4 valores finales (0 = cero, 1 = distinto de cero)."""
    cv2.imwrite(os.path.join(DEBUG_DIR, f"fila_{paso:02d}_roi.png"), fila_bgr)
    h, w = fila_bgr.shape[:2]
    if w < 20:
        return None

    porcentajes = leer_porcentajes_celdas(fila=paso)
    if porcentajes is None:
        porcentajes = [0.25] * 4
    bordes = [0] + [int(round(w * sum(porcentajes[:i + 1]))) for i in range(3)] + [w]

    mask = _mascara_texto(fila_bgr)
    if mask is None:
        return None
    cv2.imwrite(os.path.join(DEBUG_DIR, f"fila_{paso:02d}_mascara.png"), mask.astype(np.uint8) * 255)

    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    glifos = [[] for _ in range(4)]
    for k in range(1, n):
        x, y, a, alto, area = st[k]
        if area < 6 or a < 3:            # ruido / restos de línea
            continue
        centro = x + a / 2
        idx = next(i for i in range(4) if bordes[i] <= centro < bordes[i + 1]) if centro < bordes[4] else 3
        glifos[idx].append(lab[y:y + alto, x:x + a] == k)

    valores = []
    for idx in range(4):
        valor = 0 if all(_parece_digito_cero(g) for g in glifos[idx]) else 1
        valores.append(valor)
        print(f"[{paso}] Celda {idx}: x={bordes[idx]}:{bordes[idx+1]}, glifos={len(glifos[idx])}, valor={valor}")
        cv2.imwrite(os.path.join(DEBUG_DIR, f"fila_{paso:02d}_celda_{idx}.png"),
                    cv2.cvtColor(fila_bgr[:, bordes[idx]:bordes[idx + 1]], cv2.COLOR_BGR2GRAY))

    print(f"[{paso}] Valores analizados (0=cero, 1=distinto de cero): {valores}")
    return valores


def cumple_patron(valores):
    """Patrón objetivo: 0, 0, x, 0 donde x es cualquier valor distinto de 0."""
    if len(valores) != 4:
        return False
    return valores[0] == 0 and valores[1] == 0 and valores[2] != 0 and valores[3] == 0


def ejecutar_automatizacion(max_filas=20):
    y_anterior = None
    repeticiones = 0
    pulsaciones_derecha = leer_pulsaciones_derecha()

    for paso in range(max_filas):
        # Paso 1: aseguramos que estemos al extremo derecho antes de cada captura
        print(f"[{paso}] Moviendo a la derecha ({pulsaciones_derecha} pulsaciones exactas del TXT)...")
        mover_hasta_ultima_columna(pulsaciones=pulsaciones_derecha)
        time.sleep(0.2)

        # Paso 2: capturamos la pantalla completa sin depender del nombre de la ventana
        print(f"[{paso}] Capturando pantalla...")
        pantalla = pyautogui.screenshot()
        img_bgr = cv2.cvtColor(np.array(pantalla), cv2.COLOR_RGB2BGR)

        # Paso 3: detectamos la fila azul/seleccionada dentro de la imagen
        coords = detectar_fila_azul(img_bgr)
        if coords is None:
            print(f"[{paso}] No se detectó fila activa. Fin.")
            break

        x0, y0, x1, y1 = coords
        print(f"[{paso}] Fila detectada en posición: x={x0}:{x1}, y={y0}:{y1}")

        # Hacemos clic en la región detectada para dirigir la interacción a la ventana visible
        x_click = x0 + 20
        y_click = (y0 + y1) // 2
        pyautogui.click(x_click, y_click)
        time.sleep(0.15)
        print(f"[{paso}] Click hecho sobre la fila detectada para asegurar foco visual.")

        # Presionamos Home para asegurar que estamos al inicio de la fila
        pyautogui.press("home")
        time.sleep(0.1)
        print(f"[{paso}] Home presionado para resetear posición horizontal.")

        # Paso 4: Recortamos la parte derecha con los 4 valores finales
        fila_derecha = recortar_ultima_parte_fila(img_bgr, coords)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"fila_{paso:02d}_derecha.png"), fila_derecha)

        # Paso 5: Extraemos los valores
        valores = extraer_ultimos_4_valores(fila_derecha, paso=paso)

        # Paso 6: Evaluamos si cumple el patrón
        if valores is None:
            print(f"[{paso}] No se pudieron leer los 4 valores. Se pasa a la siguiente fila.")
        else:
            print(f"[{paso}] Valores leídos: {valores}")
            if cumple_patron(valores):
                print(f"[{paso}] CUMPLE patrón 0,0,x,0 -> marcando con ESPACIO")
                pyautogui.press("space")
                time.sleep(0.15)
                print(f"[{paso}] Espacio presionado. Volviendo a mover a la derecha...")
                mover_hasta_ultima_columna(pulsaciones=leer_pulsaciones_derecha())
                time.sleep(0.1)
            else:
                print(f"[{paso}] No cumple. Se continúa.")

        # Paso 7: Control de fin de grilla (si la fila no cambia, llegamos al fin)
        if y0 == y_anterior:
            repeticiones += 1
            if repeticiones >= 2:
                print("Llegó al final de la grilla. Finalizando.")
                break
        else:
            y_anterior = y0
            repeticiones = 0

        # Paso 8: Movemos a la siguiente fila con Down
        print(f"[{paso}] Presionando tecla Down para ir a la siguiente fila...")
        pyautogui.press("down")
        time.sleep(0.2)


if __name__ == "__main__":
    print("Inicializando automatización sin depender del título de la ventana...")
    time.sleep(1)
    ejecutar_automatizacion()
    input("\nProceso finalizado. Presione Enter para cerrar la consola...")
