
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt


# IMPRESIÓN DE TABLAS 


def obtener_nombres_columnas(n, m):
    # CONSTRUYE LISTA DE NOMBRES DE COLUMNAS
    cols = ["Z"]
    j = 1
    while j <= n:
        cols.append("x{}".format(j)); j += 1
    i = 1
    while i <= m:
        cols.append("s{}".format(i)); i += 1
    cols.append("RHS")
    return cols

def calcular_ancho_columna(T):
    # CALCULA ANCHO SEGÚN LONGITUD DE LA REPRESENTACIÓN DE CADA FRACCIÓN
    ancho = 6
    i = 0
    while i < len(T):
        fila = T[i]
        j = 0
        while j < len(fila):
            txt = str(fila[j])  # POR EJEMPLO: '3/5' O '2'
            if len(txt) > ancho:
                ancho = len(txt)
            j += 1
        i += 1
    return ancho

def imprimir_encabezado_tabla(n, m, rhs, ancho):
    cols = obtener_nombres_columnas(n, m)
    linea = []
    j = 0
    while j <= rhs:
        linea.append(cols[j].rjust(ancho)); j += 1
    print("    " + " ".join(linea))

def imprimir_tablero(T, n, m, rhs):
    # IMPRIME LA TABLA 
    ancho = calcular_ancho_columna(T)
    imprimir_encabezado_tabla(n, m, rhs, ancho)
    i = 0
    while i < len(T):
        fila = T[i]
        tag = "Z  " if i == 0 else ("R0"+str(i) if i < 10 else "R"+str(i))
        piezas = []
        j = 0
        while j < len(fila):
            piezas.append(str(fila[j]).rjust(ancho)); j += 1
        print(tag + " " + " ".join(piezas))
        i += 1
    print()

# ============================================================
# SIMPLEX 
# ============================================================

def leer_num(x):
    # INTENTA LEER 'a/b' O 'a.b' O ENTERO COMO FRACCION
    x = x.strip()
    if "/" in x:
        num, den = x.split("/")
        return Fraction(int(num), int(den))
    # Fraction PUEDE TOMAR STRING CON DECIMAL 
    return Fraction(x)

def armar_tablero(coef_utilidad, matriz_consumos, recursos_maximos):
    # ARMA EL TABLERO INICIAL CON FRACCIONES EXACTAS
    n = len(coef_utilidad)
    m = len(matriz_consumos)
    rhs = n + m

    z = [Fraction(0)] * (rhs + 1)
    z[0] = Fraction(1)
    j = 0
    while j < n:
        z[1 + j] = -leer_num(str(coef_utilidad[j])); j += 1

    T = [z]
    i = 0
    while i < m:
        r = [Fraction(0)] * (rhs + 1)
        j = 0
        while j < n:
            r[1 + j] = leer_num(str(matriz_consumos[i][j])); j += 1
        r[1 + n + i] = Fraction(1)
        r[rhs] = leer_num(str(recursos_maximos[i]))
        T.append(r)
        i += 1
    return T, n, m, rhs

def elegir_variable_entrante(T, n, m, rhs):
    # BUSCA EN Z EL MÁS NEGATIVO
    z = T[0]
    col = -1
    best = Fraction(0)
    j = 1
    while j < rhs:
        if z[j] < best:
            best = z[j]; col = j
        j += 1
    return col

def elegir_fila_saliente(T, col, m, rhs):
    # REGLA DE LA RAZÓN MÍNIMA EN FRACCIONES
    fila = -1
    ratio_min = None
    i = 1
    while i <= m:
        a = T[i][col]
        if a > 0:
            q = T[i][rhs] / a
            if (ratio_min is None) or (q < ratio_min) or (q == ratio_min and i < fila):
                ratio_min = q; fila = i
        i += 1
    return fila

def pivotear(T, rp, cp, rhs):
    # PIVOTEA 
    p = T[rp][cp]
    if p == 0:
        raise ValueError("pivote = 0")
    j = 0
    while j <= rhs:
        T[rp][j] = T[rp][j] / p
        j += 1
    i = 0
    while i < len(T):
        if i != rp:
            f = T[i][cp]
            if f != 0:
                j = 0
                while j <= rhs:
                    T[i][j] = T[i][j] - f * T[rp][j]
                    j += 1
        i += 1

def simplex_max(coef_utilidad, matriz_consumos, recursos_maximos, max_iter=10000):
    # RESUELVE POR SIMPLEX 
    T, n, m, rhs = armar_tablero(coef_utilidad, matriz_consumos, recursos_maximos)

    # HOLGURAS
    base = [None] * (m + 1)
    i = 1
    while i <= m:
        base[i] = n + i; i += 1

    print("======================================")
    print("TABLERO INICIAL")
    print("======================================")
    imprimir_tablero(T, n, m, rhs)

    it = 0
    while it < max_iter:
        it += 1
        c_in = elegir_variable_entrante(T, n, m, rhs)
        if c_in == -1:
            break
        r_out = elegir_fila_saliente(T, c_in, m, rhs)
        if r_out == -1:
            break
        pivotear(T, r_out, c_in, rhs)
        base[r_out] = c_in

        print("--------------------------------------")
        print("ITERACIÓN:", it)
        imprimir_tablero(T, n, m, rhs)

    # RECONSTRUYE x* EN FRACCIONES
    x = [Fraction(0)] * n
    i = 1
    while i <= m:
        col = base[i]
        val = T[i][rhs]
        if 1 <= col <= n:
            x[col - 1] = val
        i += 1
    Z = T[0][rhs]

    print("=========== RESULTADO ===========")
    print("Z*   :", str(Z))
    print("x*   :", "[" + ", ".join(str(v) for v in x) + "]")
    print("================================\n")

    return {"Z": Z, "x": x, "tabla": T, "iteraciones": it}

# ============================================================
# MÉTODO GRÁFICO 
# ============================================================

def interseccion(linea1, linea2):
    a1, b1, c1 = linea1
    a2, b2, c2 = linea2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:
        return None
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return (x, y)

def punto_dentro(A, b, pt, eps=1e-9):
    xvec = np.array([[pt[0]], [pt[1]]], dtype=float)
    lhs = (np.array(A, dtype=float) @ xvec).flatten()
    return np.all(lhs <= np.array(b, dtype=float) + eps) and pt[0] >= -eps and pt[1] >= -eps

def metodo_grafico_max_2vars(coef_utilidad, matriz_consumos, recursos_maximos, dibujar=True, verbose=True):
    if len(coef_utilidad) != 2:
        raise ValueError("EL MÉTODO GRÁFICO SOLO SOPORTA 2 VARIABLES.")
    A = np.array(matriz_consumos, dtype=float)
    b = np.array(recursos_maximos, dtype=float)
    c = np.array(coef_utilidad, dtype=float)

    lineas = []
    i = 0
    while i < A.shape[0]:
        a1, a2 = A[i, 0], A[i, 1]
        lineas.append((a1, a2, b[i])); i += 1
    lineas.append((1.0, 0.0, 0.0))
    lineas.append((0.0, 1.0, 0.0))

    candidatos = []
    i = 0
    while i < len(lineas):
        j = i + 1
        while j < len(lineas):
            P = interseccion(lineas[i], lineas[j])
            if P is not None and punto_dentro(A, b, P):
                candidatos.append((max(0.0, P[0]), max(0.0, P[1])))
            j += 1
        i += 1

    i = 0
    while i < A.shape[0]:
        a1, a2 = A[i, 0], A[i, 1]; bi = b[i]
        if abs(a1) > 1e-12:
            x_int = bi / a1
            if x_int >= 0 and punto_dentro(A, b, (x_int, 0.0)):
                candidatos.append((x_int, 0.0))
        if abs(a2) > 1e-12:
            y_int = bi / a2
            if y_int >= 0 and punto_dentro(A, b, (0.0, y_int)):
                candidatos.append((0.0, y_int))
        i += 1

    if punto_dentro(A, b, (0.0, 0.0)):
        candidatos.append((0.0, 0.0))

    unicos = []
    for p in candidatos:
        if not any(abs(p[0]-q[0])<1e-8 and abs(p[1]-q[1])<1e-8 for q in unicos):
            unicos.append(p)
    candidatos = unicos

    valores = [(p, float(c[0]*p[0] + c[1]*p[1])) for p in candidatos]
    if not valores:
        raise ValueError("REGIÓN FACTIBLE VACÍA.")

    mejor_p, mejor_z = max(valores, key=lambda t: t[1])

    if dibujar:
        V = np.array([p for p, _ in valores], dtype=float)
        cx, cy = float(np.mean(V[:,0])), float(np.mean(V[:,1]))
        ang = np.arctan2(V[:,1]-cy, V[:,0]-cx)
        Vh = V[np.argsort(ang)]
        xmax = max(1.0, float(np.max(V[:,0])*1.1))
        ymax = max(1.0, float(np.max(V[:,1])*1.1))

        plt.figure()
        xs = np.linspace(0, xmax, 400)
        for i in range(A.shape[0]):
            a1, a2 = A[i,0], A[i,1]; bi = b[i]
            if abs(a2) > 1e-12:
                ys = (bi - a1*xs)/a2; plt.plot(xs, ys, linewidth=1)
            else:
                if abs(a1) > 1e-12: plt.axvline(bi/a1, linewidth=1)
        plt.axhline(0, linewidth=1); plt.axvline(0, linewidth=1)
        if len(Vh) >= 3: plt.fill(Vh[:,0], Vh[:,1], alpha=0.2)
        plt.scatter([mejor_p[0]],[mejor_p[1]], s=60, zorder=5)
        plt.title("REGIÓN FACTIBLE Y PUNTO ÓPTIMO (MÉTODO GRÁFICO)")
        plt.xlabel("x1"); plt.ylabel("x2"); plt.xlim(0, xmax); plt.ylim(0, ymax); plt.grid(True)
        plt.show()

    print("=========== RESULTADO (GRÁFICO) ===========")
    print("Z*   :", f"{mejor_z:.4f}")
    print("x*   :", "[" + ", ".join(f"{v:.4f}" for v in [mejor_p[0], mejor_p[1]]) + "]")
    print("==========================================\n")

    return {"x": [float(mejor_p[0]), float(mejor_p[1])], "Z": float(mejor_z), "candidatos": valores}

# ============================================================
# ENTRADAS
# ============================================================

def pedir_modelo_lp():
    print("RESULCIÓN DE PROBLEMAS DE ECUACIONES LINEALES")
    n = int(input("NÚMERO DE VARIABLES (n): ").strip())
    m = int(input("NÚMERO DE RESTRICCIONES (m): ").strip())

    print("COEFICIENTES DE LA FUNCIÓN OBJETIVO (PUEDES ESCRIBIR 3/5 O 2.5 O 2):")
    partes = input(f"c1 c2 ... c{n}: ").strip().split()
    c = [leer_num(p) for p in partes]
    if len(c) != n:
        raise ValueError("NÚMERO DE COEFICIENTES DE OBJETIVO DISTINTO A n.")

    A = []; b = []
    print("\nINTRODUCE CADA RESTRICCIÓN EN FORMATO: a1 a2 ... an | bi  (TODAS EN <=)")
    i = 0
    while i < m:
        fila = input(f"R{i+1}: ").strip()
        if "|" not in fila:
            raise ValueError("FALTA EL SEPARADOR '|' PARA EL TÉRMINO INDEPENDIENTE (b).")
        lhs, rhs = fila.split("|")
        a = [leer_num(p) for p in lhs.strip().split()]
        if len(a) != n:
            raise ValueError("LA RESTRICCIÓN NO TIENE EXACTAMENTE n COEFICIENTES.")
        A.append(a); b.append(leer_num(rhs.strip()))
        i += 1
    return c, A, b

def pedir_metodo():
    print("\nSELECCIONA MÉTODO:")
    print("1) SIMPLEX (TABLAS EN FRACCIONES)")
    print("2) GRÁFICO (SOLO 2 VARIABLES)")
    return input("OPCIÓN: ").strip()

# ============================================================
# FLUJO INTERACTIVO
# ============================================================

c, A, b = pedir_modelo_lp()
opcion = pedir_metodo()

if opcion == "1":
    print("\n>> RESOLVIENDO CON SIMPLEX...")
    _ = simplex_max(c, A, b)
elif opcion == "2":
    if len(c) != 2:
        print("EL MÉTODO GRÁFICO SOLO SOPORTA 2 VARIABLES. VUELVE A INTENTAR.")
    else:
        print("\n>> RESOLVIENDO CON MÉTODO GRÁFICO...")
        _ = metodo_grafico_max_2vars([float(v) for v in c],
                                     [[float(x) for x in fila] for fila in A],
                                     [float(x) for x in b],
                                     dibujar=True, verbose=False)
else:
    print("OPCIÓN INVÁLIDA.")
