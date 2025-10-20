def armar_tablero(coef_utilidad, matriz_consumos, recursos_maximos):
    n = len(coef_utilidad)
    m = len(matriz_consumos)
    rhs = n + m

    z = [0.0] * (rhs + 1)
    z[0] = 1.0
    for j in range(n):
        z[1 + j] = -float(coef_utilidad[j])

    T = [z]
    for i in range(m):
        r = [0.0] * (rhs + 1)
        for j in range(n):
            r[1 + j] = float(matriz_consumos[i][j])
        r[1 + n + i] = 1.0
        r[rhs] = float(recursos_maximos[i])
        T.append(r)
    return T, n, m, rhs


def elegir_variable_entrante(T, n, m, rhs):
    z = T[0]
    col = -1
    best = 0.0
    for j in range(1, rhs):
        if z[j] < best:
            best = z[j]
            col = j
    return col


def elegir_fila_saliente(T, col, m, rhs):
    fila = -1
    ratio_min = None
    for i in range(1, m + 1):
        a = T[i][col]
        if a > 0:
            q = T[i][rhs] / a
            if (ratio_min is None) or (q < ratio_min - 1e-12) or (abs(q - ratio_min) < 1e-12 and i < fila):
                ratio_min = q
                fila = i
    return fila


def pivotear(T, rp, cp, rhs):
    p = T[rp][cp]
    if abs(p) < 1e-15:
        raise ValueError("pivote ~0")
    for j in range(rhs + 1):
        T[rp][j] = T[rp][j] / p
    for i in range(len(T)):
        if i != rp:
            f = T[i][cp]
            if abs(f) > 1e-15:
                for j in range(rhs + 1):
                    T[i][j] = T[i][j] - f * T[rp][j]


def _mostrar_tablero(T):
    for i, fila in enumerate(T):
        tag = "Z" if i == 0 else ("R%02d" % i)
        s = []
        for v in fila:
            if abs(v - int(v)) < 1e-12:
                s.append(str(int(round(v))))
            else:
                s.append("{:.4f}".format(v))
        print(tag, "|", " ".join(s))


def simplex_max(coef_utilidad, matriz_consumos, recursos_maximos, max_iter=10000):
    T, n, m, rhs = armar_tablero(coef_utilidad, matriz_consumos, recursos_maximos)
    base = [None] * (m + 1)
    for i in range(1, m + 1):
        base[i] = n + i

    print("TABLERO_INICIAL")
    _mostrar_tablero(T)

    it = 0
    while it < max_iter:
        it += 1
        c_in = elegir_variable_entrante(T, n, m, rhs)
        if c_in == -1:
            estado = "OPTIMO"
            break
        r_out = elegir_fila_saliente(T, c_in, m, rhs)
        if r_out == -1:
            estado = "NO_ACOTADO"
            print("NO_ACOTADO col", c_in)
            break
        print("PIVOTE fila", r_out, "col", c_in, "val", "{:.4f}".format(T[r_out][c_in]))
        pivotear(T, r_out, c_in, rhs)
        base[r_out] = c_in
        _mostrar_tablero(T)
    else:
        estado = "LIMITE_ITERACIONES"

    x = [0.0] * n
    for i in range(1, m + 1):
        col = base[i]
        val = T[i][rhs]
        if 1 <= col <= n:
            x[col - 1] = val

    Z = T[0][rhs]
    return {"Z": Z, "x": x, "tabla": T, "estado": estado, "iteraciones": it}





valores_variables = [60, 50]   
restricciones = [
    [3, 2],  
    [2, 1], 
    [1, 2],  
]
recursos_maximos = [240, 200, 100]

resultado = simplex_max(valores_variables, restricciones, recursos_maximos)
print("ESTADO:", resultado["estado"])
print("Z*:", resultado["Z"])
print("x*:", resultado["x"])
