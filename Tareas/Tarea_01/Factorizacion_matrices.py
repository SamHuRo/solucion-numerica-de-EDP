#!/usr/bin/env python
# coding: utf-8

# ---
# 
# <center>
# 
# # **Factorización de de una matriz y algoritmo de sustitución progresiva y regresiva**
# 
# **Made by:**
# 
# Samuel Huertas Rojas
# 
# </center>
# 
# ---

# A continuación, se va a desarrollar el algoritmo utilizado para obtener la factorización de una matriz y los respectivos algoritmos de sustitución progresiva y regresiva. Los cuales se utilizan para resolver ecuaciones del tipo:  <br>
# <center>
# 
# $
# A x = b
# $
# 
# 
# $A = \begin{bmatrix}
# a_{11} & a_{12} & 0      & 0      & 0      \\
# a_{21} & a_{22} & a_{23} & 0      & 0      \\
# 0      & a_{32} & a_{33} & a_{34} & 0      \\
# 0      & 0      & a_{43} & a_{44} & a_{45} \\
# 0      & 0      & 0      & a_{54} & a_{55} \\
# \end{bmatrix}, \quad
# x = \begin{bmatrix}
# x_1 \\
# x_2 \\
# x_3 \\
# x_4 \\
# x_5 
# \end{bmatrix}, \quad
# b = \begin{bmatrix}
# b_1 \\
# b_2 \\
# b_3 \\
# b_4 \\
# b_5 
# \end{bmatrix}$
# 
# </center>
# 
# La factoruzación se realiza: $A = LU$ <br>
# 
# <center>
# 
# $
# A = LU
# $
# 
# $A = \begin{bmatrix}
# a_{11} & a_{12} & 0      & 0      & 0      \\
# a_{21} & a_{22} & a_{23} & 0      & 0      \\
# 0      & a_{32} & a_{33} & a_{34} & 0      \\
# 0      & 0      & a_{43} & a_{44} & a_{45} \\
# 0      & 0      & 0      & a_{54} & a_{55} \\
# \end{bmatrix}, \quad
# L = \begin{bmatrix}
# l_{11} & 0      & 0      & 0      & 0      \\
# l_{21} & l_{22} & 0      & 0      & 0      \\
# 0      & l_{32} & l_{33} & 0      & 0      \\
# 0      & 0      & l_{43} & l_{44} & 0      \\
# 0      & 0      & 0      & l_{54} & l_{55} \\
# \end{bmatrix}, \quad
# U = \begin{bmatrix}
# 1      & u_{12} & 0      & 0      & 0      \\
# 0      & 1      & u_{23} & 0      & 0      \\
# 0      & 0      & 1      & u_{34} & 0      \\
# 0      & 0      & 0      & 1      & u_{45} \\
# 0      & 0      & 0      & 0      & 1      \\
# \end{bmatrix}$
# 
# </center>
# 
# Para no terner que guardar las matrices $A,L,U$ de forma para ahorrar almacenamiento se van a guardar sus componentes en vectores: <br>
# 
# <center>
# 
# $
# \begin{bmatrix}
# a_{1}  & c_{1}  & 0      & 0      & 0      \\
# d_{1}  & a_{2}  & c_{2}  & 0      & 0      \\
# 0      & d_{2}  & a_{3}  & c_{3}  & 0      \\
# 0      & 0      & d_{3}  & a_{4}  & c_{4} \\
# 0      & 0      & 0      & d_{4}  & a_{5} \\
# \end{bmatrix} = \begin{bmatrix}
# l_{1}  & 0      & 0      & 0      & 0      \\
# p_{1}  & l_{2}  & 0      & 0      & 0      \\
# 0      & p_{2}  & l_{3}  & 0      & 0      \\
# 0      & 0      & p_{3}  & l_{4}  & 0      \\
# 0      & 0      & 0      & p_{4}  & l_{5} \\
# \end{bmatrix} 
# \begin{bmatrix}
# 1      & u_{1} & 0      & 0      & 0      \\
# 0      & 1      & u_{2} & 0      & 0      \\
# 0      & 0      & 1      & u_{3} & 0      \\
# 0      & 0      & 0      & 1      & u_{4} \\
# 0      & 0      & 0      & 0      & 1      \\
# \end{bmatrix}
# $
# 
# </center>

# In[1]:


import numpy as np


# ## Factorización de mariz

# 
# $
# \begin{array}{|c|c|c|c|}
# \hline
# \textbf{Ecuaciones} & \textbf{l}            & \textbf{p}  & \textbf{u}     \\ \hline
# a_1 = l_1            & l_1 = a_1             &             &                \\ \hline
# c_1 = l_1 u_1       &                       &             & u_1 = c_1/l_1  \\ \hline
# d_1 = p_1           &                       &  p_1 = d_1  &                \\ \hline
# a_2 = p_1 u_1 + l_2 & l_2 = a_2 - p_1 u_1   &             &                \\ \hline
# c_2 = l_2 u_2       &                       &             & u_2 = c_2/l_2  \\ \hline
# d_2 = p_2           &                       & p_2 = d_2   &                \\ \hline
# a_3 = u_2 p_2 + l_3 & l_3 = a_3 - p_2 u_2   &             &                \\ \hline
# c_3 = u_3 l_3       &                       &             & u_3 = c_3/l_3  \\ \hline
# d_3 = p_3           &                       & p_3 = d_3   &                \\ \hline
# a_4 = u_3 p_3 + l_4 & l_4 = a_4 - p_3 u_3   &             &                \\ \hline
# c_4 = l_4u_4        &                       &             & u_4 = c_4/l_4  \\ \hline
# d_4 = p_4           &                       & p_4 = d_4   &                \\ \hline
# a_5 = u_4 p_4 + l_5 & l_5 = a_5 - p_4 u_4   &             &                \\ \hline
# \end{array}
# $

# In[ ]:


def factorizacion_matriz_nxn(
    a: np.array,  # Vector que contiene los coeficientes de la matriz (diagonal)
    c: np.array,  # Vector que contiene los coeficientes de la matriz (diagonal superior)
    d: np.array,  # Vector que contiene los coeficientes de la matriz (diagonal inferior)
):
    """
    Función para realizar la factorización de una matriz tridiagonal A = L*U
    donde L es una matriz triangular inferior con 1's en la diagonal principal y
    U es una matriz triangular superior.
    Parámetros:
    a : np.array
        Vector que contiene los coeficientes de la matriz (diagonal)
    c : np.array
        Vector que contiene los coeficientes de la matriz (diagonal superior)
    d : np.array
        Vector que contiene los coeficientes de la matriz (diagonal inferior)
    Retorna:
    l : np.array
        Vector que contiene los coeficientes de la matriz L (diagoanl)
    p : np.array
        Vector que contiene los coeficientes de la matriz L (diagonal inferior)
    u : np.array
        Vector que contiene los coeficientes de la matriz U (diagonal)
    """
    l_dig = np.zeros(len(a))
    p = np.zeros(len(d))
    u = np.zeros(len(c))

    l_dig[0] = a[0]
    p = d.copy()
    for i in range(0, len(c)):
        u[i] = c[i] / l_dig[i]
        l_dig[i + 1] = a[i + 1] - p[i] * u[i]

    return l_dig, p, u


# ### Ejemplo

# In[ ]:


def tridiagonal_matrix(a, c, d):
    """
    Construye una matriz tridiagonal a partir de tres vectores:
    a -> diagonal principal
    c -> diagonal superior
    d -> diagonal inferior
    """
    n = len(a)
    M = np.zeros((n, n))

    # Diagonal principal
    np.fill_diagonal(M, a)

    # Diagonal superior
    np.fill_diagonal(M[:-1, 1:], c)

    # Diagonal inferior
    np.fill_diagonal(M[1:, :-1], d)

    return M


# Ejemplo
a = np.array([4, 4, 4, 4, 4])  # diagonal principal
c = np.array([1, 1, 1, 1])  # diagonal superior
d = np.array([-1, -1, -1, -1])  # diagonal inferior

l_diagonal, p, u = factorizacion_matriz_nxn(a, c, d)


# Construcción de las matrices A, L y U
A = tridiagonal_matrix(a, c, d)
print("Matriz a factorizar (A):")
print(A)

L = tridiagonal_matrix(l_diagonal, np.zeros(len(c)), p)
U = tridiagonal_matrix(np.ones(len(a)), u, np.zeros(len(d)))
print("\nMatriz L:")
print(L)

print("\nMatriz U:")
print(U)

C = np.dot(L, U)
print("\nMatriz L*U:")
print(C)


# ## Algoritmos de sustitución

# ### Regresivo

# In[ ]:


# Utilizando la matriz U y el vecotr z, resolver el sistema Ux = z
def susRe_matriz(U, z):
    U = np.array(U, dtype=float)
    z = np.array(z, dtype=float)
    n = len(z)

    x = np.zeros(n)
    x[-1] = z[-1]

    for i in range(n - 2, -1, -1):  # de n-2 hasta 0
        x[i] = z[i] - U[i, i + 1] * x[i + 1]

    return x


# Utilizando los vecotres l, p, u y un vector b, resolver el sistema Ux = z
def susRe_vector(
    u: np.array,  # Vector que contiene los coeficientes de la matriz U (diagonal superior)
    z: np.array,  # Vector que contiene los términos independientes
):
    x = np.zeros(len(z))
    x[-1] = z[-1]
    for i in range(len(z) - 2, -1, -1):  # de n-2 hasta 0
        x[i] = z[i] - u[i] * x[i + 1]

    return x


# ### Progresivo

# In[ ]:


# Utilizando la matriz L y el vector b, resolver el sistema Lz = b
def susPro_matriz(L, b):
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    z = np.zeros(n)
    z[0] = b[0] / L[0, 0]

    for i in range(1, n):
        z[i] = (b[i] - L[i, i - 1] * z[i - 1]) / L[i, i]

    return z


# Utilizando los vecotres l, p y un vector b, resolver el sistema Lz = b
def susPro_vector(
    l_dig: np.array,  # Vector que contiene los coeficientes de la matriz L (diagonal)
    p: np.array,  # Vector que contiene los coeficientes de la matriz L (diagonal inferior)
    b: np.array,  # Vector que contiene los términos independientes
):
    z = np.zeros(len(b))
    z[0] = b[0] / l_dig[0]
    for i in range(1, len(b)):
        z[i] = (b[i] - p[i - 1] * z[i - 1]) / l_dig[i]

    return z

