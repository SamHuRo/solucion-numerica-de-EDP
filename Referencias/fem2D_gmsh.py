import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Función para cargar malla GMSH (deberás implementar o adaptar según tu formato)
def load_gmsh(filename):
    """
    Esta función debe cargar el archivo .msh de GMSH
    Retorna un diccionario con:
    - nbTriangles: número de triángulos
    - nbNod: número de nodos
    - POS: coordenadas de los nodos
    - TRIANGLES: conectividad de triángulos
    - LINES: líneas de frontera
    """
    # Implementación básica - adaptar según formato específico
    import meshio
    mesh = meshio.read(filename)
    
    malla = {
        'nbNod': len(mesh.points),
        'nbTriangles': 0,
        'POS': mesh.points,
        'TRIANGLES': None,
        'LINES': None
    }
    
    # Extraer triángulos
    for cell_type, cell_data in mesh.cells:
        if cell_type == 'triangle':
            malla['TRIANGLES'] = cell_data
            malla['nbTriangles'] = len(cell_data)
        elif cell_type == 'line':
            malla['LINES'] = cell_data
    
    return malla

# Funciones del problema (definir según tu caso)
def f(x):
    """Término fuente"""
    return 0  # Ajustar según tu problema

def g(punto_medio, vec_normal):
    """Condición de Neumann"""
    return 0  # Ajustar según tu problema

def u_d(coord):
    """Condición de Dirichlet"""
    x, y = coord[0], coord[1]
    return np.sin(np.pi * x) * y

# Función exacta
def u_exact_fun(x, y):
    return np.sin(np.pi * x) * y

# ==================== PROGRAMA PRINCIPAL ====================

# Cargar malla
# malla1 = load_gmsh('cuadroD.msh')
malla1 = load_gmsh('cuadroDN.msh')

nel = malla1['nbTriangles']
nver = malla1['nbNod']
Coordinates = malla1['POS'][:, :2]
Elements3 = malla1['TRIANGLES'][:, :3]
LadosFrontera = malla1['LINES']

# Identificar nodos de frontera Dirichlet y Neumann
# Ajustar los índices según tu numeración de fronteras
Dirichlet_indices = np.where(LadosFrontera[:, 2] == 12)[0]
Dirichlet = np.unique(LadosFrontera[Dirichlet_indices, :2])

Neumann_indices = np.where(LadosFrontera[:, 2] == 13)[0]
Neumann = LadosFrontera[Neumann_indices, :2]

# Nodos libres
Freenodes = np.setdiff1d(np.arange(nver), np.unique(Dirichlet))

# Inicializar matrices (usar lil_matrix para construcción eficiente)
A = lil_matrix((nver, nver))
b = np.zeros(nver)

# Ensamblaje y fuerzas volumétricas
for j in range(nel):
    # Coordenadas del elemento
    coord = Coordinates[Elements3[j, :], :].T  # 2x3
    
    # Área del triángulo
    mat_area = np.vstack([np.ones(3), coord])
    areaT = abs(np.linalg.det(mat_area)) / 2
    
    # Matriz de gradientes
    G = np.linalg.solve(mat_area.T, np.vstack([np.zeros((1, 2)), np.eye(2)]))
    
    # Matriz local
    M = areaT * G @ G.T
    
    # Ensamblaje
    indices = Elements3[j, :]
    for i in range(3):
        for k in range(3):
            A[indices[i], indices[k]] += M[i, k]
    
    # Vector de carga (fuerza volumétrica)
    barycentro = np.mean(Coordinates[indices, :], axis=0)
    b[indices] += areaT * f(barycentro) / 3

# Condiciones de Neumann
if len(Neumann) > 0:
    for j in range(Neumann.shape[0]):
        tang = Coordinates[Neumann[j, 0], :] - Coordinates[Neumann[j, 1], :]
        vecnor = np.array([-tang[1], tang[0]])
        punto_medio = np.mean(Coordinates[Neumann[j, :], :], axis=0)
        
        norm_tang = np.linalg.norm(tang)
        flux = norm_tang * g(punto_medio, vecnor) / 2
        
        
        b[Neumann[j, :]] += flux

# Condiciones de Dirichlet
u = np.zeros(nver)
coordD = Coordinates[np.unique(Dirichlet), :].T
u[np.unique(Dirichlet)] = u_d(coordD)

# Convertir A a formato CSR para operaciones eficientes
A = A.tocsr()
b = b - A @ u

# Resolver el sistema
u[Freenodes] = spsolve(A[np.ix_(Freenodes, Freenodes)], b[Freenodes])

# ==================== VISUALIZACIÓN ====================

# Solución aproximada
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_trisurf(Coordinates[:, 0], Coordinates[:, 1], u, 
                 triangles=Elements3, cmap='viridis')
ax1.set_title('Solución aproximada')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')

# Solución exacta
ax2 = fig.add_subplot(122, projection='3d')
u_exact = u_exact_fun(Coordinates[:, 0], Coordinates[:, 1])
ax2.plot_trisurf(Coordinates[:, 0], Coordinates[:, 1], u_exact,
                 triangles=Elements3, cmap='viridis')
ax2.set_title('Solución exacta')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('u')

plt.tight_layout()
plt.show()

# ==================== CÁLCULO DEL ERROR L² ====================

eL2 = 0
slnL2 = 0

for j in range(nel):
    coord = Coordinates[Elements3[j, :], :].T  # 2x3
    
    # Área del triángulo
    mat_area = np.array([[1, 1, 1], coord[0], coord[1]])
    areaT = abs(np.linalg.det(mat_area)) / 2
    
    # Puntos medios de las aristas
    ptm = coord @ np.array([[1/2, 0, 1/2],
                            [1/2, 1/2, 0],
                            [0, 1/2, 1/2]])
    
    # Valores exactos en puntos medios
    Ue = u_exact_fun(ptm[0, :], ptm[1, :])
    
    # Valores aproximados en puntos medios
    Ua = u[Elements3[j, :]] @ np.array([[1/2, 0, 1/2],
                                        [1/2, 1/2, 0],
                                        [0, 1/2, 1/2]])
    
    # Acumular errores
    eL2 += areaT * np.sum((Ue - Ua)**2) / 3
    slnL2 += areaT * np.sum(Ue**2) / 3

print(f"Error L²: {eL2:.6e}")
print(f"Norma L² solución: {slnL2:.6e}")
print(f"Error relativo: {np.sqrt(eL2/slnL2):.6e}")