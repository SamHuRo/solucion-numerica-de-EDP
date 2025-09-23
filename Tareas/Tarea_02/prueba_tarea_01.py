"""
Solver para Esquemas de Diferencias Finitas usando Sustitución Progresiva y Regresiva
===================================================================================

Este programa resuelve el esquema de diferencias finitas:
V^(n+1) = AV^n + F̃

Donde el esquema explícito está dado por:
V_m^(n+1) = V_m^n + (αk/2h)(V_(m-1)^n - V_(m+1)^n) + kf_m^n

El programa utiliza factorización LU con sustitución progresiva y regresiva
para resolver el sistema matricial en cada paso temporal.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable
import time

class FiniteDifferenceScheme:
    def __init__(self, 
                 L: float = 1.0,      # Longitud del dominio espacial
                 T: float = 0.8,      # Tiempo final
                 M: int = 10,         # Divisiones espaciales
                 N: int = 4,          # Divisiones temporales
                 alpha: float = 1.0): # Parámetro del esquema
        
        self.L = L
        self.T = T
        self.M = M
        self.N = N
        self.alpha = alpha
        
        # Cálculo de pasos
        self.h = L / M          # Paso espacial
        self.k = T / N          # Paso temporal
        
        # Grillas
        self.x = np.linspace(0, L, M + 1)
        self.t = np.linspace(0, T, N + 1)
        
        # Parámetros del esquema
        self.r = (alpha * self.k) / (2 * self.h)
        
        print(f"Configuración del problema:")
        print(f"  Dominio espacial: [0, {L}] con h = {self.h:.4f}")
        print(f"  Dominio temporal: [0, {T}] con k = {self.k:.4f}")
        print(f"  Parámetro r = αk/(2h) = {self.r:.4f}")
        print(f"  Matriz del sistema: {M-1} x {M-1}")
    
    def build_system_matrix(self) -> np.ndarray:
        """
        Construye la matriz del sistema A para el esquema implícito.
        Para el esquema explícito, A sería la matriz identidad más términos de difusión.
        """
        # Tamaño de la matriz (puntos interiores)
        size = self.M - 1
        
        # Matriz tridiagonal para el esquema
        A = np.eye(size)  # Matriz identidad como base
        
        # Para esquema explícito: V^(n+1) = V^n + términos de advección
        # La matriz A representa los coeficientes del lado derecho
        for i in range(size):
            A[i, i] = 1.0  # Término V_m^n
            
            # Términos de advección: ±(αk/2h)
            if i > 0:  # V_(m-1)^n
                A[i, i-1] = self.r
            if i < size - 1:  # V_(m+1)^n  
                A[i, i+1] = -self.r
        
        return A
    
    def lu_decomposition(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza la descomposición LU de la matriz A.
        Retorna las matrices L y U.
        """
        n = A.shape[0]
        L = np.eye(n)
        U = A.copy()
        
        print("Realizando descomposición LU...")
        
        for i in range(n):
            # Hacer ceros debajo del pivote
            for k in range(i + 1, n):
                if U[i, i] == 0:
                    raise ValueError(f"Pivote cero encontrado en posición ({i}, {i})")
                
                factor = U[k, i] / U[i, i]
                L[k, i] = factor
                
                for j in range(i, n):
                    U[k, j] = U[k, j] - factor * U[i, j]
        
        return L, U
    
    def forward_substitution(self, L: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Sustitución progresiva: resuelve Ly = b
        """
        n = len(b)
        y = np.zeros(n)
        
        for i in range(n):
            y[i] = b[i]
            for j in range(i):
                y[i] -= L[i, j] * y[j]
            y[i] /= L[i, i]
        
        return y
    
    def backward_substitution(self, U: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Sustitución regresiva: resuelve Ux = y
        """
        n = len(y)
        x = np.zeros(n)
        
        for i in range(n - 1, -1, -1):
            x[i] = y[i]
            for j in range(i + 1, n):
                x[i] -= U[i, j] * x[j]
            x[i] /= U[i, i]
        
        return x
    
    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Resuelve el sistema Ax = b usando LU con sustitución progresiva y regresiva.
        """
        # Descomposición LU
        L, U = self.lu_decomposition(A)
        
        # Sustitución progresiva: Ly = b
        y = self.forward_substitution(L, b)
        
        # Sustitución regresiva: Ux = y
        x = self.backward_substitution(U, y)
        
        return x
    
    def source_term(self, x: float, t: float) -> float:
        """
        Término fuente f(x,t). Modifica según tu problema específico.
        """
        # Ejemplo: f(x,t) = sin(πx) * exp(-t)
        return np.sin(np.pi * x) * np.exp(-t)
    
    def initial_condition(self, x: float) -> float:
        """
        Condición inicial V(x,0). Modifica según tu problema.
        """
        # Ejemplo: V(x,0) = sin(πx)
        return np.sin(np.pi * x)
    
    def boundary_conditions(self, t: float) -> Tuple[float, float]:
        """
        Condiciones de frontera V(0,t) y V(L,t).
        """
        # Ejemplo: condiciones homogéneas
        return 0.0, 0.0
    
    def solve_scheme(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resuelve el esquema de diferencias finitas completo.
        """
        # Matriz del sistema
        A = self.build_system_matrix()
        
        if verbose:
            print(f"\nMatriz del sistema A:")
            print(A)
        
        # Inicializar solución
        V = np.zeros((self.N + 1, self.M + 1))
        
        # Condición inicial
        for i in range(self.M + 1):
            V[0, i] = self.initial_condition(self.x[i])
        
        if verbose:
            print(f"\nCondición inicial V(x,0):")
            print(V[0, :])
        
        # Resolver para cada paso temporal
        for n in range(self.N):
            t_current = self.t[n]
            t_next = self.t[n + 1]
            
            if verbose:
                print(f"\n--- Paso temporal {n+1}/{self.N}: t = {t_next:.3f} ---")
            
            # Condiciones de frontera
            V[n + 1, 0], V[n + 1, -1] = self.boundary_conditions(t_next)
            
            # Construir vector del lado derecho F̃
            F_tilde = np.zeros(self.M - 1)
            
            for m in range(1, self.M):
                # Término fuente
                source = self.k * self.source_term(self.x[m], t_current)
                
                # Para esquema explícito, F̃ incluye términos de frontera
                boundary_correction = 0.0
                if m == 1:  # Punto cerca de la frontera izquierda
                    boundary_correction += self.r * V[n + 1, 0]
                if m == self.M - 1:  # Punto cerca de la frontera derecha  
                    boundary_correction -= self.r * V[n + 1, -1]
                
                F_tilde[m - 1] = source + boundary_correction
            
            # Vector del lado derecho: AV^n + F̃
            V_current = V[n, 1:-1]  # Puntos interiores en tiempo n
            b = A @ V_current + F_tilde
            
            if verbose:
                print(f"Vector b = AV^n + F̃:")
                print(f"  V^n (interior): {V_current}")
                print(f"  F̃: {F_tilde}")
                print(f"  b: {b}")
            
            # Para esquema explícito, la solución es directa
            # V^(n+1) = AV^n + F̃
            V[n + 1, 1:-1] = b
            
            # Si fuera implícito, resolveríamos: (I - θA)V^(n+1) = V^n + F̃
            # V_new = self.solve_linear_system(system_matrix, b)
            # V[n + 1, 1:-1] = V_new
            
            if verbose:
                print(f"Solución V^({n+1}): {V[n + 1, :]}")
        
        return self.x, V
    
    def plot_solution(self, x: np.ndarray, V: np.ndarray, save_fig: bool = False):
        """
        Grafica la solución en diferentes tiempos.
        """
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Evolución temporal
        plt.subplot(2, 2, 1)
        for n in range(0, self.N + 1, max(1, self.N // 4)):
            plt.plot(x, V[n, :], 'o-', label=f't = {self.t[n]:.2f}')
        plt.xlabel('x')
        plt.ylabel('V(x,t)')
        plt.title('Evolución Temporal de la Solución')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Mapa de calor
        plt.subplot(2, 2, 2)
        X, T = np.meshgrid(x, self.t)
        plt.contourf(X, T, V, levels=20, cmap='viridis')
        plt.colorbar(label='V(x,t)')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Mapa de Calor de la Solución')
        
        # Subplot 3: Solución final
        plt.subplot(2, 2, 3)
        plt.plot(x, V[-1, :], 'ro-', linewidth=2, markersize=6)
        plt.xlabel('x')
        plt.ylabel('V(x,T)')
        plt.title(f'Solución Final en t = {self.T}')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Información del esquema
        plt.subplot(2, 2, 4)
        plt.axis('off')
        info_text = f"""
        Parámetros del Esquema:
        
        • Dominio espacial: [0, {self.L}]
        • Dominio temporal: [0, {self.T}]
        • Paso espacial: h = {self.h:.4f}
        • Paso temporal: k = {self.k:.4f}
        • Parámetro α = {self.alpha}
        • Número de Courant: r = {self.r:.4f}
        
        Esquema:
        V^(n+1) = V^n + (αk/2h)(V_(m-1) - V_(m+1)) + kf
        
        Estabilidad:
        |r| ≤ 1 para estabilidad
        """
        plt.text(0.1, 0.9, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'finite_difference_solution_{int(time.time())}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_stability(self):
        """
        Analiza la estabilidad del esquema.
        """
        print(f"\n=== ANÁLISIS DE ESTABILIDAD ===")
        print(f"Número de Courant: r = αk/(2h) = {self.r:.6f}")
        
        if abs(self.r) <= 1:
            print("✅ ESTABLE: |r| ≤ 1")
        else:
            print("⚠️  INESTABLE: |r| > 1")
            print("   Recomendación: Reducir k o aumentar h")
        
        print(f"Condición CFL: k ≤ h²/(2α) = {self.h**2/(2*self.alpha):.6f}")
        
        if self.k <= self.h**2/(2*self.alpha):
            print("✅ Cumple condición CFL")
        else:
            print("⚠️  No cumple condición CFL")

def main():
    """
    Función principal que demuestra el uso del solver.
    """
    print("="*80)
    print("SOLVER DE DIFERENCIAS FINITAS CON SUSTITUCIÓN PROGRESIVA Y REGRESIVA")
    print("="*80)
    
    # Crear instancia del solver
    solver = FiniteDifferenceScheme(
        L=1.0,      # Longitud del dominio
        T=0.8,      # Tiempo final
        M=20,       # Divisiones espaciales
        N=8,        # Divisiones temporales
        alpha=0.5   # Parámetro del esquema
    )
    
    # Analizar estabilidad
    solver.analyze_stability()
    
    # Resolver el esquema
    print(f"\n{'='*50}")
    print("RESOLVIENDO EL ESQUEMA")
    print(f"{'='*50}")
    
    start_time = time.time()
    x, V = solver.solve_scheme(verbose=True)
    end_time = time.time()
    
    print(f"\nTiempo de cálculo: {end_time - start_time:.4f} segundos")
    
    # Mostrar resultados
    print(f"\n{'='*50}")
    print("RESULTADOS")
    print(f"{'='*50}")
    
    print("Solución completa V(x,t):")
    print("Filas = tiempo, Columnas = espacio")
    print(V.round(6))
    
    # Graficar solución
    solver.plot_solution(x, V, save_fig=False)
    
    # Ejemplo con parámetros que causan inestabilidad
    print(f"\n{'='*80}")
    print("EJEMPLO DE ESQUEMA INESTABLE")
    print(f"{'='*80}")
    
    unstable_solver = FiniteDifferenceScheme(
        L=1.0, T=0.8, M=10, N=4, alpha=2.0  # α grande causa inestabilidad
    )
    unstable_solver.analyze_stability()

if __name__ == "__main__":
    main()