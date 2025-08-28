# Solución Numérica de Ecuaciones Diferenciales Parciales

Este repositorio contiene las tareas y proyectos desarrollados durante el curso de **Soluciones Numéricas de Ecuaciones Diferenciales Parciales** de la Maestría en Ciencias Físicas de la Universidad Nacional de Colombia.

## 📋 Descripción del Curso

Este curso presenta un estudio sistemático de los métodos numéricos fundamentales para la resolución de ecuaciones diferenciales parciales, enfocándose en las técnicas de diferencias finitas y elementos finitos.

### Marco Teórico

El programa académico aborda los **métodos de diferencias finitas** desde una perspectiva teórica integral que comprende cuatro pilares fundamentales del análisis numérico:

- **Consistencia**: Análisis del error de truncamiento y aproximación de derivadas
- **Estabilidad**: Estudio de la propagación de errores y condiciones de estabilidad
- **Convergencia**: Teoremas de convergencia y análisis asintótico
- **Implementación computacional**: Desarrollo de algoritmos eficientes y estructuras de datos

### Alcance Dimensional y Clasificación de EDPs

La metodología pedagógica considera diferentes dimensiones espaciales según la clasificación matemática de las ecuaciones diferenciales parciales:

- **Problemas hiperbólicos**: Tratamiento exclusivo de sistemas unidimensionales, incluyendo ecuaciones de ondas y leyes de conservación
- **Problemas parabólicos**: Extensión a sistemas bidimensionales, con énfasis en ecuaciones de difusión y transferencia de calor
- **Problemas elípticos**: Análisis bidimensional de ecuaciones de Laplace, Poisson y problemas de estado estacionario

### Objetivos de Aprendizaje
- Dominar los fundamentos teóricos de los métodos numéricos para EDPs
- Desarrollar competencias en análisis de estabilidad y convergencia
- Implementar algoritmos computacionales robustos y eficientes
- Aplicar técnicas avanzadas de aproximación numérica a problemas físicos relevantes

## 🗂️ Estructura del Repositorio

```
solucion-numerica-de-EDP/
├── README.md
├── Tareas/
|    ├── Tarea_01/
│    |      ├── Factorizacion_matrices.ipynb
|    ├── Tarea_02/
│           ├── codigo/
│           └── resultados/
├── Proyecto_Final/
│   ├── codigo/
│   └── informe_final.pdf
└── Referencias/
    └── bibliografia.md
```

## 📚 Contenido del Curso

### Métodos Numéricos Cubiertos
- **Métodos de Diferencias Finitas - Ecuaciones Diferenciales Parabolicas, Hiperbolicas y Elipticas**
  - Consistencia y convergencia de esquemas de diferencias.
  - Estabilidad.
  - Condición de Courant-Friedrichs-Levy.
  - Análisis de Fourier.
  - Análisis de Von Neumann.
  - Orden de aproximación.
  - Esquemas de diferencias.
  - Ecuaciones de convección-difusión.
  - Método ADI.
  - Estimados de regularidad y principio del máximo.
  - Condiciones de borde.
  - Esquemas de diferencias para la ecuación de Poisson.

- **Método de Elementos Finitos**
  - Introducción a espacios de Hilbert y espacios de Sobolev.
  - Introducción al método de elementos finitos para problemas elípticos.
  - Algunos espacios de elementos finitos.
  - Teoría de aproximación y estimación de error para el método de elementos finitos.
  - Introducción al método de elementos finitos para problemas parabólicos e hiperbólicos.

### Tipos de EDPs Estudiadas
- Ecuación de calor (parabólica)
- Ecuación de ondas (hiperbólica)
- Ecuación de Laplace/Poisson (elíptica)

## 💻 Tecnologías Utilizadas

- **Python** - Lenguaje principal de programación
- **NumPy** - Computación numérica
- **SciPy** - Algoritmos científicos
- **Matplotlib** - Visualización de datos
- **Pandas** - Manejo de datos
- **Jupyter Notebooks** - Desarrollo interactivo
- **Git** - Control de versiones

## 📊 Resultados y Visualizaciones

Cada tarea incluye:
- Código fuente documentado
- Análisis de resultados
- Gráficas y visualizaciones
- Comparación con soluciones analíticas (cuando disponibles)
- Análisis de error y convergencia

## 📖 Documentación

- **Informes**: Cada tarea incluye un informe detallado en formato PDF
- **Código**: Código completamente documentado con docstrings

## 🤝 Contribuciones

Este es un repositorio académico personal. Las sugerencias y discusiones son bienvenidas a través de issues.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👨‍🎓 Autor

**[Samuel Huertas Rojas]**
- Estudiante de Maestría en Ciencias Físicas
- Universidad Nacional de Colombia
- Email: [shuertasr@unal.edu.co]
- GitHub: [@SamHuRo](https://github.com/SamHuRo)

## 📚 Referencias

- Jhon C. Strikwerda, *Finite Difference Schemes and Partial Differential Equations*, 2004, SIAM Philadelphia.
- K. W. Morton, D. F. Mayers, *Numerical Solution of Partial Differential Equations. An Introduction*. Second
edition, 2005, Cambridge University Press.
- Ivo Babuska, John R. Whiteman, Theofanis Stroubolis. *Finite Elements, An Introduction to the Method and
Error Estimations*, 2011, Oxford University Press
- Claes Johnson. *Numerical Solution of Partial Differential Equations by the Finite Element Method*, 2009,
Dover Ed

## 📅 Cronograma

| Semana | Tema                                     | Entregable |
|--------|------------------------------------------|------------|
| 1      | Factorización de Matrices (Preliminares) | Tarea 1    |
| 3-4    | Elementos Finitos                        | Tarea 2    |
| 5-6    | Métodos Espectrales                      | Tarea 3    |
| 7-8    | Volúmenes Finitos                        | Tarea 4    |
| 9-12   | Proyecto Final                           | Proyecto   |

---

*Última actualización: 27 de Agosto del 2025*
