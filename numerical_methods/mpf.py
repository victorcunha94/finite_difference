import numpy as np
"""
O MPF consiste em transformar esta equação em uma equação equivalente x = psi(x) e a partir
de uma aproximação incial x0 gerar uma sequencia {xk} de aproximações, para \qsi pela relação
x_{k+1} = psi(xk), pois a função psi(x) é tal que f(\qsi) = 0 se e somente se psi(\qsi) = \qsi.
Transformamos assim o problema de encontrar um zero de f(x) no problema de encontrar um ponto
fixo de psi(x). 
"""