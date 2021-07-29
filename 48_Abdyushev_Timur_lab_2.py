##################################################################################
#   48 Группа, Абдюшев Тимур, Лабораторная работа 2

#   Численные методы решения систем линейных алгебраических уравнений.
#   Прямые методы решения СЛАУ. Итерационные методы
#
#   1) Решить систему линейных алгебраических уравнений Ax = f схемой Халецкого
#
#   2) Вычислить вектор невязки r = Ax - f, где x – полученное решение.
#
#   3) Уточнить полученное решение методом простых итераций.
#   Уточнить полученное решение методом простых итераций lyambda = 2 / ||A||
#
#   4) Вычислить число обусловленности матрицы системы Ma = ||A|| * ||A^(-1)||
#
##################################################################################

import numpy as np
import math


# Метод Холецкого 
def method_holeckogo(matrix_A, matrix_F):
    # Длина столбца и строки
    len_col = len(matrix_A)
    len_row = len(matrix_A[0])
    # Нижне-угольная матрица
    new_matrix = np.zeros([len_col, len_row])
    # Находим значения для нижне-угольной матрицы
    for i in range(len_col):
        sum_elem = matrix_A[i][i]
        for j in range(len_row):
            if j <= i:
                if i != j:
                    sum_lij = matrix_A[i][j]
                    for k in range(j):
                        sum_lij -= new_matrix[i][k] * new_matrix[j][k]
                    lij = sum_lij / new_matrix[j][j]
                    new_matrix[i][j] = lij
                    sum_elem -= new_matrix[i][j] * new_matrix[i][j]
                else:
                    lii = math.sqrt(sum_elem)
                    new_matrix[i][j] = lii
    # Трансп. нижне-угольная матрица
    new_matrix_t = new_matrix.copy().transpose()

    # Вычитаем значения и получаем решения
    # Находим вектор решений у в L * y = F
    for i in range(len_col):
        for j in range(i + 1):
            if i != j:
                matrix_F[i][0] -= new_matrix[i][j] * matrix_F[j][0]
                new_matrix[i][j] = 0
            else:
                matrix_F[i][0] /= new_matrix[i][j]
                new_matrix[i][j] = 1

    # Находим вектор решений х в LT * x = y
    for i in range(len_col):
        for j in range(i + 1):
            if i != j:
                matrix_F[len_col - 1 - i][0] -= new_matrix_t[len_col - 1 - i][len_row - 1 - j] * matrix_F[len_col - 1 - j][0]
                new_matrix_t[len_col - 1 - i][len_row - 1 - j] = 0
            else:
                matrix_F[len_col - 1 - i][0] /= new_matrix_t[len_col - 1 - i][len_row - 1 - j]
                new_matrix_t[len_col - 1 - i][len_row - 1 - j] = 1

    return matrix_F


# метод Простых Итераций
def simple_iterations(matrix_A, matrix_F):
    EPS = 1e-6
    # максимальное значение итераций
    max_iterations = 1000

    iter_param = 2 / norma_1_matrix(matrix_A)
    vector_x = np.zeros((len(matrix_F), 1))
    matrix_Eye_A = np.eye(*matrix_A.shape)

    for i in range(max_iterations):
        x_last = vector_x
        matrix_B = (matrix_Eye_A - iter_param * matrix_A)
        vector_x = matrix_B.dot(vector_x) + matrix_F * iter_param
        vector_normir = vector_norm(vector_x - x_last)
        if vector_normir < EPS:
            return vector_x, i


# Первая матричная Норма
def norma_1_matrix(matrix_A):
    max_sum = 0
    for r in matrix_A:
        max_sum = max(max_sum, np.sum(abs(r)))

    return max_sum


# Норма вектора
def vector_norm(x):
    vec_norm = max(abs(x.flatten()))

    return vec_norm


# Вычисление вектора невязки
def vector_nevyazki(matrix_A, matrix_F, vector_x):
    return matrix_A.dot(vector_x) - matrix_F


# Функция печати в консоль матрицы
def print_matrix(matrix, str = '', before=8, after=4):
    # Печать числа с настройкой чисел до и после точки
    f = f'{{: {before}.{after}f}}'
    print(str)
    print('\n'.join([f''.join(f.format(el)
                    for el in row)
                    for row in matrix]) + '\n')


# Число Обусловленности матрицы системы
def num_Ma(matrix_A):
    return norma_1_matrix(np.linalg.inv(matrix_A)) * norma_1_matrix(matrix_A)


if __name__ == '__main__':
    matrix_A_arr = [
        [19, -4, 6, -1],
        [-4, 20, -2, 7],
        [6, -2, 25, -4],
        [-1, 7, -4, 15]
    ]
    matrix_A = np.array(matrix_A_arr, float)

    matrix_F_arr = [
        [100],
        [-5],
        [34],
        [69]
    ]
    matrix_F = np.array(matrix_F_arr, float)

    # 1) Решение системы лин. ур. Ax = f схемой Халецкого
    vector_x = method_holeckogo(matrix_A.copy(), matrix_F.copy())
    print_matrix(vector_x, "Вектор решения схемой Халецкого:")
    # 2) Вычисление вектора невязки
    vector_r = vector_nevyazki(matrix_A.copy(), matrix_F.copy(), vector_x.copy())
    print_matrix(vector_r, "Вектор невязки:", 8, 24)
    norma_1_matrix(matrix_A)
    # 3) Уточнение полученного решения методом простых итераций
    matrix_F_simple, needed_iterations = simple_iterations(matrix_A.copy(), matrix_F.copy())
    print_matrix(matrix_F_simple, "Вектор решения методом простых итераций:", 8)
    # 4) вычислить число обусловленности матрицы системы
    num_M = num_Ma(matrix_A.copy())
    print(f"Число Обусловленности матрицы системы = {num_M:.4f}")