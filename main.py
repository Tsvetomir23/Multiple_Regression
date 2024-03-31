import numpy as np


def multiple_regression(n, m):
    m = np.array(m, dtype=np.float64).T
    regress = [len(m[0])]
    for i in range(n + 1):
        regress.append(np.sum(m[i]))

    for i in range(n):
        row = [np.sum(m[i])]
        for j in range(n):
            row.append(np.multiply(m[i], m[j]).sum())

        row.append(np.multiply(m[i], m[n]).sum())
        regress.append(row)

    regress = [regress[:n + 2]] + regress[n + 2:]
    return regress


def gaussian_pivot(m, n):
    for i in range(n):
        pivot_row = i

        for j in range(i + 1, n):
            if abs(m[j][i]) > abs(m[pivot_row][i]):
                pivot_row = j

        if pivot_row != i:
            m[[i, pivot_row]] = m[[pivot_row, i]]

        for j in range(i + 1, n):
            factor = m[j][i] / m[i][i]
            m[j] -= factor * m[i]


def back_substitute(m, n):
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_val = sum(m[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (m[i][n] - sum_val) / m[i][i]

    x = [f'{elem:.5f}' for elem in x]
    return x


def read_file(file_path):
    matrix = []
    try:
        with open(file_path, 'r') as file:
            for line in file:

                numbers = [num.strip() for num in line.split(',')]

                if numbers[-1] == "":
                    numbers.pop()
                    numbers = [num.strip() for num in line.split(',')]

                matrix.append(numbers)

    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)
    else:
        return matrix


def main():
    try:
        np.set_printoptions(precision=5, suppress=True)

        file_path = input("Enter the file path: ")
        n = int(input("Enter the value of independent variables N: "))
        if n == 0:
            raise SyntaxError

        csv_data = read_file(file_path)
        csv_data = np.array(csv_data, dtype=np.float64)

        print("Matrix from linear equations:\n")
        print(csv_data, "\n")

        regression = multiple_regression(n, csv_data)
        regression = np.array(regression, dtype=np.float64)

        print(regression, "\n End of multiple regression \n  ================= \n")

        gaussian_pivot(regression, n + 1)
        b = back_substitute(regression, n + 1)
        print(regression, "\n End of Gaussian elimination \n ================= \n")

    except ValueError:
        print("Number value is expected")
    except TypeError:
        print("The file path is incorrect or invalid")
    except IndexError:
        print("Number for variables is larger than these in the current file")
    except SyntaxError:
        print("N must be greater than 0")
    else:
        print("Solution for the system:")
        for i in range(n + 1):
            print(b[i])


if __name__ == "__main__":
    main()
