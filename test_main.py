import unittest
from io import StringIO
from unittest import TestCase
from unittest.mock import patch
import numpy as np
from main import gaussian_pivot, back_substitute, read_file, main, multiple_regression


class TestMultipleRegression(TestCase):
    def test_multiple_regression_three_variables_v1(self):
        m = np.array([[345, 65, 23, 31.4], [168, 18, 18, 14.6], [94, 0, 0, 6.4], [187, 185, 98, 28.3],
                      [621, 87, 10, 42.1], [255, 0, 0, 15.3]], dtype=np.float64)
        n = 3
        check = multiple_regression(n, m)
        expected_result = np.array([[6., 1670., 355., 149., 138.1], [1670., 641720., 114071., 35495., 49225.1],
                                    [355., 114071., 46343., 20819., 11202.], [149., 35495., 20819., 10557., 4179.4]],
                                   dtype=np.float64)

        np.testing.assert_array_almost_equal(check, expected_result)

    def test_multiple_regression_three_variables_v2(self):
        m = np.array([[1142, 1060, 325, 201], [863, 995, 98, 98], [1065, 3205, 23, 162],
                      [554, 120, 0, 54], [983, 2896, 120, 138], [256, 485, 88, 61]], dtype=np.float64)
        n = 3
        check = multiple_regression(n, m)
        expected_result = np.array([[6., 4863., 8761., 654., 714.],
                                    [4863., 4521899., 8519938., 620707., 667832.],
                                    [8761., 8519938., 21022091., 905925., 1265493.],
                                    [654., 620707., 905925., 137902., 100583.]],
                                   dtype=np.float64)

        np.testing.assert_array_almost_equal(check, expected_result)


class TestGaussianPivot(unittest.TestCase):
    def test_gaussian_pivot_three_variables(self):
        m = np.array([[6., 1670., 355., 149., 138.1], [1670., 641720., 114071., 35495., 49225.1],
                      [355., 114071., 46343., 20819., 11202.], [149., 35495., 20819., 10557., 4179.4]],
                     dtype=np.float64)
        n = 4
        np.set_printoptions(precision=5, suppress=True)
        gaussian_pivot(m, n)
        expected_result = np.array([[1670., 641720., 114071., 35495., 49225.1],
                                    [0., -22342.53293, 22094.37425, 13273.65569, 737.98174],
                                    [0., 0., -10877.15836, -5537.64613, -931.28903],
                                    [0., 0., 0., -8.22173, -1.24188]],
                                   dtype=np.float64)
        np.testing.assert_array_almost_equal(m, expected_result, decimal=5)

    def test_gaussian_pivot_two_variables(self):
        m = np.array([[1, 2, 5], [2, 3, 8]], dtype=np.float64)
        n = 2
        gaussian_pivot(m, n)
        expected_result = np.array([[2., 3., 8.], [0., 0.5, 1.]],
                                   dtype=np.float64)
        np.testing.assert_array_almost_equal(m, expected_result)


class TestBackSubstitute(unittest.TestCase):
    def test_back_substitute_three_variables(self):
        m = np.array([[-3., -1., 2., -11.],
                      [0., 0.66667, -0.33333, 1.],
                      [0., 0., 0.5, 1.]], dtype=np.float64)
        n = 3
        result = back_substitute(m, n)
        expected_result = ['4.16667', '2.49998', '2.00000']
        self.assertEqual(result, expected_result)

    def test_back_substitute_two_variables(self):
        m = np.array([[2., 3., 8.], [0., 0.5, 1.]], dtype=np.float64)
        n = 2
        result = back_substitute(m, n)
        expected_result = ['1.00000', '2.00000']
        self.assertEqual(result, expected_result)


class TestReadFromFile(unittest.TestCase):
    def test_read_file(self):
        file_path = "Gauss3.txt"
        result = read_file(file_path)
        self.assertIsInstance(result, list)

    def test_read_file_not_found(self):
        file_path = "C:\\Random\\Path"
        result = read_file(file_path)
        self.assertIsNone(result, "File not found")


class TestMain(unittest.TestCase):
    @patch('builtins.input', side_effect=['file/path', 'not_a_number'])
    @patch('main.read_file')
    def test_main_number_value_error(self, mock_read_file, mock_input):
        with patch('sys.stdout', new_callable=StringIO) as fake_out:
            main()
            self.assertEqual(fake_out.getvalue().strip(), "Number value is expected")

    @patch('builtins.input', side_effect=['invalid_file_path', '2'])
    @patch('main.read_file', side_effect=TypeError)
    def test_main_invalid_file_path_error(self, mock_read_file, mock_input):
        with patch('sys.stdout', new_callable=StringIO) as fake_out:
            main()
            self.assertEqual(fake_out.getvalue().strip(), "The file path is incorrect or invalid")

    @patch('builtins.input', side_effect=['Gauss2.txt', '3'])
    @patch('main.read_file', side_effect=IndexError)
    def test_main_out_of_matrix_index_error(self, mock_read_file, mock_input):
        with patch('sys.stdout', new_callable=StringIO) as fake_out:
            main()
            self.assertEqual(fake_out.getvalue().strip(), "Number for variables is larger than these in the current"
                                                          " file")

    @patch('builtins.input', side_effect=['Gauss3.txt', '3'])
    def test_main_output(self, mock_input):
        with patch('sys.stdout', new_callable=StringIO) as fake_out:
            main()
            self.assertAlmostEqual(fake_out.getvalue(), """Matrix from linear equations:

[[345.   65.   23.   31.4]
 [168.   18.   18.   14.6]
 [ 94.    0.    0.    6.4]
 [187.  185.   98.   28.3]
 [621.   87.   10.   42.1]
 [255.    0.    0.   15.3]] 

[[     6.    1670.     355.     149.     138.1]
 [  1670.  641720.  114071.   35495.   49225.1]
 [   355.  114071.   46343.   20819.   11202. ]
 [   149.   35495.   20819.   10557.    4179.4]] 
 End of multiple regression 
  ================= 

[[  1670.      641720.      114071.       35495.       49225.1    ]
 [     0.      -22342.53293  22094.37425  13273.65569    737.98174]
 [     0.           0.      -10877.15836  -5537.64613   -931.28903]
 [     0.           0.           0.          -8.22173     -1.24188]] 
 End of Gaussian elimination 
 ================= 

Solution for the system:
0.56646
0.06533
0.00872
0.15105
""")


if __name__ == '__main__':
    unittest.main()
