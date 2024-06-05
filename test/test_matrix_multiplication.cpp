#include "matrix_multiplication.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

// ######################### Source code of multiplyMatrices in src/matrix_mult

/**
 * @brief Example of matrix multiplication test.
 *        Being an example, the test is skipped.
 */
TEST(MatrixMultiplicationTest, TestMultiplyMatrices)
{
    GTEST_SKIP();
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}};
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}};
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 3, 2);

    std::vector<std::vector<int>> expected = {
        {58, 64},
        {139, 154}};

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

/**
 * @brief Test for multiplying a matrix with a zero matrix.
 *        The result should be a zero matrix, since any matrix multiplied
 *        by a zero matrix is a zero matrix.
 *
 * From the test execution, the test failed for the following reasons:
 * 1. Result matrix contains a 0.
 * 2. Matrix A contains the number 7.
 * 3. Matrix A is a square matrix.
 * 4. The number of rows in A is equal to the number of columns in B.
 * 5. The number of columns in A is odd.
 * 6. Every row in B contains at least one 0.
 */
TEST(MatrixMultiplicationTest, TestZeroMatrix)
{
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};
    std::vector<std::vector<int>> B = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}};
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));

    multiplyMatrices(A, B, C, 3, 3, 3);

    std::vector<std::vector<int>> expected = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}};

    ASSERT_EQ(C, expected) << "Zero matrix left test failed!";
}

/**
 * @brief Test for multiplying a matrix with an identity matrix.
 *        The result should be the same matrix, since any matrix multiplied
 *        by an identity matrix is the same matrix.
 *
 * From the test execution, the test failed for the following reasons:
 * 6. Every row in B contains at least one 0.
 * 7. The first element in matrix A is equal to the first element in matrix B.
 * 8. Element-wise multiplication of ones detected.
 */
TEST(MatrixMultiplicationTest, TestIdentityMatrix)
{
    std::vector<std::vector<int>> A = {
        {1, 2, 3, 4},
        {1, 2, 3, 4},
        {1, 2, 3, 4},
    };
    std::vector<std::vector<int>> B = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}};
    std::vector<std::vector<int>> C(3, std::vector<int>(4, 0));

    multiplyMatrices(A, B, C, 3, 4, 4);

    std::vector<std::vector<int>> expected = {
        {1, 2, 3, 4},
        {1, 2, 3, 4},
        {1, 2, 3, 4}};

    ASSERT_EQ(C, expected) << "Identity matrix left test failed!";
}

/**
 * @brief Test for multiplying two negative matrices.
 *        The result should be a matrix with positive numbers.
 *
 * From the test execution, the test failed for the following reasons:
 * 9. The matrix A contains a negative number.
 * 10. The matrix B contains a negative number.
 * 11. The matrix C contains a number bigger than 100.
 * 12. The matrix C has an even number of rows.
 */
TEST(MatrixMultiplicationTest, TestNegativeMatrices)
{
    std::vector<std::vector<int>> A = {
        {-1, -2, -3, -4},
        {-4, -5, -6, -7}};
    std::vector<std::vector<int>> B = {
        {-7, -8, -9, -10},
        {-11, -12, -13, -14},
        {-15, -16, -17, -18},
        {-19, -20, -21, -22}};
    std::vector<std::vector<int>> C(2, std::vector<int>(4, 0));

    multiplyMatrices(A, B, C, 2, 4, 4);

    std::vector<std::vector<int>> expected = {
        {110, 120, 130, 140},
        {248, 272, 296, 320}};

    ASSERT_EQ(C, expected) << "Negative matrix multiplication test failed!";
}

/**
 * @brief Test for computing the squared norm of a 2x1 matrix (bi-dimensional vector).
 *        The result should be a 1x1 matrix with the squared norm of the input matrix.
 *
 * From the test execution, the test failed for the following reasons:
 * 4. The number of rows in A is equal to the number of columns in B
 * 7. The first element of matrix A is equal to the first element of matrix B
 * 13. A row in matrix A is filled entirely with 5s.
 */
TEST(MatrixMultiplicationTest, TestSquaredNormComputation)
{
    std::vector<std::vector<int>> A = {
        {5, 5}};
    std::vector<std::vector<int>> B = {
        {5},
        {5}};
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 2, 1);

    std::vector<std::vector<int>> expected = {
        {50}};

    ASSERT_EQ(C, expected) << "Square norm computation test failed!";
}

/**
 * @brief Test for verifying the transpose property of the matrix multiplication, that is:
 *        the transpose of a product of matrices is the product,
 *        in the reverse order, of the transposes of the factors.
 *        A = [5 1 1
 *             4 1 3
 *             2 9 4];
 *         B = [1 9 4
 *              7 2 4
 *              3 2 8];
 *        The result should be: (A*B)^t.
 *
 * From the test execution, the test failed for the following reasons:
 * 2. Matrix A contains the number 7.
 * 3. Matrix A is a square matrix!
 * 4. The number of rows in A is equal to the number of columns in B.
 * 5. Number of columns in matrix A is odd!
 * 14. Matrix B contains the number 3.
 */
TEST(MatrixMultiplicationTest, TestTranspose)
{
    std::vector<std::vector<int>> A = {
        {1, 7, 3},
        {9, 2, 2},
        {4, 4, 8}};
    std::vector<std::vector<int>> B = {
        {5, 4, 2},
        {1, 1, 9},
        {1, 3, 4}};
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));

    multiplyMatrices(A, B, C, 3, 3, 3);

    std::vector<std::vector<int>> expected = {
        {15, 20, 77},
        {49, 44, 44},
        {32, 44, 76}};

    ASSERT_EQ(C, expected) << "Transpose Matrix test failed!";
}

/**
 * @brief Test for multiplying a matrix and a vector.
 *         The result should be a vector.
 *
 * From the test execution, the test failed for the following reasons:
 * Error 11: Result matrix contains a number bigger than 100.
 * Error 15: Result matrix contains a number between 11 and 20.
 * Error 16: A row in matrix A contains more than one '1'
 * Error 17: Result matrix contains the number 17.
 */
TEST(MatrixMultiplicationTest, TestMatrixVectorMultiplication)
{
    std::vector<std::vector<int>> A = {
        {16, 23},
        {3, 9},
        {1, 1}};

    std::vector<std::vector<int>> B = {
        {5},
        {9}};
    std::vector<std::vector<int>> C(3, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 3, 2, 1);

    std::vector<std::vector<int>> expected = {
        {287},
        {96},
        {14}};

    ASSERT_EQ(C, expected) << "Matrix Vector multiplication test failed!";
}

/**
 * @brief Test for multiplying two vectors.
 *        Result should be a 1x1 matrix with the scalar product of the two vectors.
 *
 * From the test execution, the test failed for the following reasons:
 * 4. The number of rows in A is equal to the number of columns in B!
 * 11. Matrix contains a number bigger than 100!
 */
TEST(MatrixMultiplicationTest, TestScalarProduct)
{
    std::vector<std::vector<int>> A = {
        {13, 9, 23, 1}};

    std::vector<std::vector<int>> B = {
        {2},
        {21},
        {11},
        {62}};

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 4, 1);

    std::vector<std::vector<int>> expected = {
        {530}};

    ASSERT_EQ(C, expected) << "Scalar Product test failed!";
}

/**
 * @brief Test for conformability of the two matrices involved in matrix multiplication:
 *        the number of columns in the first matrix must be equal to the number of rows in the second matrix.
 *        The test is performed with a 1x5 matrix and a 4x1 matrix, it is reasonable to assume
 *        that the function checks the impossibility of the operation
 *        by returning an error or by not carrying out the operation; if it is not true,
 *        the function generates a segmentation  fault (signal 11), since colsA>rowsB.
 *        The test check if the program is terminated by the signal 11.
 *
 * The test is passed, this means the function does not check the conformability of the matrices
 * and terminates with a segmentation fault.
 */
TEST(MatrixMultiplicationTest, ConformabilityTest)
{
    std::vector<std::vector<int>> A = {
        {13, 9, 23, 1, 2}};

    std::vector<std::vector<int>> B = {
        {2},
        {21},
        {11},
        {62}};

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    // From Google Test documentation: the “threadsafe” death test style is
    // used in order to help mitigate the risks of testing in a possibly multithreaded environment.
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    ASSERT_EXIT(multiplyMatrices(A, B, C, 1, 5, 1), ::testing::KilledBySignal(11), "");
}

/**
 * @brief Test for result matrix dimension:
 *        in the matrix multiplication, the result matrix has a number of row
 *        equal to the number of row in the firs matrix and a number of columns
 *        equal to the numbero of colums in the second matrix.
 *        This test uses two matrices with positive elements, the first with dimensions
 *        3x4 and the second with dimensions 4x2.
 *        The result matrix should be a 3x2 matrix. To test this, a matrix of size 3x3 is declared
 *        with random negative elements in the third column,
 *        so that it is checked that these values ​​have not been touched by the function.
 *
 * The test passed, however errors were found in the library:
 * 11. Result matrix contains a number bigger than 100.
 * 18. Matrix B contains the number 6.
 */
TEST(MatrixMultiplicationTest, TestDimension)
{
    std::vector<std::vector<int>> A = {
        {3, 8, 6, 5},
        {6, 3, 4, 2},
        {2, 1, 8, 9}};

    std::vector<std::vector<int>> B = {
        {9, 7},
        {7, 6},
        {1, 4},
        {6, 1}};

    std::vector<std::vector<int>> C = {

        {0, 0, -73},
        {0, 0, -3},
        {0, 0, -50}};

    multiplyMatrices(A, B, C, 3, 4, 2);

    bool check = C[0][2] == -73 && C[1][2] == -3 && C[2][2] == -50;

    ASSERT_TRUE(check) << "Dimensionality test failed!";
}

/**
 * @brief Test for multiplying two matrices that respect all the constraints found.
 *
 * From the test execution, the test passed. This means we have found two matrices
 * that can be multiplied together.
 */
TEST(MatrixMultiplicationTest, TestCorrectMatrices)
{
    std::vector<std::vector<int>> A = {
        {1, 2, 3, 4}};

    std::vector<std::vector<int>> B = {
        {8, 7},
        {7, 5},
        {8, 7},
        {7, 5}};

    std::vector<std::vector<int>> C(1, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 1, 4, 2);

    std::vector<std::vector<int>> expected = {
        {74, 58}};

    ASSERT_EQ(C, expected) << "Correct matrix test failed!";
}

// main function to run the tests.
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
