#!/usr/bin/env python
# coding: utf-8

# # Programming Assignment - Gaussian Elimination
# 
# 
# Welcome to the programming assignment on Gaussian Elimination! In this assignment, you will implement the Gaussian elimination method, a foundational algorithm for solving systems of linear equations.
# 
# Linear algebra is fundamental to machine learning, serving as the basis for numerous algorithms. Gaussian elimination, while not the most advanced method used today, is a classical and essential technique for solving systems of linear equations. It provides valuable insights into the core principles of linear algebra and lays the groundwork for more advanced numerical methods.
# 
# ## Why should you care?
# 
# - **Foundational Skills**: Strengthen your understanding of key linear algebra concepts.
# - **Programming Practice**: Enhance your coding skills by implementing a classical mathematical algorithm.
# - **Historical Significance**: Gaussian elimination, though not the most cutting-edge method today, is historically significant and provides a solid starting point for understanding the evolution of linear algebra techniques.
# 
# 
# 
# 
# # Outline
# - [ 1 - Introduction ](#1)
#   - [ 1.1 How to complete this assignment](#1.1)
#   - [ 1.2 Gaussian Elimination Algorithm](#1.2)
# - [ 2 - Necessary imports](#2)
# - [ 3 - Auxiliary functions](#3)
#   - [ 3.1 - Function swap rows](#3.1)
#   - [ 3.2 - Finding the first non-zero value in a column starting from a specific value](#3.2)
#   - [ 3.3 - Find the first non zero element for any row](#3.3)
#   - [ 3.4 Moving one row to the bottom](#3.4)
#   - [ 3.5 - Constructing the Augmented Matrix](#3.5)
# - [ 4 - Row echelon form](#4)
#   - [ 4.1 - Row Echelon Form](#4.1)
#   - [ 4.2 - A worked example ](#4.2)
#     - [ Exercise 1](#ex01)
# - [ 5 - Back substitution](#5)
#   - [ Exercise 2](#ex02)
# - [ 6 - The Gaussian Elimination](#6)
#   - [ 6.1 - Bringing it all together](#6.1)
#     - [ Exercise 3](#ex03)
# - [ 7 - Test with any system of equations!](#7)
# 

# <a name="1"></a>
# ## 1 - Introduction 
# 
# 
# <a name="1.1"></a>
# ### 1.1 How to complete this assignment
# 
# This is the first assignment in the Math for Machine Learning and Data Science specialization! Let's quickly go over how it works.
# 
# This assignment has $3$ graded functions. Each graded function will have some parts replaced as `None`. These parts you have to replace with the proper value. For instance, in the first graded function there is this line of code:
# 
# ```Python
# pivot_candidate = M[None, None]
# ```
# 
# This means that you must replace the correct values for the row (first None) and column (second None). Do not worry, the functions have comments on every line of code so you won't get lost! 
# 
# After each graded function, there is a code to test your solution. It will test your function with some basic and quick tests to assure you are in the right path! **Note that these tests perform only basic tests, so you may pass in the unit tests but fail when submitting your code.** This is because when grading we perform more complex tests. However, in any case you will be provided with feedbacks so it will help you debugging your code.
# 
# <a name="1.2"></a>
# ### 1.2 Gaussian Elimination Algorithm
# 
# Gaussian elimination offers a systematic approach to solving systems of linear equations by transforming an augmented matrix into row-echelon form, thereby enabling the determination of variables. The algorithm comprises several essential steps:
# 
# **NOTE**: 
# 
# - For simplicity, the algorithm you will develop here will only work on **non-singular** systems of equations, i.e., equations that have a unique solution.
# - Remember you can check if a matrix is singular or not by computing its determinant.
# 
# ### Step 1: Augmented Matrix
# 
# Consider a system of linear equations:
# 
# $$
# \begin{align*}
# 2x_1 + 3x_2 + 5x_3&= 12 \\
# -3x_1 - 2x_2 + 4x_3 &= -2 \\
# x_1 + x_2 - 2x_3  &= 8 \\
# 
# \end{align*}
# $$
# Create the augmented matrix \([A | B]\), where \(A\) represents the coefficient matrix and \(B\) denotes the column vector of constants:
# 
# $$
# A = \begin{bmatrix}
# \phantom{-}2 & \phantom{-}3 & \phantom{-}5  \\
# -3 & -2 & \phantom-4 \\
# \phantom{-}1 & \phantom{-}1 & -2  \\
# \end{bmatrix}
# $$
# 
# $$
# B = \begin{bmatrix}
# \phantom-12 \\ -2 \\ \phantom-8 
# \end{bmatrix}
# $$
# 
# Thus, \([A | B]\) is represented as:
# 
# $$
# \begin{bmatrix}
# \phantom{-}2 & \phantom{-}3 & \phantom{-}5 & | & \phantom{-}12 \\
# -3 & -2 & \phantom-4 & | & -2 \\
# \phantom{-}1 & \phantom{-}1 & -2 & | & \phantom{-}8 \\
# \end{bmatrix}
# $$
# 
# Note: For this assignment, matrix \(A\) is **always square**, accommodating scenarios with \(n\) equations and \(n\) variables.
# 
# ### Step 2: Transform Matrix into Row Echelon Form
# Initiate row operations to convert the augmented matrix into row-echelon form. The objective is to introduce zeros below the leading diagonal.
# 
# - **Row Switching:** Rearrange rows to position the leftmost non-zero entry at the top.
# - **Row Scaling:** Multiply a row by a non-zero scalar.
# - **Row Replacement:** Substitute a row with the sum of itself and a multiple of another row.
# 
# ### Step 3: Back Substitution
# 
# After attaining the row-echelon form, solve for variables starting from the last row and progressing upwards.
# 
# Remember, the aim is to simplify the system for easy determination of solutions!
# 
# ### Step 4: Compile the Gaussian Elimination Algorithm
# 
# Combine each function related to the aforementioned steps into a single comprehensive function.

# <a name="2"></a>
# ## 2 - Necessary imports
# 
# Next codeblock will import the necessary libraries to run this assignment. Please do not add nor remove any value there.

# In[68]:


import numpy as np


# In[69]:


import w2_unittest


# <a name="3"></a>
# ## 3 - Auxiliary functions
# 
# This section introduces three auxiliary functions crucial for facilitating your assignment. These functions have already been coded, eliminating the need for your concern regarding their implementation. However, it's essential to examine them carefully to grasp their appropriate usage.
# 
# **Note: In Python, indices commence at $0$ rather than $1$. Therefore, a matrix with $n$ rows is indexed as $0, 1, 2, \ldots, n-1$.**
# 

# <a name="3.1"></a>
# ### 3.1 - Function swap rows
# 
# This function has as input a [numpy array](https://numpy.org/doc/stable/reference/generated/numpy.array.html) and two indexes to swap the rows corresponding to those indexes. It **does not change the original matrix**, but returns a new one.

# In[70]:


def swap_rows(M, row_index_1, row_index_2):
    """
    Swap rows in the given matrix.

    Parameters:
    - matrix (numpy.array): The input matrix to perform row swaps on.
    - row_index_1 (int): Index of the first row to be swapped.
    - row_index_2 (int): Index of the second row to be swapped.
    """

    # Copy matrix M so the changes do not affect the original matrix. 
    M = M.copy()
    # Swap indexes
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M


# Let's practice with some examples. Consider the following matrix $M$.

# In[71]:


M = np.array([
[1, 3, 6],
[0, -5, 2],
[-4, 5, 8]
])
print(M)


# Swapping row $0$ with row $2$:

# In[72]:


M_swapped = swap_rows(M, 0, 2)
print(M_swapped)


# <a name="3.2"></a>
# ### 3.2 - Finding the first non-zero value in a column starting from a specific value
# 
# This function becomes essential when encountering a $0$ value during row operations. It determines whether a non-zero value exists below the encountered zero, allowing for potential row swaps. Consider the following scenario within a square matrix (non-augmented):
# 
# Let's say, during a specific step of the row-echelon form process, you've successfully reduced the first 2 rows, but you encounter a zero pivot (highlighted in red) in the third row. The task is to search, **solely in entries below the pivot**, for a potential row swap.
# 
# $$
# \begin{bmatrix}
# 6 & 4 & 8 & 1 \\
# 0 & 8 & 6 & 4 \\
# \color{darkred}0 & \color{darkred}0 & \color{red}0 & \color{darkred}3 \\
# 0 & 0 & 5 & 9 
# \end{bmatrix}
# $$
# 
# Performing a row swap between indexes 2 and 3 (remember, indexing starts at 0!), the matrix transforms into:
# 
# $$
# \begin{bmatrix}
# 6 & 4 & 8 & 1 \\
# 0 & 8 & 6 & 4 \\
# 0 & 0 & 5 & 9  \\
# 0 & 0 & 0 & 3 
# \end{bmatrix}
# $$
# 
# Resulting in the matrix achieving the row-echelon form.

# In[73]:


def get_index_first_non_zero_value_from_column(M, column, starting_row):
    """
    Retrieve the index of the first non-zero value in a specified column of the given matrix.

    Parameters:
    - matrix (numpy.array): The input matrix to search for non-zero values.
    - column (int): The index of the column to search.
    - starting_row (int): The starting row index for the search.

    Returns:
    int: The index of the first non-zero value in the specified column, starting from the given row.
                Returns -1 if no non-zero value is found.
    """
    # Get the column array starting from the specified row
    column_array = M[starting_row:,column]
    for i, val in enumerate(column_array):
        # Iterate over every value in the column array. 
        # To check for non-zero values, you must always use np.isclose instead of doing "val == 0".
        if not np.isclose(val, 0, atol = 1e-5):
            # If one non zero value is found, then adjust the index to match the correct index in the matrix and return it.
            index = i + starting_row
            return index
    # If no non-zero value is found below it, return -1.
    return -1


# Let's practice with this function. Consider the following matrix.

# In[74]:


N = np.array([
[0, 5, -3 ,6 ,8],
[0, 6, 3, 8, 1],
[0, 0, 0, 0, 0],
[0, 0, 0 ,0 ,7],
[0, 2, 1, 0, 4]
]
)
print(N)


# If you search for a value below the first column starting at the first row, the function should return -1:

# In[75]:


print(get_index_first_non_zero_value_from_column(N, column = 0, starting_row = 0))


# Searching for the first non zero value in the last column starting from row with index 2, it should return 3 (index corresponding to the value 7).

# In[76]:


print(get_index_first_non_zero_value_from_column(N, column = -1, starting_row = 2))


# <a name="3.3"></a>
# ### 3.3 - Find the first non zero element for any row
# 
# This function aids in locating the pivot within a designated row of a matrix. It identifies the index of the first non-zero element in the desired row. If no non-zero value is present, it returns -1.

# In[77]:


def get_index_first_non_zero_value_from_row(M, row, augmented = False):
    """
    Find the index of the first non-zero value in the specified row of the given matrix.

    Parameters:
    - matrix (numpy.array): The input matrix to search for non-zero values.
    - row (int): The index of the row to search.
    - augmented (bool): Pass this True if you are dealing with an augmented matrix, 
                        so it will ignore the constant values (the last column in the augmented matrix).

    Returns:
    int: The index of the first non-zero value in the specified row.
                Returns -1 if no non-zero value is found.
    """

    # Create a copy to avoid modifying the original matrix
    M = M.copy()


    # If it is an augmented matrix, then ignore the constant values
    if augmented == True:
        # Isolating the coefficient matrix (removing the constant terms)
        M = M[:,:-1]
        
    # Get the desired row
    row_array = M[row]
    for i, val in enumerate(row_array):
        # If finds a non zero value, returns the index. Otherwise returns -1.
        if not np.isclose(val, 0, atol = 1e-5):
            return i
    return -1


# Let's practice with the same matrix as before:

# In[78]:


print(N)


# If not passing the argument `augmented`, then it is assumed the matrix is not augmented. 
# 
# Looking for the first non-zero index in row $2$ must return -1 whereas in row $3$, the value returned must be $4$ (the index for the value $7$ in that row). 

# In[79]:


print(f'Output for row 2: {get_index_first_non_zero_value_from_row(N, 2)}')
print(f'Output for row 3: {get_index_first_non_zero_value_from_row(N, 3)}')


# Now, let's pass the argument `augmented = True`. This will make the algorithm consider $N$ an augmented matrix, therefore the last column will be removed from consideration. Now, the output for row 3 (starting from 0) should be different, excluding the last column, the output should be `-1` as well, since in the coefficient matrix (the matrix without the last column) there is no non-zero element:

# In[80]:


print(f'Output for row 3: {get_index_first_non_zero_value_from_row(N, 3, augmented = True)}')


# <a name="3.5"></a>
# ### 3.4 - Constructing the Augmented Matrix
# 
# This function constructs the augmented matrix by combining a square matrix of size $n \times n$, representing $n$ equations with $n$ variables each, with an $n \times 1$ matrix that denotes its constant values. The function concatenates both matrices to form the augmented matrix and returns the result.

# In[81]:


def augmented_matrix(A, B):
    """
    Create an augmented matrix by horizontally stacking two matrices A and B.

    Parameters:
    - A (numpy.array): First matrix.
    - B (numpy.array): Second matrix.

    Returns:
    - numpy.array: Augmented matrix obtained by horizontally stacking A and B.
    """
    augmented_M = np.hstack((A,B))
    return augmented_M


# In[82]:


A = np.array([[1,2,3], [3,4,5], [4,5,6]])
B = np.array([[1], [5], [7]])

print(augmented_matrix(A,B))


# <a name="4"></a>
# ## 4 - Row echelon form
# 
# <a name="4.1"></a>
# 
# <a name="4.1"></a>
# ### 4.1 - Row Echelon Form
# 
# As discussed in the lectures, a matrix in row echelon form adheres to the following conditions:
# 
# - Rows consisting entirely of zeroes should be positioned at the bottom.
# - Each non-zero row must have its left-most non-zero coefficient (termed as a **pivot**) located to the right of any row above it. Consequently, all elements below the pivot within the same column should be 0.
# 
# 
# **NOTE:**
# 
# - The algorithm you will build will consider only non-singular system of equations, this implies that the coefficient matrix must have determinant different from $0$. Also, it implies one very important property: **the matrix's row echelon form will have all its pivots in the main diagonal**. This is an important property because it will significantly simplify the computation.
# 
# 
# This form ensures a structured arrangement facilitating subsequent steps in the Gaussian elimination process.
# 
# 
# Example of matrix **in row echelon form**
# 
# $$M =
# \begin{bmatrix}
# 7 & 2 & 3 \\
# 0 & 9 & 4 \\
# 0 & 0 & 3 \\
# \end{bmatrix}
# $$
# 
# Examples of matrices that **are not in row echelon form**
# 
# $$
# A = \begin{bmatrix}
# 1 & 2 & 2 \\
# 0 & 5 & 3 \\
# 1 & 0 & 8 \\
# \end{bmatrix}
# $$
# 
# $$B = 
# \begin{bmatrix}
# 1 & 2 & 3 \\
# 0 & 0 & 4 \\
# 0 & 0 & 7 \\
# \end{bmatrix}
# $$
# 
# Matrix $A$ fails to satisfy the criteria for row echelon form as there exists a non-zero element below the first pivot (located in row 0). Similarly, matrix $B$ does not meet the requirements as the second pivot (in row 1 with a value of 4) has a non-zero element below it.

# <a name="4.2"></a>
# ### 4.2 - A worked example 
# 
# In this section, you'll revisit an example from the lecture to facilitate the implementation of an algorithm. If you feel confident in proceeding with the algorithm, you may skip this section.
# 
# Consider matrix $M$ given by:
# 
# $$
# M = 
# \begin{bmatrix} 
# * & * & * & \\
# 0 & \text{pivot} & * \\
# 0 & \text{value} & * 
# \end{bmatrix}
# $$
# 
# Here, the asterisk (*) denotes any number. To nullify the last row (row $2$), two steps are required:
# 
# - Scale $R_1$ by the inverse of the pivot:
# 
# $$
# \text{Row 1} \rightarrow \frac{1}{\text{pivot}} \cdot \text{Row } 
# $$
# 
# Resulting in the updated matrix with the pivot for row $1$ set to $1$:
# 
# $$
# M = 
# \begin{bmatrix} 
# * & * & * & \\
# 0 & 1 & * \\
# 0 & \text{value} & * 
# \end{bmatrix}
# $$
# 
# Next, to eliminate the value below the pivot in row $1$, apply the following formula:
# 
# $$
# \text{Row 2} \rightarrow \text{Row 2} - \text{value}\cdot \text{Row 1}
# $$
# 
# This transformation yields the modified matrix:
# 
# $$
# M = 
# \begin{bmatrix} 
# * & * & * & \\
# 0 & 1 & * \\
# 0 & 0 & * 
# \end{bmatrix}
# $$

# **Note that the square matrix $A$ needs to be in row-echelon form. However, every row operation conducted must also affect the augmented (constant) part. This ensures that you are effectively preserving the solutions for the entire system!** 

# Consider the following system of equations:
# 
# 
# $$
# \begin{align*}
# 2x_2 + x_3 &= 3 \\
# x_1 + x_2 +x_3 &= 6 \\
# x_1 + 2x_2 + 1x_3 &= 12
# \end{align*}
# $$
# 
# Consequently, the square matrix $A$ is formulated as:
# 
# $$
# A = 
# \begin{bmatrix} 
# 0 & 2 & 1 & \\
# 1 & 1 & 1 & \\
# 1 & 2 & 1 & 
# \end{bmatrix}
# $$
# 
# The column vector (a matrix of size $n \times 1$) is represented by:
# 
# $$
# B = 
# \begin{bmatrix} 
# 3\\
# 6\\
# 12
# \end{bmatrix}
# $$
# 
# Combining matrices $A$ and $B$ yields the augmented matrix $M$:
# 
# $$
# M = 
# \begin{bmatrix} 
# 0 & 2 & 1 & | & 3 \\
# 1 & 1 & 1 & | & 6 \\
# 1 & 2 & 1 & | & 12 
# \end{bmatrix}
# $$

# **Step 1:**
# 
# Commencing with row $0$: The initial candidate for the pivot is always the value in the main diagonal of the matrix. Denoting row $0$ as $R_0$:
# 
# $$R_0= \begin{bmatrix} 0 & 2 & 1 & | & 3 \end{bmatrix}$$
# 
# The value in the main diagonal is the element $M[0,0]$ (the first element of the first column). The first row can be accessed by performing $M[0]$, i.e., $M[0] = R_0$.
# 
# The first row operation involves **scaling by the pivot's inverse**. Since the value in the main diagonal is $0$, necessitating a non-zero value for scaling by its inverse, you must switch rows in this case. Note that $R_1$ has a value different from $0$ in the required index. Consequently, switching rows $0$ and $1$:
# 
# $$R_0 \rightarrow R_1$$
# $$R_1 \rightarrow R_0$$
# 
# Resulting in the updated augmented matrix:
# 
# $$
# M = 
# \begin{bmatrix} 
# 1 & 1 & 1 & | & 6 \\
# 0 & 2 & 1 & | & 3 \\
# 1 & 2 & 1 & | & 12 
# \end{bmatrix}
# $$
# 
# Now, the pivot is already $1$, eliminating the need for row scaling. Following the formula:
# 
# $$ R_1 \rightarrow  R_1 - 0 \cdot R_0 = R_1$$
# 
# Therefore, the second row remains unchanged. Moving to the third row ($R_2$), the value in the augmented matrix below the pivot from $R_0$ is $M[2,0]$, which is $1$.
# 
# $$R_2 = R_2 - 1 \cdot R_0 = \begin{bmatrix} 0 & 1 & 0 & | & 6  \end{bmatrix}$$
# 
# Resulting in the modified augmented matrix:
# 
# $$
# M = 
# \begin{bmatrix} 
# 1 & 1 & 1 & | & 6 \\
# 0 & 2 & 1 & | & 3 \\
# 0 & 1 & 0 & | & 6
# \end{bmatrix}
# $$
# 
# Progressing to the second row ($R_1$), the value in the main diagonal is $2$, different from zero. Scaling it by $\frac{1}{2}$:
# 
# $$R_1 = \frac{1}{2}R_1$$
# 
# Resulting in the augmented matrix:
# 
# $$
# M = 
# \begin{bmatrix} 
# 1 & 1 & 1 & | & 6 \\
# 0 & 1 & \frac{1}{2} & | & \frac{3}{2} \\
# 0 & 1 & 0 & | & 6
# \end{bmatrix}
# $$
# 
# Now, there's only one row below it for row replacement. The value just below the pivot is located at $M[2,1]$, which is $1$. Thus:
# 
# $$R_2 = R_2 - 1 \cdot R_1 = \begin{bmatrix} \phantom{-}0 & \phantom{-}0 & -\frac{1}{2} & | & \phantom{-}\frac{9}{2} \end{bmatrix} $$
# 
# Resulting in the augmented matrix:
# 
# 
# $$
# M = 
# \begin{bmatrix} 
# \phantom{-}1 & \phantom{-}1 & \phantom{-}1 & | & \phantom{-}6 \\
# \phantom{-}0 & \phantom{-}1 & \phantom{-}\frac{1}{2} & | & \phantom{-}\frac{3}{2} \\
# \phantom{-}0 & \phantom{-}0 & -\frac{1}{2} & | & \phantom{-}\frac{9}{2} 
# \end{bmatrix}
# $$
# 
# Finally, normalizing the last row as
# 
# $$R_2 = -2 \cdot R_2$$
# 
# The resulting matrix is
# 
# $$
# M = 
# \begin{bmatrix} 
# \phantom{-}1 & \phantom{-}1 & \phantom{-}1 & | & \phantom{-}6 \\
# \phantom{-}0 & \phantom{-}1 & \phantom{-}\frac{1}{2} & | & \phantom{-}\frac{3}{2} \\
# \phantom{-}0 & \phantom{-}0 & \phantom{-}1 & | & -9 
# \end{bmatrix}
# $$
# 
# Thus, the matrix is now in row echelon form with unitary pivots.

# Now you are ready to go! You will implement such algorithm in the following exercise.
# 
# <a name="ex01"></a>
# ### Exercise 1
# 
# This exercise involves implementing the elimination method to convert a matrix into row-echelon form. As discussed in lectures, the primary approach involves inspecting the values along the diagonal. If they equate to $0$, an attempt to swap rows should be made to obtain a non-zero value.
# 

# In[83]:


# GRADED FUNCTION: row_echelon_form

def row_echelon_form(A, B):
    """
    Utilizes elementary row operations to transform a given set of matrices, 
    which represent the coefficients and constant terms of a linear system, into row echelon form.

    Parameters:
    - A (numpy.array): The input square matrix of coefficients.
    - B (numpy.array): The input column matrix of constant terms

    Returns:
    numpy.array: A new augmented matrix in row echelon form with pivots as 1.
    """
    
    # Before any computation, check if matrix A (coefficient matrix) has non-zero determinant. 
    # It will be used the numpy sub library np.linalg to compute it.

    det_A = np.linalg.det(A)

    # Returns "Singular system" if determinant is zero
    if np.isclose(det_A, 0) == True:
        return 'Singular system'

    # Make copies of the input matrices to avoid modifying the originals
    A = A.copy()
    B = B.copy()


    # Convert matrices to float to prevent integer division
    A = A.astype('float64')
    B = B.astype('float64')

    # Number of rows in the coefficient matrix
    num_rows = len(A) 

    ### START CODE HERE ###

    # Transform matrices A and B into the augmented matrix M
    M = augmented_matrix(A,B)
    # Iterate over the rows.
    for row in range(num_rows):

        # The first pivot candidate is always in the main diagonal, let's get it. 
        # Remember that the diagonal elements in a matrix has the same index for row and column. 
        # You may access a matrix value by typing M[row, column]. In this case, column = None
        pivot_candidate = M[row, row]

        # If pivot_candidate is zero, it cannot be a pivot for this row. 
        # So the first step you need to take is to look at the rows below it to check if there is a non-zero element in the same column.
        # The usage of np.isclose is a good practice when comparing two floats.
        if np.isclose(pivot_candidate, 0) == True: 
            # Get the index of the first non-zero value below the pivot_candidate. 
            first_non_zero_value_below_pivot_candidate = get_index_first_non_zero_value_from_column(M, row, row) 

            # Swap rows
            M = swap_rows(M, row, first_non_zero_value_below_pivot_candidate) 

            # Get the pivot, which is in the main diagonal now 
            pivot = M[row,row] 
        
        # If pivot_candidate is already non-zero, then it is the pivot for this row
        else:
            pivot = pivot_candidate
        
        # Now you are ready to apply the row reduction in every row below the current
            
        # Divide the current row by the pivot, so the new pivot will be 1. You may use the formula current_row -> 1/pivot * current_row
        # Where current_row can be accessed using M[row].
        M[row] = (1/pivot) * M[row]

        # Perform row reduction for rows below the current row
        for j in range(row + 1, num_rows): 
            # Get the value in the row that is below the pivot value. 
            # Remember that, since you are dealing only with non-singular matrices, the pivot is in the main diagonal.
            # Therefore, the values in row j that are below the pivot, must have column index the same index as the column index for the pivot.
            value_below_pivot = M[j,row]

            # Perform row reduction using the formula:
            # row_to_reduce -> row_to_reduce - value_below_pivot * pivot_row
            M[j] = M[j] - value_below_pivot*M[row]
            
    ### END CODE HERE ###

    return M
            


# In[84]:


A = np.array([[1,2,3],[0,1,0], [0,0,5]])
B = np.array([[1], [2], [4]])
row_echelon_form(A,B)


# In[85]:


w2_unittest.test_row_echelon_form(row_echelon_form)


# <a name="5"></a>
# ## 5 - Back substitution
# 
# The final step of the algorithm involves back substitution, a crucial process in obtaining solutions for the linear system. As discussed in the lectures, this method initiates from the bottom and moves upwards. Utilizing elementary row operations, it aims to convert every element above the pivot into zeros, ending with a matrix in **reduced row echelon form**. The formula employed is:
# 
# 
# $$\text{Row above} \rightarrow \text{Row above} - \text{value} \cdot \text{Row pivot}$$
# 
# In this equation, $\text{value}$ denotes the value above the pivot, which initially equals 1. To illustrate this process, let's consider the following matrix:

# $$
# M = 
# \begin{bmatrix} 
# \phantom{-}1 & -1 & \phantom{-}\frac{1}{2} & | & \phantom{-}\frac{1}{2} \\
# \phantom{-}0 & \phantom{-}1 & \phantom{-}1 & | & -1 \\
# \phantom{-}0 & \phantom{-}0 & \phantom{-}1 & | & -1 
# \end{bmatrix}
# $$
# 
# Starting from bottom to top:
# 
# - $R_2$:
# 
# - -  $R_1 = R_1 - 1 \cdot R_2 = \begin{bmatrix} 0 & 1 & 0 & | & 0 \end{bmatrix}$
# - - $R_0 = R_0 - \frac{1}{2} \cdot R_2 = \begin{bmatrix} 1 & -1& 0 & | & 1 \end{bmatrix}$
# 
# The resulting matrix is then
# 
# $$
# M = 
# \begin{bmatrix} 
# \phantom{-}1 & -1 & \phantom{-}0 & | & \phantom{-}1  \\
# \phantom{-}0 & \phantom{-}1 & \phantom{-}0 & | & \phantom{-}0 \\
# \phantom{-}0 & \phantom{-}0 & \phantom{-}1 & | & -1 
# \end{bmatrix}
# $$
# 
# Moving to $R_1$:
# 
# - $R_1$:
# 
# - - $R_0 = R_0 - \left(-1 \cdot R_1 \right) = \begin{bmatrix} 1 & 0 & 0 & | & 1 \end{bmatrix}$
# 
# And the final matrix is
# 
# $$
# M = 
# \begin{bmatrix} 
# \phantom{-}1 & \phantom{-}0 & \phantom{-}0 & | & \phantom{-}1  \\
# \phantom{-}0 & \phantom{-}1 & \phantom{-}0 & | & \phantom{-}0 \\
# \phantom{-}0 & \phantom{-}0 & \phantom{-}1 & | & -1
# \end{bmatrix}
# $$

# Note that after back substitution, the solution is just the values in the augmented column! In this case,
# 
# $$
# x_0 = 1 \\ x_1 =0\\ x_2 = -1
# $$

# <a name="ex02"></a>
# ### Exercise 2
# 
# In this exercise you will implement a function to perform back substitution in an **augmented matrix with unique solution and in row echelon form with unitary pivots**

# In[86]:


# GRADED FUNCTION: back_substitution

def back_substitution(M):
    """
    Perform back substitution on an augmented matrix (with unique solution) in reduced row echelon form to find the solution to the linear system.

    Parameters:
    - M (numpy.array): The augmented matrix in row echelon form with unitary pivots (n x n+1).

    Returns:
    numpy.array: The solution vector of the linear system.
    """
    
    # Make a copy of the input matrix to avoid modifying the original
    M = M.copy()

    # Get the number of rows (and columns) in the matrix of coefficients
    num_rows = M.shape[0]

    ### START CODE HERE ####
    
    # Iterate from bottom to top
    for row in reversed(range(num_rows)): 
        substitution_row = row

        
        # Get the index of the first non-zero element in the substitution row. Remember to pass the correct value to the argument augmented.
        index = get_index_first_non_zero_value_from_row(M, substitution_row, augmented = True)

        # Iterate over the rows above the substitution_row
        for j in range(row): 

            # Get the row to be reduced. The indexing here is similar as above, with the row variable replaced by the j variable.
            row_to_reduce = M[j]

            # Get the value of the element at the found index in the row to reduce
            value = row_to_reduce[substitution_row]

            # Perform the back substitution step using the formula row_to_reduce -> row_to_reduce - value * substitution_row
            row_to_reduce = row_to_reduce - value * M[substitution_row]

            # Replace the updated row in the matrix, be careful with indexing!
            M[j,:] = row_to_reduce

    ### END CODE HERE ####

     # Extract the solution from the last column
    solution = M[:,-1]    
    return solution


# In[87]:


w2_unittest.test_back_substitution(back_substitution)


# <a name="6"></a>
# ## 6 - The Gaussian Elimination
# 
# <a name="6.1"></a>
# ### 6.1 - Bringing it all together
# 
# Your task now is to integrate all the steps achieved thus far. Start with a square matrix $A$ of size $ n \times n$ and a column matrix $B$ of size $n \times 1$ and transform the augmented matrix $[A | B]$ into reduced row echelon form. Subsequently, verify the existence of solutions. If solutions are present, proceed to perform back substitution to obtain the values. In scenarios where there are no solutions or an infinite number of solutions, handle and indicate these outcomes accordingly.

# <a name="ex03"></a>
# ### Exercise 3
# 
# In this exercise you will combine every function you just wrote to finish the Gaussian Elimination algorithm.

# In[88]:


# GRADED FUNCTION: gaussian_elimination

def gaussian_elimination(A, B):
    """
    Solve a linear system represented by an augmented matrix using the Gaussian elimination method.

    Parameters:
    - A (numpy.array): Square matrix of size n x n representing the coefficients of the linear system
    - B (numpy.array): Column matrix of size 1 x n representing the constant terms.

    Returns:
    numpy.array or str: The solution vector if a unique solution exists, or a string indicating the type of solution.
    """

    ### START CODE HERE ###

    # Get the matrix in row echelon form
    row_echelon_M = row_echelon_form(A,B)

    # If the system is non-singular, then perform back substitution to get the result. 
    # Since the function row_echelon_form returns a string if there is no solution, let's check for that.
    # The function isinstance checks if the first argument has the type as the second argument, returning True if it does and False otherwise.
    if not isinstance(row_echelon_M, str): 
        solution = back_substitution(row_echelon_M)

    ### END SOLUTION HERE ###

    return solution
        


# In[89]:


w2_unittest.test_gaussian_elimination(gaussian_elimination)


# <a name="7"></a>
# ## 7 - Test with any system of equations!
# 
# The code below will allow you to write any equation in the format it is given below (any unknown lower case variables are accepted, in any order) and transform it in its respective augmented matrix so you can solve it using the functions you just wrote in this assignment!
# 
# You just need to change the equations variable, always keeping * to indicate product between unknowns and variables and one equation in each line!

# In[90]:


from utils import string_to_augmented_matrix


# In[91]:


equations = """
3*x + 6*y + 6*w + 8*z = 1
5*x + 3*y + 6*w = -10
4*y - 5*w + 8*z = 8
4*w + 8*z = 9
"""

variables, A, B = string_to_augmented_matrix(equations)

sols = gaussian_elimination(A, B)

if not isinstance(sols, str):
    for variable, solution in zip(variables.split(' '),sols):
        print(f"{variable} = {solution:.4f}")
else:
    print(sols)


# Congratulations! You have finished the first assignment of this course! You built from scratch a linear system solver!
