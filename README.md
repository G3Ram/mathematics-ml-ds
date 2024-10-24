# Course Title: **Mathematics for Machine Learning and Data Science**

## Python (Pre-requisite)

Here are a few recommended learning materials

1. [Google's crash course in Python](https://www.coursera.org/learn/python-crash-course)
2. [University of Michigan's Python for everybody](https://www.coursera.org/learn/python)
3. [Python Data Structures](https://www.coursera.org/learn/python-data)
4. [CodeAcademy's learn Python](https://www.codecademy.com/learn/learn-python-3)
5. [Microsoft's Python for beginners](https://learn.microsoft.com/en-us/training/paths/beginner-python/)
6. [W3School's Python tutorial](https://www.w3schools.com/python/default.asp)
7. [Official Python documentation](https://docs.python.org/3/)

## Section 1: **Linear Algebra**

1. **Module 1.1: System of Linear Equations**

   Matrices are commonly used in machine learning and data science to represent data and its transformations. In this week, you will learn how matrices naturally arise from systems of equations and how certain matrix properties can be thought in terms of operations on system of equations.

   **Learning Objectives**

   - Form and graphically interpret 2x2 and 3x3 systems of linear equations
   - Determine the number of solutions to a 2x2 and 3x3 system of linear equations
   - Distinguish between singular and non-singular systems of equations
   - Determine the singularity of 2x2 and 3x3 system of equations by calculating the determinant

2. **Module 1.2: Solving Systems of Linear Equations**

   In this week, you will learn how to solve a system of linear equations using the elimination method and the row echelon form. You will also learn about an important property of a matrix: the rank. The concept of the rank of a matrix is useful in computer vision for compressing images.

   **Learning Objectives**

   - Solve a system of linear equations using the elimination method.
   - Use a matrix to represent a system of linear equations and solve it using matrix row reduction.
   - Solve a system of linear equations by calculating the matrix in the row echelon form.
   - Calculate the rank of a system of linear equations and use the rank to determine the number of solutions of the system.

3. **Module 1.3: Solving Systems of Linear Equations**

   An individual instance (observation) of data is typically represented as a vector in machine learning. In this week, you will learn about properties and operations of vectors. You will also learn about linear transformations, matrix inverse, and one of the most important operations on matrices: the matrix multiplication. You will see how matrix multiplication naturally arises from composition of linear transformations. Finally, you will learn how to apply some of the properties of matrices and vectors that you have learned so far to neural networks.

   **Learning Objectives**

   - Perform common operations on vectors like sum, difference, and dot product.
   - Multiply matrices and vectors.
   - Represent a system of linear equations as a linear transformation on a vector.
   - Calculate the inverse of a matrix, if it exists.

4. **Module 1.4: Solving Systems of Linear Equations**

   In this final week, you will take a deeper look at determinants. You will learn how determinants can be geometrically interpreted as an area and how to calculate determinant of product and inverse of matrices. We conclude this course with eigenvalues and eigenvectors. Eigenvectors are used in dimensionality reduction in machine learning. You will see how eigenvectors naturally follow from the concept of eigenbases.

   **Learning Objectives**

   - Interpret the determinant of a matrix as an area and calculate determinant of an inverse of a matrix and a product of matrices.
   - Determine the bases and span of vectors.
   - Find eigenbases for a special type of linear transformations commonly used in machine learning.
   - Calculate the eignenvalues and eigenvectors of a linear transformation (matrix).

## Section 2: **Calculus**

In this course, you will:

**Learning Objectives**

- Perform gradient descent in neural networks with different activation and cost functions
- Visually interpret differentiation of different types of functions commonly used in machine learning
- Approximately optimize different types of functions commonly used in machine learning using first-order (gradient descent) and second-order (Newton’s method) iterative methods
- Analytically optimize different types of functions commonly used in machine learning using properties of derivatives and gradients.

## Section 3: **Probability**

1. **Module 3.1: Introduction to Probability and Probability Distribution**

   In this week, you will learn about probability of events and various rules of probability to correctly do arithmetic with probabilities. You will learn the concept of conditional probability and the key idea behind Bayes theorem. In lesson 2, we generalize the concept of probability of events to probability distribution over random variables. You will learn about some common probability distributions like the Binomial distribution and the Normal distribution.

2. **Module 3.2: Describing Probability Distribution and Probability Distribution with Multi Variables**

   This week you will learn about different measures to describe probability distributions as well as any dataset. These include measures of central tendency (mean, median, and mode), variance, skewness, and kurtosis. The concept of the expected value of a random variable is introduced to help you understand each of these measures. You will also learn about some visual tools to describe data and distributions. In lesson 2, you will learn about the probability distribution of two or more random variables using concepts like joint distribution, marginal distribution, and conditional distribution. You will end the week by learning about covariance: a generalization of variance to two or more random variables.

3. **Module 3.3: Sampling and Point Estimation**

   This week shifts its focus from probability to statistics. You will start by learning the concept of a sample and a population and two fundamental results from statistics that concern samples and population: the law of large numbers and the central limit theorem. In lesson 2, you will learn the first and the simplest method of estimation in statistics: point estimation. You will see how maximum likelihood estimation, the most common point estimation method, works and how regularization helps prevent overfitting. You'll then learn how Bayesian Statistics incorporates the concept of prior beliefs into the way data is evaluated and conclusions are reached.

4. **Module 3.4: Confidence Intervals and Hypothesis Testing**

   This week you will learn another estimation method called interval estimation. The most common interval estimates are confidence intervals and you will see how they are calculated and how to correctly interpret them. In lesson 2, you will learn about hypothesis testing where estimates are formulated as a hypothesis and then tested in the presence of available evidence or a sample of data. You will learn the concept of p-value that helps in making a decision about a hypothesis test and also learn some common tests like the t-test, two-sample t-test, and the paired t-test. You will end the week with an interesting application of hypothesis testing in data science: A/B testing.
