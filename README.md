# Solving Sparse Linear Systems

There are two types of matrices—and corresponding types of linear systems—based on the frequency of zeroes in a matrix: **sparse** and **dense**.  
  
A **sparse matrix** is an _m_ x _n_ matrix containing a large number of zeroes which stores fewer than _m_ x _n_ entries.<sup><a href="#ref1">[1]</a></sup> More strictly defined, a sparse matrix must have _O_(min(_m_, _n_)) nonzero entries.<sup><a href="#ref1">[1]</a></sup>

A **dense matrix**, on the other hand, is an _m_x_n_ matrix that stores all _m_x_n_ elements.<sup><a href="#ref1">[1]</a></sup>  In practice, when properly and optimally implemented, dense matrices do not contain a large number of zeroes, since the matrix could otherwise be represented as a sparse matrix. While dense matrices can have entries equal to zero, they have _O_(_n<sup>2</sup>_) nonzero entries.<sup><a href="#ref8">[8]</a></sup>

Sparse matrices can be stored more efficiently than dense matrices by taking advantage of repeated patterns of zeroes to reduce the number of entries that need to be stored.<sup><a href="#ref1">[1]</a></sup>  
  
A **sparse linear system** is a linear system of the form Ax = b where A is a sparse matrix, while a **dense linear system** has a dense matrix as the A term.  
  
Given an organized structure of nonzero elements in a sparse matrix, there may additionally be specialized algorithms to solve a sparse linear system more optimally (i.e. with a faster runtime).<sup><a href="#ref8">[8]</a></sup>  

## Contents
[History](#history)
<br />
[Drawbacks of Dense Linear Systems and Advantages of Sparse Systems](#drawbacks-of-dense-linear-systems-and-advantages-of-sparse-systems)
<br />
[Triangular Linear Systems](#triangular-linear-systems)
<br />
[Banded Linear Systems](#banded-linear-systems)
<br />
[Tridiagonal Linear Systems](#tridiagonal-linear-systems)
<br />
[Pentadiagonal Linear Systems](#pentadiagonal-linear-systems)
<br />
[Hessenberg Linear Systems](#hessenberg-linear-systems)
<br />
[Direct vs. Iterative Methods](#direct-vs.-iterative-methods)
<br />
[Conjugate Gradient Method](#conjugate-gradient-method)
<br />
[References](#references)

## History
Interest in sparse matrices developed in the mid-twentieth century during the adoption of early computers to solve linear systems, first when dealing with electrical grids and later in engineering, finance, and scheduling.<sup><a href="#ref8">[8]</a></sup> In practice, the systems that resulted from these scenarios were filled with patterns of zeroes, leading researchers to realize they could significantly increase the efficiency of computation by taking advantage of these patterns.<sup><a href="#ref8">[8]</a></sup> 

## Drawbacks of Dense Linear Systems and Advantages of Sparse Systems

Dense linear systems are solved using algorithms including, but not limited to, LU decomposition, Cholesky decomposition, singular value decomposition (SVD), and QR iteration.<sup><a href="#ref7">[7]</a></sup> While implementation of these algorithms is not important for this article, it is important to understand the implication of their time complexities. With _O_(_n<sup>3</sup>_) runtimes,<sup><a href="#ref7">[7]</a></sup> these algorithms can be incredibly computationally expensive. For relatively small matrices, the computational benefits of sparse matrices are negligible, but for large matrices with dimensions in the thousands or millions, any improvement that can be made due to a pattern of zeroes in a sparse matrix can improve the runtime to solve a system by orders of magnitude.<sup><a href="#ref8">[8]</a></sup>

## Triangular Linear Systems

There are two types of **triangular matrices**: **upper triangular** and **lower triangular**.<sup><a href="#ref2">[2]</a></sup> An upper triangular matrix contains only zeroes below the **main diagonal**—the diagonal going from the top left to the bottom right corner—while a lower triangular matrix contains only zeroes above the main diagonal.<sup><a href="#ref2">[2]</a></sup> Note that an upper triangular matrix can have any number of zeroes below the main diagonal and a lower triangular matrix can have any number of zeroes above the main diagonal; thus, while all triangular matrices may not be considered sparse by the strictest of definitions,<sup><a href="#ref1">[1]</a></sup> many sparse matrices meet the definition of being triangular and can take advantage of methods to solve triangular linear systems.

<br />

$$\begin{bmatrix}a & b & c \\ 0 & d & e \\ 0 & 0 & f \end{bmatrix}$$

Upper Triangular Matrix

<br />

$\begin{bmatrix}a & 0 & 0\\b & c & 0\\d & e & f \end{bmatrix}$

Lower Triangular Matrix

<br />
  
Upper triangular linear systems are solved using a technique called **backward substitution**. Backward substitution starts from the last row of an upper triangular matrix (i.e., the one with only one entry, in the last column) and solves for the corresponding entry. Then, it substitutes this value into row above and solves for the corresponding entry. This process continues for all rows. Below is an example of the algorithm in Python:

```
import numpy as np  # for matrix operations

def backward_substitution(U, b):  # function to solve an upper triangular matrix
	n = U.shape[0]  # get the size of the matrix
	x = np.zeros_like(b)  # initialize the solution vector with zeros

	# Start with the last row and solve upwards
	x[n - 1] = b[n - 1] / U[n - 1, n - 1]  # solve for the last variable
	for i in range(n - 2, -1, -1):  # iterate from the second-to-last row upwards
		sum_ax = np.dot(U[i, i+1:], x[i+1:])  # compute the sum of known terms
		x[i] = (b[i] - sum_ax) / U[i, i]  # isolate and solve for x[i]

	return x  # return the solution  
```

Lower triangular systems are solved using **forward substitution**. As can be inferred by its name, this technique is similar to backward substitution, working from top to bottom instead of bottom to top. Below is an example of the algorithm in Python:  

```
import numpy as np  # for matrix operations

def forward_substitution(L, b):  # function to solve a lower triangular matrix
	n = L.shape[0]  # get the size of the matrix
	x = np.zeros_like(b)  # initialize the solution vector with zeros

	# Start with the first row and solve downwards
	x[0] = b[0] / L[0, 0]  # solve for the first variable
	for i in range(1, n):  # iterate from the second row downwards
		sum_ax = np.dot(L[i, :i], x[:i])  # compute the sum of known terms
		x[i] = (b[i] - sum_ax) / L[i, i]  # isolate and solve for x[i]

	return x  # return the solution
```

Both backward and forward substitution have _O_(_n_) time complexity.

## Banded Linear Systems

**Banded matrices** are a type of sparse matrix in which only the main diagonal and   several diagonals directly above and below the main diagonal have nonzero entries.<sup><a href="#ref9">[9]</a></sup> The **bandwidth** of a banded matrix refers to the number of diagonals in either direction (above or below the main diagonal) that are nonzero.<sup><a href="#ref9">[9]</a></sup>

## Tridiagonal Linear Systems

A **tridiagonal matrix** is an _n_ x _n_ banded matrix with a bandwidth of one (i.e., the only nonzero entries are in the main diagonal, the diagonal directly above the main diagonal, and the diagonal directly below the main diagonal).<sup><a href="#ref3">[3]</a></sup>

<br />

$\begin{bmatrix}a & b & 0 & 0 & 0\\c & d & e & 0 & 0\\0 & f & g & h & 0\\0 & 0 & i & j & k\\0 & 0 & 0 & l & m\end{bmatrix}$

Tridiagonal Matrix

<br />

 Although many algorithms for solving tridiagonal linear systems exist, one of the most common is the Thomas algorithm,<sup><a href="#ref3">[3]</a></sup> created by Llewellyn Thomas, a physicist at Columbia University’s Watson Scientific Computing Laboratory,<sup><a href="#ref5">[5]</a></sup> in 1949<sup><a href="#ref4">[4]</a></sup>. The Thomas algorithm creates an upper diagonal matrix before performing back substitution,<sup><a href="#ref3">[3]</a></sup> giving it _O_(_n_) time complexity. Below is an implementation of the algorithm in Python:  

```
import numpy as np  #for matrix operations  
  
def thomas_algorithm(A, b): #function to solve a tridiagonal matrix
	A[0][1] /= A[0][0] #normalize
	b[0] /= A[0][0] #normalize
	A[0][0] = 1 #normalize

	for i in range(1, A.shape[0]): #for each row
		if i < A.shape[0] - 1: #for all but last row
			A[i][i+1] /= A[i][i] - A[i][i-1] * A[i-1][i]
		b[i] -= A[i][i-1] * b[i - 1] #subtract product of subdiagonal and previous vector element
		b[i] /= A[i][i] - A[i][i-1] * A[i-1][i] #normalize
		A[i][i-1] = 0 #set subdiagonal to zero
		A[i][i] = 1 #set diagonal to one
		
	for i in range(A.shape[0] - 2, -1, -1): #iterate from bottom row up
		b[i] -= b[i + 1] * A[i][i+1] #isolate diagonal
		A[i][i+1] = 0 #set superdiagonal to 0

return b #return solution  
```

## Pentadiagonal Linear Systems
Pentadiagonal matrices are banded matrices with a bandwidth of two.<sup><a href="#ref9">[9]</a></sup>

<br />

$\begin{bmatrix}x_1 & x_2 & x_3 & 0 & 0 & 0 & 0\\x_4 & x_5 & x_6 & x_7 & 0 & 0 & 0\\x_8 & x_9 & x_{10} & x_{11} & x_{12} & 0 & 0\\0 & x_{13} & x_{14} & x_{15} & x_{16} & x_{17} & 0\\0 & 0 & x_{18} & x_{19} & x_{20} & x_{21} & x_{22}\\0 & 0 & 0 & x_{23} & x_{24} & x_{25} & x_{26}\\0 & 0 & 0 & 0 & x_{27} & x_{28} & x_{29}\end{bmatrix}$

Pentadiagonal Matrix

<br />

Pentadiagonal matrices can be solved using a similar algorithm to the Thomas algorithm,<sup><a href="#ref10">[10]</a></sup> as shown below in Python:

```
import numpy as np  #for matrix operations

#function to solve a pentadiagonal system
def solve_pentadiagonal_system(A, b):
    n = len(A)  #number of rows in A
    x = np.zeros_like(b)  #solution vector
    y = np.zeros_like(b)  #intermediate vector

    #forward substitution
    for i in range(n): #for each row
        y[i] = b[i]  #initialize y[i]
        
        #subtract contributions from lower diagonals
        if i > 0: #starting at 1st row
            y[i] -= A[i, i - 1] * y[i - 1] #subtract diagonal
        if i > 1: #starting at 2nd row
            y[i] -= A[i, i - 2] * y[i - 2] #subtract diagonal
        
        #subtract contributions from upper diagonals
        if i < n - 1: #for all but last row
            y[i] -= A[i, i + 1] * y[i + 1] #subtract diagonal
        if i < n - 2: #for all but second-to-last row
            y[i] -= A[i, i + 2] * y[i + 2] #subtract diagonal
        
        #normalize by diagonal
        y[i] /= A[i, i]

    # backward substitution
    for i in range(n - 1, -1, -1): #for each row (bottom to top)
        x[i] = y[i]  #initialize x[i]
        
        #subtract contributions from upper diagonals
        if i < n - 1: #for all but last row
            x[i] -= A[i, i + 1] * x[i + 1] #subtract diagonal
        if i < n - 2: #for all but second-to-last row
            x[i] -= A[i, i + 2] * x[i + 2] #subtract diagonal
        
        #subtract contributions from lower diagonals
        if i > 0: #starting at 1st row
            x[i] -= A[i, i - 1] * x[i - 1] #subtract diagonal
        if i > 1: #starting at 2nd row
            x[i] -= A[i, i - 2] * x[i - 2] #subtract diagonal
        
        x[i] /= A[i, i] #normalize by diagonal

    return x  #return solution vector

```

## Hessenberg Linear Systems

**Hessenberg matrices** are nearly triangular matrices. Similar to diagonal matrices, there are two types of Hessenberg matrices: **upper Hessenberg** and **lower Hessenberg**. An upper Hessenberg matrix contains only zeroes for all entries two or more diagonals below the main diagonal. A lower Hessenberg matrix contains only zeroes for all entries two or more diagonals above the main diagonal.

<br />

$\begin{bmatrix}a & b & c & d & e\\f & g & h & i & j\\0 & k & l & m & n\\0 & 0 & o & p & q\\0 & 0 & 0 & r & s\end{bmatrix}$

Upper Hessenberg Matrix

<br />

$\begin{bmatrix}a & b & 0 & 0 & 0\\c & d & e & 0 & 0\\f & g & h & i & 0\\j & k & l & m & n\\o & p & q & r & s\end{bmatrix}$

Lower Hessenberg Matrix

<br />

Hessenberg linear systems are solved in two steps: first, the Hessenberg matrix is converted to a triangular matrix. Then, the system is solved as a triangular system. For the first step, methods such as LU decomposition and QR factorization are used.

## Direct vs. Iterative Methods
All algorithms described up until  this point use **direct methods** with a predetermined number of steps. Alternatively, sparse linear systems can be solved using **iterative methods** that converge.<sup><a href="#ref8">[8]</a></sup> Though iterative methods can be slow, they are easy to implement.<sup><a href="#ref8">[8]</a></sup>

## Conjugate Gradient Method

The Conjugate Gradient Method is a frequently used iterative method to solve large sparse linear systems. It approaches solving the linear system as an optimization problem minimizing

$f(x) = \frac{1}{2}x^TAx - b^Tx+c$.

At each iteration, the algorithm computes the residual

$r_{(i)} = b - Ax_{(i)}$,

uses this residual to pick a search direction and updates x. The algorithm continues until the residual is below a certain tolerance.<sup><a href="#ref11">[11]</a></sup>

The conjugate gradient method has a time complexity of *O*(*m*$\sqrt{k}$), where *m* is the number of nonzero entries and *k* is the number of iterations performed.<sup><a href="#ref11">[11]</a></sup>
 
## References

1. CS 357 @ UIUC Course Staff. Sparse Matrices · CS 357 Textbook [Internet]. 2024 [cited 2024 Dec 13]. Available from: <a id="ref1" href="https://cs357.cs.illinois.edu/textbook/notes/sparse.html">https://cs357.cs.illinois.edu/textbook/notes/sparse.html</a>

2. Heroux MA. Solving Sparse Linear Systems Part 1: Basic Concepts [Internet]. Sandia National Laboratories; [cited 2024 Dec 14]. Available from: <a id="ref2" href="https://faculty.csbsju.edu/mheroux/HartreeTutorialPart1.pdf">https://faculty.csbsju.edu/mheroux/HartreeTutorialPart1.pdf</a>

3. Lee C-R, Chen Y-C. Augmented Block Cimmino Distributed Algorithm for solving tridiagonal systems on GPU [Internet]. Advances in GPU Research and Practice; 2017 [cited 2024 Dec 13]. Available from: <a id="ref3" href="https://www.sciencedirect.com/topics/computer-science/tridiagonal-matrix">https://www.sciencedirect.com/topics/computer-science/tridiagonal-matrix</a>

4. <p id="ref4">Thomas LH. Elliptic Problems in Linear Difference Equations Over a Network. Watson Scientific Computing Lab Report. 1949.</p>

5. Special Collections Research Center Staff, editor. Llewellyn Hilleth Thomas Papers, 1921-1989 [Internet]. [cited 2024 Dec 13]. Available from: <a id="ref5" href="https://www.lib.ncsu.edu/findingaids/mc00210">https://www.lib.ncsu.edu/findingaids/mc00210</a>

6. Licht M. Triangular Systems of Equations [Internet]. 2021 [cited 2024 Dec 14]. Available from: <a id="ref6" href="https://mathweb.ucsd.edu/~mlicht/wina2021/pdf/lecture05.pdf">https://mathweb.ucsd.edu/~mlicht/wina2021/pdf/lecture05.pdf</a>

7. Gates M. Dense Linear Algebra Part 2 [Internet]. University of Tennessee; 2019 [cited 2024 Dec 14]. Available from: <a id="ref7" href="https://icl.utk.edu/~mgates3/files/lect10-dla-part2-2019.pdf">https://icl.utk.edu/~mgates3/files/lect10-dla-part2-2019.pdf</a>

8. O’Connor D. An Introduction to Sparse Matrices [Internet]. [cited 2024 Dec 14]. Available from: <a id="ref8" href="https://www.irishmathsoc.org/nl15/nl15_6-30.pdf">https://www.irishmathsoc.org/nl15/nl15_6-30.pdf</a>

9. Yano M, Patera AT, Konidaris G, Penn JD. Banded Matrices [Internet]. MIT OpenCourseWare; 2022 [cited 2024 Dec 13]. Available from: <a id="ref9" href="https://eng.libretexts.org/Bookshelves/Mechanical_Engineering/Math_Numerics_and_Programming_(for_Mechanical_Engineers)/05%3A_Unit_V_-_(Numerical)_Linear_Algebra_2_-_Solution_of_Linear_Systems/27%3A_Gaussian_Elimination_-_Sparse_Matrices/27.01%3A_Banded_Matrices">https://eng.libretexts.org/Bookshelves/Mechanical_Engineering/Math_Numerics_and_Programming_(for_Mechanical_Engineers)/05%3A_Unit_V_-_(Numerical)_Linear_Algebra_2_-_Solution_of_Linear_Systems/27%3A_Gaussian_Elimination_-_Sparse_Matrices/27.01%3A_Banded_Matrices</a>

10. Karawia AA, Askar SS. On Solving Pentadiagonal Linear Systems via Transformations [Internet]. 2015 [cited 2024 Dec 14]. Available from: <a id="ref10" href="https://onlinelibrary.wiley.com/doi/10.1155/2015/232456">https://onlinelibrary.wiley.com/doi/10.1155/2015/232456</a>

11. Shewchuk JR. An Introduction to the Conjugate Gradient Method Without The Agonizing Pain [Internet]. 1994 [cited 2024 Dec 14]. Available from: <a id="ref11" href="https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf">https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf</a>
