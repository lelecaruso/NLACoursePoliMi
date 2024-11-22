#include <Eigen/SparseCore>
#include <Eigen/SparseQR>
#include <iostream>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include "grad.hpp"
#include "cg.hpp"

using namespace Eigen; //shorcut for eigen declaration
using SpMat=SparseMatrix<double,RowMajor>;
using namespace std;

int main(){
using namespace LinearAlgebra;
//Loading the matrix using loadmarket functions
SpMat mat;
loadMarket(mat,"diffreact.mtx");
cout<<"matrix dimensions are: "<<mat.rows()<<"X"<<mat.cols()<<endl;
//check if matrix is symm
double tol_symm = 1.0e-14;
SpMat matT = mat.transpose();
SpMat B = mat - matT;
if(B.norm()<= tol_symm){
    cout<<"The matrix is symm since the norm of mat-matT is zero"<<endl;
}
else{
    cout<<"The matrix isn't symm since the norm of mat-matT is bigger than zero"<<endl;
}
cout<<"matrix A norm is: "<<mat.norm()<<endl;
SpMat Ass = B*0.5;
cout<<"The norm of Ass is: "<<Ass.norm()<<endl;

//define b = A*xe 
VectorXd xe = VectorXd:: Ones(mat.cols());
VectorXd b = mat * xe;
cout<<"the norm of b is: "<<b.norm()<<endl;

/*
 Solve the linear system Ax = b using both the Gradient Method and Conjugate Gradi-
ent method (implemented in the grad.hpp and cg.hpp templates, respectively). Fix a
maximum number of iterations which is sufficient to reduce the (relative) residual below
than 10−8. Use the diagonal preconditioner provided by Eigen. Report on the sheet the
iteration counts and the relative residual at the last iteration.
*/

int result1,result2;
int maxit = int(mat.cols());;
double tol = 1.0e-8; // relative residual <= tol as a stopping crit
Eigen::DiagonalPreconditioner<double> D(mat); // Create diagonal preconditioner
VectorXd x(mat.cols());

result2 = CG(mat,x,b,D,maxit,tol);
cout << "CG flag = " << result2 << endl;
cout << "tollerance achieved: " << tol << endl;
cout << "iterations performed: " << maxit << endl;


x = x * 0;
tol = 1.0e-8;
maxit = 10000; //for grad method

result1 = GRAD(mat,x,b,D,maxit,tol);
cout << "GRAD flag = " << result1 << endl;
cout << "tollerance achieved: " << tol << endl;
cout << "iterations performed: " << maxit << endl;

/*
test # mpirun -n 4 ./test1 diffreact.mtx 2 sol.txt hist.txt -i cg -adds true -p ssor -adds_iter 3 -tol 1.0e-9 

number of processes = 4
matrix size = 256 x 256 (3976 nonzero entries)

initial vector x      : all components set to 0
precision             : double
linear solver         : CG
preconditioner        : SSOR + Additive Schwarz
convergence condition : ||b-Ax||_2 <= 1.0e-09 * ||b-Ax_0||_2
matrix storage format : CSR
linear solver status  : normal end

CG: number of iterations = 28
CG:   double             = 28
CG:   quad               = 0
CG: elapsed time         = 7.998150e-04 sec.
CG:   preconditioner     = 3.986630e-04 sec.
CG:     matrix creation  = 2.300000e-07 sec.
CG:   linear solver      = 4.011520e-04 sec.
CG: relative residual    = 7.381524e-10
*/

//Compare
//Clearly the ADD Schwartz precond. is super efficient, and the number of iterations drammatically decrease.

    return 0;
}