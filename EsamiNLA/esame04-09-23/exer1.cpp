#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/IterativeLinearSolvers>
#include "gmres.hpp"
#include "jacobi.hpp"

using std::endl;
using std::cout;

// Some useful alias
using namespace Eigen;
using SpMat=Eigen::SparseMatrix<double,RowMajor>;


int main(int argc, char** argv)
{   
    using namespace LinearAlgebra;
    SpMat A;
    loadMarket(A,"Aex1.mtx"); //size 500 x 500

    //Check if symm and report the norm 2 of A
    SpMat AT = A.transpose();
    SpMat B = (A-AT);
    double tol_symm = 1.0e-14;
    if(B.norm()<= tol_symm){
        cout<<"La matrice A è simmetrica"<<endl;
    }else{
        cout<<"La matrice A NON è simmetrica"<<endl;
    }

    cout<<"La norma di A è: "<<A.norm()<<endl;

    
    /*
    Fix a maximum
    number of iterations which is sufficient to reduce the (relative) residual below than 10−5
    and take x0 = b as initial guess. Report on the sheet the iteration counts and ‖xj ‖
    */
   //No preconditioner as it is not spec, so i will use Identity matrix


    VectorXd b = VectorXd:: Ones(A.cols());
    VectorXd xj(A.cols());

    xj = b;
    int max_iter = 500;   //sono sufficienti 15;
    double tol = 1.0e-5;
    int result;

    Eigen::DiagonalPreconditioner<double> D(A);// Create Diagonal precond
    // For jacobi it's not a real preconditioner but it used to compute the iteration x(k+1) = x(k) + D^-1rk
    result = Jacobi(A,xj,b,D,max_iter,tol);
    cout << "Jacobi flag = " << result << endl;
    cout << "iterations performed: " << max_iter << endl;
    cout << "tolerance achieved  : " << tol << endl;

    cout <<"La norma di xj è: "<<xj.norm()<<endl;

    /*
    Compute the approximate solution xg of Ax = b obtained using the GMRES method
    without restart. Fix a maximum number of iterations which is sufficient to reduce the
    residual below than 10−10 considering xj (computed in the previous point) as initial guess.
    Use the Eigen diagonal preconditioner. Report the iteration counts and ‖xj −xg‖ 
    */

   VectorXd xg(A.cols());
   xg = xj;
   tol = 1.0e-10;
   int no_restart = A.cols(); //not using restart because i have fixed it to the maximum size that the krylov space could reach
   int resultg;
   int maxit = A.cols();

   resultg = GMRES(A, xg, b, D, no_restart, maxit, tol);
    cout << "GMRES   flag = " << result << endl;
    cout << "iterations performed: " << maxit << endl;
    cout << "tolerance achieved  : " << tol << endl;
    
    /*
    mpirun -n 4 ./test1 Aex1.mtx  1 sol.txt hist.txt -i jacobi -tol 1.0e-5

number of processes = 4
matrix size = 500 x 500 (7708 nonzero entries)

initial vector x      : all components set to 0
precision             : double
linear solver         : Jacobi


Jacobi: number of iterations = 14

Jacobi: relative residual    = 9.603771e-06



root@0bc56b79e068 test # mpirun -n 4 ./test1 Aex1.mtx  1 sol.txt hist.txt -i gmres -tol 1.0e-10

number of processes = 4
matrix size = 500 x 500 (7708 nonzero entries)

initial vector x      : all components set to 0
precision             : double
linear solver         : GMRES
preconditioner        : none


GMRES: number of iterations = 17

GMRES: relative residual    = 2.509352e-11

mpirun -n 4 ./test1 Aex1.mtx  1 sol.txt hist.txt -i gmres -tol 1.0e-10 -p ssor  

number of processes = 4
matrix size = 500 x 500 (7708 nonzero entries)


linear solver         : GMRES
preconditioner        : SSOR


GMRES: number of iterations = 12

GMRES: relative residual    = 5.336642e-11
    
    
    */







    return 0;
}
