#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "cg.hpp"

using std::endl;
using std::cout;

// Some useful alias
using namespace Eigen;
using SpMat=Eigen::SparseMatrix<double,RowMajor>;


int main(int argc, char** argv)
{   
    using namespace LinearAlgebra;

    int n = 800;
    SpMat A(n,n);
    for(int i = 0; i<n ; i++){
        A.coeffRef(i,i) = 4.0;
        if(i>0){A.coeffRef(i,i-1) = -1.0;}
        if(i>1){A.coeffRef(i,i-2) = -1.0;}
        if(i<n-1){A.coeffRef(i,i+1) = -1.0;}
        if(i<n-2){A.coeffRef(i,i+2) = -1.0;}
    }
    VectorXd v = VectorXd:: Ones(n);
    
    //OBS: la funzione x1.dot(x2) restituisce il prodotto x1T * x2
    cout<<"La norma A del vettore v è: "<< v.dot(A*v) <<endl;

    VectorXd b = VectorXd:: Ones(n);
    for(int i = 0; i<n; i++){
        if(i%2==1){b(i) = 0.0;}
    }
    cout<<"La norma del vettore b è: "<<b.norm()<<endl;

    /*Solve the linear system Ax = b using the Conjugate Gradient method implemented in the
    cg.hpp template. Fix a maximum number of iterations which is sufficient to reduce the
    (relative) residual below than 10−12. Use the diagonal preconditioner provided by Eigen.
    Report on the sheet the iteration counts and the relative residual at the last iteration
    */

    VectorXd x(n);
    double tol = 1.0e-12;
    Eigen:: DiagonalPreconditioner<double>D(A);
    int result, maxit;
    maxit = n;
    result = CG(A,x,b,D,maxit,tol);
    cout<<"Solving with CG, flag: "<<result<<endl;
    cout<<"Iterations: "<<maxit<<endl;
    cout<<"relative residual: "<<tol<<endl;


/*
 export matrix A and vector b in
the matrix market format (save as matA.mtx and vecb.mtx) and move them to the folder
lis-2.0.34/test. Solve the same linear system using the Conjugate Gradient method of
the LIS library setting a tolerance of 10−12. Report on the sheet the iteration counts and
the relative residual at the last iteration
*/
    saveMarket(A,"matA.mtx");
    FILE* out = fopen("vecb.mtx","w");
    fprintf(out,"%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out,"%d\n", n);
    for (int i=0; i<n; i++) {
        fprintf(out,"%d %f\n", i ,b(i));
    }
    fclose(out);

    /*
    CG: number of iterations = 458
    CG: relative residual    = 9.096176e-13
    */

return 0;
}