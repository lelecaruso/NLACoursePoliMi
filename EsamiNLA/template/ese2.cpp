/*
 mpirun -n 4 ./eigen1 gr_30_30.mtx eigvec.txt hist.txt -e pi -etol 1.0e-8 -emaxiter 1800

number of processes = 4
matrix size = 900 x 900 (4322 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Power
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-08 * ||lx||_2
matrix storage format : CSR
shift                 : 0.000000e+00
eigensolver status    : LIS_MAXITER(code=4)

Power: mode number          = 0
Power: eigenvalue           = 1.186734e+01
Power: number of iterations = 1800
Power: elapsed time         = 1.894669e-02 sec.
Power:   preconditioner     = 0.000000e+00 sec.
Power:     matrix creation  = 0.000000e+00 sec.
Power:   linear solver      = 0.000000e+00 sec.
Power: relative residual    = 1.506219e-08

*/


/*
//to find a set of smallest eigenvalues see lis_ug -e si -ss n

mpirun -n 4 ./eigen1  gr_30_30.mtx eigvec.txt hist.txt -e si -ss 8 -i cg
mpirun -n 4 ./eigen1 gr_30_30.mtx eigvec.txt hist.txt -e si -ss 8  -i bicgstab -p jacobi
mpirun -n 4 ./eigen1 gr_30_30.mtx eigvec.txt hist.txt -e si -ss 8 -i gmres -p ssor

*/