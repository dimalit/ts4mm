/*
 * solver.cpp
 *
 *  Created on: Feb 19, 2015
 *      Author: dimalit
 */

#include "solver.h"
#include <cassert>

// externally-visible:
double init_step;
double atolerance, rtolerance;
int N;
int n0, k;
double beta;
double alpha;
double gamma_omega;

const double PI = 4*atan(1.0);

PetscErrorCode RHSFunction(TS ts, PetscReal t,Vec in,Vec out,void*);
PetscErrorCode solve(Vec initial_state,
		   int max_steps, double max_time,
		   bool (*step_func)(Vec state, Vec rhs, int steps, double time));
PetscErrorCode step_monitor(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx);

PetscErrorCode solve(Vec initial_state,
		   int max_steps, double max_time,
		   bool (*step_func)(Vec state, Vec rhs, int steps, double time))
{
//	VecView(initial_state, PETSC_VIEWER_STDERR_WORLD);

	PetscErrorCode ierr;

	int lo, hi;
	VecGetOwnershipRange(initial_state, &lo, &hi);

	TS ts;
	ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
	ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);

	ierr = TSSetType(ts, TSRK);CHKERRQ(ierr);
	ierr = TSRKSetType(ts, TSRK4);CHKERRQ(ierr);
	// XXX: strange cast - should work without it too!
	ierr = TSSetRHSFunction(ts, NULL, (PetscErrorCode (*)(TS,PetscReal,Vec,Vec,void*))RHSFunction, 0);CHKERRQ(ierr);

	ierr = TSSetInitialTimeStep(ts, 0.0, init_step);CHKERRQ(ierr);
	ierr = TSSetTolerances(ts, atolerance, NULL, rtolerance, NULL);CHKERRQ(ierr);
//	fprintf(stderr, "steps=%d time=%lf ", max_steps, max_time);

	ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

	ierr = TSSetSolution(ts, initial_state);CHKERRQ(ierr);

	ierr = TSSetDuration(ts, max_steps, max_time);CHKERRQ(ierr);

	ierr = TSMonitorSet(ts, step_monitor, (void*)step_func, NULL);
	ierr = TSSolve(ts, initial_state);CHKERRQ(ierr);			// results are "returned" in step_monitor

	double tstep;
	TSGetTimeStep(ts, &tstep);

	TSDestroy(&ts);
	return 0;
}

PetscErrorCode step_monitor(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx){
	bool (*step_func)(Vec state, Vec rhs, int steps, double time) = (bool (*)(Vec state, Vec rhs, int steps, double time)) mctx;

	PetscErrorCode ierr;

//	VecView(u, PETSC_VIEWER_STDERR_WORLD);

	// get final RHS
	Vec rhs;
	TSRHSFunction func;
	ierr = TSGetRHSFunction(ts, &rhs, &func, NULL);CHKERRQ(ierr);
	func(ts, time, u, rhs, NULL);	// XXX: why I need to call func instead of getting rhs from TSGetRhsFunction??

//	VecView(u, PETSC_VIEWER_STDERR_WORLD);
//	VecView(rhs, PETSC_VIEWER_STDERR_WORLD);

	PetscInt true_steps;
	TSGetTimeStepNumber(ts, &true_steps);

	bool res = step_func(u, rhs, true_steps, time);
	if(!res){
		//TSSetConvergedReason(ts, TS_CONVERGED_USER);
		TSSetDuration(ts, steps, time);
	}

	return 0;
}

PetscErrorCode RHSFunction(TS ts, PetscReal t,Vec in,Vec out,void*){
//	fprintf(stderr, "%s\n", __FUNCTION__);
	PetscErrorCode ierr;

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int addition = 4*k+2;
	static double* fields = new double[addition];

	// bcast E and phi
	const double* data;
	ierr = VecGetArrayRead(in, &data);CHKERRQ(ierr);
	if(rank == 0){
		memcpy(fields, data, addition*sizeof(double));
	}
	ierr = VecRestoreArrayRead(in, &data);CHKERRQ(ierr);
	MPI_Bcast(fields, addition, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

	// compute sums
	int lo, hi;
	VecGetOwnershipRange(in, &lo, &hi);
	if(lo < addition)	// exclude Es and phis
		lo = addition;

	static double* sum_sin = new double[2*k+1];
		memset(sum_sin, 0, sizeof(double)*(2*k+1));
	static double* sum_cos = new double[2*k+1];
		memset(sum_cos, 0, sizeof(double)*(2*k+1));
	for(int i=lo; i<hi; i+=3){
		double apz[3];
		int indices[] = {i, i+1, i+2};
		VecGetValues(in, 3, indices, apz);
		for(int j=0; j<2*k+1; j++){
			double n = n0 - k + j;
			double phi_n = fields[j*2+1];
			sum_sin[j] += apz[0]*sin(apz[1] - apz[2]*n/n0 - phi_n);
			sum_cos[j] += apz[0]*cos(apz[1] - apz[2]*n/n0 - phi_n);
		}
	}

	static double* sum_sin_out = NULL;
	static double* sum_cos_out = NULL;
	if(rank==0){
		sum_sin_out = new double[2*k+1];
			memset(sum_sin_out, 0, sizeof(double)*(2*k+1));
		sum_cos_out = new double[2*k+1];
			memset(sum_cos_out, 0, sizeof(double)*(2*k+1));
	}

	MPI_Reduce(sum_sin, sum_sin_out, 2*k+1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
	MPI_Reduce(sum_cos, sum_cos_out, 2*k+1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);

	// compute derivatives of E_e and phi_e
	if(rank == 0){
		for(int j=0; j<2*k+1; j++){
			double E = fields[2*j+0];

			double dE = 1.0/2.0/N*sum_sin_out[j];
			double dphi = - 1.0/2.0/N*sum_cos[j] / E;
			VecSetValue(out, 2*j+0, dE,   INSERT_VALUES);
			VecSetValue(out, 2*j+1, dphi, INSERT_VALUES);
		}
	}
	VecAssemblyBegin(out);
	VecAssemblyEnd(out);

	// compute n, a, k
	VecGetOwnershipRange(out, &lo, &hi);
	if(lo < addition)
		lo = addition;

	for(int i=lo; i<hi; i+=3){
		double apz[3];
		int indices[] = {i, i+1, i+2};
		VecGetValues(in, 3, indices, apz);

		double da = 0;
		double dp = 0;

		for(int j=0; j<2*k+1; j++){
			double n = n0 - k + j;
			double E = fields[2*j+0];
			double phi = fields[2*j+1];

			da += -0.5 * E * sin(apz[1] - apz[2]*n/n0 - phi);
			dp += -0.5 * E * cos(apz[1] - apz[2]*n/n0 - phi) / apz[0] - alpha*apz[0]*apz[0];
		}// for mode

		//double da = - 0.5*E*sin(apzd[1]-apz[2] - phi);
		//double dp = - 0.5*E/apz[0]*cos(apz[1] - apz[2] - phi) - alpha*apz[0]*apz[0];

		VecSetValue(out, i, da, INSERT_VALUES);
		VecSetValue(out, i+1, dp, INSERT_VALUES);
		VecSetValue(out, i+2, 0, INSERT_VALUES);
	}

	VecAssemblyBegin(out);
	VecAssemblyEnd(out);

//	VecView(in, PETSC_VIEWER_STDERR_WORLD);
//	VecView(out, PETSC_VIEWER_STDERR_WORLD);

	return 0;
}
