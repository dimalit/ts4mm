/*
 * main.cpp
 *
 *  Created on: Aug 20, 2014
 *      Author: dimalit
 */

#include <model_e4mm.pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
using namespace pb;

#include "solver.h"

#include <petscts.h>
#include <mpi.h>
#include <cstring>
#include <unistd.h>

E4mmConfig pconfig;
EXPetscSolverConfig sconfig;
E4mmState state;

int rank;
int size;
clock_t t1;		// for time counting in step_func
int max_steps; double max_time;
bool use_step = false;

void vec_to_state(Vec v, E4mmState*);
void state_to_vec(const E4mmState* state, Vec v);
bool step_func(Vec u, Vec rhs, int steps, double time);

// TMP
//#include <fcntl.h>

void broadcast_message(google::protobuf::Message& msg){
	char* buf; int buf_size;

	if(rank == 0)
		buf_size = msg.ByteSize();
	MPI_Bcast(&buf_size, 1, MPI_INT, 0, PETSC_COMM_WORLD);

	buf = new char[buf_size];

	if(rank == 0)
		msg.SerializeToArray(buf, buf_size);

	MPI_Bcast(buf, buf_size, MPI_BYTE, 0, PETSC_COMM_WORLD);

	if(rank != 0)
		msg.ParseFromArray(buf, buf_size);

	delete[] buf;
}

void parse_with_prefix(google::protobuf::Message& msg, int fd){
	int size;
	int ok = read(fd, &size, sizeof(size));
	assert(ok == sizeof(size));

	//TODO:without buffer cannot read later bytes
	char *buf = (char*)malloc(size);
	int read_size = 0;
	while(read_size != size){
		ok = read(fd, buf+read_size, size-read_size);
		read_size+=ok;
		assert(ok > 0 || read_size==size);
	}
	msg.ParseFromArray(buf, size);
	free(buf);
}

int main(int argc, char** argv){

//	int go = 0;
//	while(go==0){
//		sleep(1);
//	}

//	close(0);
//	open("../ode-env/all.tmp", O_RDONLY);

	if(argc > 1 && strcmp(argv[1], "use_step")==0){
		use_step = true;
		argc--;
		argv++;
	}

	PetscErrorCode ierr;
	ierr = PetscInitialize(&argc, &argv, (char*)0, (char*)0);CHKERRQ(ierr);

	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
	ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);

	if(rank==0){
		E4mmModel all;
		//all.ParseFromFileDescriptor(0);
		parse_with_prefix(all, 0);

		sconfig.CopyFrom(all.sconfig());
		pconfig.CopyFrom(all.pconfig());
		state.CopyFrom(all.state());
	}

	broadcast_message(sconfig);
	broadcast_message(pconfig);

	// set global parameters
	init_step = sconfig.init_step();
	atolerance = sconfig.atol();
	rtolerance = sconfig.rtol();
	N = pconfig.n();
	n0 = pconfig.n0();
	k = pconfig.k();
	beta = pconfig.beta();
	alpha = pconfig.alpha();
	gamma_omega = pconfig.gamma_omega();

	Vec u;
	VecCreate(PETSC_COMM_WORLD, &u);
	VecSetType(u, VECMPI);

	int addition = rank==0 ? 4*k+2 : 0;
	VecSetSizes(u, addition+pconfig.n()*3/size, PETSC_DECIDE);

	state_to_vec(&state, u);

	if(rank == 0){
		int ok;
		ok = read(0, &max_steps, sizeof(max_steps));
			assert(ok==sizeof(max_steps));
		ok = read(0, &max_time, sizeof(max_time));
			assert(ok==sizeof(max_time));
	}

	MPI_Bcast(&max_steps, 1, MPI_INT, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&max_time, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

	t1 = clock();

	solve(u, max_steps, max_time, step_func);

	VecDestroy(&u);

	ierr = PetscFinalize();CHKERRQ(ierr);
	return 0;
}

bool step_func(Vec res, Vec res_rhs, int passed_steps, double passed_time){
	clock_t t2 = clock();
	double dtime = (double)(t2-t1)/CLOCKS_PER_SEC;

//	VecView(res_rhs, PETSC_VIEWER_STDERR_(PETSC_COMM_WORLD));

	// return if not using steps
	if(!use_step && passed_steps < max_steps && passed_time < max_time)
		return true;
	// wait if using steps
	else if(use_step){
		int ok;
		char c;
		if(rank==0){
			ok = read(0, &c, sizeof(c));
				assert(ok==sizeof(c));
		}
		MPI_Bcast(&c, 1, MPI_BYTE, 0, PETSC_COMM_WORLD);
		if(c=='f')
			return false;			// finish!
		assert(c=='s');
	}// if use_steps


	E4mmSolution sol;
	if(rank==0){
		for(int i=0; i<N; i++){
			sol.mutable_state()->add_particles();
			sol.mutable_d_state()->add_particles();
		}
	}

	vec_to_state(res, sol.mutable_state());
	vec_to_state(res_rhs, sol.mutable_d_state());

	if(rank == 0){
		// 1 write time and steps
		write(1, &passed_steps, sizeof(passed_steps));
		write(1, &passed_time, sizeof(passed_time));

		// 2 write state
		int size = sol.ByteSize();
		write(1, &size, sizeof(size));
		sol.SerializeToFileDescriptor(1);
	}

	t1 = t2;
	return true;
}

void state_to_vec(const E4mmState* state, Vec v){
	int vecsize;
	VecGetSize(v, &vecsize);
	assert(vecsize == pconfig.n()*3+(pconfig.k()*2+1)*2);

	double *arr;
	VecGetArray(v, &arr);

	PetscInt* borders;
	VecGetOwnershipRanges(v, (const PetscInt**)&borders);

	if(rank == 0){

		int addition = 4*k+2;

		// write Es and phis
		for(int i=0; i<2*k+1; i++){
			const E4mmState::Fields& f = state->fields(i);
			arr[2*i] = f.e();
			arr[2*i+1] = f.phi();
		}

		for(int r = size-1; r>=0; r--){		// go down - because last will be mine
			int lo = borders[r];
			int hi = borders[r+1];
			if(r==0)
				lo += addition;

			assert((lo-addition)%3 == 0);
			assert((hi-addition)%3 == 0);

			int first = (lo-addition) / 3;
			int count = (hi - lo) / 3;

			for(int i=0; i<count; i++){
				E4mmState::Particles p = state->particles(first+i);
				arr[addition + i*3+0] = p.a();
				arr[addition + i*3+1] = p.psi();
				arr[addition + i*3+2] = p.z();
			}

			if(r!=0)
				MPI_Send(arr+addition, count*3, MPI_DOUBLE, r, 0, PETSC_COMM_WORLD);

		}// for

	}// if rank == 0
	else{
		int count3 = borders[rank+1] - borders[rank];
		MPI_Status s;
		int ierr = MPI_Recv(arr, count3, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD, &s);
		assert(MPI_SUCCESS == ierr);
	}// if rank != 0

	VecRestoreArray(v, &arr);
}

void vec_to_state(Vec v, E4mmState* state){
	const double *arr;
	VecGetArrayRead(v, &arr);

	PetscInt* borders;
	VecGetOwnershipRanges(v, (const PetscInt**)&borders);

	if(rank == 0){

		PetscScalar* buf = (PetscScalar*)malloc(sizeof(PetscScalar)*(borders[1]-borders[0]));

		int addition = 4*k+2;

		// read Es and phis
		for(int i=0; i<2*k+1; i++){
			E4mmState::Fields* f = state->mutable_fields(i);
			f->set_e(arr[2*i]);
			f->set_phi(arr[2*i+1]);
		}

		for(int r = 0; r<size; r++){
			int lo = borders[r];
			int hi = borders[r+1];

			assert((lo-addition)%3 == 0 || lo==0);
			assert((hi-addition)%3 == 0);

			int first = (lo-addition) / 3;
			int count = (hi - lo) / 3;

			MPI_Status s;
			if(r!=0){
				int ok = MPI_Recv(buf, count*3, MPI_DOUBLE, r, 0, PETSC_COMM_WORLD, &s);
				assert(MPI_SUCCESS == ok);
			}
			else	// copy only particles (arr+2)
				memcpy(buf, arr+addition, sizeof(PetscScalar)*(hi-lo-addition));

			for(int i=0; i<count; i++){
				E4mmState::Particles* p = state->mutable_particles(first+i);
				p->set_a(buf[i*3+0]);

				double psi = buf[i*3+1];
				if(psi > 2*M_PI)
					psi -= 2*M_PI;
				else if(psi < 0)
					psi += 2*M_PI;

				p->set_psi(psi);
				p->set_z(buf[i*3+2]);
			}
		}// for

		free(buf);
	}// if rank == 0
	else{
		int count3 = borders[rank+1] - borders[rank];
		int ierr = MPI_Send((double*)arr, count3, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD);
		assert(MPI_SUCCESS == ierr);
	}// if rank != 0

	VecRestoreArrayRead(v, &arr);
}

