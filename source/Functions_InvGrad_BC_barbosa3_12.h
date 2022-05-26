
#include <stdio.h>

using namespace std;

double cal_Magnetic_mag_Farq_Barbosa_noneffsto(int N_data, int N_mod, double miu,double * deltaT,double * G,
	double * Wd,double * deltaTHax,double * GHax,double * WdHax,
	double * deltaTHay,double * GHay,double * WdHay,double * deltaTZa,double * GZa,double * WdZa,
	double * deltaTTa,double * GTa,double * WdTa,
	double * deltaT_2, double * G_2, double * Wd_2, double * geomag_ref_x, double * geomag_ref_y, double * geomag_ref_z,
	int * paraInvOrNot,
	double * Ws,double * Wx,MKL_INT * columns_x,MKL_INT * rowIndex_x,double * Wy,MKL_INT * columns_y,MKL_INT * rowIndex_y,
	double * Wz,MKL_INT * columns_z,MKL_INT * rowIndex_z,double * W_deepw,double * alpha,
	double * m_ref_n,double * m_min_n,double * m_max_n,double * m_0_n,double * Rd,
	double * RdHax,double * RdHay,double * RdZa,double * RdTa,
	double * Rd_2,
	double * Rs,double * Rx,double * Ry,double * Rz,double * m_result,
	double * m_alpha_s,double * m_alpha_x,double * m_alpha_y,double * m_alpha_z,
	double * tran_r11,double * tran_r12,double * tran_r13,double * tran_r21,
	double * tran_r22,double * tran_r23,double * tran_r31,double * tran_r32,double * tran_r33)
{

	double * m_tran_0;
	m_tran_0 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (m_tran_0 == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix m_tran_0. Aborting... \n\n");
		mkl_free(m_tran_0);
		return 1;
	}
	cal_tran_m(N_mod,m_0_n,m_min_n,m_max_n,"forw",m_tran_0);

	double * T;
	T = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (T == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix T. Aborting... \n\n");
		mkl_free(T);
		return 1;
	}
	cal_diag_mtx2(N_mod, m_0_n, m_min_n, m_max_n, T,W_deepw);


	int i;
	double * temp_m;
	temp_m = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_m == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_m. Aborting... \n\n");
		mkl_free(temp_m);
		system ("PAUSE");
	}
	#pragma omp parallel for shared(N_mod,m_0_n,m_ref_n,temp_m,W_deepw) private(i)
	for(i=0;i<N_mod;i++) {
		temp_m[i] = (m_0_n[i]-m_ref_n[i]);
	}

    struct matrix_descr descrA;
	descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
	sparse_matrix_t csr_Wx;
	mkl_sparse_d_create_csr(&csr_Wx, SPARSE_INDEX_BASE_ZERO,
		N_mod, // number of rows
		N_mod, // number of cols
		rowIndex_x, rowIndex_x + 1, columns_x, Wx);
	mkl_sparse_optimize(csr_Wx);
	
	sparse_matrix_t csr_Wy;
	mkl_sparse_d_create_csr(&csr_Wy, SPARSE_INDEX_BASE_ZERO,
		N_mod, // number of rows
		N_mod, // number of cols
		rowIndex_y, rowIndex_y + 1, columns_y, Wy);
	mkl_sparse_optimize(csr_Wy);

	sparse_matrix_t csr_Wz;
	mkl_sparse_d_create_csr(&csr_Wz, SPARSE_INDEX_BASE_ZERO,
		N_mod, // number of rows
		N_mod, // number of cols
		rowIndex_z, rowIndex_z + 1, columns_z, Wz);
	mkl_sparse_optimize(csr_Wz);

	double * temp_x,* temp_y,* temp_z,* temp_s;
	double * temp_xy, *temp_xz, *temp_yx, *temp_zx, *temp_yz, *temp_zy;

	temp_s = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	temp_x = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	temp_y = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	temp_z = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );

	temp_xy = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	temp_yx = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	temp_xz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	temp_zx = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	temp_yz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	temp_zy = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );

	if (temp_s == NULL || temp_x==NULL || temp_y==NULL || temp_z==NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_s temp_x temp_y temp_z. Aborting... \n\n");
		mkl_free(temp_s);
		mkl_free(temp_x);
		mkl_free(temp_y);
		mkl_free(temp_z);
		system ("PAUSE");
		return 1;
	}

	if (temp_xy == NULL || temp_yx==NULL || temp_xz==NULL || temp_zx==NULL || temp_zy == NULL || temp_yz == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_xy temp_yx temp_xz temp_zx. Aborting... \n\n");
		mkl_free(temp_xy);
		mkl_free(temp_yx);
		mkl_free(temp_xz);
		mkl_free(temp_zx);
		mkl_free(temp_yz);
		mkl_free(temp_zy);
		system ("PAUSE");
		return 1;
	}

	#pragma omp parallel for shared(N_mod,temp_s,miu,W_deepw,alpha,Ws,Rs,temp_m) private(i)
	for(i=0;i<N_mod;i++){
		temp_s[i] = miu*alpha[0]*Ws[i]*Rs[i]*Ws[i]*temp_m[i];
		// printf("temp_s\n");
        temp_x[i] = 0;
        temp_y[i] = 0;
        temp_z[i] = 0;

        temp_xy[i] = 0;
        temp_yx[i] = 0;
        temp_xz[i] = 0;
        temp_zx[i] = 0;
        temp_yz[i] = 0;
        temp_zy[i] = 0;
    }

	char transa = 'n';
	double * temp_x1;
	temp_x1 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_x1 == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_x1. Aborting... \n\n");
		mkl_free(temp_x1);
		system ("PAUSE");
		return 1;
	}
    #pragma omp parallel for shared(N_mod,temp_x1) private(i)
    for(i=0;i<N_mod;i++) {
        temp_x1[i] = 0;
    }
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_x1);
    // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_x1);
	#pragma omp parallel for shared(N_mod,temp_x1,Rx) private(i)
	for(i=0;i<N_mod;i++)
		temp_x1[i] = Rx[i]*temp_x1[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_x1, 0.0, temp_x);
    // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_x1,temp_x);

	#pragma omp parallel for shared(N_mod,temp_x,miu,W_deepw,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_x[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r11[i] 
			+ m_alpha_y[i]*tran_r21[i]*tran_r21[i] 
			+ m_alpha_z[i]*tran_r31[i]*tran_r31[i])*temp_x[i];
	mkl_free(temp_x1);


	transa = 'n';
	double * temp_y1;
	temp_y1 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_y1 == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_y1. Aborting... \n\n");
		mkl_free(temp_y1);
		system ("PAUSE");
		return 1;
	}
	#pragma omp parallel for shared(N_mod,temp_y1) private(i)
    for(i=0;i<N_mod;i++) {
        temp_y1[i] = 0;
    }
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_y1);
    // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_y1);
	#pragma omp parallel for shared(N_mod,temp_y1,Ry) private(i)
	for(i=0;i<N_mod;i++)
		temp_y1[i] = Ry[i]*temp_y1[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wy, descrA, temp_y1, 0.0, temp_y);
    // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_y1,temp_y);

	#pragma omp parallel for shared(N_mod,temp_y,miu,W_deepw,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_y[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r12[i] 
			+ m_alpha_y[i]*tran_r22[i]*tran_r22[i] 
			+ m_alpha_z[i]*tran_r32[i]*tran_r32[i])*temp_y[i];
	mkl_free(temp_y1);

	transa = 'n';
	double * temp_z1;
	temp_z1 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_z1 == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_z1. Aborting... \n\n");
		mkl_free(temp_z1);
		system ("PAUSE");
		return 1;
	}
    #pragma omp parallel for shared(N_mod,temp_z1) private(i)
    for(i=0;i<N_mod;i++) {
        temp_z1[i] = 0;
    }
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_z1);
    // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_z1);
	#pragma omp parallel for shared(N_mod,temp_z1,Rz) private(i)
	for(i=0;i<N_mod;i++)
		temp_z1[i] = Rz[i]*temp_z1[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wz, descrA, temp_z1, 0.0, temp_z);
    // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_z1,temp_z);
	#pragma omp parallel for shared(N_mod,temp_z,miu,W_deepw,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_z[i] = miu*(m_alpha_x[i]*tran_r13[i]*tran_r13[i] 
			+ m_alpha_y[i]*tran_r23[i]*tran_r23[i] 
			+ m_alpha_z[i]*tran_r33[i]*tran_r33[i])*temp_z[i];
	mkl_free(temp_z1);


	transa = 'n';
	double * temp_x1_xy;
	temp_x1_xy = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_x1_xy == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_x1_xy. Aborting... \n\n");
		mkl_free(temp_x1_xy);
		system ("PAUSE");
		return 1;
	}
    #pragma omp parallel for shared(N_mod,temp_x1_xy) private(i)
    for(i=0;i<N_mod;i++) {
        temp_x1_xy[i] = 0;
    }
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_x1_xy);
    // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_x1_xy);
	#pragma omp parallel for shared(N_mod,temp_x1_xy,Rx) private(i)
	for(i=0;i<N_mod;i++)
		temp_x1_xy[i] = sqrt(Rx[i])*sqrt(Ry[i])*temp_x1_xy[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_x1_xy, 0.0, temp_xy);
    // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_x1_xy,temp_xy);

	#pragma omp parallel for shared(N_mod,temp_xy,miu,W_deepw,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_xy[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r12[i] 
			+ m_alpha_y[i]*tran_r21[i]*tran_r22[i] 
			+ m_alpha_z[i]*tran_r31[i]*tran_r32[i])*temp_xy[i];
	mkl_free(temp_x1_xy);


	transa = 'n';
	double * temp_x1_yx;
	temp_x1_yx = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_x1_yx == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_x1_yx. Aborting... \n\n");
		mkl_free(temp_x1_yx);
		system ("PAUSE");
		return 1;
	}
    #pragma omp parallel for shared(N_mod,temp_x1_yx) private(i)
    for(i=0;i<N_mod;i++) {
        temp_x1_yx[i] = 0;
    }
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_x1_yx);
    // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_x1_yx);
	#pragma omp parallel for shared(N_mod,temp_x1_yx,Rx) private(i)
	for(i=0;i<N_mod;i++)
		temp_x1_yx[i] = sqrt(Rx[i])*sqrt(Ry[i])*temp_x1_yx[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wy, descrA, temp_x1_yx, 0.0, temp_yx);
    // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_x1_yx,temp_yx);

	#pragma omp parallel for shared(N_mod,temp_yx,miu,W_deepw,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_yx[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r12[i] 
			+ m_alpha_y[i]*tran_r21[i]*tran_r22[i] 
			+ m_alpha_z[i]*tran_r31[i]*tran_r32[i])*temp_yx[i];
	mkl_free(temp_x1_yx);


	transa = 'n';
	double * temp_x1_xz;
	temp_x1_xz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_x1_xz == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_x1_xz. Aborting... \n\n");
		mkl_free(temp_x1_xz);
		system ("PAUSE");
		return 1;
	}
    #pragma omp parallel for shared(N_mod,temp_x1_xz) private(i)
    for(i=0;i<N_mod;i++) {
        temp_x1_xz[i] = 0;
    }
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_x1_xz);
    // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_x1_xz);
	#pragma omp parallel for shared(N_mod,temp_x1_xz,Rx) private(i)
	for(i=0;i<N_mod;i++)
		temp_x1_xz[i] = sqrt(Rx[i])*sqrt(Rz[i])*temp_x1_xz[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_x1_xz, 0.0, temp_xz);
    // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_x1_xz,temp_xz);

	#pragma omp parallel for shared(N_mod,temp_xz,miu,W_deepw,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_xz[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r13[i] 
			+ m_alpha_y[i]*tran_r21[i]*tran_r23[i] 
			+ m_alpha_z[i]*tran_r31[i]*tran_r33[i])*temp_xz[i];
	mkl_free(temp_x1_xz);


	transa = 'n';
	double * temp_x1_zx;
	temp_x1_zx = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_x1_zx == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_x1_zx. Aborting... \n\n");
		mkl_free(temp_x1_zx);
		system ("PAUSE");
		return 1;
	}
    #pragma omp parallel for shared(N_mod,temp_x1_zx) private(i)
    for(i=0;i<N_mod;i++) {
        temp_x1_zx[i] = 0;
    }
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_x1_zx);
    // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_x1_zx);
	#pragma omp parallel for shared(N_mod,temp_x1_zx,Rx) private(i)
	for(i=0;i<N_mod;i++)
		temp_x1_zx[i] = sqrt(Rx[i])*sqrt(Rz[i])*temp_x1_zx[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wz, descrA, temp_x1_zx, 0.0, temp_zx);
    // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_x1_zx,temp_zx);

	#pragma omp parallel for shared(N_mod,temp_zx,miu,W_deepw,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_zx[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r13[i] 
			+ m_alpha_y[i]*tran_r21[i]*tran_r23[i] 
			+ m_alpha_z[i]*tran_r31[i]*tran_r33[i])*temp_zx[i];
	mkl_free(temp_x1_zx);


	transa = 'n';
	double * temp_x1_yz;
	temp_x1_yz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_x1_yz == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_x1_yz. Aborting... \n\n");
		mkl_free(temp_x1_yz);
		system ("PAUSE");
		return 1;
	}
    #pragma omp parallel for shared(N_mod,temp_x1_yz) private(i)
    for(i=0;i<N_mod;i++) {
        temp_x1_yz[i] = 0;
    }
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_x1_yz);
    // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_x1_yz);
	#pragma omp parallel for shared(N_mod,temp_x1_yz,Rx) private(i)
	for(i=0;i<N_mod;i++)
		temp_x1_yz[i] = sqrt(Ry[i])*sqrt(Rz[i])*temp_x1_yz[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wy, descrA, temp_x1_yz, 0.0, temp_yz);
    // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_x1_yz,temp_yz);

	#pragma omp parallel for shared(N_mod,temp_yz,miu,W_deepw,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_yz[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r13[i] 
			+ m_alpha_y[i]*tran_r22[i]*tran_r23[i] 
			+ m_alpha_z[i]*tran_r32[i]*tran_r33[i])*temp_yz[i];
	mkl_free(temp_x1_yz);


	transa = 'n';
	double * temp_x1_zy;
	temp_x1_zy = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_x1_zy == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_x1_zy. Aborting... \n\n");
		mkl_free(temp_x1_zy);
		system ("PAUSE");
		return 1;
	}
    #pragma omp parallel for shared(N_mod,temp_x1_zy) private(i)
    for(i=0;i<N_mod;i++) {
        temp_x1_zy[i] = 0;
    }
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_x1_zy);
    // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_x1_zy);
	#pragma omp parallel for shared(N_mod,temp_x1_zy,Rx) private(i)
	for(i=0;i<N_mod;i++)
		temp_x1_zy[i] = sqrt(Ry[i])*sqrt(Rz[i])*temp_x1_zy[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wz, descrA, temp_x1_zy, 0.0, temp_zy);
    // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_x1_zy,temp_zy);

	#pragma omp parallel for shared(N_mod,temp_zy,miu,W_deepw,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_zy[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r13[i] 
			+ m_alpha_y[i]*tran_r22[i]*tran_r23[i] 
			+ m_alpha_z[i]*tran_r32[i]*tran_r33[i])*temp_zy[i];
	mkl_free(temp_x1_zy);



	double * temp_d;
	double * temp_d1;
	if(paraInvOrNot[0] != 0){
		temp_d1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		if (temp_d1 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_d1. Aborting... \n\n");
			mkl_free(temp_d1);
			system ("PAUSE");
			return 1;
		}
		cblas_dcopy(N_data,deltaT,1,temp_d1,1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,G,N_mod,m_0_n,1,-1,temp_d1,1);

		#pragma omp parallel for shared(N_data,temp_d1,Wd) private(i)
		for(i=0;i<N_data;i++)
			temp_d1[i] = Wd[i]*Rd[i]*Wd[i]*temp_d1[i];
		
		temp_d = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_mod,temp_d) private(i)
		for(i=0;i<N_mod;i++)
			temp_d[i] = 0;
		cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,G,N_mod,temp_d1,1,0,temp_d,1);
		mkl_free(temp_d1);
	}
	else{
		temp_d = (double *)mkl_malloc(N_mod*sizeof( double ), 64);
		#pragma omp parallel for shared(N_mod,temp_d) private(i)
		for(i=0;i<N_mod;i++)
			temp_d[i] = 0;
	}
	
	double * temp_dHax;
	double * temp_dHax1;
	if(paraInvOrNot[1] != 0){
		temp_dHax1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		if (temp_dHax1 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_d1. Aborting... \n\n");
			mkl_free(temp_dHax1);
			system ("PAUSE");
			return 1;
		}
		cblas_dcopy(N_data,deltaTHax,1,temp_dHax1,1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,GHax,N_mod,m_0_n,1,-1,temp_dHax1,1);
		#pragma omp parallel for shared(N_data,temp_dHax1,WdHax) private(i)
		for(i=0;i<N_data;i++)
			temp_dHax1[i] = WdHax[i]*RdHax[i]*WdHax[i]*temp_dHax1[i];
		
		temp_dHax = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_mod,temp_dHax) private(i)
		for(i=0;i<N_mod;i++)
			temp_dHax[i] = 0;
		cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,GHax,N_mod,temp_dHax1,1,0,temp_dHax,1);
		mkl_free(temp_dHax1);
	}
	else{
		temp_dHax = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_mod,temp_dHax) private(i)
		for(i=0;i<N_mod;i++)
			temp_dHax[i] = 0;
	}
	
	double * temp_dHay;
	double * temp_dHay1;
	if(paraInvOrNot[2] != 0){
		temp_dHay1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		if (temp_dHay1 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_d1. Aborting... \n\n");
			mkl_free(temp_dHay1);
			system ("PAUSE");
			return 1;
		}
		cblas_dcopy(N_data,deltaTHay,1,temp_dHay1,1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,GHay,N_mod,m_0_n,1,-1,temp_dHay1,1);
		#pragma omp parallel for shared(N_data,temp_dHay1,WdHay) private(i)
		for(i=0;i<N_data;i++)
			temp_dHay1[i] = WdHay[i]*RdHay[i]*WdHay[i]*temp_dHay1[i];
		
		temp_dHay = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_mod,temp_dHay) private(i)
		for(i=0;i<N_mod;i++)
			temp_dHay[i] = 0;
		cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,GHay,N_mod,temp_dHay1,1,0,temp_dHay,1);
		mkl_free(temp_dHay1);
	}
	else{
		temp_dHay = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_mod,temp_dHay) private(i)
		for(i=0;i<N_mod;i++)
			temp_dHay[i] = 0;
	}
	

	double * temp_dZa;
	double * temp_dZa1;
	if(paraInvOrNot[3] != 0){
		temp_dZa1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		if (temp_dZa1 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_d1. Aborting... \n\n");
			mkl_free(temp_dZa1);
			system ("PAUSE");
			return 1;
		}
		cblas_dcopy(N_data,deltaTZa,1,temp_dZa1,1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,GZa,N_mod,m_0_n,1,-1,temp_dZa1,1);
		#pragma omp parallel for shared(N_data,temp_dZa1,WdZa) private(i)
		for(i=0;i<N_data;i++)
			temp_dZa1[i] = WdZa[i]*RdZa[i]*WdZa[i]*temp_dZa1[i];
		
		temp_dZa = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_mod,temp_dZa) private(i)
		for(i=0;i<N_mod;i++)
			temp_dZa[i] = 0;
		cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,GZa,N_mod,temp_dZa1,1,0,temp_dZa,1);
		mkl_free(temp_dZa1);
	}
	else{
		temp_dZa = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_mod,temp_dZa) private(i)
		for(i=0;i<N_mod;i++)
			temp_dZa[i] = 0;
	}
	
	double * temp_dTa;
	if(paraInvOrNot[4] != 0){
		double * temp_dxyz1,* temp_dx10,* temp_dy1,* temp_dz1;
		temp_dxyz1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_dx10 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_dy1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_dz1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );

		if (temp_dxyz1 == NULL || temp_dx10 == NULL || temp_dy1 == NULL || temp_dz1 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_dxyz1. Aborting... \n\n");
			mkl_free(temp_dxyz1);
			mkl_free(temp_dx10);
			mkl_free(temp_dy1);
			mkl_free(temp_dz1);
			system ("PAUSE");
			return 1;
		}
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,GHax,N_mod,m_0_n, 1, 0.0, temp_dx10, 1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,GHay,N_mod,m_0_n, 1, 0.0, temp_dy1, 1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,GZa,N_mod,m_0_n, 1, 0.0, temp_dz1, 1);

		#pragma omp parallel for shared(N_data,temp_dxyz1,temp_dx10,temp_dy1,temp_dz1) private(i)
		for(i = 0; i < N_data; i++){
			temp_dxyz1[i] = sqrt(Txyz_weight[0]*temp_dx10[i]*temp_dx10[i] + Txyz_weight[1]*temp_dy1[i]*temp_dy1[i] + Txyz_weight[2]*temp_dz1[i]*temp_dz1[i]);
		}

		// cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,Gxyz,N_mod,m_0_n,1,-1,temp_dxyz1,1);


		#pragma omp parallel for shared(N_data,temp_dxyz1,RdTa,WdTa,deltaTTa) private(i)
		for(i=0;i<N_data;i++)
			temp_dxyz1[i] = WdTa[i]*RdTa[i]*WdTa[i]*(temp_dxyz1[i]-deltaTTa[i]);
		
		temp_dTa = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_mod,temp_dTa) private(i)
		for(i=0;i<N_mod;i++)
			temp_dTa[i] = 0;
		cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,GTa,N_mod,temp_dxyz1,1,0,temp_dTa,1);
		mkl_free(temp_dxyz1);
		mkl_free(temp_dy1);
		mkl_free(temp_dx10);
		mkl_free(temp_dz1);
	}
	else{
		temp_dTa = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_mod,temp_dTa) private(i)
		for(i=0;i<N_mod;i++)
			temp_dTa[i] = 0;
	}
	
	double * temp_deltaT_2;
	if(paraInvOrNot[5] != 0){
		double * temp_dxyz1,* temp_dx10,* temp_dy1,* temp_dz1;
		temp_dxyz1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_dx10 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_dy1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_dz1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );

		if (temp_dxyz1 == NULL || temp_dx10 == NULL || temp_dy1 == NULL || temp_dz1 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_dxyz1. Aborting... \n\n");
			mkl_free(temp_dxyz1);
			mkl_free(temp_dx10);
			mkl_free(temp_dy1);
			mkl_free(temp_dz1);
			system ("PAUSE");
			return 1;
		}
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,GHax,N_mod,m_0_n, 1, 0.0, temp_dx10, 1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,GHay,N_mod,m_0_n, 1, 0.0, temp_dy1, 1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,GZa,N_mod,m_0_n, 1, 0.0, temp_dz1, 1);

		#pragma omp parallel for shared(N_data,temp_dxyz1,temp_dx10,temp_dy1,temp_dz1) private(i)
		for(i = 0; i < N_data; i++){
			temp_dxyz1[i] = sqrt((temp_dx10[i]+geomag_ref_x[i])*(temp_dx10[i]+geomag_ref_x[i]) 
				+ (temp_dy1[i]+geomag_ref_y[i])*(temp_dy1[i]+geomag_ref_y[i]) 
				+ (temp_dz1[i]+geomag_ref_z[i])*(temp_dz1[i]+geomag_ref_z[i]))
				- sqrt(geomag_ref_x[i]*geomag_ref_x[i] + geomag_ref_y[i]*geomag_ref_y[i] 
					+ geomag_ref_z[i]*geomag_ref_z[i]);
		}

		// cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,Gxyz,N_mod,m_0_n,1,-1,temp_dxyz1,1);

		#pragma omp parallel for shared(N_data,temp_dxyz1,Rd_2,Wd_2,deltaT_2) private(i)
		for(i=0;i<N_data;i++)
			temp_dxyz1[i] = Wd_2[i]*Rd_2[i]*Wd_2[i]*(temp_dxyz1[i]-deltaT_2[i]);
		
		temp_deltaT_2 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_mod,temp_deltaT_2) private(i)
		for(i=0;i<N_mod;i++)
			temp_deltaT_2[i] = 0;
		cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,G_2,N_mod,temp_dxyz1,1,0,temp_deltaT_2,1);
		mkl_free(temp_dxyz1);
		mkl_free(temp_dy1);
		mkl_free(temp_dx10);
		mkl_free(temp_dz1);
	}
	else{
		temp_deltaT_2 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_mod,temp_deltaT_2) private(i)
		for(i=0;i<N_mod;i++)
			temp_deltaT_2[i] = 0;
	}


	double * B;
	B = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (B == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix B. Aborting... \n\n");
		mkl_free(B);
		system ("PAUSE");
		return 1;
	}
	#pragma omp parallel for shared(N_mod,B,temp_s,temp_x,temp_y,temp_z,temp_d,temp_dTa) private(i)
	for(i=0;i<N_mod;i++) {
		B[i] = (-1)*(temp_s[i]+temp_x[i]+temp_y[i]+temp_z[i]
			+ temp_xy[i] + temp_yx[i] + temp_xz[i] + temp_zx[i] + temp_yz[i] + temp_zy[i]
			+ temp_d[i]
			+ temp_dHax[i]+temp_dHay[i]+temp_dZa[i]+temp_dTa[i]
			+ temp_deltaT_2[i]);
	}
	// printf("B\n");
	mkl_free(temp_m);
	mkl_free(temp_s);
	mkl_free(temp_x);
	mkl_free(temp_y);
	mkl_free(temp_z);

	mkl_free(temp_xy);
	mkl_free(temp_yx);
	mkl_free(temp_yz);
	mkl_free(temp_zy);
	mkl_free(temp_xz);
	mkl_free(temp_zx);

	mkl_free(temp_d);
	mkl_free(temp_deltaT_2);
	
	mkl_free(temp_dHax);
	mkl_free(temp_dHay);
	mkl_free(temp_dZa);
	mkl_free(temp_dTa);


	/////////////////////////////////////////////////////////////////
	double * X0;
	X0 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	double * X1;
	X1 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (X0 == NULL || X1 == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix X0 X1. Aborting... \n\n");
		mkl_free(X1);
		mkl_free(X0);
		system ("PAUSE");
		return 1;
	}

	#pragma omp parallel for shared(N_mod,X0) private(i)
	for(i=0;i<N_mod;i++)
		X0[i] = 0;
	double t_stop = 0.001;
	// C G 
	double * R0, * P0;
	R0 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	P0 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (P0 == NULL || R0 == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix P0 R0. Aborting... \n\n");
		mkl_free(P0);
		mkl_free(R0);
		system ("PAUSE");
		return 1;
	}

	#pragma omp parallel for shared(N_mod,R0,P0,B) private(i)
	for(i=0;i<N_mod;i++) {
		R0[i] = (-1)*B[i];
		P0[i] = (-1)*R0[i];
	}

	// --------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------
	// --------------------    CG method for linear inversion    ----------------------
	// --------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------

	for(int j = 1;j<N_data;j++) {
		double alpha_k;
		double temp1,temp2;
		
		double * AP0;
		AP0 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (AP0 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix AP0. Aborting... \n\n");
			mkl_free(AP0);
			system ("PAUSE");
			return 1;
		}

		temp_m = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_m == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_m. Aborting... \n\n");
			mkl_free(temp_m);
			system ("PAUSE");
			return 1;
		}
		#pragma omp parallel for shared(N_mod,temp_m,P0) private(i)
		for(i=0;i<N_mod;i++)
			temp_m[i] = P0[i];

		temp_s = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		temp_x = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		temp_y = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		temp_z = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );


		temp_xy = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		temp_yz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		temp_zx = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		temp_yx = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		temp_zy = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		temp_xz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );

		if (temp_s == NULL || temp_x==NULL || temp_y==NULL || temp_z==NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_s temp_x temp_y temp_z. Aborting... \n\n");
			mkl_free(temp_s);
			mkl_free(temp_x);
			mkl_free(temp_y);
			mkl_free(temp_z);
			system ("PAUSE");
			return 1;
		}
		if (temp_xy == NULL || temp_yx==NULL || temp_yz==NULL || temp_zy==NULL || temp_xz==NULL || temp_zx==NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_s temp_x temp_y temp_z. Aborting... \n\n");
			mkl_free(temp_xy);
			mkl_free(temp_yx);
			mkl_free(temp_yz);
			mkl_free(temp_zy);
			mkl_free(temp_xz);
			mkl_free(temp_zx);
			system ("PAUSE");
			return 1;
		}

		#pragma omp parallel for shared(N_mod,temp_s,miu,W_deepw,alpha,Ws,Rs,temp_m) private(i)
		for(i=0;i<N_mod;i++){
			temp_s[i] = miu*alpha[0]*Ws[i]*Rs[i]*Ws[i]*temp_m[i];
            temp_x[i] = 0;
            temp_y[i] = 0;
            temp_z[i] = 0;
            temp_xy[i] = 0;
            temp_yz[i] = 0;
            temp_zx[i] = 0;
            temp_yx[i] = 0;
            temp_zy[i] = 0;
            temp_xz[i] = 0;
        }

		char transa = 'n';

		temp_x1 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_x1 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_x1. Aborting... \n\n");
			mkl_free(temp_x1);
			system ("PAUSE");
			return 1;
		}
        #pragma omp parallel for shared(N_mod,temp_x1) private(i)
        for(i=0;i<N_mod;i++) {
            temp_x1[i] = 0;
        }
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_x1);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_x1);
		#pragma omp parallel for shared(N_mod,temp_x1,Rx) private(i)
		for(i=0;i<N_mod;i++)
			temp_x1[i] = Rx[i]*temp_x1[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_x1, 0.0, temp_x);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_x1,temp_x);

		#pragma omp parallel for shared(N_mod,temp_x,miu,W_deepw,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_x[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r11[i] 
				+ m_alpha_y[i]*tran_r21[i]*tran_r21[i] 
				+ m_alpha_z[i]*tran_r31[i]*tran_r31[i])*temp_x[i];
		mkl_free(temp_x1);


		transa = 'n';
		temp_y1 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_y1 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_y1. Aborting... \n\n");
			mkl_free(temp_y1);
			system ("PAUSE");
			return 1;
		}
        #pragma omp parallel for shared(N_mod,temp_y1) private(i)
        for(i=0;i<N_mod;i++) {
            temp_y1[i] = 0;
        }
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_y1);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_y1);
		#pragma omp parallel for shared(N_mod,temp_y1,Ry) private(i)
		for(i=0;i<N_mod;i++)
			temp_y1[i] = Ry[i]*temp_y1[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wy, descrA, temp_y1, 0.0, temp_y);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_y1,temp_y);

		#pragma omp parallel for shared(N_mod,temp_y,miu,W_deepw,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_y[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r12[i] 
				+ m_alpha_y[i]*tran_r22[i]*tran_r22[i] 
				+ m_alpha_z[i]*tran_r32[i]*tran_r32[i])*temp_y[i];
		mkl_free(temp_y1);


		transa = 'n';
		temp_z1 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_z1 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_z1. Aborting... \n\n");
			mkl_free(temp_z1);
			system ("PAUSE");
			return 1;
		}
        #pragma omp parallel for shared(N_mod,temp_z1) private(i)
        for(i=0;i<N_mod;i++) {
            temp_z1[i] = 0;
        }
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_z1);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_z1);
		#pragma omp parallel for shared(N_mod,temp_z1,Rz) private(i)
		for(i=0;i<N_mod;i++)
			temp_z1[i] = Rz[i]*temp_z1[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wz, descrA, temp_z1, 0.0, temp_z);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_z1,temp_z);
		#pragma omp parallel for shared(N_mod,temp_z,miu,W_deepw,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_z[i] = miu*(m_alpha_x[i]*tran_r13[i]*tran_r13[i] 
				+ m_alpha_y[i]*tran_r23[i]*tran_r23[i] 
				+ m_alpha_z[i]*tran_r33[i]*tran_r33[i])*temp_z[i];
		mkl_free(temp_z1);


		transa = 'n';
		temp_x1_xy = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_x1_xy == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_x1_xy. Aborting... \n\n");
			mkl_free(temp_x1_xy);
			system ("PAUSE");
			return 1;
		}
        #pragma omp parallel for shared(N_mod,temp_x1_xy) private(i)
        for(i=0;i<N_mod;i++) {
            temp_x1_xy[i] = 0;
        }
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_x1_xy);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_x1_xy);
		#pragma omp parallel for shared(N_mod,temp_x1_xy,Rx) private(i)
		for(i=0;i<N_mod;i++)
			temp_x1_xy[i] = sqrt(Rx[i])*sqrt(Ry[i])*temp_x1_xy[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_x1_xy, 0.0, temp_xy);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_x1_xy,temp_xy);

		#pragma omp parallel for shared(N_mod,temp_xy,miu,W_deepw,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_xy[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r12[i] 
				+ m_alpha_y[i]*tran_r21[i]*tran_r22[i] 
				+ m_alpha_z[i]*tran_r31[i]*tran_r32[i])*temp_xy[i];
		mkl_free(temp_x1_xy);


		transa = 'n';
		temp_x1_yx = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_x1_yx == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_x1_yx. Aborting... \n\n");
			mkl_free(temp_x1_yx);
			system ("PAUSE");
			return 1;
		}
        #pragma omp parallel for shared(N_mod,temp_x1_yx) private(i)
        for(i=0;i<N_mod;i++) {
            temp_x1_yx[i] = 0;
        }
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_x1_yx);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_x1_yx);
		#pragma omp parallel for shared(N_mod,temp_x1_yx,Rx) private(i)
		for(i=0;i<N_mod;i++)
			temp_x1_yx[i] = sqrt(Rx[i])*sqrt(Ry[i])*temp_x1_yx[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wy, descrA, temp_x1_yx, 0.0, temp_yx);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_x1_yx,temp_yx);

		#pragma omp parallel for shared(N_mod,temp_yx,miu,W_deepw,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_yx[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r12[i] 
				+ m_alpha_y[i]*tran_r21[i]*tran_r22[i] 
				+ m_alpha_z[i]*tran_r31[i]*tran_r32[i])*temp_yx[i];
		mkl_free(temp_x1_yx);


		transa = 'n';
		temp_x1_xz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_x1_xz == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_x1_xz. Aborting... \n\n");
			mkl_free(temp_x1_xz);
			system ("PAUSE");
			return 1;
		}
        #pragma omp parallel for shared(N_mod,temp_x1_xz) private(i)
        for(i=0;i<N_mod;i++) {
            temp_x1_xz[i] = 0;
        }
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_x1_xz);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_x1_xz);
		#pragma omp parallel for shared(N_mod,temp_x1_xz,Rx) private(i)
		for(i=0;i<N_mod;i++)
			temp_x1_xz[i] = sqrt(Rx[i])*sqrt(Rz[i])*temp_x1_xz[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_x1_xz, 0.0, temp_xz);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_x1_xz,temp_xz);

		#pragma omp parallel for shared(N_mod,temp_xz,miu,W_deepw,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_xz[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r13[i] 
				+ m_alpha_y[i]*tran_r21[i]*tran_r23[i] 
				+ m_alpha_z[i]*tran_r31[i]*tran_r33[i])*temp_xz[i];
		mkl_free(temp_x1_xz);


		transa = 'n';
		temp_x1_zx = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_x1_zx == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_x1_zx. Aborting... \n\n");
			mkl_free(temp_x1_zx);
			system ("PAUSE");
			return 1;
		}
        #pragma omp parallel for shared(N_mod,temp_x1_zx) private(i)
        for(i=0;i<N_mod;i++) {
            temp_x1_zx[i] = 0;
        }
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_x1_zx);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_x1_zx);
		#pragma omp parallel for shared(N_mod,temp_x1_zx,Rx) private(i)
		for(i=0;i<N_mod;i++)
			temp_x1_zx[i] = sqrt(Rx[i])*sqrt(Rz[i])*temp_x1_zx[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wz, descrA, temp_x1_zx, 0.0, temp_zx);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_x1_zx,temp_zx);

		#pragma omp parallel for shared(N_mod,temp_zx,miu,W_deepw,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_zx[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r13[i] 
				+ m_alpha_y[i]*tran_r21[i]*tran_r23[i] 
				+ m_alpha_z[i]*tran_r31[i]*tran_r33[i])*temp_zx[i];
		mkl_free(temp_x1_zx);


		transa = 'n';
		temp_x1_yz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_x1_yz == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_x1_yz. Aborting... \n\n");
			mkl_free(temp_x1_yz);
			system ("PAUSE");
			return 1;
		}
        #pragma omp parallel for shared(N_mod,temp_x1_yz) private(i)
        for(i=0;i<N_mod;i++) {
            temp_x1_yz[i] = 0;
        }
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_x1_yz);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_x1_yz);
		#pragma omp parallel for shared(N_mod,temp_x1_yz,Rx) private(i)
		for(i=0;i<N_mod;i++)
			temp_x1_yz[i] = sqrt(Ry[i])*sqrt(Rz[i])*temp_x1_yz[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wy, descrA, temp_x1_yz, 0.0, temp_yz);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_x1_yz,temp_yz);

		#pragma omp parallel for shared(N_mod,temp_yz,miu,W_deepw,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_yz[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r13[i] 
				+ m_alpha_y[i]*tran_r22[i]*tran_r23[i] 
				+ m_alpha_z[i]*tran_r32[i]*tran_r33[i])*temp_yz[i];
		mkl_free(temp_x1_yz);


		transa = 'n';
		temp_x1_zy = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_x1_zy == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_x1_zy. Aborting... \n\n");
			mkl_free(temp_x1_zy);
			system ("PAUSE");
			return 1;
		}
        #pragma omp parallel for shared(N_mod,temp_x1_zy) private(i)
        for(i=0;i<N_mod;i++) {
            temp_x1_zy[i] = 0;
        }
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_x1_zy);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_x1_zy);
		#pragma omp parallel for shared(N_mod,temp_x1_zy,Rx) private(i)
		for(i=0;i<N_mod;i++)
			temp_x1_zy[i] = sqrt(Ry[i])*sqrt(Rz[i])*temp_x1_zy[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wz, descrA, temp_x1_zy, 0.0, temp_zy);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_x1_zy,temp_zy);

		#pragma omp parallel for shared(N_mod,temp_zy,miu,W_deepw,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_zy[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r13[i] 
				+ m_alpha_y[i]*tran_r22[i]*tran_r23[i] 
				+ m_alpha_z[i]*tran_r32[i]*tran_r33[i])*temp_zy[i];
		mkl_free(temp_x1_zy);


		#pragma omp parallel for shared(N_mod,temp_m,P0) private(i)
		for(i=0;i<N_mod;i++)
			temp_m[i] = P0[i];
		
		
		temp_d = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_d == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_d. Aborting... \n\n");
			mkl_free(temp_d);
			system ("PAUSE");
			return 1;
		}
		if(paraInvOrNot[0] != 0){
			double * temp_d1;
			temp_d1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			if (temp_d1 == NULL) {
				printf( "\n ERROR: Can't allocate memory for matrix temp_d1. Aborting... \n\n");
				mkl_free(temp_d1);
				system ("PAUSE");
				return 1;
			}
			#pragma omp parallel for shared(N_data,temp_d1) private(i)
			for(i = 0;i < N_data;i++)
				temp_d1[i] = 0;
			cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,G,N_mod,temp_m,1,0,temp_d1,1);
			#pragma omp parallel for shared(N_data,temp_d1,Wd) private(i)
			for(i=0;i<N_data;i++)
				temp_d1[i] = Wd[i]*Rd[i]*Wd[i]*temp_d1[i];
			
			#pragma omp parallel for shared(N_mod,temp_d) private(i)
			for(i=0;i<N_mod;i++)
				temp_d[i] = 0;
			cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,G,N_mod,temp_d1,1,0,temp_d,1);
			mkl_free(temp_d1);
		}
		else{
			#pragma omp parallel for shared(N_mod,temp_d) private(i)
			for(i=0;i<N_mod;i++)
				temp_d[i] = 0;
		}
		
		
		temp_dHax = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_dHax == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_dHax. Aborting... \n\n");
			mkl_free(temp_dHax);
			system ("PAUSE");
			return 1;
		}
		if(paraInvOrNot[1] != 0){
			double * temp_dHax1;
			temp_dHax1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			if (temp_dHax1 == NULL) {
				printf( "\n ERROR: Can't allocate memory for matrix temp_dHax1. Aborting... \n\n");
				mkl_free(temp_dHax1);
				system ("PAUSE");
				return 1;
			}
			#pragma omp parallel for shared(N_data,temp_dHax1) private(i)
			for(i = 0;i < N_data;i++)
				temp_dHax1[i] = 0;
			cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,GHax,N_mod,temp_m,1,0,temp_dHax1,1);
			#pragma omp parallel for shared(N_data,temp_dHax1,WdHax) private(i)
			for(i=0;i<N_data;i++)
				temp_dHax1[i] = WdHax[i]*RdHax[i]*WdHax[i]*temp_dHax1[i];
			
			#pragma omp parallel for shared(N_mod,temp_dHax) private(i)
			for(i=0;i<N_mod;i++)
				temp_dHax[i] = 0;
			cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,GHax,N_mod,temp_dHax1,1,0,temp_dHax,1);
			mkl_free(temp_dHax1);
		}
		else{
			#pragma omp parallel for shared(N_mod,temp_dHax) private(i)
			for(i=0;i<N_mod;i++)
				temp_dHax[i] = 0;
		}

		temp_dHay = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_dHay == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_dHay. Aborting... \n\n");
			mkl_free(temp_dHay);
			system ("PAUSE");
			return 1;
		}
		if(paraInvOrNot[2] != 0){
			double * temp_dHay1;
			temp_dHay1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			if (temp_dHay1 == NULL) {
				printf( "\n ERROR: Can't allocate memory for matrix temp_dHay1. Aborting... \n\n");
				mkl_free(temp_dHay1);
				system ("PAUSE");
				return 1;
			}
			#pragma omp parallel for shared(N_data,temp_dHay1) private(i)
			for(i = 0;i < N_data;i++)
				temp_dHay1[i] = 0;
			cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,GHay,N_mod,temp_m,1,0,temp_dHay1,1);
			#pragma omp parallel for shared(N_data,temp_dHay1,WdHay) private(i)
			for(i=0;i<N_data;i++)
				temp_dHay1[i] = WdHay[i]*RdHay[i]*WdHay[i]*temp_dHay1[i];
			
			#pragma omp parallel for shared(N_mod,temp_dHay) private(i)
			for(i=0;i<N_mod;i++)
				temp_dHay[i] = 0;
			cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,GHay,N_mod,temp_dHay1,1,0,temp_dHay,1);
			mkl_free(temp_dHay1);
		}
		else{
			#pragma omp parallel for shared(N_mod,temp_dHay) private(i)
			for(i=0;i<N_mod;i++)
				temp_dHay[i] = 0;
		}

		temp_dZa = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_dZa == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_dZa. Aborting... \n\n");
			mkl_free(temp_dZa);
			system ("PAUSE");
			return 1;
		}
		if(paraInvOrNot[3] != 0){
			double * temp_dZa1;
			temp_dZa1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			if (temp_dZa1 == NULL) {
				printf( "\n ERROR: Can't allocate memory for matrix temp_dZa1. Aborting... \n\n");
				mkl_free(temp_dZa1);
				system ("PAUSE");
				return 1;
			}
			#pragma omp parallel for shared(N_data,temp_dZa1) private(i)
			for(i = 0;i < N_data;i++)
				temp_dZa1[i] = 0;
			cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,GZa,N_mod,temp_m,1,0,temp_dZa1,1);
			#pragma omp parallel for shared(N_data,temp_dZa1,WdZa) private(i)
			for(i=0;i<N_data;i++)
				temp_dZa1[i] = WdZa[i]*RdZa[i]*WdZa[i]*temp_dZa1[i];
			
			#pragma omp parallel for shared(N_mod,temp_dZa) private(i)
			for(i=0;i<N_mod;i++)
				temp_dZa[i] = 0;
			cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,GZa,N_mod,temp_dZa1,1,0,temp_dZa,1);
			mkl_free(temp_dZa1);
		}
		else{
			#pragma omp parallel for shared(N_mod,temp_dZa) private(i)
			for(i=0;i<N_mod;i++)
				temp_dZa[i] = 0;
		}


		temp_dTa = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_dTa == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_dTa. Aborting... \n\n");
			mkl_free(temp_dTa);
			system ("PAUSE");
			return 1;
		}
		if(paraInvOrNot[4] != 0){
			double * temp_dTa1;
			temp_dTa1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			if (temp_dTa1 == NULL) {
				printf( "\n ERROR: Can't allocate memory for matrix temp_dTa1. Aborting... \n\n");
				mkl_free(temp_dTa1);
				system ("PAUSE");
				return 1;
			}
			#pragma omp parallel for shared(N_data,temp_dTa1) private(i)
			for(i = 0;i < N_data;i++)
				temp_dTa1[i] = 0;

			cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,GTa,N_mod,temp_m,1,0,temp_dTa1,1);
			#pragma omp parallel for shared(N_data,temp_dTa1,WdTa,RdTa) private(i)
			for(i=0;i<N_data;i++)
				temp_dTa1[i] = WdTa[i]*RdTa[i]*WdTa[i]*temp_dTa1[i];

			temp_dTa = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
			if (temp_dTa == NULL) {
				printf( "\n ERROR: Can't allocate memory for matrix temp_dTa. Aborting... \n\n");
				mkl_free(temp_dTa);
				system ("PAUSE");
				return 1;
			}
			#pragma omp parallel for shared(N_mod,temp_dTa) private(i)
			for(i=0;i<N_mod;i++)
				temp_dTa[i] = 0;
			cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,GTa,N_mod,temp_dTa1,1,0,temp_dTa,1);
			mkl_free(temp_dTa1);
		}
		else{
			#pragma omp parallel for shared(N_mod,temp_dTa) private(i)
			for(i=0;i<N_mod;i++)
				temp_dTa[i] = 0;
		}


		temp_deltaT_2 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_deltaT_2 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_deltaT_2. Aborting... \n\n");
			mkl_free(temp_deltaT_2);
			system ("PAUSE");
			return 1;
		}
		if(paraInvOrNot[5] != 0){
			double * temp_deltaT_21;
			temp_deltaT_21 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			if (temp_deltaT_21 == NULL) {
				printf( "\n ERROR: Can't allocate memory for matrix temp_deltaT_21. Aborting... \n\n");
				mkl_free(temp_deltaT_21);
				system ("PAUSE");
				return 1;
			}
			#pragma omp parallel for shared(N_data,temp_deltaT_21) private(i)
			for(i = 0;i < N_data;i++)
				temp_deltaT_21[i] = 0;

			cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,G_2,N_mod,temp_m,1,0,temp_deltaT_21,1);
			#pragma omp parallel for shared(N_data,temp_deltaT_21,Wd_2,Rd_2) private(i)
			for(i=0;i<N_data;i++)
				temp_deltaT_21[i] = Wd_2[i]*Rd_2[i]*Wd_2[i]*temp_deltaT_21[i];

			temp_deltaT_2 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
			if (temp_deltaT_2 == NULL) {
				printf( "\n ERROR: Can't allocate memory for matrix temp_deltaT_2. Aborting... \n\n");
				mkl_free(temp_deltaT_2);
				system ("PAUSE");
				return 1;
			}
			#pragma omp parallel for shared(N_mod,temp_deltaT_2) private(i)
			for(i=0;i<N_mod;i++)
				temp_deltaT_2[i] = 0;
			cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,G_2,N_mod,temp_deltaT_21,1,0,temp_deltaT_2,1);
			mkl_free(temp_deltaT_21);
		}
		else{
			#pragma omp parallel for shared(N_mod,temp_deltaT_2) private(i)
			for(i=0;i<N_mod;i++)
				temp_deltaT_2[i] = 0;
		}


		#pragma omp parallel for shared(N_mod,temp_s,AP0,temp_x,temp_y,temp_z,temp_d) private(i)
		for(i=0;i<N_mod;i++) {
			AP0[i] = (temp_s[i]+temp_x[i]+temp_y[i]+temp_z[i]
				+temp_xy[i]+temp_yx[i]+temp_xz[i]+temp_zx[i]+temp_yz[i]+temp_zy[i]
				+temp_d[i]
				+temp_dHax[i]+temp_dHay[i]+temp_dZa[i]+temp_dTa[i]
				+temp_deltaT_2[i]);

		}
		mkl_free(temp_m);
		mkl_free(temp_s);
		mkl_free(temp_x);
		mkl_free(temp_y);
		mkl_free(temp_z);

		mkl_free(temp_xy);
		mkl_free(temp_yx);
		mkl_free(temp_zx);
		mkl_free(temp_xz);
		mkl_free(temp_yz);
		mkl_free(temp_zy);

		mkl_free(temp_d);
		mkl_free(temp_deltaT_2);
		mkl_free(temp_dHax);
		mkl_free(temp_dHay);
		mkl_free(temp_dZa);
		mkl_free(temp_dTa);


		temp1 = 0;
		temp2 = 0;
		// #pragma omp parallel for shared(N_mod,R0,P0,AP0) private(i,temp1)
		temp1 = cblas_ddot(N_mod,R0,1,R0,1);
		temp2 = cblas_ddot(N_mod,P0,1,AP0,1);
		alpha_k = temp1/(temp2+eps1);


		double * R1;
		R1 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (R1 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix R1. Aborting... \n\n");
			mkl_free(R1);
			system ("PAUSE");
			return 1;
		}
		cblas_dcopy(N_mod,X0,1,X1,1);
		cblas_daxpy(N_mod,alpha_k,P0,1,X1,1);

		cblas_dcopy(N_mod,R0,1,R1,1);
		cblas_daxpy(N_mod,alpha_k,AP0,1,R1,1);
		double err;
		temp1 = 0;temp2 = 0;

		temp1 = cblas_ddot(N_mod,R1,1,R1,1);
		temp2 = cblas_ddot(N_mod,B,1,B,1);
		err = temp1/(temp2+eps1);

		if(err<t_stop)
			break;
		double beta ;
		temp2 = 0;
		temp2 = cblas_ddot(N_mod,R0,1,R0,1);
		beta = temp1/(temp2+eps1);

		double * P1;
		P1 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (P1 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix P1. Aborting... \n\n");
			mkl_free(P1);
			system ("PAUSE");
			return 1;
		}
		#pragma omp parallel for shared(N_mod,P1,R1) private(i)
		for(i=0;i<N_mod;i++)
			P1[i] = (-1)*R1[i];
		cblas_daxpy(N_mod,beta,P0,1,P1,1);

		cblas_dcopy(N_mod,X1,1,X0,1);
		cblas_dcopy(N_mod,R1,1,R0,1);
		cblas_dcopy(N_mod,P1,1,P0,1);
		mkl_free(AP0);
		mkl_free(R1);
		mkl_free(P1);

	}
	// C G :end
	
	
	#pragma omp parallel for shared(N_mod,T,X1) private(i)
	for(i=0;i<N_mod;i++){
		X1[i] = X1[i]/T[i];
	}

	double * m_tran_1;
	m_tran_1 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	#pragma omp parallel for shared(N_mod,m_tran_1,m_tran_0,X1) private(i)
	for(i=0;i<N_mod;i++)
		m_tran_1[i] = m_tran_0[i] + X1[i];


	cal_tran_m(N_mod,m_tran_1,m_min_n,m_max_n,"back",m_result);


	mkl_free(X0);
	mkl_free(X1);
	mkl_free(P0);
	mkl_free(R0);

	mkl_free(m_tran_1);
	mkl_free(m_tran_0);
	mkl_free(T);

	mkl_free(B);

    mkl_sparse_destroy(csr_Wx);
	mkl_sparse_destroy(csr_Wy);
	mkl_sparse_destroy(csr_Wz);
	return 0;
}


/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/


int cal_datamisfit_objmodfun_Barbosa_noneffsto(int N_data, int N_mod,double miu,double * deltaT,
	double * G,double * Wd,double * deltaTHax,double * GHax,double * WdHax,
	double * deltaTHay,double * GHay,double * WdHay,double * deltaTZa,double * GZa,
	double * WdZa,double * deltaTTa,double * GTa,double * WdTa,
	double * deltaT_2, double * G_2, double * Wd_2, double * geomag_ref_x, double * geomag_ref_y, double * geomag_ref_z,
	int * paraInvOrNot,
	double * Ws,double * Wx,MKL_INT * columns_x,MKL_INT * rowIndex_x,double * Wy,MKL_INT * columns_y,MKL_INT * rowIndex_y,
	double * Wz,MKL_INT * columns_z,MKL_INT * rowIndex_z,double * W_deepw,double * alpha,
	double * m_ref_n,double * m_1_n,double * m_min_n,double * m_max_n,
	double * p1,double * p2,double * px2,double * py2,double * pz2,
	double * epsilon1,double * epsilon2,double * epsilonx2,double * epsilony2,double * epsilonz2,
	double * m_result,
	double * m_alpha_s, double * m_alpha_x,double * m_alpha_y, double * m_alpha_z,
	double * tran_r11,double * tran_r12,double * tran_r13,double * tran_r21,
	double * tran_r22,double * tran_r23,double * tran_r31,double * tran_r32,double * tran_r33)
{
	int i;

	double phid_sum = 0;
	if(paraInvOrNot[0] != 0){
		double * temp_d1;
		temp_d1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		if (temp_d1 == NULL /*|| temp_d2 == NULL*/) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_d1 temp_d2. Aborting... \n\n");
			mkl_free(temp_d1);
			return 1;
		}
		cblas_dcopy(N_data,deltaT,1,temp_d1,1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,G,N_mod,m_1_n,1,-1,temp_d1,1);
		for(i=0;i<N_data;i++)
			phid_sum += temp_d1[i]*Wd[i]*Wd[i]*temp_d1[i];
		mkl_free(temp_d1);
	}
	else{
		phid_sum = 0;
	}
	
	double phid_sumHax = 0;
	if(paraInvOrNot[1] != 0){
		double * temp_d1;
		temp_d1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		if (temp_d1 == NULL /*|| temp_d2 == NULL*/) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_d1 temp_d2. Aborting... \n\n");
			mkl_free(temp_d1);
			return 1;
		}
		cblas_dcopy(N_data,deltaTHax,1,temp_d1,1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,GHax,N_mod,m_1_n,1,-1,temp_d1,1);
		for(i=0;i<N_data;i++)
			phid_sumHax += temp_d1[i]*WdHax[i]*WdHax[i]*temp_d1[i];
		mkl_free(temp_d1);
	}
	else{
		phid_sumHax = 0;
	}
	
	double phid_sumHay = 0;
	if(paraInvOrNot[2] != 0){
		double * temp_d1;
		temp_d1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		if (temp_d1 == NULL /*|| temp_d2 == NULL*/) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_d1 temp_d2. Aborting... \n\n");
			mkl_free(temp_d1);
			return 1;
		}
		cblas_dcopy(N_data,deltaTHay,1,temp_d1,1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,GHay,N_mod,m_1_n,1,-1,temp_d1,1);
		for(i=0;i<N_data;i++)
			phid_sumHay += temp_d1[i]*WdHay[i]*WdHay[i]*temp_d1[i];
		mkl_free(temp_d1);
	}
	else{
		phid_sumHay = 0;
	}
	
	double phid_sumZa = 0;
	if(paraInvOrNot[3] != 0){
		double * temp_d1;
		temp_d1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		if (temp_d1 == NULL /*|| temp_d2 == NULL*/) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_d1 temp_d2. Aborting... \n\n");
			mkl_free(temp_d1);
			return 1;
		}
		cblas_dcopy(N_data,deltaTZa,1,temp_d1,1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,GZa,N_mod,m_1_n,1,-1,temp_d1,1);
		for(i=0;i<N_data;i++)
			phid_sumZa += temp_d1[i]*WdZa[i]*WdZa[i]*temp_d1[i];
		mkl_free(temp_d1);
	}
	else{
		phid_sumZa = 0;
	}

	
	double phid_sumTa = 0;
	if(paraInvOrNot[4] != 0){
		double * temp_d1;
		double * temp_dx1,* temp_dy1,* temp_dz1;
		temp_d1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_dx1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_dy1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_dz1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		if (temp_dx1 == NULL || temp_dy1 == NULL || temp_dz1 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_dxyz1. Aborting... \n\n");
			mkl_free(temp_dx1);
			mkl_free(temp_dy1);
			mkl_free(temp_dz1);
			system ("PAUSE");
			return 1;
		}
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,GHax,N_mod,m_1_n, 1, 0.0, temp_dx1, 1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,GHay,N_mod,m_1_n, 1, 0.0, temp_dy1, 1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,GZa,N_mod,m_1_n, 1, 0.0, temp_dz1, 1);

		#pragma omp parallel for shared(N_data,temp_d1,temp_dx1,temp_dy1,temp_dz1) private(i)
		for(i = 0; i < N_data; i++){
			temp_d1[i] = sqrt(Txyz_weight[0]*temp_dx1[i]*temp_dx1[i] + Txyz_weight[1]*temp_dy1[i]*temp_dy1[i] + Txyz_weight[2]*temp_dz1[i]*temp_dz1[i]) - deltaTTa[i];
		}

		for(i=0;i<N_data;i++)
			phid_sumTa += pow( temp_d1[i]*WdTa[i]*WdTa[i]*temp_d1[i]+epsilon1[i]*epsilon1[i],p1[i]/2) ;
		mkl_free(temp_d1);
		mkl_free(temp_dx1);
		mkl_free(temp_dy1);
		mkl_free(temp_dz1);

	}
	else{
		phid_sumTa = 0;
	}

	double phid_sum_2 = 0;
	if(paraInvOrNot[5] != 0){
		double * temp_d1;
		double * temp_dx1,* temp_dy1,* temp_dz1;
		temp_d1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_dx1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_dy1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_dz1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		if (temp_dx1 == NULL || temp_dy1 == NULL || temp_dz1 == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_dxyz1. Aborting... \n\n");
			mkl_free(temp_dx1);
			mkl_free(temp_dy1);
			mkl_free(temp_dz1);
			system ("PAUSE");
			return 1;
		}
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,GHax,N_mod,m_1_n, 1, 0.0, temp_dx1, 1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,GHay,N_mod,m_1_n, 1, 0.0, temp_dy1, 1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,GZa,N_mod,m_1_n, 1, 0.0, temp_dz1, 1);

		#pragma omp parallel for shared(N_data,temp_d1,temp_dx1,temp_dy1,temp_dz1) private(i)
		for(i = 0; i < N_data; i++){
			temp_d1[i] = sqrt((temp_dx1[i]+geomag_ref_x[i])*(temp_dx1[i]+geomag_ref_x[i]) 
				+ (temp_dy1[i]+geomag_ref_y[i])*(temp_dy1[i]+geomag_ref_y[i]) 
				+ (temp_dz1[i]+geomag_ref_z[i])*(temp_dz1[i]+geomag_ref_z[i])) 
				- sqrt(geomag_ref_x[i]*geomag_ref_x[i] + geomag_ref_y[i]*geomag_ref_y[i] 
					+ geomag_ref_z[i]*geomag_ref_z[i])
				- deltaT_2[i];
		}

		for(i=0;i<N_data;i++)
			phid_sum_2 += pow( temp_d1[i]*Wd_2[i]*Wd_2[i]*temp_d1[i]+epsilon1[i]*epsilon1[i],p1[i]/2) ;
		mkl_free(temp_d1);
		mkl_free(temp_dx1);
		mkl_free(temp_dy1);
		mkl_free(temp_dz1);

	}
	else{
		phid_sum_2 = 0;
	}

	m_result[0] = phid_sum + phid_sumHax + phid_sumHay + phid_sumZa + phid_sumTa + phid_sum_2;


	double * temp_m;
	temp_m = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_m == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_m. Aborting... \n\n");
		mkl_free(temp_m);
		system ("PAUSE");
		return 1;
	}
	#pragma omp parallel for shared(N_mod,m_1_n,m_ref_n,temp_m,W_deepw) private(i)
	for(i=0;i<N_mod;i++) {
		temp_m[i] = (m_1_n[i]-m_ref_n[i]);
	}

    struct matrix_descr descrA;
	descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
	sparse_matrix_t csr_Wx;
	mkl_sparse_d_create_csr(&csr_Wx, SPARSE_INDEX_BASE_ZERO,
		N_mod, // number of rows
		N_mod, // number of cols
		rowIndex_x, rowIndex_x + 1, columns_x, Wx);
	mkl_sparse_optimize(csr_Wx);
	
	sparse_matrix_t csr_Wy;
	mkl_sparse_d_create_csr(&csr_Wy, SPARSE_INDEX_BASE_ZERO,
		N_mod, // number of rows
		N_mod, // number of cols
		rowIndex_y, rowIndex_y + 1, columns_y, Wy);
	mkl_sparse_optimize(csr_Wy);

	sparse_matrix_t csr_Wz;
	mkl_sparse_d_create_csr(&csr_Wz, SPARSE_INDEX_BASE_ZERO,
		N_mod, // number of rows
		N_mod, // number of cols
		rowIndex_z, rowIndex_z + 1, columns_z, Wz);
	mkl_sparse_optimize(csr_Wz);

	double * temp_x,* temp_y,* temp_z,* temp_s;
	double * temp_xy, * temp_xz, * temp_yz;

	temp_s = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	temp_x = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	temp_y = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	temp_z = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );

	temp_xy = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	temp_xz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	temp_yz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );

	if (temp_s == NULL || temp_x==NULL || temp_y==NULL || temp_z==NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_s temp_x temp_y temp_z. Aborting... \n\n");
		mkl_free(temp_s);
		mkl_free(temp_x);
		mkl_free(temp_y);
		mkl_free(temp_z);
		system ("PAUSE");
		return 1;
	}

	if (temp_xy == NULL || temp_xz==NULL || temp_yz == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_xy temp_yx temp_xz temp_zx. Aborting... \n\n");
		mkl_free(temp_xy);
		mkl_free(temp_xz);
		mkl_free(temp_yz);
		system ("PAUSE");
		return 1;
	}

	double phim_s_sum,phim_x_sum,phim_y_sum,phim_z_sum;
	double phim_xy_sum,phim_yz_sum,phim_xz_sum;
	phim_s_sum = 0;
	//#pragma omp parallel for shared(N_mod,temp_s,miu,W_deepw,xlength,ylength,zlength,alpha,Ws,Rs,temp_m) private(i)

	for(i=0;i<N_mod;i++)
		phim_s_sum += alpha[0]*pow(temp_m[i]*Ws[i]*Ws[i]*temp_m[i] + epsilon2[i]*epsilon2[i],p2[i]/2);

	mkl_free(temp_s);

	char transa = 'n';
	double * temp_x1;
	temp_x1 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_x1 == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_x1. Aborting... \n\n");
		mkl_free(temp_x1);
		system ("PAUSE");
		return 1;
	}
    #pragma omp parallel for shared(N_mod,temp_x1) private(i)
    for(i=0;i<N_mod;i++) {
        temp_x1[i] = 0;
    }
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_x1);
	// mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_x1);

	double * temp_y1;
	temp_y1 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_y1 == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_y1. Aborting... \n\n");
		mkl_free(temp_y1);
		system ("PAUSE");
		return 1;
	}
    #pragma omp parallel for shared(N_mod,temp_y1) private(i)
    for(i=0;i<N_mod;i++) {
        temp_y1[i] = 0;
    }
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_y1);
    // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_y1);

	double * temp_z1;
	temp_z1 = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (temp_z1 == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_z1. Aborting... \n\n");
		mkl_free(temp_z1);
		system ("PAUSE");
		return 1;
	}
    #pragma omp parallel for shared(N_mod,temp_z1) private(i)
    for(i=0;i<N_mod;i++) {
        temp_z1[i] = 0;
    }
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_z1);
    // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_z1);



	phim_x_sum = 0;
	phim_y_sum = 0;
	phim_z_sum = 0;

	phim_xy_sum = 0;
	phim_xz_sum = 0;
	phim_yz_sum = 0;

	
	for(i=0;i<N_mod;i++) {

        phim_x_sum += m_alpha_x[i]*pow(temp_x1[i]*temp_x1[i]+epsilonx2[i]*epsilonx2[i],px2[i]/2);

		phim_y_sum += m_alpha_y[i]*pow(temp_y1[i]*temp_y1[i]+epsilony2[i]*epsilony2[i],py2[i]/2);

		phim_z_sum += m_alpha_z[i]*pow(temp_z1[i]*temp_z1[i]+epsilonz2[i]*epsilonz2[i],pz2[i]/2);
	}

	mkl_free(temp_x1);
	mkl_free(temp_x);
	
	mkl_free(temp_y1);
	mkl_free(temp_y);

	mkl_free(temp_z1);
	mkl_free(temp_z);

	mkl_free(temp_xy);
	mkl_free(temp_xz);
	mkl_free(temp_yz);

	mkl_free(temp_m);


	m_result[1] = phim_s_sum+phim_x_sum+phim_y_sum+phim_z_sum+phim_xy_sum+phim_xz_sum+phim_yz_sum;


	m_result[2] = m_result[0] + miu*m_result[1];

    mkl_sparse_destroy(csr_Wx);
	mkl_sparse_destroy(csr_Wy);
	mkl_sparse_destroy(csr_Wz);

	return 0;
}

/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/

double InversionMagSus_Barbosa_noneffsto(TESSEROID *model, OBSMAGPOS *obs_pos_mag, int N_data, int N_mod, 
	double beta,double miu,double * obs_deltaT, double * obs_deltaTerr,
	double * obs_deltaTHax, double * obs_deltaTerrHax,
	double * obs_deltaTHay, double * obs_deltaTerrHay,double * obs_deltaTZa, double * obs_deltaTerrZa,
	double * obs_deltaTTa, double * obs_deltaTerrTa,
	double * obs_deltaT_2, double * obs_deltaTerr_2, double * geomag_ref_x, double * geomag_ref_y, double * geomag_ref_z,
	int * paraInvOrNot,
	double * modelsmoothfun_s,double * modelsmoothfun_x,double * modelsmoothfun_y,double * modelsmoothfun_z,
	int * modelsmoothfun_xdiffdrct,int * modelsmoothfun_ydiffdrct,int * modelsmoothfun_zdiffdrct,
	double * G_THax,double * G_THay,double * G_TZa,
	double * p1,double * p2,double * px2,double * py2,double * pz2,
	double * epsilon1,double * epsilon2,double * epsilonx2,double * epsilony2,double * epsilonz2,
	double * alpha,double * m_ref_n,double * m_min_n,double * m_max_n,double * m_0_n,double * m_1_n,
	double * phi_result,char * fileLog,
	int withCalGCVcurve,int para_deepweight_type,double z0_deepw_real,
	double * m_alpha_s,double * m_alpha_x,double * m_alpha_y,double * m_alpha_z,
	double * m_angle_phi,double * m_angle_theta,double * m_angle_psi,
	int * index_lon, int * index_lat, int * index_depth, unsigned long long N_data_ULL, unsigned long long N_mod_ULL)
{
	int i,j,k;
	unsigned long long temp3;
	unsigned long long i_tmp, j_tmp;

	double * tran_r11 = new double [N_mod];
	double * tran_r12 = new double [N_mod];
	double * tran_r13 = new double [N_mod];
	double * tran_r21 = new double [N_mod];
	double * tran_r22 = new double [N_mod];
	double * tran_r23 = new double [N_mod];
	double * tran_r31 = new double [N_mod];
	double * tran_r32 = new double [N_mod];
	double * tran_r33 = new double [N_mod];

	double angle_phi; 
	double angle_theta;
	double angle_psi;

	#pragma omp parallel for shared(N_mod) private(i,angle_phi,angle_theta,angle_psi)
	for(i=0; i<N_mod; i++){
		angle_phi   = m_angle_phi[i] * PI / 180;
		angle_theta = (m_angle_theta[i]) * PI / 180;
		angle_psi   = m_angle_psi[i] * PI / 180;

		tran_r11[i] = cos(angle_phi)*cos(angle_psi) - sin(angle_phi)*cos(angle_theta)*sin(angle_psi);
		tran_r12[i] = sin(angle_phi)*cos(angle_psi) + cos(angle_phi)*cos(angle_theta)*sin(angle_psi);
		tran_r13[i] = sin(angle_theta)*sin(angle_psi);
		tran_r21[i] = (-1)*sin(angle_phi)*sin(angle_theta);
		tran_r22[i] = cos(angle_phi)*sin(angle_theta);
		tran_r23[i] = (-1)*cos(angle_theta);
		tran_r31[i] = (-1)*cos(angle_phi)*sin(angle_psi) - sin(angle_phi)*cos(angle_theta)*cos(angle_psi);
		tran_r32[i] = sin(angle_phi)*sin(angle_psi) + cos(angle_phi)*cos(angle_theta)*cos(angle_psi);
		tran_r33[i] = sin(angle_theta)*cos(angle_psi);
	}

	// Construct the Wd matrix: Start
	double * Wd;
	if (paraInvOrNot[0] != 0) {
        Wd = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
        if (Wd == NULL) {
            printf( "\n ERROR: Can't allocate memory for matrix Wd. Aborting... \n\n");
            mkl_free(Wd);
            return 1;
        }
        #pragma omp parallel for shared(Wd,N_data,obs_deltaTerr) private(i)
        for (i=0;i<N_data;i++)
            Wd[i] = 1/obs_deltaTerr[i];
    }
	else{
		Wd = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		#pragma omp parallel for shared(Wd,N_data,obs_deltaTerr) private(i)
        for (i=0;i<N_data;i++)
            Wd[i] = 0;
	}
	
	double * WdHax;
    if (paraInvOrNot[1] != 0) {
		WdHax = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		if (WdHax == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix Wd. Aborting... \n\n");
			mkl_free(WdHax);
			return 1;
		}
		#pragma omp parallel for shared(WdHax,N_data,obs_deltaTerrHax) private(i)
		for (i=0;i<N_data;i++)
			WdHax[i] = 1/obs_deltaTerrHax[i];
    }else{
		WdHax = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		#pragma omp parallel for shared(WdHax,N_data,obs_deltaTerrHax) private(i)
		for (i=0;i<N_data;i++)
			WdHax[i] = 0;
	}
	
	double * WdHay;
    if (paraInvOrNot[2] != 0){
        
        WdHay = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
        if (WdHay == NULL) {
            printf( "\n ERROR: Can't allocate memory for matrix Wd. Aborting... \n\n");
            mkl_free(WdHay);
            return 1;
        }
        #pragma omp parallel for shared(WdHay,N_data,obs_deltaTerrHay) private(i)
        for (i=0;i<N_data;i++)
            WdHay[i] = 1/obs_deltaTerrHay[i];
    }
	else{
		WdHay = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		#pragma omp parallel for shared(WdHay,N_data,obs_deltaTerrHay) private(i)
        for (i=0;i<N_data;i++)
            WdHay[i] = 0;
	}
	
	double * WdZa;
    if (paraInvOrNot[3] != 0) {
        WdZa = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
        if (WdZa == NULL) {
            printf( "\n ERROR: Can't allocate memory for matrix Wd. Aborting... \n\n");
            mkl_free(WdZa);
            return 1;
        }
        #pragma omp parallel for shared(WdZa,N_data,obs_deltaTerrZa) private(i)
        for (i=0;i<N_data;i++)
            WdZa[i] = 1/obs_deltaTerrZa[i];
    }else{
		WdZa = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		#pragma omp parallel for shared(WdZa,N_data,obs_deltaTerrZa) private(i)
        for (i=0;i<N_data;i++)
            WdZa[i] = 0;
	}

	double * WdTa;
    if (paraInvOrNot[4] != 0) {
        WdTa = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
        if (WdTa == NULL) {
            printf( "\n ERROR: Can't allocate memory for matrix Wd. Aborting... \n\n");
            mkl_free(WdTa);
            return 1;
        }
        #pragma omp parallel for shared(WdTa,N_data,obs_deltaTerrTa) private(i)
        for (i=0;i<N_data;i++)
            WdTa[i] = 1/obs_deltaTerrTa[i];
    }else{
		WdTa = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		#pragma omp parallel for shared(WdTa,N_data,obs_deltaTerrTa) private(i)
        for (i=0;i<N_data;i++)
            WdTa[i] = 0;
	}

	double * Wd_2;
	if (paraInvOrNot[5] != 0) {
        Wd_2 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
        if (Wd_2 == NULL) {
            printf( "\n ERROR: Can't allocate memory for matrix Wd_2. Aborting... \n\n");
            mkl_free(Wd_2);
            return 1;
        }
        #pragma omp parallel for shared(Wd_2,N_data,obs_deltaTerr_2) private(i)
        for (i=0;i<N_data;i++)
            Wd_2[i] = 1/obs_deltaTerr_2[i];
    }
	else{
		Wd_2 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		#pragma omp parallel for shared(Wd_2,N_data,obs_deltaTerr) private(i)
        for (i=0;i<N_data;i++)
            Wd_2[i] = 0;
	}
	// Construct the Wd matrix: End


	// Construct the Ws Wx Wy Wz matrix: Start
	int num_values_x, num_columns_x, num_rowIndex;
	int num_values_y, num_columns_y;
	int num_values_z, num_columns_z;
	
	num_rowIndex = N_mod + 1;
	int total_tmp_x = 0, total_tmp_y = 0, total_tmp_z = 0;
	for(i = 0; i < N_mod; i++){
		if(index_lon[i] >= 0)
			total_tmp_y ++;

		if(index_lat[i] >= 0)
			total_tmp_x ++;

		if(index_depth[i] >= 0)
			total_tmp_z ++;
	}

	num_values_x = 2*total_tmp_x;
	num_columns_x = 2*total_tmp_x;

	num_values_y = 2*total_tmp_y;
	num_columns_y = 2*total_tmp_y;

	num_values_z = 2*total_tmp_z;
	num_columns_z = 2*total_tmp_z;


	double * Ws;
	Ws = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (Ws == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix Ws. Aborting... \n\n");
		mkl_free(Ws);
		return 1;
	}

	// Ws 1
	double center_lon, center_lat, center_depth, tmp_spacing_lon, tmp_spacing_lat, tmp_spacing_depth;
	
	for(i = 0; i < N_mod; i++){
        tmp_spacing_lon = model[i].e - model[i].w;
        tmp_spacing_lat = model[i].n - model[i].s;
        tmp_spacing_depth = model[i].top - model[i].bot;

		Ws[i] = 1*modelsmoothfun_s[i]*sqrt(tmp_spacing_lon*tmp_spacing_lat*tmp_spacing_depth*100000*100000);
	}
	// Ws 0

	// ----------------------------------------------------------------------
	// ----------------------------------------------------------------------
	// ----------------------++++++++++++++++++++++++++----------------------
	// ----------------------------------------------------------------------
	// ----------------------++++++++++++++++++++++++++----------------------
	// ----------------------------------------------------------------------
	// ----------------------------------------------------------------------
	int sum;

	double * Wx;
	double * Wy;
	double * Wz;
	MKL_INT * columns_x;
	MKL_INT * columns_y;
	MKL_INT * columns_z;

	if(num_values_x == 0){
		Wx = new double[1];
		columns_x  = new MKL_INT[1];
	}
	else{
		Wx = new double[num_values_x];
		columns_x  = new MKL_INT[num_columns_x];
	}
	
	if(num_values_y == 0){
		Wy = new double[1];
		columns_y  = new MKL_INT[1];
	}
	else{
		Wy = new double[num_values_y];
		columns_y  = new MKL_INT[num_columns_y];
	}

	if(num_values_z == 0){
		Wz = new double[1];
		columns_z  = new MKL_INT[1];
	}
	else{
		Wz = new double[num_values_z];
		columns_z  = new MKL_INT[num_columns_z];
	}
	
	MKL_INT * rowIndex_x = new MKL_INT[num_rowIndex];
	MKL_INT * rowIndex_y = new MKL_INT[num_rowIndex];	
	MKL_INT * rowIndex_z = new MKL_INT[num_rowIndex];


	sum = 0;
	rowIndex_x[0] = 0;
	if(num_values_x == 0){
		Wx[0] = 0;
		columns_x[0] = 0;
		for(i = 0; i < N_mod; i++){
			rowIndex_x[i+1] = 1;
		} 
	}
	else{
		for(i = 0; i < N_mod; i++){
			center_lon = (model[i].w + model[i].e)/2;
			center_lat = (model[i].s + model[i].n)/2;
			center_depth = (model[i].top + model[i].bot)/2;

			tmp_spacing_lon = model[i].e - model[i].w;
			tmp_spacing_lat = model[i].n - model[i].s;
			tmp_spacing_depth = model[i].top - model[i].bot;
			if(index_lat[i] >= 0){
				if(i < index_lat[i]){
					Wx[sum] = 1*modelsmoothfun_x[i]*sqrt(tmp_spacing_depth*tmp_spacing_lon/tmp_spacing_lat);
					columns_x[sum] = i;
					sum++;
					Wx[sum] = (-1)*modelsmoothfun_x[i]*sqrt(tmp_spacing_depth*tmp_spacing_lon/tmp_spacing_lat);
					columns_x[sum] = index_lat[i];
					sum++;

				}
				else if(i > index_lat[i]){
					Wx[sum] = 1*modelsmoothfun_x[i]*sqrt(tmp_spacing_depth*tmp_spacing_lon/tmp_spacing_lat);
					columns_x[sum] = index_lat[i];
					sum++;
					Wx[sum] = (-1)*modelsmoothfun_x[i]*sqrt(tmp_spacing_depth*tmp_spacing_lon/tmp_spacing_lat);
					columns_x[sum] = i;
					sum++;
				}
				else{
					cout<<endl<<"-----  Warning!!!! -----"<<endl;
					cout<<endl<<"-----  Warning!!!! -----"<<endl;
					cout<<" Wx 存在差分索引号与对应单元号 完全一致的情况，有问题，请查证！"<<endl;			
				}
			}
			rowIndex_x[i+1] = sum;
		}
	}


	sum = 0;
	rowIndex_y[0] = 0;
	if(num_values_y == 0){
		Wy[0] = 0;
		columns_y[0] = 0;
		for(i = 0; i < N_mod; i++){
			rowIndex_y[i+1] = 1;
		} 
	}
	else{
		for(i = 0; i < N_mod; i++){
			center_lon = (model[i].w + model[i].e)/2;
			center_lat = (model[i].s + model[i].n)/2;
			center_depth = (model[i].top + model[i].bot)/2;

			tmp_spacing_lon = model[i].e - model[i].w;
			tmp_spacing_lat = model[i].n - model[i].s;
			tmp_spacing_depth = model[i].top - model[i].bot;
			if(index_lon[i] >= 0){
				if(i < index_lon[i]){
					Wy[sum] = 1*modelsmoothfun_y[i]*sqrt(tmp_spacing_depth*tmp_spacing_lat/tmp_spacing_lon);
					columns_y[sum] = i;
					sum++;
					Wy[sum] = (-1)*modelsmoothfun_y[i]*sqrt(tmp_spacing_depth*tmp_spacing_lat/tmp_spacing_lon);
					columns_y[sum] = index_lon[i];
					sum++;

				}
				else if(i > index_lon[i]){
					Wy[sum] = 1*modelsmoothfun_y[i]*sqrt(tmp_spacing_depth*tmp_spacing_lat/tmp_spacing_lon);
					columns_y[sum] = index_lon[i];
					sum++;
					Wy[sum] = (-1)*modelsmoothfun_y[i]*sqrt(tmp_spacing_depth*tmp_spacing_lat/tmp_spacing_lon);
					columns_y[sum] = i;
					sum++;
				}
				else{
					cout<<endl<<"-----  Warning!!!! -----"<<endl;
					cout<<endl<<"-----  Warning!!!! -----"<<endl;
					cout<<" Wy 存在差分索引号与对应单元号 完全一致的情况，有问题，请查证！"<<endl;			
				}
			}
			rowIndex_y[i+1] = sum;
		}
	}
	
	sum = 0;
	rowIndex_z[0] = 0;
	if(num_values_z == 0){
		Wz[0] = 0;
		columns_z[0] = 0;
		for(i = 0; i < N_mod; i++){
			rowIndex_z[i+1] = 1;
		} 
	}
	else{
		for(i = 0; i < N_mod; i++){
			center_lon = (model[i].w + model[i].e)/2;
			center_lat = (model[i].s + model[i].n)/2;
			center_depth = (model[i].top + model[i].bot)/2;

			tmp_spacing_lon = model[i].e - model[i].w;
			tmp_spacing_lat = model[i].n - model[i].s;
			tmp_spacing_depth = model[i].top - model[i].bot;
			if(index_depth[i] >= 0){
				if(i < index_depth[i]){
					Wz[sum] = 1*modelsmoothfun_z[i]*sqrt(tmp_spacing_lat*tmp_spacing_lon*100000*100000/tmp_spacing_depth);
					columns_z[sum] = i;
					sum++;
					Wz[sum] = (-1)*modelsmoothfun_z[i]*sqrt(tmp_spacing_lat*tmp_spacing_lon*100000*100000/tmp_spacing_depth);
					columns_z[sum] = index_depth[i];
					sum++;

				}
				else if(i > index_depth[i]){
					Wz[sum] = 1*modelsmoothfun_z[i]*sqrt(tmp_spacing_lat*tmp_spacing_lon*100000*100000/tmp_spacing_depth);
					columns_z[sum] = index_depth[i];
					sum++;
					Wz[sum] = (-1)*modelsmoothfun_z[i]*sqrt(tmp_spacing_lat*tmp_spacing_lon*100000*100000/tmp_spacing_depth);
					columns_z[sum] = i;
					sum++;
				}
				else{
					cout<<endl<<"-----  Warning!!!! -----"<<endl;
					cout<<endl<<"-----  Warning!!!! -----"<<endl;
					cout<<" Wz 存在差分索引号与对应单元号 完全一致的情况，有问题，请查证！"<<endl;			
				}
			}
			rowIndex_z[i+1] = sum;
		}
	}


	unsigned long long N_Gsensitivity = N_data_ULL*N_mod_ULL;
	double * G_T;
	if(paraInvOrNot[0] != 0){
		G_T = new double [N_Gsensitivity];
		double cosIsinA, cosIcosA, sinI;
        double total_Bxyz;
        unsigned long long tmp_G_T;
		for(unsigned long long i = 0ULL; i < N_data_ULL; i++){
			total_Bxyz = sqrt(obs_pos_mag[i].Bx*obs_pos_mag[i].Bx + obs_pos_mag[i].By*obs_pos_mag[i].By
				+ obs_pos_mag[i].Bz*obs_pos_mag[i].Bz);
			cosIcosA = obs_pos_mag[i].Bx / total_Bxyz;
			cosIsinA = obs_pos_mag[i].By / total_Bxyz;
			sinI     = obs_pos_mag[i].Bz / total_Bxyz;

			for(unsigned long long j = 0ULL; j < N_mod_ULL; j++){
				tmp_G_T = i*N_mod_ULL + j;
				G_T[tmp_G_T] = G_THax[tmp_G_T]*cosIcosA + G_THay[tmp_G_T]*cosIsinA + G_TZa[tmp_G_T]*sinI;
			}
		}
		cout<<"The sensitivity of total-field anomaly to unit susceptibility : G_T done."<<endl<<endl;
	}
	else{
		G_T = new double [2];
		G_T[0] = 0;
		G_T[1] = 0;
		cout<<"The sensitivity of total-field anomaly to unit susceptibility : G_T done."<<endl<<endl;
	}

	double * G_TTa;
	if(paraInvOrNot[4] != 0){
		G_TTa = new double [N_Gsensitivity];
		cal_Magnetic_mag_TotalGradSen_noneffsto(N_data_ULL,N_mod_ULL,G_THax,G_THay,G_TZa,G_TTa,m_0_n);
	}
	else{
		G_TTa = new double [2];
		G_TTa[0] = 0;
		G_TTa[1] = 0;
	}

	double * G_T_2;
	if(paraInvOrNot[5] != 0){
		G_T_2 = new double [N_Gsensitivity];
		cal_Magnetic_mag_TotalFieldAnomalySen_2_noneffsto(N_data_ULL,N_mod_ULL,
			G_THax,G_THay,G_TZa,G_T_2,m_0_n,geomag_ref_x,geomag_ref_y,geomag_ref_z);
	}
	else{
		G_T_2 = new double [2];
		G_T_2[0] = 0;
		G_T_2[1] = 0;
	}
	
	
	// Calculate the deep-weight matrix: Start(20131028 14:21)
	double * W_deepw;
	W_deepw = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (W_deepw == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix W_deepw. Aborting... \n\n");
		mkl_free(W_deepw);
		return 1;
	}
	// z0_deepw beta are two parameters in calculating the deep weight matrix
	
	// = 1 general depth weighting function 
	// = 2 sensitivity based weighting function
	if(para_deepweight_type == 1){
		
		int temp_nn;
		temp_nn = 0;
		for(i = 0; i < N_mod; i++) {
			W_deepw[i] = pow(fabs(z0_deepw_real - (model[i].top+model[i].bot)/2), -beta/2);
		}
	}
	else if(para_deepweight_type == 2){
		double temp_wdeepsum;
		for(i_tmp=0ULL;i_tmp<N_mod_ULL;i_tmp++){
			temp_wdeepsum = 0;
			for(j_tmp=0ULL;j_tmp<N_data_ULL;j_tmp++){
				unsigned long long temp_index = j_tmp*N_mod_ULL+i_tmp;
				if(paraInvOrNot[0] != 0) {
					temp_wdeepsum = temp_wdeepsum + G_T[temp_index]*G_T[temp_index];
				}
			}
			W_deepw[i_tmp] = pow(temp_wdeepsum, beta/4);
		}
	}
	cout<<"Generalized depth weighting function : W_deepw done."<<endl<<endl;


	#pragma omp parallel for shared(N_data,N_mod,G_T,G_THax,G_THay,G_TZa,G_TTa,W_deepw) private(i_tmp,j_tmp,temp3)
	for(i_tmp=0ULL;i_tmp<N_data_ULL;i_tmp++){
		for(j_tmp=0ULL;j_tmp<N_mod_ULL;j_tmp++){
			temp3 = i_tmp*N_mod_ULL+j_tmp;
			int j = j_tmp;
			if(paraInvOrNot[0] != 0){
				G_T[temp3] = G_T[temp3]/W_deepw[j];
			}
		}
	}
	#pragma omp parallel for shared(N_mod,m_0_n,m_min_n,m_max_n,m_ref_n,W_deepw) private(i)
	for(i=0;i<N_mod;i++){
		m_0_n[i] = m_0_n[i]*W_deepw[i];
		m_min_n[i] = m_min_n[i]*W_deepw[i];
		m_max_n[i] = m_max_n[i]*W_deepw[i];
		m_ref_n[i] = m_ref_n[i]*W_deepw[i];
	}
	


	int internum;

	internum = 1;
	// Construct the Rd Rs Rx Ry Rz matrix: Start
	double * Rd,* Rs,* Rx,* Ry,* Rz;
	double * RdHax, * RdHay,* RdZa,* RdTa;

	double * Rd_2;

	Rs = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	Rx = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	Ry = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	Rz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
	if (Rs==NULL || Rx==NULL || Ry==NULL || Rz==NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix m_ref_n m_min_n/m_max_n. Aborting... \n\n");
		mkl_free(Rs);
		mkl_free(Rx);
		mkl_free(Ry);
		mkl_free(Rz);
		return 1;
	}
	// Rd: Start
	if(paraInvOrNot[0] != 0){
		Rd = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		double * temp_m;
		temp_m = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
	
		if (temp_m == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_m. Aborting... \n\n");
			mkl_free(temp_m);
		}
		cblas_dcopy(N_data,obs_deltaT,1,temp_m,1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,-1,G_T,N_mod,m_0_n,1,1,temp_m,1);
		double temp;
		#pragma omp parallel for shared(N_data,Wd,temp_m,p1,Rd,epsilon1) private(i,temp)
		for(i=0;i<N_data;i++) {
			temp = Wd[i]*temp_m[i];
			Rd[i] = p1[i]*pow(temp*temp+epsilon1[i]*epsilon1[i],p1[i]/2-1);
		}
		mkl_free(temp_m);
	}
	else{
		Rd = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_data,Rd) private(i)
		for(i=0;i<N_data;i++)
			Rd[0] = 0;
	}

	if(paraInvOrNot[1] != 0){
		RdHax = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		double * temp_m;
		temp_m = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
	
		if (temp_m == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_m. Aborting... \n\n");
			mkl_free(temp_m);
		}
		cblas_dcopy(N_data,obs_deltaTHax,1,temp_m,1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,-1,G_THax,N_mod,m_0_n,1,1,temp_m,1);
		double temp;
		#pragma omp parallel for shared(N_data,WdHax,temp_m,p1,RdHax,epsilon1) private(i,temp)
		for(i=0;i<N_data;i++) {
			temp = WdHax[i]*temp_m[i];
			RdHax[i] = p1[i]*pow(temp*temp+epsilon1[i]*epsilon1[i],p1[i]/2-1);
		}
		mkl_free(temp_m);
	}
	else{
		RdHax = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_data,RdHax) private(i)
		for(i=0;i<N_data;i++)
			RdHax[0] = 0;
	}
	
	if(paraInvOrNot[2] != 0){
		RdHay = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		double * temp_m;
		temp_m = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
	
		if (temp_m == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_m. Aborting... \n\n");
			mkl_free(temp_m);
		}
		cblas_dcopy(N_data,obs_deltaTHay,1,temp_m,1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,-1,G_THay,N_mod,m_0_n,1,1,temp_m,1);

		double temp;
		#pragma omp parallel for shared(N_data,WdHay,temp_m,p1,RdHay,epsilon1) private(i,temp)
		for(i=0;i<N_data;i++) {
			temp = WdHay[i]*temp_m[i];
			RdHay[i] = p1[i]*pow(temp*temp+epsilon1[i]*epsilon1[i],p1[i]/2-1);
		}

		mkl_free(temp_m);
	}
	else{
		RdHay = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_data,RdHay) private(i)
		for(i=0;i<N_data;i++)
			RdHay[0] = 0;
	}
	
	if(paraInvOrNot[3] != 0){
		RdZa = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		double * temp_m;
		temp_m = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
	
		if (temp_m == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_m. Aborting... \n\n");
			mkl_free(temp_m);
		}
		cblas_dcopy(N_data,obs_deltaTZa,1,temp_m,1);

		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,-1,G_TZa,N_mod,m_0_n,1,1,temp_m,1);

		double temp;
		#pragma omp parallel for shared(N_data,WdZa,temp_m,p1,RdZa,epsilon1) private(i,temp)
		for(i=0;i<N_data;i++) {
			temp = WdZa[i]*temp_m[i];
			RdZa[i] = p1[i]*pow(temp*temp+epsilon1[i]*epsilon1[i],p1[i]/2-1);
		}

		mkl_free(temp_m);
	}
	else{
		RdZa = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_data,RdZa) private(i)
		for(i=0;i<N_data;i++)
			RdZa[0] = 0;
	}
	
	if(paraInvOrNot[4] != 0){
		RdTa = (double *)mkl_malloc( N_data*sizeof( double ), 64 );

		double * temp_mTa1, * temp_mx1, * temp_my1, * temp_mz1;
		temp_mTa1  = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_mx1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_my1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_mz1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );

		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,G_THax,N_mod,m_0_n, 1, 0.0, temp_mx1, 1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,G_THay,N_mod,m_0_n, 1, 0.0, temp_my1, 1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,G_TZa,N_mod,m_0_n, 1, 0.0, temp_mz1, 1);
		
		#pragma omp parallel for shared(N_data,temp_mTa1,temp_mx1,temp_my1,temp_mz1,obs_deltaTTa,Txyz_weight) private(i)
		for(i = 0; i < N_data; i++){
			temp_mTa1[i] = sqrt(Txyz_weight[0]*temp_mx1[i]*temp_mx1[i] + Txyz_weight[1]*temp_my1[i]*temp_my1[i] + Txyz_weight[2]*temp_mz1[i]*temp_mz1[i]) - obs_deltaTTa[i];
		}
		mkl_free(temp_mx1);
		mkl_free(temp_my1);
		mkl_free(temp_mz1);
		double temp;
		#pragma omp parallel for shared(N_data,WdTa,temp_mTa1,p1,RdTa,epsilon1) private(i,temp)
		for(i=0;i<N_data;i++) {
			temp = WdTa[i]*temp_mTa1[i];
			RdTa[i] = p1[i]*pow(temp*temp+epsilon1[i]*epsilon1[i],p1[i]/2-1);
		}

		mkl_free(temp_mTa1);
	}
	else{
		RdTa = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_data,RdTa) private(i)
		for(i=0;i<N_data;i++)
			RdTa[0] = 0;
	}

	// 2017-12-29 16:29:29 Rd_2
	if(paraInvOrNot[5] != 0){
		Rd_2 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );

		double * temp_mTa1, * temp_mx1, * temp_my1, * temp_mz1;
		temp_mTa1  = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_mx1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_my1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		temp_mz1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );

		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,G_THax,N_mod,m_0_n, 1, 0.0, temp_mx1, 1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,G_THay,N_mod,m_0_n, 1, 0.0, temp_my1, 1);
		cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,G_TZa,N_mod,m_0_n, 1, 0.0, temp_mz1, 1);
		
		#pragma omp parallel for shared(N_data,temp_mTa1,temp_mx1,temp_my1,temp_mz1,obs_deltaTTa,Txyz_weight) private(i)
		for(i = 0; i < N_data; i++){
			temp_mTa1[i] = sqrt((temp_mx1[i]+geomag_ref_x[i])*(temp_mx1[i]+geomag_ref_x[i]) 
				+ (temp_my1[i]+geomag_ref_y[i])*(temp_my1[i]+geomag_ref_y[i]) 
				+ (temp_mz1[i]+geomag_ref_z[i])*(temp_mz1[i]+geomag_ref_z[i])) 
				- sqrt(geomag_ref_x[i]*geomag_ref_x[i] + geomag_ref_y[i]*geomag_ref_y[i] + geomag_ref_z[i]*geomag_ref_z[i])
				- obs_deltaT_2[i];
		}
		mkl_free(temp_mx1);
		mkl_free(temp_my1);
		mkl_free(temp_mz1);
		double temp;
		#pragma omp parallel for shared(N_data,Wd_2,temp_mTa1,p1,Rd_2,epsilon1) private(i,temp)
		for(i=0;i<N_data;i++) {
			temp = Wd_2[i]*temp_mTa1[i];
			Rd_2[i] = p1[i]*pow(temp*temp+epsilon1[i]*epsilon1[i],p1[i]/2-1);
		}

		mkl_free(temp_mTa1);
	}
	else{
		Rd_2 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
		#pragma omp parallel for shared(N_data,Rd_2) private(i)
		for(i=0;i<N_data;i++)
			Rd_2[0] = 0;
	}
	// Rd: End

    struct matrix_descr descrA;
	descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
	sparse_matrix_t csr_Wx;
	mkl_sparse_d_create_csr(&csr_Wx, SPARSE_INDEX_BASE_ZERO,
		N_mod, // number of rows
		N_mod, // number of cols
		rowIndex_x, rowIndex_x + 1, columns_x, Wx);
	mkl_sparse_optimize(csr_Wx);
	
	sparse_matrix_t csr_Wy;
	mkl_sparse_d_create_csr(&csr_Wy, SPARSE_INDEX_BASE_ZERO,
		N_mod, // number of rows
		N_mod, // number of cols
		rowIndex_y, rowIndex_y + 1, columns_y, Wy);
	mkl_sparse_optimize(csr_Wy);

	sparse_matrix_t csr_Wz;
	mkl_sparse_d_create_csr(&csr_Wz, SPARSE_INDEX_BASE_ZERO,
		N_mod, // number of rows
		N_mod, // number of cols
		rowIndex_z, rowIndex_z + 1, columns_z, Wz);
	mkl_sparse_optimize(csr_Wz);

	double * temp_m;
	temp_m = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );

	if (temp_m == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_m. Aborting... \n\n");
		mkl_free(temp_m);
	}
	#pragma omp parallel for shared(N_mod,m_0_n,m_ref_n,temp_m,W_deepw) private(i)
	for(i=0;i<N_mod;i++) {
		temp_m[i] = (m_0_n[i]-m_ref_n[i]);
	}


	// Rs: Start
	double temp;
	#pragma omp parallel for shared(N_mod,Ws,W_deepw,m_0_n,m_ref_n,p2,Rs,epsilon2) private(i,temp)
	for(i=0;i<N_mod;i++) {
		// double temp;
		temp = Ws[i]*temp_m[i];
		Rs[i] = p2[i]*pow(temp*temp+epsilon2[i]*epsilon2[i],p2[i]/2-1);
	}

	// Rs: End
	char transa = 'n';
	double * temp_mx;
	temp_mx = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );

	if (temp_mx == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_mx. Aborting... \n\n");
		mkl_free(temp_mx);
	}
    #pragma omp parallel for shared(N_mod,temp_mx) private(i)
    for(i=0;i<N_mod;i++) {
        temp_mx[i] = 0;
    }
	// Rx: Start
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_mx);
    // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_mx);
	#pragma omp parallel for shared(N_mod,Wx,temp_mx,columns_x,px2,Rx,epsilonx2) private(i)
	for(i=0;i<N_mod;i++) {
		Rx[i] = px2[i]*pow(temp_mx[i]*temp_mx[i]+epsilonx2[i]*epsilonx2[i],px2[i]/2-1);
	}
	mkl_free(temp_mx);
	// Rx: End


	// Ry: Start
	double * temp_my;
	temp_my = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );

	if (temp_my == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_my. Aborting... \n\n");
		mkl_free(temp_my);
	}
    #pragma omp parallel for shared(N_mod,temp_my) private(i)
    for(i=0;i<N_mod;i++) {
        temp_my[i] = 0;
    }
	// Rx: Start
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_my);
    // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_my);
	#pragma omp parallel for shared(N_mod,Wy,temp_my,columns_y,py2,Ry,epsilony2) private(i)
	for(i=0;i<N_mod;i++) {
		Ry[i] = py2[i]*pow(temp_my[i]*temp_my[i]+epsilony2[i]*epsilony2[i],py2[i]/2-1);
	}
	mkl_free(temp_my);
	// Ry: End

	// Rz: Start
	double * temp_mz;
	temp_mz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );

	if (temp_mz == NULL) {
		printf( "\n ERROR: Can't allocate memory for matrix temp_mz. Aborting... \n\n");
		mkl_free(temp_mz);
	}
    #pragma omp parallel for shared(N_mod,temp_mz) private(i)
    for(i=0;i<N_mod;i++) {
        temp_mz[i] = 0;
    }
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_mz);
    // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_mz);
	#pragma omp parallel for shared(N_mod,Wz,temp_mz,columns_z,pz2,Rz,epsilonz2) private(i)
	for(i=0;i<N_mod;i++) {
		Rz[i] = pz2[i]*pow(temp_mz[i]*temp_mz[i]+epsilonz2[i]*epsilonz2[i],pz2[i]/2-1);
	}
	mkl_free(temp_mz);
	mkl_free(temp_m);
	// Rz: End
	// Construct the Rd Rs Rx Ry Rz  matrix: End(From Line:436)

	// Inversion Beginning

	// choose a regularization parameter(use the GCV method): Begin
	
	if(withCalGCVcurve == 1){
		miu = auto_cal_RegulPara_GCV_nonlinar_noneffsto(para_deepweight_type, N_data,N_mod, G_T,Wd,obs_deltaT,
			Ws,Wx, columns_x, rowIndex_x,Wy, columns_y, rowIndex_y,Wz, columns_z, rowIndex_z,
			alpha,Rs,Rx,Ry,Rz,Rd,m_0_n,m_ref_n,m_alpha_s,m_alpha_x,m_alpha_y, m_alpha_z,
			tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);
		
		cout<<"GCV calculation : done."<<endl<<endl;
	}

	FILE * fRstLog;
	fRstLog = fopen(fileLog,"a");
	fprintf(fRstLog,"The optimal regularization parameter is 10^ %.2f \n\n",log10(miu));
	fclose(fRstLog);
	
	
	fRstLog = fopen(fileLog,"a");

	fprintf(fRstLog,"************************** ");
	fprintf(fRstLog,"************************** \n\n");
	fprintf(fRstLog,"       internum : %d \n\n",internum);

	cout<<"************************** ";
	cout<<"************************** "<<endl<<endl;
	cout<<"              internum : "<<internum<<endl<<endl;

	clock_t start,finish;
	start = clock();

	cal_Magnetic_mag_Farq_Barbosa_noneffsto(N_data,N_mod,miu,obs_deltaT,G_T,Wd,
		obs_deltaTHax,G_THax,WdHax,obs_deltaTHay,G_THay,WdHay,obs_deltaTZa,G_TZa,WdZa,obs_deltaTTa,G_TTa,WdTa,
		obs_deltaT_2, G_T_2, Wd_2, geomag_ref_x, geomag_ref_y, geomag_ref_z,
		paraInvOrNot,Ws,Wx,
		columns_x,rowIndex_x,Wy,columns_y,rowIndex_y,Wz,columns_z,rowIndex_z,W_deepw,
		alpha,m_ref_n,m_min_n,m_max_n,m_0_n,Rd,RdHax,RdHay,RdZa,RdTa,Rd_2,Rs,Rx,Ry,Rz,m_1_n,
		m_alpha_s,m_alpha_x,m_alpha_y, m_alpha_z,
		tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);

	// 开始计算phi_d phi_m部分
	mkl_free(Rs);
	mkl_free(Rx);
	mkl_free(Ry);
	mkl_free(Rz);

	mkl_free(Rd);
	mkl_free(RdHax);
	mkl_free(RdHay);
	mkl_free(RdZa);
	mkl_free(RdTa);
	mkl_free(Rd_2);
	// cout<<"	"<<lamoda1<<endl;
	
	clock_t start1,finish1;
	start1 = clock();

	cal_datamisfit_objmodfun_Barbosa_noneffsto(N_data,N_mod,miu,obs_deltaT,G_T,Wd,
		obs_deltaTHax,G_THax,WdHax,obs_deltaTHay,G_THay,WdHay,obs_deltaTZa,G_TZa,WdZa,obs_deltaTTa,G_TTa,WdTa,
		obs_deltaT_2, G_T_2, Wd_2, geomag_ref_x, geomag_ref_y, geomag_ref_z,
		paraInvOrNot,Ws,Wx,
		columns_x,rowIndex_x,Wy,columns_y,rowIndex_y,Wz,columns_z,rowIndex_z,W_deepw,
		alpha,m_ref_n,m_1_n,m_min_n,m_max_n,p1,p2,px2,py2,pz2,epsilon1,epsilon2,epsilonx2,epsilony2,epsilonz2,phi_result,
		m_alpha_s,m_alpha_x,m_alpha_y, m_alpha_z,
		tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);

	finish1 = clock();

	finish=clock();
	fprintf(fRstLog,"     datamisfit : %.12f\n\n",phi_result[0]);
	fprintf(fRstLog,"    modelobjfun : %.12f\n\n",phi_result[1]);
	fprintf(fRstLog,"     sum_objfun : %.12f\n\n",phi_result[2]);
	fprintf(fRstLog,"     time_spent : %.12f\n\n",(double)(finish-start)/CLOCKS_PER_SEC);

	cout<<"           data_misfit : "<<phi_result[0]<<endl<<endl;
	cout<<"           modelobjfun : "<<phi_result[1]<<endl<<endl;
	cout<<"            sum_objfun : "<<phi_result[2]<<endl<<endl;
	
	
	double phy12_corr; 

	FILE * fDatamisfitTime;
	fDatamisfitTime = fopen("save_DatamisfitTime.dat","a");
	fprintf(fDatamisfitTime,"num----------data_misfit--model_objective_fun---sum_datamisfit_mdf----------------Time-----------phy12_corr\n");
	fprintf(fDatamisfitTime,"%3d %20.12f %20.12f %20.12f %20.12f %20.12f\n",internum,phi_result[0],phi_result[1],
		phi_result[2],(double)(finish-start)/CLOCKS_PER_SEC,phy12_corr);
	fclose(fDatamisfitTime);


	char tempfilename[256];
	memset(tempfilename, 0, sizeof(tempfilename));
	sprintf(tempfilename,"%3d",internum);
	ofstream ftemp;
	ftemp.open(tempfilename);
	#pragma omp parallel for shared(N_mod,m_1_n,W_deepw) private(i)
	for(i=0;i<N_mod;i++){
		m_1_n[i] = m_1_n[i]/W_deepw[i];
	}

	for( i = 0; i < N_mod; i++){
		ftemp<<m_1_n[i]<<endl;
	}
	cout<<"       min_sus max_sus : "<<min_value_vector(N_mod, m_1_n,0)<< "  "<<max_value_vector(N_mod, m_1_n,0)<<endl<<endl;

	ftemp<<flush; 
	ftemp.close();
	fprintf(fRstLog,"min_sus max_sus : %.8f\t%.8f\n\n",min_value_vector(N_mod, m_1_n,0),max_value_vector(N_mod, m_1_n,0));
	fclose(fRstLog);
	// saving the temp results (End)

	#pragma omp parallel for shared(N_data_ULL,N_mod_ULL,G_T,W_deepw) private(i_tmp,j_tmp,temp3)
	for(i_tmp=0ULL;i_tmp<N_data_ULL;i_tmp++){
		for(j_tmp=0ULL;j_tmp<N_mod_ULL;j_tmp++){
			temp3 = i_tmp*N_mod_ULL+j_tmp;
			int j = j_tmp;
			if(paraInvOrNot[0] != 0){
				G_T[temp3] = G_T[temp3]*W_deepw[j];
			}
		}
	}
	#pragma omp parallel for shared(N_mod,m_0_n,m_min_n,m_max_n,m_ref_n,W_deepw) private(i)
	for(i=0;i<N_mod;i++){
		// m_0_n[i] = m_0_n[i]*W_deepw[i];
		m_min_n[i] = m_min_n[i]/W_deepw[i];
		m_max_n[i] = m_max_n[i]/W_deepw[i];
		m_ref_n[i] = m_ref_n[i]/W_deepw[i];
	}

	double bianhualiang;
	bianhualiang = 1;

	// system("PAUSE");
	double phi_dtemp = 0;
	for (int ii=0;ii<500;ii++) {
		
		cblas_dcopy(N_mod,m_1_n,1,m_0_n,1);

		if(paraInvOrNot[4] != 0){
			cal_Magnetic_mag_TotalGradSen_noneffsto(N_data_ULL,N_mod_ULL,G_THax,G_THay,G_TZa,G_TTa,m_0_n);
		}

		if(paraInvOrNot[5] != 0){
			cal_Magnetic_mag_TotalFieldAnomalySen_2_noneffsto(N_data_ULL,N_mod_ULL,
				G_THax,G_THay,G_TZa,G_T_2,m_0_n,geomag_ref_x,geomag_ref_y,geomag_ref_z);
		}

		
		#pragma omp parallel for shared(N_mod,m_0_n,W_deepw) private(i)
		for(i=0;i<N_mod;i++){
			m_0_n[i] = m_0_n[i]*W_deepw[i];
		}

		#pragma omp parallel for shared(N_data,N_mod,G_T,W_deepw) private(i_tmp,j_tmp,temp3)
		for(i_tmp=0ULL;i_tmp<N_data_ULL;i_tmp++){
			for(j_tmp=0ULL;j_tmp<N_mod_ULL;j_tmp++){
				temp3 = i_tmp*N_mod_ULL+j_tmp;
				int j = j_tmp;
				if(paraInvOrNot[0] != 0){
					G_T[temp3] = G_T[temp3]/W_deepw[j];
				}
			}
		}
		#pragma omp parallel for shared(N_mod,m_0_n,m_min_n,m_max_n,m_ref_n,W_deepw) private(i)
		for(i=0;i<N_mod;i++){
			// m_0_n[i] = m_0_n[i]*W_deepw[i];
			m_min_n[i] = m_min_n[i]*W_deepw[i];
			m_max_n[i] = m_max_n[i]*W_deepw[i];
			m_ref_n[i] = m_ref_n[i]*W_deepw[i];
		}

		// Construct the Rd Rs Rx Ry Rz matrix: Start
		// Rd: Start

		if(paraInvOrNot[0] != 0){
			Rd = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			double * temp_m;
			temp_m = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			if (temp_m == NULL) {
				printf( "\n ERROR: Can't allocate memory for matrix temp_m. Aborting... \n\n");
				mkl_free(temp_m);
			}
			cblas_dcopy(N_data,obs_deltaT,1,temp_m,1);
			cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,-1,G_T,N_mod,m_0_n,1,1,temp_m,1);

			double temp;
			#pragma omp parallel for shared(N_data,Wd,temp_m,p1,Rd,epsilon1) private(i,temp)
			for(i=0;i<N_data;i++) {
				temp = Wd[i]*temp_m[i];
				Rd[i] = p1[i]*pow(temp*temp+epsilon1[i]*epsilon1[i],p1[i]/2-1);
			}
			mkl_free(temp_m);
		}
		else{
			Rd = (double *)mkl_malloc( 1*sizeof( double ), 64 );
			Rd[0] = 0;
		}


		if(paraInvOrNot[1] != 0){
			RdHax = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			double * temp_m;
			temp_m = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			if (temp_m == NULL) {
				printf( "\n ERROR: Can't allocate memory for matrix temp_m. Aborting... \n\n");
				mkl_free(temp_m);
			}
			cblas_dcopy(N_data,obs_deltaTHax,1,temp_m,1);

			cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,-1,G_THax,N_mod,m_0_n,1,1,temp_m,1);

			#pragma omp parallel for shared(N_data,WdHax,temp_m,p1,RdHax,epsilon1) private(i,temp)
			for(i=0;i<N_data;i++) {
				temp = WdHax[i]*temp_m[i];
				RdHax[i] = p1[i]*pow(temp*temp+epsilon1[i]*epsilon1[i],p1[i]/2-1);
			}
			mkl_free(temp_m);
		}
		else{
			RdHax = (double *)mkl_malloc( 1*sizeof( double ), 64 );
			RdHax[0] = 0;
		}

		if(paraInvOrNot[2] != 0){
			RdHay = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			double * temp_m;
			temp_m = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			if (temp_m == NULL) {
				printf( "\n ERROR: Can't allocate memory for matrix temp_m. Aborting... \n\n");
				mkl_free(temp_m);
			}
			cblas_dcopy(N_data,obs_deltaTHay,1,temp_m,1);

			cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,-1,G_THay,N_mod,m_0_n,1,1,temp_m,1);

			#pragma omp parallel for shared(N_data,WdHay,temp_m,p1,RdHay,epsilon1) private(i,temp)
			for(i=0;i<N_data;i++) {
				temp = WdHay[i]*temp_m[i];
				RdHay[i] = p1[i]*pow(temp*temp+epsilon1[i]*epsilon1[i],p1[i]/2-1);
			}
			mkl_free(temp_m);
		}
		else{
			RdHay = (double *)mkl_malloc( 1*sizeof( double ), 64 );
			RdHay[0] = 0;
		}


		if(paraInvOrNot[3] != 0){
			RdZa = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			double * temp_m;
			temp_m = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			if (temp_m == NULL) {
				printf( "\n ERROR: Can't allocate memory for matrix temp_m. Aborting... \n\n");
				mkl_free(temp_m);
			}
			cblas_dcopy(N_data,obs_deltaTZa,1,temp_m,1);
			cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,-1,G_TZa,N_mod,m_0_n,1,1,temp_m,1);

			#pragma omp parallel for shared(N_data,WdZa,temp_m,p1,RdZa,epsilon1) private(i,temp)
			for(i=0;i<N_data;i++) {
				temp = WdZa[i]*temp_m[i];
				RdZa[i] = p1[i]*pow(temp*temp+epsilon1[i]*epsilon1[i],p1[i]/2-1);
			}
			mkl_free(temp_m);
		}
		else{
			RdZa = (double *)mkl_malloc( 1*sizeof( double ), 64 );
			RdZa[0] = 0;
		}


		if(paraInvOrNot[4] != 0){
			RdTa = (double *)mkl_malloc( N_data*sizeof( double ), 64 );

			double * temp_mTa1, * temp_mx1, * temp_my1, * temp_mz1;
			temp_mTa1  = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			temp_mx1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			temp_my1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			temp_mz1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );

			cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,G_THax,N_mod,m_0_n, 1, 0.0, temp_mx1, 1);
			cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,G_THay,N_mod,m_0_n, 1, 0.0, temp_my1, 1);
			cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,G_TZa,N_mod,m_0_n, 1, 0.0, temp_mz1, 1);
			
			#pragma omp parallel for shared(N_data,temp_mTa1,temp_mx1,temp_my1,temp_mz1,obs_deltaTTa,Txyz_weight) private(i)
			for(i = 0; i < N_data; i++){
				temp_mTa1[i] = sqrt(Txyz_weight[0]*temp_mx1[i]*temp_mx1[i] + Txyz_weight[1]*temp_my1[i]*temp_my1[i] + Txyz_weight[2]*temp_mz1[i]*temp_mz1[i]) - obs_deltaTTa[i];
			}
			mkl_free(temp_mx1);
			mkl_free(temp_my1);
			mkl_free(temp_mz1);

			#pragma omp parallel for shared(N_data,WdTa,temp_mTa1,p1,RdTa,epsilon1) private(i,temp)
			for(i=0;i<N_data;i++) {
				temp = WdTa[i]*temp_mTa1[i];
				RdTa[i] = p1[i]*pow(temp*temp+epsilon1[i]*epsilon1[i],p1[i]/2-1);
			}
			mkl_free(temp_mTa1);
		}
		else{
			RdTa = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			#pragma omp parallel for shared(N_data,RdTa) private(i)
			for(i=0;i<N_data;i++)
				RdTa[0] = 0;
		}	


		if(paraInvOrNot[5] != 0){
			Rd_2 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );

			double * temp_mTa1, * temp_mx1, * temp_my1, * temp_mz1;
			temp_mTa1  = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			temp_mx1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			temp_my1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			temp_mz1 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );

			cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,G_THax,N_mod,m_0_n, 1, 0.0, temp_mx1, 1);
			cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,G_THay,N_mod,m_0_n, 1, 0.0, temp_my1, 1);
			cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,G_TZa,N_mod,m_0_n, 1, 0.0, temp_mz1, 1);
			
			#pragma omp parallel for shared(N_data,temp_mTa1,temp_mx1,temp_my1,temp_mz1,obs_deltaT_2,Txyz_weight) private(i)
			for(i = 0; i < N_data; i++){
				temp_mTa1[i] = sqrt((temp_mx1[i]+geomag_ref_x[i])*(temp_mx1[i]+geomag_ref_x[i]) 
					+ (temp_my1[i]+geomag_ref_y[i])*(temp_my1[i]+geomag_ref_y[i]) 
					+ (temp_mz1[i]+geomag_ref_z[i])*(temp_mz1[i]+geomag_ref_z[i])) 
					- sqrt(geomag_ref_x[i]*geomag_ref_x[i] + geomag_ref_y[i]*geomag_ref_y[i] 
						+ geomag_ref_z[i]*geomag_ref_z[i])
					- obs_deltaT_2[i];
			}
			mkl_free(temp_mx1);
			mkl_free(temp_my1);
			mkl_free(temp_mz1);

			#pragma omp parallel for shared(N_data,Wd_2,temp_mTa1,p1,Rd_2,epsilon1) private(i,temp)
			for(i=0;i<N_data;i++) {
				temp = Wd_2[i]*temp_mTa1[i];
				Rd_2[i] = p1[i]*pow(temp*temp+epsilon1[i]*epsilon1[i],p1[i]/2-1);
			}
			mkl_free(temp_mTa1);
		}
		else{
			Rd_2 = (double *)mkl_malloc( N_data*sizeof( double ), 64 );
			#pragma omp parallel for shared(N_data,Rd_2) private(i)
			for(i=0;i<N_data;i++)
				Rd_2[0] = 0;
		}
		// Rd: End


		double * temp_m;
		temp_m = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (temp_m == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_m. Aborting... \n\n");
			mkl_free(temp_m);
		}
		#pragma omp parallel for shared(N_mod,m_0_n,m_ref_n,temp_m) private(i)
		for(i=0;i<N_mod;i++) {
			temp_m[i] = (m_0_n[i]-m_ref_n[i]);
		}

		Rs = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		Rx = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		Ry = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		Rz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (Rs==NULL || Rx==NULL || Ry==NULL || Rz==NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix m_ref_n m_min_n/m_max_n. Aborting... \n\n");
			mkl_free(Rs);
			mkl_free(Rx);
			mkl_free(Ry);
			mkl_free(Rz);
			return 1;
		}
		// Rs: Start
		#pragma omp parallel for shared(N_mod,Ws,W_deepw,m_0_n,m_ref_n,p2,Rs,epsilon2) private(i,temp)
		for(i=0;i<N_mod;i++) {
			temp = Ws[i]*temp_m[i];
			Rs[i] = p2[i]*pow(temp*temp+epsilon2[i]*epsilon2[i],p2[i]/2-1);
		}
		// Rs: End

		
		char transa = 'n';

		temp_mx = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );

		if (temp_mx == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_mx. Aborting... \n\n");
			mkl_free(temp_mx);
		}
        #pragma omp parallel for shared(N_mod,temp_mx) private(i)
        for(i=0;i<N_mod;i++) {
            temp_mx[i] = 0;
        }
		// Rx: Start
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_mx);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_mx);
		#pragma omp parallel for shared(N_mod,Wx,temp_mx,columns_x,px2,Rx,epsilonx2) private(i)
		for(i=0;i<N_mod;i++) {
			Rx[i] = px2[i]*pow(temp_mx[i]*temp_mx[i]+epsilonx2[i]*epsilonx2[i],px2[i]/2-1);
		}
		mkl_free(temp_mx);

		// Rx: End


		// Ry: Start
		temp_my = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );

		if (temp_my == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_my. Aborting... \n\n");
			mkl_free(temp_my);
		}
        #pragma omp parallel for shared(N_mod,temp_my) private(i)
        for(i=0;i<N_mod;i++) {
            temp_my[i] = 0;
        }
		// Rx: Start
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_my);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_my);
		#pragma omp parallel for shared(N_mod,Wy,temp_my,columns_y,py2,Ry,epsilony2) private(i)
		for(i=0;i<N_mod;i++) {
			Ry[i] = py2[i]*pow(temp_my[i]*temp_my[i]+epsilony2[i]*epsilony2[i],py2[i]/2-1);
		}
		mkl_free(temp_my);
		// Ry: End

		// Rz: Start
		temp_mz = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );

		if (temp_mz == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix temp_mz. Aborting... \n\n");
			mkl_free(temp_mz);
		}
        #pragma omp parallel for shared(N_mod,temp_mz) private(i)
        for(i=0;i<N_mod;i++) {
            temp_mz[i] = 0;
        }
		// Rx: Start
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_mz);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_mz);
		#pragma omp parallel for shared(N_mod,Wz,temp_mz,columns_z,pz2,Rz,epsilonz2) private(i)
		for(i=0;i<N_mod;i++) {
			Rz[i] = pz2[i]*pow(temp_mz[i]*temp_mz[i]+epsilonz2[i]*epsilonz2[i],pz2[i]/2-1);
		}
		mkl_free(temp_mz);
		mkl_free(temp_m);
		// Rx: End

		// Construct the Rd Rs Rx Ry Rz  matrix: End(From Line:436)
		start = clock();
		clock_t start1,finish1;
		start1 = clock();

		cal_Magnetic_mag_Farq_Barbosa_noneffsto(N_data,N_mod,miu,obs_deltaT,G_T,Wd,
			obs_deltaTHax,G_THax,WdHax,obs_deltaTHay,G_THay,WdHay,obs_deltaTZa,G_TZa,WdZa,obs_deltaTTa,G_TTa,WdTa,
			obs_deltaT_2, G_T_2, Wd_2, geomag_ref_x, geomag_ref_y, geomag_ref_z,
			paraInvOrNot,Ws,Wx,
			columns_x,rowIndex_x,Wy,columns_y,rowIndex_y,Wz,columns_z,rowIndex_z,W_deepw,
			alpha,m_ref_n,m_min_n,m_max_n,m_0_n,Rd,RdHax,RdHay,RdZa,RdTa,Rd_2,Rs,Rx,Ry,Rz,m_1_n,
			m_alpha_s,m_alpha_x,m_alpha_y, m_alpha_z,
			tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);

		finish1 = clock();
		mkl_free(Rs);
		mkl_free(Rx);
		mkl_free(Ry);
		mkl_free(Rz);

		mkl_free(Rd);
		mkl_free(RdHax);
		mkl_free(RdHay);
		mkl_free(RdZa);
		mkl_free(RdTa);
		mkl_free(Rd_2);
		internum ++;

		cout<<"************************** ";
		cout<<"************************** "<<endl<<endl;
		cout<<"              internum : "<<internum<<endl<<endl;
		cout<<"            time_spent : "<<(double)(finish1-start1)/CLOCKS_PER_SEC<<endl<<endl;


		
		FILE * fRstLog;
		fRstLog = fopen(fileLog,"a");
		fprintf(fRstLog,"************************** ");
		fprintf(fRstLog,"************************** \n\n");
		fprintf(fRstLog,"       internum : %d \n\n",internum);


		start1 = clock();

		cal_datamisfit_objmodfun_Barbosa_noneffsto(N_data,N_mod,miu,obs_deltaT,G_T,Wd,
			obs_deltaTHax,G_THax,WdHax,obs_deltaTHay,G_THay,WdHay,obs_deltaTZa,G_TZa,WdZa,obs_deltaTTa,G_TTa,WdTa,
			obs_deltaT_2, G_T_2, Wd_2, geomag_ref_x, geomag_ref_y, geomag_ref_z,
			paraInvOrNot,Ws,Wx,
			columns_x,rowIndex_x,Wy,columns_y,rowIndex_y,Wz,columns_z,rowIndex_z,W_deepw,
			alpha,m_ref_n,m_1_n,m_min_n,m_max_n,p1,p2,px2,py2,pz2,epsilon1,epsilon2,epsilonx2,epsilony2,epsilonz2,phi_result,
			m_alpha_s,m_alpha_x,m_alpha_y, m_alpha_z,
			tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);

		finish1 = clock();

		finish=clock();

		cout<<"           data_misfit : "<<phi_result[0]<<endl<<endl;
		cout<<"           modelobjfun : "<<phi_result[1]<<endl<<endl;
		cout<<"            sum_objfun : "<<phi_result[2]<<endl<<endl;

		fprintf(fRstLog,"     datamisfit : %.12f\n\n",phi_result[0]);
		fprintf(fRstLog,"    modelobjfun : %.12f\n\n",phi_result[1]);
		fprintf(fRstLog,"     sum_objfun : %.12f\n\n",phi_result[2]);
		fprintf(fRstLog,"     time_spent : %.12f\n\n",(double)(finish-start)/CLOCKS_PER_SEC);
		

		FILE * fDatamisfitTime;
		fDatamisfitTime = fopen("save_DatamisfitTime.dat","a");
		fprintf(fDatamisfitTime,"%3d %20.12f %20.12f %20.12f %20.12f %20.12f\n",internum,phi_result[0],phi_result[1],
			phi_result[2],(double)(finish-start)/CLOCKS_PER_SEC,phy12_corr);
		fclose(fDatamisfitTime);
		
		double phi_derr;
		phi_derr = fabs(phi_result[2] - phi_dtemp)/phi_result[2];

		cout<<"       sum_objfun_diff : "<<phi_derr*100<<" (<= 0.5 %)"<<endl<<endl;

		
		#pragma omp parallel for shared(N_mod,m_1_n,m_0_n,W_deepw) private(i)
		for(i=0;i<N_mod;i++){
			m_1_n[i] = m_1_n[i]/W_deepw[i];
			m_0_n[i] = m_0_n[i]/W_deepw[i];
		}
		#pragma omp parallel for shared(N_data,N_mod,G_T,W_deepw) private(i_tmp,j_tmp,temp3)
		for(i_tmp=0ULL;i_tmp<N_data_ULL;i_tmp++){
			for(j_tmp=0ULL;j_tmp<N_mod_ULL;j_tmp++){
				temp3 = i_tmp*N_mod_ULL+j_tmp;
				int j = j_tmp;
				if(paraInvOrNot[0] != 0){
					G_T[temp3] = G_T[temp3]*W_deepw[j];
				}
			}
		}

		#pragma omp parallel for shared(N_mod,m_ref_n,m_min_n,m_max_n,W_deepw) private(i)
		for(i=0;i<N_mod;i++){
			m_ref_n[i] = m_ref_n[i]/W_deepw[i];
			m_min_n[i] = m_min_n[i]/W_deepw[i];
			m_max_n[i] = m_max_n[i]/W_deepw[i];
		}
		
		if (internum%1 == 0){
			char tempfilename[256];
			memset(tempfilename, 0, sizeof(tempfilename));
			sprintf(tempfilename,"%3d",internum);
			ofstream ftemp;
			ftemp.open(tempfilename);

			
			for( i = 0; i < N_mod; i++){
				ftemp<<m_1_n[i]<<endl;
			}
			
			ftemp<<flush; 
			ftemp.close();
		}

		double * m_bianhua_n;
		m_bianhua_n = (double *)mkl_malloc( N_mod*sizeof( double ), 64 );
		if (m_bianhua_n == NULL) {
			printf( "\n ERROR: Can't allocate memory for matrix m_bianhua_n. Aborting... \n\n");
			mkl_free(m_bianhua_n);
		}
		//cblas_daxpy(N_mod,-1,m_0_n,1,m_1_n,1);
		#pragma omp parallel for shared(N_mod,m_bianhua_n,m_1_n,m_0_n) private(i)
		for(i=0;i<N_mod;i++)
			m_bianhua_n[i] = abs(m_1_n[i] - m_0_n[i]);
		
		bianhualiang = max_value_vector(N_mod,m_bianhua_n,0);

		cout<<"        sus_model_diff : "<<bianhualiang<<" (<= 0.001 SI)"<<endl<<endl;
		cout<<"       min_sus max_sus : "<<min_value_vector(N_mod, m_1_n,0)<< "  "<<max_value_vector(N_mod, m_1_n,0)<<" SI"<<endl<<endl;

		fprintf(fRstLog,"min_sus max_sus : %.8f\t%.8f\n\n",max_value_vector(N_mod, m_1_n,0),min_value_vector(N_mod, m_1_n,0));
		fclose(fRstLog);
		
		if(bianhualiang<0.001 && phi_derr < 0.005){
		//if(phi_log<0.01 && phi_derr < 0.01){
			mkl_free(m_bianhua_n);
			break;
		}
		phi_dtemp = phi_result[2];
		mkl_free(m_bianhua_n);

	}

    mkl_sparse_destroy(csr_Wx);
	mkl_sparse_destroy(csr_Wy);
	mkl_sparse_destroy(csr_Wz);
    
	return log10(miu);
}