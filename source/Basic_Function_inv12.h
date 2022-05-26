#include <fstream>
#include <math.h>
#include <iomanip>
#include <ctime>
#include <iostream>
#include <string.h>
#include <map>
#include <omp.h>
#include <stdlib.h>
#include <mkl_blas.h>
#include <mkl_spblas.h>
#include "mkl.h"

#define eps1 0.00000000000000001
#define random(x) (rand()%x)
double Txyz_weight[3] = {1,1,1};
using namespace std;

// quickly save a double vector for debugging
void SaveFile_double (char * tempfilename, int num, double * result)
{
	ofstream ftemp;
	ftemp.open(tempfilename);
	
	for(int i=0;i<num;i++){
		ftemp<<result[i]<<endl;
	}
	ftemp<<flush; 
	ftemp.close();
}

// quickly save a int vector for debugging
void SaveFile_int (char * tempfilename, int num, int * result)
{
	ofstream ftemp;
	ftemp.open(tempfilename);
	
	for(int i=0;i<num;i++){
		ftemp<<result[i]<<endl;
	}
	ftemp<<flush; 
	ftemp.close();
}

// the logarithmic transformation related functions (Barbosa et al., 1999) : 1
void cal_tran_m(int N_mod, double * m, double * m_min_n,double * m_max_n,char * transtype, double * result)
{
	if(strcmp(transtype,"forw") == 0) {
		int i;
		#pragma omp parallel for shared(result,N_mod,m_max_n,m,m_min_n) private(i)
		for(i=0;i<N_mod;i++)
			result[i] = (-1)*log((m_max_n[i]-m[i])/(m[i]-m_min_n[i]+eps1));
	}
	else if(strcmp(transtype,"back") == 0) {
		int i;
		#pragma omp parallel for shared(result,N_mod,m_max_n,m,m_min_n) private(i)
		for(i=0;i<N_mod;i++)
			result[i] = (m_max_n[i] - m_min_n[i])/(1 + exp(-m[i])+eps1) + m_min_n[i];
	}
}

void cal_diag_mtx(int N_mod, double * m,double * m_min,double * m_max,double * T)
{
	int i;
	#pragma omp parallel for shared(T,N_mod,m_max,m,m_min) private(i)
	for(i=0;i<N_mod;i++)
		T[i] = ((m_max[i]-m[i])*(m[i]-m_min[i])+m_max[i]/5)/(m_max[i]-m_min[i]);
}

void cal_diag_mtx2(int N_mod, double * m,double * m_min,double * m_max,double * T,double * Wdeepw)
{
	int i;
	#pragma omp parallel for shared(T,N_mod,m_max,m,m_min) private(i)
	for(i=0;i<N_mod;i++)
		T[i] = ((m_max[i]-m[i])*(m[i]-m_min[i])+m_max[i]*Wdeepw[i]/5)/(m_max[i]-m_min[i]);
}
// the logarithmic transformation related functions (Barbosa et al., 1999) : 2

// correlation coefficient between two physical property models
double cal_phys_corr(int n_mod,double * phy1,double * phy2)
{
	double phy1_sum,phy2_sum;
    double phy1_mean,phy2_mean;

    phy1_sum = 0;
    phy2_sum = 0;

	int i;
    for(i=0;i<n_mod;i++){
        phy1_sum = phy1_sum + phy1[i];
        phy2_sum = phy2_sum + phy2[i];
    }
    phy1_mean = phy1_sum/n_mod;
	phy2_mean = phy2_sum/n_mod;

	double temp_up,temp_dn1,temp_dn2;
	temp_up = 0;
	temp_dn1 = 0;
	temp_dn2 = 0;

	// #pragma omp parallel for shared(n_mod) private(i)
	for(i=0;i<n_mod;i++){
		temp_up += (phy1[i] - phy1_mean)*(phy2[i] - phy2_mean);
		temp_dn1 += (phy1[i] - phy1_mean)*(phy1[i] - phy1_mean);
		temp_dn2 += (phy2[i] - phy2_mean)*(phy2[i] - phy2_mean);
	}
	double phy12_corr;
	phy12_corr = temp_up/sqrt(temp_dn1*temp_dn2);
	
	return phy12_corr;
}


// Euclidean Distance between two physical property models
double cal_phys_EuclideanDist(int n_mod, double * phy1, double * phy2)
{
	double result_EculDist = 0;
	for(int i = 0; i < n_mod; ++i) {
		result_EculDist = result_EculDist + (phy1[i] - phy2[i])*(phy1[i] - phy2[i]);
	}
	// result_EculDist = sqrt(result_EculDist);

	return sqrt(result_EculDist);
}


// Manhattan Distance between two physical property models
double cal_phys_ManhattanDist(int n_mod, double * phy1, double * phy2)
{
	double result_ManhDist = 0;
	for(int i = 0; i < n_mod; ++i) {
		result_ManhDist = result_ManhDist + fabs(phy1[i] - phy2[i]);
	}

	return result_ManhDist;
}

// Cosine angle between two physical property models
double cal_phys_cos(int n_mod, double * phy1, double * phy2)
{
	double temp_up,temp_dn1,temp_dn2;
	temp_up = 0;
	temp_dn1 = 0;
	temp_dn2 = 0;
	int i;
	for(i=0;i<n_mod;i++){
		temp_up += phy1[i]*phy2[i];
		temp_dn1 += phy1[i]*phy1[i];
		temp_dn2 += phy2[i]*phy2[i];
	}

	return temp_up/sqrt(temp_dn1*temp_dn2);
}


// calculating the total gradient sensitivity or total amplitude of anamalous vector sensitivity to the susceptibility
int cal_Magnetic_mag_TotalGradSen_noneffsto(unsigned long long N_data_ULL,unsigned long long N_mod_ULL,
	double * Gx,double * Gy,double * Gz,double * Gxyz,double * m)
{
	int N_data = N_data_ULL;
	int N_mod = N_mod_ULL;
	int i,j;
	double * deltaTx = new double[N_data];
	double * deltaTy = new double[N_data];
	double * deltaTz = new double[N_data];
	double * deltaTxyz = new double[N_data];

	cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,Gx,N_mod,m, 1, 0.0, deltaTx, 1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,Gy,N_mod,m, 1, 0.0, deltaTy, 1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,Gz,N_mod,m, 1, 0.0, deltaTz, 1);

	unsigned long long temp_num;
	unsigned long long i_tmp, j_tmp;

	for(i_tmp=0ULL;i_tmp<N_data_ULL;i_tmp++){
		deltaTxyz[i_tmp] = sqrt(Txyz_weight[0]*deltaTx[i_tmp]*deltaTx[i_tmp] + Txyz_weight[1]*deltaTy[i_tmp]*deltaTy[i_tmp] 
			+ Txyz_weight[2]*deltaTz[i_tmp]*deltaTz[i_tmp]);
		#pragma omp parallel for shared(i_tmp,N_mod_ULL,deltaTxyz,deltaTx,deltaTy,deltaTz,Gxyz,Gx,Gy,Gz) private(j_tmp,temp_num)
		for(j_tmp=0ULL;j_tmp<N_mod_ULL;j_tmp++){
			temp_num = i_tmp*N_mod_ULL+j_tmp;
			Gxyz[temp_num] = (Txyz_weight[0]*Gx[temp_num]*deltaTx[i_tmp] + Txyz_weight[1]*Gy[temp_num]*deltaTy[i_tmp] 
				+ Txyz_weight[2]*Gz[temp_num]*deltaTz[i_tmp])/deltaTxyz[i_tmp];
		}
	}
	delete [] deltaTx;
	delete [] deltaTy;
	delete [] deltaTz;
	delete [] deltaTxyz;

	return 0;
}

// calculating the total-field anomaly sensitivity to the susceptibility based on its obtaining process
int cal_Magnetic_mag_TotalFieldAnomalySen_2_noneffsto(unsigned long long N_data_ULL,unsigned long long N_mod_ULL,
	double * Gx,double * Gy,double * Gz,double * Gxyz,double * m,
	double * geomag_ref_x, double * geomag_ref_y, double * geomag_ref_z)
{
	int N_data = N_data_ULL;
	int N_mod = N_mod_ULL;
	int i,j;
	double * deltaTx = new double[N_data];
	double * deltaTy = new double[N_data];
	double * deltaTz = new double[N_data];
	double * deltaTxyz = new double[N_data];

	cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,Gx,N_mod,m, 1, 0.0, deltaTx, 1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,Gy,N_mod,m, 1, 0.0, deltaTy, 1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,Gz,N_mod,m, 1, 0.0, deltaTz, 1);

	unsigned long long temp_num;
	unsigned long long i_tmp, j_tmp;

	for(i_tmp=0ULL;i_tmp<N_data_ULL;i_tmp++){
		deltaTxyz[i_tmp] = sqrt((deltaTx[i_tmp]+geomag_ref_x[i_tmp])*(deltaTx[i_tmp]+geomag_ref_x[i_tmp]) 
			+ (deltaTy[i_tmp]+geomag_ref_y[i_tmp])*(deltaTy[i_tmp]+geomag_ref_y[i_tmp]) 
			+ (deltaTz[i_tmp]+geomag_ref_z[i_tmp])*(deltaTz[i_tmp]+geomag_ref_z[i_tmp]));

		#pragma omp parallel for shared(i_tmp,N_mod,deltaTxyz,deltaTx,deltaTy,deltaTz,Gxyz,Gx,Gy,Gz) private(j_tmp,temp_num)
		for(j_tmp=0ULL;j_tmp<N_mod_ULL;j_tmp++){
			temp_num = i_tmp*N_mod_ULL+j_tmp;
			Gxyz[temp_num] = (Gx[temp_num]*(deltaTx[i_tmp]+geomag_ref_x[i_tmp]) 
				+ Gy[temp_num]*(deltaTy[i_tmp]+geomag_ref_y[i_tmp]) 
				+ Gz[temp_num]*(deltaTz[i_tmp]+geomag_ref_z[i_tmp]))/deltaTxyz[i_tmp];
		}
	}

	delete [] deltaTx;
	delete [] deltaTy;
	delete [] deltaTz;
	delete [] deltaTxyz;

	return 0;
}

// calculating the GCV value for a determined regularization paramter
double cal_RegulPara_GCV_nonlinar_noneffsto(double miu, int N_data, int N_mod, double * G,double * Wd,double * deltaT,
	double * Ws,double * Wx,MKL_INT * columns_x,MKL_INT * rowIndex_x,
	double * Wy,MKL_INT * columns_y,MKL_INT * rowIndex_y,
	double * Wz,MKL_INT * columns_z,MKL_INT * rowIndex_z,
	double * alpha,double * Rs,double * Rx,double * Ry,double * Rz,double * Rd,
	double * m_0_n,double * m_ref_n,
	double * m_alpha_s, double * m_alpha_x,double * m_alpha_y, double * m_alpha_z,
	double * tran_r11,double * tran_r12,double * tran_r13,double * tran_r21,
	double * tran_r22,double * tran_r23,double * tran_r31,double * tran_r32,double * tran_r33)
{
	int i;

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

	double * temp_m = new double [N_mod];
	#pragma omp parallel for shared(N_mod,m_0_n,m_ref_n,temp_m) private(i)
	for(i=0;i<N_mod;i++) {
		temp_m[i] = (m_0_n[i]-m_ref_n[i]);
	}

	double * temp_s = new double[N_mod];
	double * temp_x = new double[N_mod];
	double * temp_y = new double[N_mod];
	double * temp_z = new double[N_mod];

	double * temp_xy = new double[N_mod];
	double * temp_yx = new double[N_mod];
	double * temp_zx = new double[N_mod];
	double * temp_xz = new double[N_mod];
	double * temp_yz = new double[N_mod];
	double * temp_zy = new double[N_mod];

	#pragma omp parallel for shared(N_mod,temp_s,miu,alpha,Ws,Rs,temp_m) private(i)
	for(i=0;i<N_mod;i++){
		temp_s[i] = miu*m_alpha_s[i]*Ws[i]*Rs[i]*Ws[i]*temp_m[i];
        temp_x[i] = 0;
        temp_y[i] = 0;
        temp_z[i] = 0;

        temp_xy[i] = 0;
        temp_yx[i] = 0;
        temp_zx[i] = 0;
        temp_xz[i] = 0;
        temp_yz[i] = 0;
        temp_zy[i] = 0;
    }

	char transa = 'n';
	double * temp_x10 = new double [N_mod];
    #pragma omp parallel for shared(N_mod,temp_x10) private(i)
	for(i=0;i<N_mod;i++) {
		temp_x10[i] = 0;
	}
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_x10);
	// mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_x10); the sparse matrix multiplying vector function for intel 2015
	#pragma omp parallel for shared(N_mod,temp_x10,Rx) private(i)
	for(i=0;i<N_mod;i++)
		temp_x10[i] = Rx[i]*temp_x10[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_x10, 0.0, temp_x);
    // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_x10,temp_x);
	#pragma omp parallel for shared(N_mod,temp_x,miu,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_x[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r11[i] + m_alpha_y[i]*tran_r21[i]*tran_r21[i] + m_alpha_z[i]*tran_r31[i]*tran_r31[i])*temp_x[i];
	delete [] temp_x10;

	transa = 'n';
	double * temp_y1 = new double [N_mod];
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
	#pragma omp parallel for shared(N_mod,temp_y,miu,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_y[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r12[i] + m_alpha_y[i]*tran_r22[i]*tran_r22[i] + m_alpha_z[i]*tran_r32[i]*tran_r32[i])*temp_y[i];
	delete [] temp_y1;
	
	
	transa = 'n';
	double * temp_z1 = new double [N_mod];
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
	#pragma omp parallel for shared(N_mod,temp_z,miu,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_z[i] = miu*(m_alpha_x[i]*tran_r13[i]*tran_r13[i] + m_alpha_y[i]*tran_r23[i]*tran_r23[i] + m_alpha_z[i]*tran_r33[i]*tran_r33[i])*temp_z[i];
	delete [] temp_z1;

	transa = 'n';
	double * temp_z1_xy = new double [N_mod];
    #pragma omp parallel for shared(N_mod,temp_z1_xy) private(i)
	for(i=0;i<N_mod;i++) {
		temp_z1_xy[i] = 0;
	}
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_z1_xy);
    // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_z1_xy);
	#pragma omp parallel for shared(N_mod,temp_z1_xy,Rz) private(i)
	for(i=0;i<N_mod;i++)
		temp_z1_xy[i] = sqrt(Rx[i])*sqrt(Ry[i])*temp_z1_xy[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_z1_xy, 0.0, temp_xy);
    // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_z1_xy,temp_xy);
	#pragma omp parallel for shared(N_mod,temp_xy,miu,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_xy[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r12[i] + m_alpha_y[i]*tran_r21[i]*tran_r22[i] + m_alpha_z[i]*tran_r31[i]*tran_r32[i])*temp_xy[i];
	delete [] temp_z1_xy;



	transa = 'n';
	double * temp_z1_yx = new double [N_mod];
    #pragma omp parallel for shared(N_mod,temp_z1_yx) private(i)
	for(i=0;i<N_mod;i++) {
		temp_z1_yx[i] = 0;
	}
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_z1_yx);
    // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_z1_yx);
	#pragma omp parallel for shared(N_mod,temp_z1_yx,Rz) private(i)
	for(i=0;i<N_mod;i++)
		temp_z1_yx[i] = sqrt(Rx[i])*sqrt(Ry[i])*temp_z1_yx[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wy, descrA, temp_z1_yx, 0.0, temp_yx);
    // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_z1_yx,temp_yx);
	#pragma omp parallel for shared(N_mod,temp_yx,miu,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_yx[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r12[i] + m_alpha_y[i]*tran_r21[i]*tran_r22[i] + m_alpha_z[i]*tran_r31[i]*tran_r32[i])*temp_yx[i];
	delete [] temp_z1_yx;

	transa = 'n';
	double * temp_z1_zx = new double [N_mod];
	#pragma omp parallel for shared(N_mod,temp_z1_zx) private(i)
	for(i=0;i<N_mod;i++) {
		temp_z1_zx[i] = 0;
	}
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_z1_zx);
    // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_z1_zx);
	#pragma omp parallel for shared(N_mod,temp_z1_zx,Rz) private(i)
	for(i=0;i<N_mod;i++)
		temp_z1_zx[i] = sqrt(Rz[i])*sqrt(Rx[i])*temp_z1_zx[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wz, descrA, temp_z1_zx, 0.0, temp_zx);
    // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_z1_zx,temp_zx);
	#pragma omp parallel for shared(N_mod,temp_zx,miu,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_zx[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r13[i] + m_alpha_y[i]*tran_r21[i]*tran_r23[i] + m_alpha_z[i]*tran_r31[i]*tran_r33[i])*temp_zx[i];
	delete [] temp_z1_zx;

	transa = 'n';
	double * temp_z1_xz = new double [N_mod];
	#pragma omp parallel for shared(N_mod,temp_z1_xz) private(i)
	for(i=0;i<N_mod;i++) {
		temp_z1_xz[i] = 0;
	}
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_z1_xz);
    // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_z1_xz);
	#pragma omp parallel for shared(N_mod,temp_z1_xz,Rz) private(i)
	for(i=0;i<N_mod;i++)
		temp_z1_xz[i] = sqrt(Rz[i])*sqrt(Rx[i])*temp_z1_xz[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_z1_xz, 0.0, temp_xz);
    // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_z1_xz,temp_xz);
	#pragma omp parallel for shared(N_mod,temp_xz,miu,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_xz[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r13[i] + m_alpha_y[i]*tran_r21[i]*tran_r23[i] + m_alpha_z[i]*tran_r31[i]*tran_r33[i])*temp_xz[i];
	delete [] temp_z1_xz;


	transa = 'n';
	double * temp_z1_yz = new double [N_mod];
	#pragma omp parallel for shared(N_mod,temp_z1_yz) private(i)
	for(i=0;i<N_mod;i++) {
		temp_z1_yz[i] = 0;
	}
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_z1_yz);
    // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_z1_yz);
	#pragma omp parallel for shared(N_mod,temp_z1_yz,Rz) private(i)
	for(i=0;i<N_mod;i++)
		temp_z1_yz[i] = sqrt(Rz[i])*sqrt(Ry[i])*temp_z1_yz[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wy, descrA, temp_z1_yz, 0.0, temp_yz);
    // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_z1_yz,temp_yz);
	#pragma omp parallel for shared(N_mod,temp_yz,miu,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_yz[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r13[i] + m_alpha_y[i]*tran_r22[i]*tran_r23[i] + m_alpha_z[i]*tran_r32[i]*tran_r33[i])*temp_yz[i];
	delete [] temp_z1_yz;

	transa = 'n';
	double * temp_z1_zy = new double [N_mod];
	#pragma omp parallel for shared(N_mod,temp_z1_zy) private(i)
	for(i=0;i<N_mod;i++) {
		temp_z1_zy[i] = 0;
	}
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_z1_zy);
    // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_z1_zy);
	#pragma omp parallel for shared(N_mod,temp_z1_zy,Rz) private(i)
	for(i=0;i<N_mod;i++)
		temp_z1_zy[i] = sqrt(Rz[i])*sqrt(Ry[i])*temp_z1_zy[i];
	transa = 't';
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wz, descrA, temp_z1_zy, 0.0, temp_zy);
    // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_z1_zy,temp_zy);
	#pragma omp parallel for shared(N_mod,temp_zy,miu,alpha) private(i)
	for(i=0;i<N_mod;i++)
		temp_zy[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r13[i] + m_alpha_y[i]*tran_r22[i]*tran_r23[i] + m_alpha_z[i]*tran_r32[i]*tran_r33[i])*temp_zy[i];
	delete [] temp_z1_zy;


	double * temp_d1 = new double [N_data];
	double * temp_dobs_dpre = new double [N_data];
	cblas_dcopy(N_data,deltaT,1,temp_d1,1);
	cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,G,N_mod,m_0_n,1,-1,temp_d1,1);

	#pragma omp parallel for shared(N_data,temp_d1,Rd,Wd,temp_dobs_dpre) private(i)
	for(i=0;i<N_data;i++){
		temp_dobs_dpre[i] = (-1)*temp_d1[i];
		temp_d1[i] = Wd[i]*Rd[i]*Wd[i]*temp_d1[i];
	}

	double * temp_d = new double [N_mod];
	#pragma omp parallel for shared(N_mod,temp_d) private(i)
	for(i=0;i<N_mod;i++)
		temp_d[i] = 0;
	cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,G,N_mod,temp_d1,1,0,temp_d,1);
	delete [] temp_d1;

	double * B = new double[N_mod];
	#pragma omp parallel for shared(N_mod,B) private(i)
	for(i=0;i<N_mod;i++){
		B[i] = (-1)*(temp_s[i]+temp_x[i]+temp_y[i]+temp_z[i]+temp_d[i]
			+temp_xy[i]+temp_yx[i]+temp_yz[i]+temp_zy[i]+temp_xz[i]+temp_zx[i]);
	}
	// delete [] temp_up1;
	delete [] temp_m;
	delete [] temp_s;
	delete [] temp_x;
	delete [] temp_y;
	delete [] temp_z;
	delete [] temp_d;
	delete [] temp_xy;
	delete [] temp_yx;
	delete [] temp_yz;
	delete [] temp_zy;
	delete [] temp_xz;
	delete [] temp_zx;


	double * X0 = new double[N_mod];
	double * X1 = new double[N_mod];
	#pragma omp parallel for shared(N_mod,X0) private(i)
	for(i=0;i<N_mod;i++)
		X0[i] = 0;
	double t_stop = 0.0001;
	// Conjugate Gradient method
	double * R0 = new double[N_mod]; 
	double * P0 = new double[N_mod];
	#pragma omp parallel for shared(N_mod,R0,P0,B) private(i)
	for(i=0;i<N_mod;i++) {
		R0[i] = (-1)*B[i];
		P0[i] = (-1)*R0[i];
	}

	for(int j = 1;j<N_data;j++){
		double alpha_k;
		double temp1,temp2;

		double * AP0 = new double[N_mod];
		double * temp_m = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_m,P0) private(i)
		for(i=0;i<N_mod;i++)
			temp_m[i] = P0[i];

		double * temp_s = new double [N_mod];
		double * temp_x = new double [N_mod];
		double * temp_y = new double [N_mod];
		double * temp_z = new double [N_mod];

		double * temp_xy = new double[N_mod];
		double * temp_yx = new double[N_mod];
		double * temp_zx = new double[N_mod];
		double * temp_xz = new double[N_mod];
		double * temp_yz = new double[N_mod];
		double * temp_zy = new double[N_mod];

		#pragma omp parallel for shared(N_mod,temp_s,miu,alpha,Ws,Rs,temp_m) private(i)
		for(i=0;i<N_mod;i++){
			temp_s[i] = miu*m_alpha_s[i]*Ws[i]*Rs[i]*Ws[i]*temp_m[i];
            temp_x[i] = 0;
            temp_y[i] = 0;
            temp_z[i] = 0;
            temp_xy[i] = 0;
            temp_yx[i] = 0;
            temp_zx[i] = 0;
            temp_xz[i] = 0;
            temp_yz[i] = 0;
            temp_zy[i] = 0;
        }

		transa = 'n';
		double * temp_x10 = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_x10) private(i)
        for(i=0;i<N_mod;i++) {
            temp_x10[i] = 0;
        }
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_x10);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_x10);
		#pragma omp parallel for shared(N_mod,temp_x10,Rx) private(i)
		for(i=0;i<N_mod;i++)
			temp_x10[i] = Rx[i]*temp_x10[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_x10, 0.0, temp_x);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_x10,temp_x);
		#pragma omp parallel for shared(N_mod,temp_x,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_x[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r11[i] + m_alpha_y[i]*tran_r21[i]*tran_r21[i] + m_alpha_z[i]*tran_r31[i]*tran_r31[i])*temp_x[i];
		delete [] temp_x10;

		transa = 'n';
		double * temp_y1 = new double [N_mod];
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
		#pragma omp parallel for shared(N_mod,temp_y,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_y[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r12[i] + m_alpha_y[i]*tran_r22[i]*tran_r22[i] + m_alpha_z[i]*tran_r32[i]*tran_r32[i])*temp_y[i];
		delete [] temp_y1;
		
		
		transa = 'n';
		double * temp_z1 = new double [N_mod];
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
		#pragma omp parallel for shared(N_mod,temp_z,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_z[i] = miu*(m_alpha_x[i]*tran_r13[i]*tran_r13[i] + m_alpha_y[i]*tran_r23[i]*tran_r23[i] + m_alpha_z[i]*tran_r33[i]*tran_r33[i])*temp_z[i];
		delete [] temp_z1;

		transa = 'n';
		double * temp_z1_xy = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_z1_xy) private(i)
        for(i=0;i<N_mod;i++) {
            temp_z1_xy[i] = 0;
        }
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_z1_xy);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_z1_xy);
		#pragma omp parallel for shared(N_mod,temp_z1_xy,Rz) private(i)
		for(i=0;i<N_mod;i++)
			temp_z1_xy[i] = sqrt(Rx[i])*sqrt(Ry[i])*temp_z1_xy[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_z1_xy, 0.0, temp_xy);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_z1_xy,temp_xy);
		#pragma omp parallel for shared(N_mod,temp_xy,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_xy[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r12[i] + m_alpha_y[i]*tran_r21[i]*tran_r22[i] + m_alpha_z[i]*tran_r31[i]*tran_r32[i])*temp_xy[i];
		delete [] temp_z1_xy;

		transa = 'n';
		double * temp_z1_yx = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_z1_yx) private(i)
        for(i=0;i<N_mod;i++) {
            temp_z1_yx[i] = 0;
        }
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_z1_yx);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_z1_yx);
		#pragma omp parallel for shared(N_mod,temp_z1_yx,Rz) private(i)
		for(i=0;i<N_mod;i++)
			temp_z1_yx[i] = sqrt(Rx[i])*sqrt(Ry[i])*temp_z1_yx[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wy, descrA, temp_z1_yx, 0.0, temp_yx);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_z1_yx,temp_yx);
		#pragma omp parallel for shared(N_mod,temp_yx,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_yx[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r12[i] + m_alpha_y[i]*tran_r21[i]*tran_r22[i] + m_alpha_z[i]*tran_r31[i]*tran_r32[i])*temp_yx[i];
		delete [] temp_z1_yx;

		transa = 'n';
		double * temp_z1_zx = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_z1_zx) private(i)
        for(i=0;i<N_mod;i++) {
            temp_z1_zx[i] = 0;
        }
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_z1_zx);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_z1_zx);
		#pragma omp parallel for shared(N_mod,temp_z1_zx,Rz) private(i)
		for(i=0;i<N_mod;i++)
			temp_z1_zx[i] = sqrt(Rz[i])*sqrt(Rx[i])*temp_z1_zx[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wz, descrA, temp_z1_zx, 0.0, temp_zx);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_z1_zx,temp_zx);
		#pragma omp parallel for shared(N_mod,temp_zx,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_zx[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r13[i] + m_alpha_y[i]*tran_r21[i]*tran_r23[i] + m_alpha_z[i]*tran_r31[i]*tran_r33[i])*temp_zx[i];
		delete [] temp_z1_zx;

		transa = 'n';
		double * temp_z1_xz = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_z1_xz) private(i)
        for(i=0;i<N_mod;i++) {
            temp_z1_xz[i] = 0;
        }
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_z1_xz);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_z1_xz);
		#pragma omp parallel for shared(N_mod,temp_z1_xz,Rz) private(i)
		for(i=0;i<N_mod;i++)
			temp_z1_xz[i] = sqrt(Rz[i])*sqrt(Rx[i])*temp_z1_xz[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_z1_xz, 0.0, temp_xz);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_z1_xz,temp_xz);
		#pragma omp parallel for shared(N_mod,temp_xz,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_xz[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r13[i] + m_alpha_y[i]*tran_r21[i]*tran_r23[i] + m_alpha_z[i]*tran_r31[i]*tran_r33[i])*temp_xz[i];
		delete [] temp_z1_xz;


		transa = 'n';
		double * temp_z1_yz = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_z1_yz) private(i)
        for(i=0;i<N_mod;i++) {
            temp_z1_yz[i] = 0;
        }
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_z1_yz);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_z1_yz);
		#pragma omp parallel for shared(N_mod,temp_z1_yz,Rz) private(i)
		for(i=0;i<N_mod;i++)
			temp_z1_yz[i] = sqrt(Rz[i])*sqrt(Ry[i])*temp_z1_yz[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wy, descrA, temp_z1_yz, 0.0, temp_yz);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_z1_yz,temp_yz);
		#pragma omp parallel for shared(N_mod,temp_yz,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_yz[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r13[i] + m_alpha_y[i]*tran_r22[i]*tran_r23[i] + m_alpha_z[i]*tran_r32[i]*tran_r33[i])*temp_yz[i];
		delete [] temp_z1_yz;

		transa = 'n';
		double * temp_z1_zy = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_z1_zy) private(i)
        for(i=0;i<N_mod;i++) {
            temp_z1_zy[i] = 0;
        }
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_z1_zy);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_z1_zy);
		#pragma omp parallel for shared(N_mod,temp_z1_zy,Rz) private(i)
		for(i=0;i<N_mod;i++)
			temp_z1_zy[i] = sqrt(Rz[i])*sqrt(Ry[i])*temp_z1_zy[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wz, descrA, temp_z1_zy, 0.0, temp_zy);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_z1_zy,temp_zy);
		#pragma omp parallel for shared(N_mod,temp_zy,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_zy[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r13[i] + m_alpha_y[i]*tran_r22[i]*tran_r23[i] + m_alpha_z[i]*tran_r32[i]*tran_r33[i])*temp_zy[i];
		delete [] temp_z1_zy;


		double * temp_d1 = new double [N_data];
		#pragma omp parallel for shared(N_data,temp_d1) private(i)
		for(i = 0;i < N_data;i++)
			temp_d1[i] = 0;

		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,G,N_mod,temp_m,1,0,temp_d1,1);
		#pragma omp parallel for shared(N_data,temp_d1,Wd,Rd) private(i)
		for(i=0;i<N_data;i++)
			temp_d1[i] = Wd[i]*Rd[i]*Wd[i]*temp_d1[i];
 		double * temp_d = new double[N_mod];
		#pragma omp parallel for shared(N_mod,temp_d) private(i)
		for(i=0;i<N_mod;i++)
			temp_d[i] = 0;
		cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,G,N_mod,temp_d1,1,0,temp_d,1);
		delete []temp_d1;

		// #pragma omp parallel for shared(N_mod,temp_s,AP0,temp_x,temp_y,temp_z,temp_d,temp_lbm_AP0,temp_dx,temp_dz,temp_dy) private(i)
		#pragma omp parallel for shared(N_mod,temp_s,AP0,temp_x,temp_y,temp_z,temp_d) private(i)		
		for(i=0;i<N_mod;i++) {
			AP0[i] = (temp_s[i]+temp_x[i]+temp_y[i]+temp_z[i]+temp_d[i]
				+temp_xy[i]+temp_yx[i]+temp_yz[i]+temp_zy[i]+temp_xz[i]+temp_zx[i]);
		}
		delete [] temp_m;
		delete [] temp_s;
		delete [] temp_x;
		delete [] temp_y;
		delete [] temp_z;
		delete [] temp_d;

		delete [] temp_xy;
		delete [] temp_yx;
		delete [] temp_zx;
		delete [] temp_xz;
		delete [] temp_yz;
		delete [] temp_zy;

		temp1 = 0;
		temp2 = 0;

		temp1 = cblas_ddot(N_mod,R0,1,R0,1);
		temp2 = cblas_ddot(N_mod,P0,1,AP0,1);
		alpha_k = temp1/(temp2+eps1);

		double * R1 = new double [N_mod];
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

		double * P1 = new double [N_mod];
		#pragma omp parallel for shared(N_mod,P1,R1) private(i)
		for(i=0;i<N_mod;i++)
			P1[i] = (-1)*R1[i];
		cblas_daxpy(N_mod,beta,P0,1,P1,1);

		cblas_dcopy(N_mod,X1,1,X0,1);
		cblas_dcopy(N_mod,R1,1,R0,1);
		cblas_dcopy(N_mod,P1,1,P0,1);

		delete [] AP0;
		delete [] R1;
		delete [] P1;
	}
	delete [] X0;

	double * temp_up2 = new double [N_data];
	#pragma omp parallel for shared(N_data,temp_up2) private(i)
	for(i = 0;i < N_data;i++)
		temp_up2[i] = 0;
	cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,G,N_mod,X1,1,0,temp_up2,1);
	delete [] X1;

	#pragma omp parallel for shared(N_data,temp_up2,Wd) private(i)
	for(i=0;i<N_data;i++){
		temp_up2[i] = Wd[i]*temp_up2[i];
	}


	double * temp_up3 = new double [N_data];
	#pragma omp parallel for shared(N_data,temp_up2,temp_up3,deltaT) private(i)
	for(i=0;i<N_data;i++)
		temp_up3[i] = Wd[i]*temp_dobs_dpre[i] - temp_up2[i];
	delete [] temp_up2;
	delete [] temp_dobs_dpre;

	double UP = cblas_ddot(N_data,temp_up3,1,temp_up3,1);
	delete [] temp_up3;
	// UP part finish


	// DOWN part begin
	// Tao Lei called me for dinner
	double * u_pbb = new double [N_data]; 
	srand((int)time(0));
	#pragma omp parallel for shared(u_pbb,N_data)  private(i)
	for(i=0;i<N_data;i++){
		if(random(100)>=50)
			u_pbb[i] = 1*Wd[i];
		else
			u_pbb[i] = -1*Wd[i];
	}


	double * temp_dn1 = new double[N_mod];
	#pragma omp parallel for shared(N_mod, temp_dn1) private(i)
	for(i=0;i<N_mod;i++){
		temp_dn1[i] = 0;
	}
	cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,G,N_mod,u_pbb,1,0,temp_dn1,1);

	#pragma omp parallel for shared(N_mod,temp_dn1,B) private(i)
	for(i=0;i<N_mod;i++){
		B[i] = temp_dn1[i];
	}
	delete [] temp_dn1;

	double * X00 = new double[N_mod];
	double * X10 = new double[N_mod];
	#pragma omp parallel for shared(N_mod,X00) private(i)
	for(i=0;i<N_mod;i++)
		X00[i] = 0;

	double * R00 = new double[N_mod]; 
	double * P00 = new double[N_mod];
	#pragma omp parallel for shared(N_mod,R00,P00,B) private(i)
	for(i=0;i<N_mod;i++) {
		R00[i] = (-1)*B[i];
		P00[i] = (-1)*R00[i];
	}
	for(int j = 1;j<N_data;j++){
		double alpha_k;
		double temp1,temp2;

		double * AP0 = new double[N_mod];
		double * temp_m = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_m,P00) private(i)
		for(i=0;i<N_mod;i++)
			temp_m[i] = P00[i];
		double * temp_s = new double [N_mod];
		double * temp_x = new double [N_mod];
		double * temp_y = new double [N_mod];
		double * temp_z = new double [N_mod];

		double * temp_xy = new double [N_mod];
		double * temp_yx = new double [N_mod];
		double * temp_zx = new double [N_mod];
		double * temp_xz = new double [N_mod];
		double * temp_yz = new double [N_mod];
		double * temp_zy = new double [N_mod];

		#pragma omp parallel for shared(N_mod,temp_s,miu,alpha,Ws,Rs,temp_m) private(i)
		for(i=0;i<N_mod;i++){
			temp_s[i] = miu*m_alpha_s[i]*Ws[i]*Rs[i]*Ws[i]*temp_m[i];
            temp_x[i] = 0;
            temp_y[i] = 0;
            temp_z[i] = 0;
            temp_xy[i] = 0;
            temp_yx[i] = 0;
            temp_zx[i] = 0;
            temp_xz[i] = 0;
            temp_yz[i] = 0;
            temp_zy[i] = 0;
        }


		transa = 'n';
		double * temp_x10 = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_x10) private(i)
        for(i=0;i<N_mod;i++) {
            temp_x10[i] = 0;
        }
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_x10);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_x10);
		#pragma omp parallel for shared(N_mod,temp_x10,Rx) private(i)
		for(i=0;i<N_mod;i++)
			temp_x10[i] = Rx[i]*temp_x10[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_x10, 0.0, temp_x);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_x10,temp_x);
		#pragma omp parallel for shared(N_mod,temp_x,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_x[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r11[i] + m_alpha_y[i]*tran_r21[i]*tran_r21[i] + m_alpha_z[i]*tran_r31[i]*tran_r31[i])*temp_x[i];
		delete [] temp_x10;

		transa = 'n';
		double * temp_y1 = new double [N_mod];
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
		#pragma omp parallel for shared(N_mod,temp_y,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_y[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r12[i] + m_alpha_y[i]*tran_r22[i]*tran_r22[i] + m_alpha_z[i]*tran_r32[i]*tran_r32[i])*temp_y[i];
		delete [] temp_y1;
		
		
		transa = 'n';
		double * temp_z1 = new double [N_mod];
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
		#pragma omp parallel for shared(N_mod,temp_z,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_z[i] = miu*(m_alpha_x[i]*tran_r13[i]*tran_r13[i] + m_alpha_y[i]*tran_r23[i]*tran_r23[i] + m_alpha_z[i]*tran_r33[i]*tran_r33[i])*temp_z[i];
		delete [] temp_z1;


		transa = 'n';
		double * temp_z1_xy = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_z1_xy) private(i)
        for(i=0;i<N_mod;i++) {
            temp_z1_xy[i] = 0;
        }
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_z1_xy);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_z1_xy);
		#pragma omp parallel for shared(N_mod,temp_z1_xy,Rz) private(i)
		for(i=0;i<N_mod;i++)
			temp_z1_xy[i] = sqrt(Rx[i])*sqrt(Ry[i])*temp_z1_xy[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_z1_xy, 0.0, temp_xy);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_z1_xy,temp_xy);
		#pragma omp parallel for shared(N_mod,temp_xy,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_xy[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r12[i] + m_alpha_y[i]*tran_r21[i]*tran_r22[i] + m_alpha_z[i]*tran_r31[i]*tran_r32[i])*temp_xy[i];
		delete [] temp_z1_xy;

		transa = 'n';
		double * temp_z1_yx = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_z1_yx) private(i)
        for(i=0;i<N_mod;i++) {
            temp_z1_yx[i] = 0;
        }
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_z1_yx);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_z1_yx);
		#pragma omp parallel for shared(N_mod,temp_z1_yx,Rz) private(i)
		for(i=0;i<N_mod;i++)
			temp_z1_yx[i] = sqrt(Rx[i])*sqrt(Ry[i])*temp_z1_yx[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wy, descrA, temp_z1_yx, 0.0, temp_yx);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_z1_yx,temp_yx);
		#pragma omp parallel for shared(N_mod,temp_yx,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_yx[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r12[i] + m_alpha_y[i]*tran_r21[i]*tran_r22[i] + m_alpha_z[i]*tran_r31[i]*tran_r32[i])*temp_yx[i];
		delete [] temp_z1_yx;

		transa = 'n';
		double * temp_z1_zx = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_z1_zx) private(i)
        for(i=0;i<N_mod;i++) {
            temp_z1_zx[i] = 0;
        }
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wx, descrA, temp_m, 0.0, temp_z1_zx);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_m,temp_z1_zx);
		#pragma omp parallel for shared(N_mod,temp_z1_zx,Rz) private(i)
		for(i=0;i<N_mod;i++)
			temp_z1_zx[i] = sqrt(Rz[i])*sqrt(Rx[i])*temp_z1_zx[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wz, descrA, temp_z1_zx, 0.0, temp_zx);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_z1_zx,temp_zx);
		#pragma omp parallel for shared(N_mod,temp_zx,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_zx[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r13[i] + m_alpha_y[i]*tran_r21[i]*tran_r23[i] + m_alpha_z[i]*tran_r31[i]*tran_r33[i])*temp_zx[i];
		delete [] temp_z1_zx;

		transa = 'n';
		double * temp_z1_xz = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_z1_xz) private(i)
        for(i=0;i<N_mod;i++) {
            temp_z1_xz[i] = 0;
        }
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_z1_xz);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_z1_xz);
		#pragma omp parallel for shared(N_mod,temp_z1_xz,Rz) private(i)
		for(i=0;i<N_mod;i++)
			temp_z1_xz[i] = sqrt(Rz[i])*sqrt(Rx[i])*temp_z1_xz[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wx, descrA, temp_z1_xz, 0.0, temp_xz);
        // mkl_dcsrgemv(&transa, &N_mod,Wx,rowIndex_x,columns_x,temp_z1_xz,temp_xz);
		#pragma omp parallel for shared(N_mod,temp_xz,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_xz[i] = miu*(m_alpha_x[i]*tran_r11[i]*tran_r13[i] + m_alpha_y[i]*tran_r21[i]*tran_r23[i] + m_alpha_z[i]*tran_r31[i]*tran_r33[i])*temp_xz[i];
		delete [] temp_z1_xz;


		transa = 'n';
		double * temp_z1_yz = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_z1_yz) private(i)
        for(i=0;i<N_mod;i++) {
            temp_z1_yz[i] = 0;
        }
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wz, descrA, temp_m, 0.0, temp_z1_yz);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_m,temp_z1_yz);
		#pragma omp parallel for shared(N_mod,temp_z1_yz,Rz) private(i)
		for(i=0;i<N_mod;i++)
			temp_z1_yz[i] = sqrt(Rz[i])*sqrt(Ry[i])*temp_z1_yz[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wy, descrA, temp_z1_yz, 0.0, temp_yz);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_z1_yz,temp_yz);
		#pragma omp parallel for shared(N_mod,temp_yz,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_yz[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r13[i] + m_alpha_y[i]*tran_r22[i]*tran_r23[i] + m_alpha_z[i]*tran_r32[i]*tran_r33[i])*temp_yz[i];
		delete [] temp_z1_yz;

		transa = 'n';
		double * temp_z1_zy = new double [N_mod];
		#pragma omp parallel for shared(N_mod,temp_z1_zy) private(i)
        for(i=0;i<N_mod;i++) {
            temp_z1_zy[i] = 0;
        }
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csr_Wy, descrA, temp_m, 0.0, temp_z1_zy);
        // mkl_dcsrgemv(&transa, &N_mod,Wy,rowIndex_y,columns_y,temp_m,temp_z1_zy);
		#pragma omp parallel for shared(N_mod,temp_z1_zy,Rz) private(i)
		for(i=0;i<N_mod;i++)
			temp_z1_zy[i] = sqrt(Rz[i])*sqrt(Ry[i])*temp_z1_zy[i];
		transa = 't';
		mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, csr_Wz, descrA, temp_z1_zy, 0.0, temp_zy);
        // mkl_dcsrgemv(&transa, &N_mod,Wz,rowIndex_z,columns_z,temp_z1_zy,temp_zy);
		#pragma omp parallel for shared(N_mod,temp_zy,miu,alpha) private(i)
		for(i=0;i<N_mod;i++)
			temp_zy[i] = miu*(m_alpha_x[i]*tran_r12[i]*tran_r13[i] + m_alpha_y[i]*tran_r22[i]*tran_r23[i] + m_alpha_z[i]*tran_r32[i]*tran_r33[i])*temp_zy[i];
		delete [] temp_z1_zy;


		double * temp_d1 = new double [N_data];
		#pragma omp parallel for shared(N_data,temp_d1) private(i)
		for(i = 0;i < N_data;i++)
			temp_d1[i] = 0;

		cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,G,N_mod,temp_m,1,0,temp_d1,1);
		#pragma omp parallel for shared(N_data,temp_d1,Wd,Rd) private(i)
		for(i=0;i<N_data;i++)
			temp_d1[i] = Wd[i]*Rd[i]*Wd[i]*temp_d1[i];
 		double * temp_d = new double[N_mod];
		#pragma omp parallel for shared(N_mod,temp_d) private(i)
		for(i=0;i<N_mod;i++)
			temp_d[i] = 0;
		cblas_dgemv(CblasRowMajor, CblasTrans,N_data,N_mod,1,G,N_mod,temp_d1,1,0,temp_d,1);
		delete []temp_d1;


		// #pragma omp parallel for shared(N_mod,temp_s,AP0,temp_x,temp_y,temp_z,temp_d,temp_lbm_AP0,temp_dx,temp_dz,temp_dy) private(i)
		#pragma omp parallel for shared(N_mod,temp_s,AP0,temp_x,temp_y,temp_z,temp_d) private(i)		
		for(i=0;i<N_mod;i++) {
			AP0[i] = (temp_s[i]+temp_x[i]+temp_y[i]+temp_z[i]+temp_d[i]
				+temp_xy[i]+temp_yx[i]+temp_yz[i]+temp_zy[i]+temp_xz[i]+temp_zx[i]);
		}
		delete [] temp_m;
		delete [] temp_s;
		delete [] temp_x;
		delete [] temp_y;
		delete [] temp_z;
		delete [] temp_d;

		delete [] temp_xy;
		delete [] temp_yz;
		delete [] temp_zx;
		delete [] temp_xz;
		delete [] temp_yx;
		delete [] temp_zy;

		temp1 = 0;
		temp2 = 0;

		temp1 = cblas_ddot(N_mod,R00,1,R00,1);
		temp2 = cblas_ddot(N_mod,P00,1,AP0,1);
		alpha_k = temp1/(temp2+eps1);

		double * R1 = new double [N_mod];
		cblas_dcopy(N_mod,X00,1,X10,1);
		cblas_daxpy(N_mod,alpha_k,P00,1,X10,1);

		cblas_dcopy(N_mod,R00,1,R1,1);
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

		temp2 = cblas_ddot(N_mod,R00,1,R00,1);
		beta = temp1/(temp2+eps1);

		double * P1 = new double [N_mod];
		#pragma omp parallel for shared(N_mod,P1,R1) private(i)
		for(i=0;i<N_mod;i++)
			P1[i] = (-1)*R1[i];
		cblas_daxpy(N_mod,beta,P00,1,P1,1);

		cblas_dcopy(N_mod,X10,1,X00,1);
		cblas_dcopy(N_mod,R1,1,R00,1);
		cblas_dcopy(N_mod,P1,1,P00,1);

		delete [] AP0;
		delete [] R1;
		delete [] P1;
	}
	delete [] X00;

	double * temp_dn2 = new double [N_data];
	#pragma omp parallel for shared(N_data,temp_dn2) private(i)
	for(i = 0;i < N_data;i++)
		temp_dn2[i] = 0;
	cblas_dgemv(CblasRowMajor, CblasNoTrans,N_data,N_mod,1,G,N_mod,X10,1,0,temp_dn2,1);
	delete [] X10;


	double temp_dn3;
	temp_dn3 = cblas_ddot(N_data,u_pbb,1,temp_dn2,1);
	delete [] temp_dn2;
	double DOWN;
	DOWN = (N_data - temp_dn3)*(N_data - temp_dn3);

	double gcv_miu;
	gcv_miu = UP/(DOWN+eps1);
	// DOWN part finished
    mkl_sparse_destroy(csr_Wx);
	mkl_sparse_destroy(csr_Wy);
	mkl_sparse_destroy(csr_Wz);

	return gcv_miu;
}


// calculating the optimal regularization parameter using GCV method
double auto_cal_RegulPara_GCV_nonlinar_noneffsto(int para_deepweight_type, int N_data, int N_mod, double * G_T,double * Wd,double * obs_deltaT,
	double * Ws,double * Wx,MKL_INT * columns_x,MKL_INT * rowIndex_x,
	double * Wy,MKL_INT * columns_y,MKL_INT * rowIndex_y,
	double * Wz,MKL_INT * columns_z,MKL_INT * rowIndex_z,
	double * alpha,double * Rs,double * Rx,double * Ry,double * Rz,double * Rd,
	double * m_0_n,double * m_ref_n,double * m_alpha_s, double * m_alpha_x,double * m_alpha_y, double * m_alpha_z,
	double * tran_r11,double * tran_r12,double * tran_r13,double * tran_r21,
	double * tran_r22,double * tran_r23,double * tran_r31,double * tran_r32,double * tran_r33)
{
	cout<<"GCV calculation begins : : : : : !!!"<<endl;
	
	double ori_para;
	if(para_deepweight_type == 1)
		ori_para = 20;
	else if(para_deepweight_type == 2)
		ori_para = 20;

	ofstream resultfileGCV;
	resultfileGCV.open("result_test_gcv.dat");
	double result_temp;
	int sum_temp = 0;
	double temp_gcv[5][2];
	double para;
	para = pow(10,ori_para);

	result_temp = cal_RegulPara_GCV_nonlinar_noneffsto(para, N_data,N_mod, G_T,Wd,obs_deltaT,
			Ws,Wx, columns_x, rowIndex_x,Wy, columns_y, rowIndex_y,Wz, columns_z, rowIndex_z,
			alpha,Rs,Rx,Ry,Rz,Rd,m_0_n,m_ref_n,m_alpha_s,m_alpha_x,m_alpha_y, m_alpha_z,
			tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);
	resultfileGCV<<ori_para<<" "<<log10(result_temp)<<endl;
	temp_gcv[0][0] = ori_para;
	temp_gcv[0][1] = log10(result_temp);
	sum_temp++;
	cout<<"Calculation number : "<<sum_temp<<" -- Regu_para : GCV value "<<temp_gcv[0][0]<<"    "<<temp_gcv[0][1]<<endl;

	para = pow(10,ori_para-2);
	result_temp = cal_RegulPara_GCV_nonlinar_noneffsto(para, N_data,N_mod, G_T,Wd,obs_deltaT,
			Ws,Wx, columns_x, rowIndex_x,Wy, columns_y, rowIndex_y,Wz, columns_z, rowIndex_z,
			alpha,Rs,Rx,Ry,Rz,Rd,m_0_n,m_ref_n,m_alpha_s, m_alpha_x,m_alpha_y, m_alpha_z,
			tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);
	resultfileGCV<<ori_para-2<<" "<<log10(result_temp)<<endl;
	temp_gcv[1][0] = ori_para-2;
	temp_gcv[1][1] = log10(result_temp);
	sum_temp++;
	cout<<"Calculation number : "<<sum_temp<<" -- Regu_para : GCV value "<<temp_gcv[1][0]<<"   "<<temp_gcv[1][1]<<endl;

	para = pow(10,ori_para-4);
	result_temp = cal_RegulPara_GCV_nonlinar_noneffsto(para, N_data,N_mod, G_T,Wd,obs_deltaT,
			Ws,Wx, columns_x, rowIndex_x,Wy, columns_y, rowIndex_y,Wz, columns_z, rowIndex_z,
			alpha,Rs,Rx,Ry,Rz,Rd,m_0_n,m_ref_n,m_alpha_s, m_alpha_x,m_alpha_y, m_alpha_z,
			tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);
	resultfileGCV<<ori_para-4<<" "<<log10(result_temp)<<endl;
	temp_gcv[2][0] = ori_para-4;
	temp_gcv[2][1] = log10(result_temp);
	sum_temp++;
	cout<<"Calculation number : "<<sum_temp<<" -- Regu_para : GCV value "<<temp_gcv[2][0]<<"   "<<temp_gcv[2][1]<<endl;

	para = pow(10,ori_para-6);
	result_temp = cal_RegulPara_GCV_nonlinar_noneffsto(para, N_data,N_mod, G_T,Wd,obs_deltaT,
			Ws,Wx, columns_x, rowIndex_x,Wy, columns_y, rowIndex_y,Wz, columns_z, rowIndex_z,
			alpha,Rs,Rx,Ry,Rz,Rd,m_0_n,m_ref_n,m_alpha_s, m_alpha_x,m_alpha_y, m_alpha_z,
			tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);
	resultfileGCV<<ori_para-6<<" "<<log10(result_temp)<<endl;
	temp_gcv[3][0] = ori_para-6;
	temp_gcv[3][1] = log10(result_temp);
	sum_temp++;
	cout<<"Calculation number : "<<sum_temp<<" -- Regu_para : GCV value "<<temp_gcv[3][0]<<"   "<<temp_gcv[3][1]<<endl;

	para = pow(10,ori_para-8);
	result_temp = cal_RegulPara_GCV_nonlinar_noneffsto(para, N_data,N_mod, G_T,Wd,obs_deltaT,
			Ws,Wx, columns_x, rowIndex_x,Wy, columns_y, rowIndex_y,Wz, columns_z, rowIndex_z,
			alpha,Rs,Rx,Ry,Rz,Rd,m_0_n,m_ref_n,m_alpha_s, m_alpha_x,m_alpha_y, m_alpha_z,
			tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);
	resultfileGCV<<ori_para-8<<" "<<log10(result_temp)<<endl;
	temp_gcv[4][0] = ori_para-8;
	temp_gcv[4][1] = log10(result_temp);
	sum_temp++;
	cout<<"Calculation number : "<<sum_temp<<" -- Regu_para : GCV value "<<temp_gcv[4][0]<<"   "<<temp_gcv[4][1]<<endl;


	double temp_max = 0;
	temp_max = fabs(temp_gcv[0][1]-temp_gcv[2][1]);
	if (temp_max < fabs(temp_gcv[1][1]-temp_gcv[2][1])){
		temp_max = fabs(temp_gcv[1][1]-temp_gcv[2][1]);
	}
	if (temp_max < fabs(temp_gcv[3][1]-temp_gcv[2][1])){
		temp_max = fabs(temp_gcv[3][1]-temp_gcv[2][1]);
	}
	if (temp_max < fabs(temp_gcv[4][1]-temp_gcv[2][1])){
		temp_max = fabs(temp_gcv[4][1]-temp_gcv[2][1]);
	}

	int interuptOrnot = 0; 
	while(!(temp_gcv[0][1]>temp_gcv[2][1] && temp_gcv[1][1]>temp_gcv[2][1] && 
		temp_gcv[3][1]>temp_gcv[2][1] && temp_gcv[4][1]>temp_gcv[2][1] && temp_max > 0.5) 
		&& sum_temp <= 20){
		temp_gcv[0][0] = temp_gcv[1][0];
		temp_gcv[0][1] = temp_gcv[1][1];
		temp_gcv[1][0] = temp_gcv[2][0];
		temp_gcv[1][1] = temp_gcv[2][1];
		temp_gcv[2][0] = temp_gcv[3][0];
		temp_gcv[2][1] = temp_gcv[3][1];
		temp_gcv[3][0] = temp_gcv[4][0];
		temp_gcv[3][1] = temp_gcv[4][1];

		para = pow(10,temp_gcv[3][0]-2);
		result_temp = cal_RegulPara_GCV_nonlinar_noneffsto(para, N_data,N_mod, G_T,Wd,obs_deltaT,
			Ws,Wx, columns_x, rowIndex_x,Wy, columns_y, rowIndex_y,Wz, columns_z, rowIndex_z,
			alpha,Rs,Rx,Ry,Rz,Rd,m_0_n,m_ref_n,m_alpha_s, m_alpha_x,m_alpha_y, m_alpha_z,
			tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);
		resultfileGCV<<temp_gcv[3][0]-2<<" "<<log10(result_temp)<<endl;
		temp_gcv[4][0] = temp_gcv[3][0]-2;
		temp_gcv[4][1] = log10(result_temp);
		sum_temp++;
		cout<<"Calculation number : "<<sum_temp<<" -- Regu_para : GCV value "<<temp_gcv[4][0]<<"   "<<temp_gcv[4][1]<<endl;

		temp_max = fabs(temp_gcv[0][1]-temp_gcv[2][1]);
		if (temp_max < fabs(temp_gcv[1][1]-temp_gcv[2][1]))
			temp_max = fabs(temp_gcv[1][1]-temp_gcv[2][1]);
		if (temp_max < fabs(temp_gcv[3][1]-temp_gcv[2][1]))
			temp_max = fabs(temp_gcv[3][1]-temp_gcv[2][1]);
		if (temp_max < fabs(temp_gcv[4][1]-temp_gcv[2][1]))
			temp_max = fabs(temp_gcv[4][1]-temp_gcv[2][1]);
	}
	if(sum_temp >= 20)
		interuptOrnot = 1;

	double miu;
	if(interuptOrnot == 0){
		double i_temp;
		for(i_temp=temp_gcv[4][0]+0.5;i_temp<temp_gcv[3][0]-0.1;i_temp=i_temp+0.5){
			para = pow(10,i_temp);
			result_temp = cal_RegulPara_GCV_nonlinar_noneffsto(para, N_data,N_mod, G_T,Wd,obs_deltaT,
				Ws,Wx, columns_x, rowIndex_x,Wy, columns_y, rowIndex_y,Wz, columns_z, rowIndex_z,
				alpha,Rs,Rx,Ry,Rz,Rd,m_0_n,m_ref_n,m_alpha_s, m_alpha_x,m_alpha_y, m_alpha_z,
				tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);
			resultfileGCV<<i_temp<<" "<<log10(result_temp)<<endl;
			sum_temp++;
			cout<<"Calculation number : "<<sum_temp<<" -- Regu_para : GCV value "<<i_temp<<"   "<<log10(result_temp)<<endl;
		}
		for(i_temp=temp_gcv[3][0]+0.2;i_temp<temp_gcv[2][0]-0.05;i_temp=i_temp+0.2){
			para = pow(10,i_temp);
			result_temp = cal_RegulPara_GCV_nonlinar_noneffsto(para, N_data,N_mod, G_T,Wd,obs_deltaT,
				Ws,Wx, columns_x, rowIndex_x,Wy, columns_y, rowIndex_y,Wz, columns_z, rowIndex_z,
				alpha,Rs,Rx,Ry,Rz,Rd,m_0_n,m_ref_n,m_alpha_s, m_alpha_x,m_alpha_y, m_alpha_z,
				tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);
			resultfileGCV<<i_temp<<" "<<log10(result_temp)<<endl;
			sum_temp++;
			cout<<"Calculation number : "<<sum_temp<<" -- Regu_para : GCV value "<<i_temp<<"   "<<log10(result_temp)<<endl;
		}
		for(i_temp=temp_gcv[2][0]+0.2;i_temp<temp_gcv[1][0]-0.05;i_temp=i_temp+0.2){
			para = pow(10,i_temp);
			result_temp = cal_RegulPara_GCV_nonlinar_noneffsto(para, N_data,N_mod, G_T,Wd,obs_deltaT,
				Ws,Wx, columns_x, rowIndex_x,Wy, columns_y, rowIndex_y,Wz, columns_z, rowIndex_z,
				alpha,Rs,Rx,Ry,Rz,Rd,m_0_n,m_ref_n,m_alpha_s, m_alpha_x,m_alpha_y, m_alpha_z,
				tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);
			resultfileGCV<<i_temp<<" "<<log10(result_temp)<<endl;
			sum_temp++;
			cout<<"Calculation number : "<<sum_temp<<" -- Regu_para : GCV value "<<i_temp<<"   "<<log10(result_temp)<<endl;
		}
		for(i_temp=temp_gcv[1][0]+0.5;i_temp<temp_gcv[0][0]-0.1;i_temp=i_temp+0.5){
			para = pow(10,i_temp);
			result_temp = cal_RegulPara_GCV_nonlinar_noneffsto(para, N_data,N_mod, G_T,Wd,obs_deltaT,
				Ws,Wx, columns_x, rowIndex_x,Wy, columns_y, rowIndex_y,Wz, columns_z, rowIndex_z,
				alpha,Rs,Rx,Ry,Rz,Rd,m_0_n,m_ref_n,m_alpha_s, m_alpha_x,m_alpha_y, m_alpha_z,
				tran_r11,tran_r12,tran_r13,tran_r21,tran_r22,tran_r23,tran_r31,tran_r32,tran_r33);
			resultfileGCV<<i_temp<<" "<<log10(result_temp)<<endl;
			sum_temp++;
			cout<<"Calculation number : "<<sum_temp<<" -- Regu_para : GCV value "<<i_temp<<"   "<<log10(result_temp)<<endl;
		}
		resultfileGCV.close();

		int num_cal = sum_temp;

		double * gcv_vlu = new double [num_cal];
		double * regpara = new double [num_cal];

		FILE * fp_gcv_read = fopen("result_test_gcv.dat","r");
		int i,j,k;
		for(i=0;i<num_cal;i++){
			fscanf(fp_gcv_read,"%lf %lf\n",regpara+i,gcv_vlu+i);
		}
		fclose(fp_gcv_read);

		double min_gcv,min_para;
		min_gcv = 10000;
		for (i=0; i<num_cal; i++) {
			if(gcv_vlu[i]<min_gcv){
				min_gcv = gcv_vlu[i];
				min_para = regpara[i];
			}
		}
		cout<<endl<<"  The optimal regularization parameter is 10 ^ "<<min_para<<endl;
		miu = pow(10,min_para);
	}
	else{
		resultfileGCV.close();
		int num_cal = sum_temp;
		double * gcv_vlu = new double [num_cal];
		double * regpara = new double [num_cal];

		FILE * fp_gcv_read = fopen("result_test_gcv.dat","r");
		int i,j,k;
		double gcv_min = 10000, gcv_max = -10000;

		for(i=0;i<num_cal;i++){
			fscanf(fp_gcv_read,"%lf %lf\n",regpara+i,gcv_vlu+i);
			gcv_min = min(gcv_min, gcv_vlu[i]);
			gcv_max = max(gcv_max, gcv_vlu[i]);
		}
		fclose(fp_gcv_read);

		for (i = 0; i < num_cal; i++) {
			gcv_vlu[i] = regpara[num_cal - 1] + (gcv_vlu[i] - gcv_min) / (gcv_max - gcv_min)*2*(regpara[0] - regpara[num_cal - 1]);
		}

		double min_gcv, min_para;

		for(i=2;i<num_cal-2;i++){
			double xtemp1,xtemp2,ytemp1,ytemp2;
			xtemp1 = regpara[i-2] - regpara[i];
			ytemp1 = gcv_vlu[i-2] - gcv_vlu[i];

			xtemp2 = regpara[i+2] - regpara[i];
			ytemp2 = gcv_vlu[i+2] - gcv_vlu[i];
			
			double dist1,dist2;
			dist1 = sqrt(xtemp1*xtemp1 + ytemp1*ytemp1);
			dist2 = sqrt(xtemp2*xtemp2 + ytemp2*ytemp2);
			double theta;
			theta = acos((xtemp1*xtemp2+ytemp1*ytemp2)/(dist1*dist2));
			theta = fabs(theta/PI*180);
			double theta2;
			theta2 = acos((-1)*xtemp2/dist2);
			theta2 = fabs(theta2 / PI * 180);
			cout << theta << endl;
			cout << theta2 << endl;
			if(theta >= 60 && theta <= 120 && theta2 < 256){
				min_para = regpara[i];
				break;
			}
		}

		cout<<"The optimal regularization parameter is 10 ^ "<<min_para<<endl;
		miu = pow(10,min_para);
	}
	cout<<"GCV calculation finish : : : : : !!!"<<endl;
	return miu;
}