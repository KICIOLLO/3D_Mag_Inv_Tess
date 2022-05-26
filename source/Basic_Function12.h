#include <fstream>
#include <math.h>
#include <iomanip>
#include <ctime>
#include <iostream>
#include <string>
#include <map>
#include <stdlib.h>
#include <omp.h>
#include "getopt_cmdline.h"

#include "Para_inv_Variable12.h"
#include "Functions_InvGrad_BC_lbm_12.h"
#include "Functions_InvGrad_BC_barbosa3_12.h"
#include "Functions_InvGrad_BC_compulsory2_12.h"

using namespace std;

// create a log file of which name is the time when inversion started
void BuildFileFolderandFile(char * fileLog)
{
	time_t timep;
	struct tm *p;
	time(&timep);
	p = localtime(&timep); // getting the local time
	int year = 1900 + p->tm_year;
	int month = p->tm_mon + 1;
	int day = p->tm_mday;
	int hour = p->tm_hour;
	int minute = p->tm_min;
	int second = p->tm_sec;


	char str_year[8];
	char str_month[8];
	char str_day[8];
	char str_hour[8];
	char str_minute[8];
	char str_second[8];


	memset(str_year, 0, sizeof(str_year));
	sprintf(str_year, "%d", year);

	memset(str_month, 0, sizeof(str_month));
	if (month < 10)
		sprintf(str_month, "%d%d", 0, month);
	else
		sprintf(str_month, "%d", month);

	memset(str_day, 0, sizeof(str_day));
	if (day < 10)
		sprintf(str_day, "%d%d", 0, day);
	else
		sprintf(str_day, "%d", day);

	memset(str_hour, 0, sizeof(str_hour));
	if (hour < 10)
		sprintf(str_hour, "%d%d", 0, hour);
	else
		sprintf(str_hour, "%d", hour);

	memset(str_minute, 0, sizeof(str_minute));
	if (minute < 10)
		sprintf(str_minute, "%d%d", 0, minute);
	else
		sprintf(str_minute, "%d", minute);

	memset(str_second, 0, sizeof(str_second));
	if (second < 10)
		sprintf(str_second, "%d%d", 0, second);
	else
		sprintf(str_second, "%d", second);

	memset(fileLog, 0, sizeof(fileLog));
	strcpy(fileLog, str_year);
	strcat(fileLog, str_month);
	strcat(fileLog, str_day);

	strcat(fileLog, "_");
	strcat(fileLog, str_hour);
	strcat(fileLog, "_");
	strcat(fileLog, str_minute);
	strcat(fileLog, "_");
	strcat(fileLog, str_second);
	strcat(fileLog, "_InvLog.dat");

	FILE * fRstLog;
	fRstLog = fopen(fileLog, "a");
	fprintf(fRstLog, "The Inversion Starts at %d-%s-%s %s:%s:%s\n\n\n", year, str_month, str_day, str_hour, str_minute, str_second);
	fclose(fRstLog);
}


void save_invrslt_predata_Lcrv(TESSEROID *model, OBSMAGPOS *obs_pos_mag, int N_data, int N_mod,
    double para_new,double * m_1_n,int * paraInvOrNot,
    double * deltaT_pre,double * deltaT_preHax,double * deltaT_preHay,
    double * deltaT_preZa,double * deltaT_preTa,
    double * deltaT_pre_2,
    double * phi_result, char * fileLog)
{

	int i,j,k,l;

    char fileresult[256];
    memset(fileresult, 0, sizeof(fileresult));
    sprintf(fileresult,"%s_sus_result.dat",fileLog);


    ofstream ftemp1;
    ftemp1.open(fileresult);
    for (j=0; j<N_mod; j++)
        ftemp1<<m_1_n[j]<<endl;
    
    ftemp1<<flush;
    ftemp1.close();
    cout<<"\t The final susceptibility inversion result saves in -- "<<fileresult<<" --"<<endl<<endl;



    char fileresult_pre[256];
    memset(fileresult_pre, 0, sizeof(fileresult_pre));
    sprintf(fileresult_pre,"%s_sus_result_pre.dat", fileLog);
    ofstream resultfile;
    resultfile.open(fileresult_pre);

    resultfile<<"# obs_pos_lon"<<" "<<"obs_pos_lat"<<" "<<"obs_pos_height"<<" ";
    resultfile<<"obs_pos_IGRF_Blon"<<" "<<"obs_pos_IGRF_Blat"<<" "<<"obs_pos_IGRF_Br"<<" ";
    resultfile<<"deltaT_pre"<<" ";
    resultfile<<"deltaT_preBlon"<<" ";
    resultfile<<"deltaT_preBlat"<<" ";
    resultfile<<"deltaT_preBr"<<endl;

    for (i=0; i<N_data; i++) {
        resultfile<<obs_pos_mag[i].lon<<" "<<obs_pos_mag[i].lat<<" "<<obs_pos_mag[i].height<<" ";
        resultfile<<obs_pos_mag[i].Bx<<" "<<obs_pos_mag[i].By<<" "<<obs_pos_mag[i].Bz<<" ";
        resultfile<<deltaT_pre[i]<<" ";
        resultfile<<deltaT_preHax[i]<<" ";
        resultfile<<deltaT_preHay[i]<<" ";
        resultfile<<deltaT_preZa[i]<<" ";
        resultfile<<endl;
    }
    resultfile.close();

    
    FILE * fLcurve;
    fLcurve = fopen("Lcurve.dat","a");
    fprintf(fLcurve,"%.12f\t%.12f\t%.12f\t%f\t%s\n",phi_result[0],phi_result[1],phi_result[2],para_new, fileLog);
    fclose(fLcurve);
}


// 20160218 10:53 Calculate the correlation bwtween the inv_result and the the
void CalPhyCorrAndDist(char * ModelFile, double * m_1_n, int N_mod, char * fileLog, double * result)
{
	
    double * Modelphy = new double [N_mod];
	int i;
    FILE * fpmodelphy = fopen(ModelFile,"r");
    for (i=0;i<N_mod;i++){
        fscanf(fpmodelphy,"%lf\n",Modelphy+i);
    }

    double PhysCorr, PhysEuclDist, PhysManhDist, PhysCos;
    PhysCorr = cal_phys_corr(N_mod, Modelphy, m_1_n);
    PhysEuclDist = cal_phys_EuclideanDist(N_mod, Modelphy, m_1_n);
    PhysManhDist = cal_phys_ManhattanDist(N_mod, Modelphy, m_1_n);
    PhysCos = cal_phys_cos(N_mod, Modelphy, m_1_n);

	result[0] = PhysCorr;
	result[1] = PhysEuclDist;
	result[2] = PhysManhDist;
	result[3] = PhysCos;


    double PhysTotalResult;
    PhysTotalResult = 1;

    FILE* fpOut = fopen("PhysCorrTotal.dat", "a");

    fprintf(fpOut, "%12.8f  %12.8f  %12.8f  %12.8f  ",PhysCorr,PhysEuclDist,PhysManhDist,PhysCos);
    fprintf(fpOut, "%s %s\n\n",ModelFile,fileLog);

    fclose(fpOut);
}

// The inversion part of the main function
void Inversion(Para_inv_open * paraOpen, Para_inv_Variable * ParaVariable)
{
    char fileLog[256];
    BuildFileFolderandFile(fileLog);

    paraOpen->Print_Log(fileLog);

    clock_t start,finish;
    start = clock();
    int i,j,k,l;
    int N_data = ParaVariable->N_data;
    int N_mod = ParaVariable->N_mod;

    double phi_result[4];
    double * deltaT_pre = new double [N_data];

    double * deltaT_preHax = new double [N_data];
    double * deltaT_preHay = new double [N_data];
    double * deltaT_preZa = new double [N_data];
    double * deltaT_preTa = new double [N_data];

    double * deltaT_pre_2 = new double [N_data];

    double * m_0_n1 = new double [N_mod];
    double * m_1_n = new double [N_mod];

    double para_new; 
    

    #pragma omp parallel for shared(m_0_n1) private(i)
    for(i=0;i<N_mod;i++){
        m_0_n1[i] = ParaVariable->m_0_n[i];
    }

    double miu = pow(10,paraOpen->para_new);
    if(paraOpen->whichBoundConstrainMthd == 1){
        para_new = InversionMagSus_compulsory_noneffsto(ParaVariable->model,ParaVariable->obs_pos_mag,N_data,N_mod,
            paraOpen->beta,miu,ParaVariable->obs_deltaT,ParaVariable->obs_deltaTerr,
            ParaVariable->obs_deltaTHax,ParaVariable->obs_deltaTerrHax,ParaVariable->obs_deltaTHay,ParaVariable->obs_deltaTerrHay,
            ParaVariable->obs_deltaTZa,ParaVariable->obs_deltaTerrZa,ParaVariable->obs_deltaTTa,ParaVariable->obs_deltaTerrTa,
            ParaVariable->obs_deltaT_2,ParaVariable->obs_deltaTerr_2,
            ParaVariable->geomag_ref_x,ParaVariable->geomag_ref_y,ParaVariable->geomag_ref_z,
            paraOpen->paraInvOrNot,
            ParaVariable->modelsmoothfun_s,ParaVariable->modelsmoothfun_x,ParaVariable->modelsmoothfun_y,ParaVariable->modelsmoothfun_z,
            ParaVariable->modelsmoothfun_xdiffdrct,ParaVariable->modelsmoothfun_ydiffdrct,ParaVariable->modelsmoothfun_zdiffdrct,
            ParaVariable->G_THax,ParaVariable->G_THay,ParaVariable->G_TZa,
            ParaVariable->p1,ParaVariable->p2,ParaVariable->px2,ParaVariable->py2,ParaVariable->pz2,ParaVariable->epsilon1,ParaVariable->epsilon2,
            ParaVariable->epsilonx2,ParaVariable->epsilony2,ParaVariable->epsilonz2,
            paraOpen->alpha,ParaVariable->m_ref_n,ParaVariable->m_min_n,ParaVariable->m_max_n,m_0_n1,m_1_n,phi_result,fileLog,
            paraOpen->withCalGCVcurve,paraOpen->para_deepweight_type,paraOpen->z0_deepw_real,
            ParaVariable->m_alpha_s, ParaVariable->m_alpha_x, ParaVariable->m_alpha_y, ParaVariable->m_alpha_z,
            ParaVariable->m_angle_phi, ParaVariable->m_angle_theta, ParaVariable->m_angle_psi,
            ParaVariable->index_lon, ParaVariable->index_lat, ParaVariable->index_depth, 
            ParaVariable->N_data, ParaVariable->N_mod);
    }
    else if(paraOpen->whichBoundConstrainMthd == 2){
        para_new = InversionMagSus_lbm_noneffsto(ParaVariable->model,ParaVariable->obs_pos_mag,N_data,N_mod,
            paraOpen->beta,miu,ParaVariable->obs_deltaT,ParaVariable->obs_deltaTerr,
            ParaVariable->obs_deltaTHax,ParaVariable->obs_deltaTerrHax,ParaVariable->obs_deltaTHay,ParaVariable->obs_deltaTerrHay,
            ParaVariable->obs_deltaTZa,ParaVariable->obs_deltaTerrZa,ParaVariable->obs_deltaTTa,ParaVariable->obs_deltaTerrTa,
            ParaVariable->obs_deltaT_2,ParaVariable->obs_deltaTerr_2,
            ParaVariable->geomag_ref_x,ParaVariable->geomag_ref_y,ParaVariable->geomag_ref_z,
            paraOpen->paraInvOrNot,
            ParaVariable->modelsmoothfun_s,ParaVariable->modelsmoothfun_x,ParaVariable->modelsmoothfun_y,ParaVariable->modelsmoothfun_z,
            ParaVariable->modelsmoothfun_xdiffdrct,ParaVariable->modelsmoothfun_ydiffdrct,ParaVariable->modelsmoothfun_zdiffdrct,
            ParaVariable->G_THax,ParaVariable->G_THay,ParaVariable->G_TZa,
            ParaVariable->p1,ParaVariable->p2,ParaVariable->px2,ParaVariable->py2,ParaVariable->pz2,ParaVariable->epsilon1,ParaVariable->epsilon2,
            ParaVariable->epsilonx2,ParaVariable->epsilony2,ParaVariable->epsilonz2,
            paraOpen->alpha,ParaVariable->m_ref_n,ParaVariable->m_min_n,ParaVariable->m_max_n,m_0_n1,m_1_n,phi_result,fileLog,
            paraOpen->withCalGCVcurve,paraOpen->para_deepweight_type,paraOpen->z0_deepw_real,
            ParaVariable->m_alpha_s,ParaVariable->m_alpha_x,ParaVariable->m_alpha_y,ParaVariable->m_alpha_z,
            ParaVariable->m_angle_phi,ParaVariable->m_angle_theta,ParaVariable->m_angle_psi,
            ParaVariable->index_lon, ParaVariable->index_lat, ParaVariable->index_depth,
            ParaVariable->N_data, ParaVariable->N_mod);
    }
    else if(paraOpen->whichBoundConstrainMthd == 3){
        para_new = InversionMagSus_Barbosa_noneffsto(ParaVariable->model,ParaVariable->obs_pos_mag,N_data,N_mod,
            paraOpen->beta,miu,ParaVariable->obs_deltaT,ParaVariable->obs_deltaTerr,
            ParaVariable->obs_deltaTHax,ParaVariable->obs_deltaTerrHax,ParaVariable->obs_deltaTHay,ParaVariable->obs_deltaTerrHay,
            ParaVariable->obs_deltaTZa,ParaVariable->obs_deltaTerrZa,ParaVariable->obs_deltaTTa,ParaVariable->obs_deltaTerrTa,
            ParaVariable->obs_deltaT_2,ParaVariable->obs_deltaTerr_2,
            ParaVariable->geomag_ref_x,ParaVariable->geomag_ref_y,ParaVariable->geomag_ref_z,
            paraOpen->paraInvOrNot,
            ParaVariable->modelsmoothfun_s,ParaVariable->modelsmoothfun_x,ParaVariable->modelsmoothfun_y,ParaVariable->modelsmoothfun_z,
            ParaVariable->modelsmoothfun_xdiffdrct,ParaVariable->modelsmoothfun_ydiffdrct,ParaVariable->modelsmoothfun_zdiffdrct,
            ParaVariable->G_THax,ParaVariable->G_THay,ParaVariable->G_TZa,
            ParaVariable->p1,ParaVariable->p2,ParaVariable->px2,ParaVariable->py2,ParaVariable->pz2,ParaVariable->epsilon1,ParaVariable->epsilon2,
            ParaVariable->epsilonx2,ParaVariable->epsilony2,ParaVariable->epsilonz2,
            paraOpen->alpha,ParaVariable->m_ref_n,ParaVariable->m_min_n,ParaVariable->m_max_n,m_0_n1,m_1_n,phi_result,fileLog,
            paraOpen->withCalGCVcurve,paraOpen->para_deepweight_type,paraOpen->z0_deepw_real,
            ParaVariable->m_alpha_s, ParaVariable->m_alpha_x, ParaVariable->m_alpha_y, ParaVariable->m_alpha_z,
            ParaVariable->m_angle_phi, ParaVariable->m_angle_theta, ParaVariable->m_angle_psi,
            ParaVariable->index_lon, ParaVariable->index_lat, ParaVariable->index_depth,
            ParaVariable->N_data, ParaVariable->N_mod);
    }
    else{
        cout<<endl<<endl<<" ----------  Warning !! ----------"<<endl;
        cout<<endl<<endl<<" ----------  Warning !! ----------"<<endl;
        cout<< "  Please choose the suitable bound constrained method and the corresponding value!"<<endl;
        cout<< "    Its value must be the below values!"<<endl;
        cout<< "    1. general or compulsury constraint!"<<endl;
        cout<< "    2. logarithmic barrier method (LBM, e.g., Li and Oldenburg, 2003)"<<endl;
        cout<< "    3. logarithmic transform method (e.g., Barbosa et al., 1999)"<<endl;
    }

    cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,ParaVariable->G_THax,ParaVariable->N_mod,m_1_n, 1, 0.0, deltaT_preHax, 1);
    cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,ParaVariable->G_THay,ParaVariable->N_mod,m_1_n, 1, 0.0, deltaT_preHay, 1);
    cblas_dgemv(CblasRowMajor,CblasNoTrans,N_data,N_mod,1.0,ParaVariable->G_TZa,ParaVariable->N_mod,m_1_n, 1, 0.0, deltaT_preZa, 1);

    double cosIsinA, cosIcosA, sinI;
    double total_Bxyz;
        
    #pragma omp parallel for shared(N_data,deltaT_preHax,deltaT_preHay,deltaT_preZa,deltaT_preTa) private(i)
    for(i=0;i<N_data;i++){
        
        total_Bxyz = sqrt(ParaVariable->geomag_ref_x[i]*ParaVariable->geomag_ref_x[i] 
            + ParaVariable->geomag_ref_y[i]*ParaVariable->geomag_ref_y[i]
            + ParaVariable->geomag_ref_z[i]*ParaVariable->geomag_ref_z[i]);
        cosIcosA = ParaVariable->geomag_ref_x[i] / total_Bxyz;
        cosIsinA = ParaVariable->geomag_ref_y[i] / total_Bxyz;
        sinI     = ParaVariable->geomag_ref_z[i] / total_Bxyz;
        deltaT_pre[i] = deltaT_preHax[i]*cosIcosA + deltaT_preHay[i]*cosIsinA + deltaT_preZa[i]*sinI;
    }

    save_invrslt_predata_Lcrv(ParaVariable->model,ParaVariable->obs_pos_mag,N_data, N_mod,
        para_new,m_1_n,paraOpen->paraInvOrNot,
        deltaT_pre,deltaT_preHax,deltaT_preHay,deltaT_preZa,deltaT_preTa,
        deltaT_pre_2,
        phi_result,fileLog);

	finish = clock();
	cout << "The present inversion totally cost : " << (double)(finish - start) / CLOCKS_PER_SEC << "seconds" << endl;
	FILE * fRstLog1;
	fRstLog1 = fopen(fileLog, "a");

	fprintf(fRstLog1, "************************** ");
	fprintf(fRstLog1, "************************** \n\n");

	fprintf(fRstLog1, "************************** ");
	fprintf(fRstLog1, "************************** \n\n");

	fprintf(fRstLog1, "The present inversion totally cost : %lf seconds\n\n", (double)(finish - start) / CLOCKS_PER_SEC);
	fclose(fRstLog1);


    delete[] deltaT_pre;
    delete[] deltaT_pre_2;

    delete[] deltaT_preHax;
    delete[] deltaT_preHay;
    delete[] deltaT_preZa;
    delete[] deltaT_preTa;

    delete[] m_0_n1;
    delete[] m_1_n;
	return;
}


void ForwardModel_using_InvsSen(int N_data, int N_mod, TESSEROID *model, OBSMAGPOS *obs_pos, 
    double * G_Hax, double * G_Hay, double * G_Za)
{
    int i,j;
    double * mag = new double[N_mod];
    for (i = 0; i < N_mod; i++) {
        mag[i] = model[i].suscept;
    }
    double * Hax = new double [N_data];
    double * Hay = new double [N_data];
    double * Za = new double [N_data];
    double * deltaT = new double [N_data];
    double * Ta = new double [N_data];
    double * deltaT_2 = new double [N_data];
    
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N_data, N_mod, 1.0, G_Hax, N_mod, mag, 1, 0.0, Hax, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N_data, N_mod, 1.0, G_Hay, N_mod, mag, 1, 0.0, Hay, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N_data, N_mod, 1.0, G_Za, N_mod, mag, 1, 0.0, Za, 1);

    FILE * fpOut = fopen("result_deltaT_ALL_prediction.dat","w");

    for(i=0; i<N_data;i++){
        double cosIsinA, cosIcosA, sinI;
        double total_Bxyz;
        total_Bxyz = sqrt(obs_pos[i].Bx*obs_pos[i].Bx + obs_pos[i].By*obs_pos[i].By
            + obs_pos[i].Bz*obs_pos[i].Bz);
        cosIcosA = obs_pos[i].Bx / total_Bxyz;
        cosIsinA = obs_pos[i].By / total_Bxyz;
        sinI     = obs_pos[i].Bz / total_Bxyz;
        // multiply (-1) cause the input Bz component is Down positive
        // while the programe of Eldar Baykiev is Up positive
        deltaT[i]   = Hax[i]*cosIcosA + Hay[i]*cosIsinA + Za[i]*sinI;
        Ta[i]       = sqrt(Hax[i]*Hax[i] + Hay[i]*Hay[i] + Za[i]*Za[i]);
        deltaT_2[i] = sqrt( (Hax[i]+obs_pos[i].Bx)*(Hax[i]+obs_pos[i].Bx) 
            + (Hay[i]+obs_pos[i].By)*(Hay[i]+obs_pos[i].By) 
            + (Za[i] +obs_pos[i].Bz)*(Za[i] +obs_pos[i].Bz) )
            - total_Bxyz;

        fprintf(fpOut, "%12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n",
            obs_pos[i].lon, obs_pos[i].lat, obs_pos[i].height, deltaT[i], Hax[i], Hay[i], 
            Za[i], Ta[i], deltaT_2[i]);
    }
    fclose(fpOut);
}