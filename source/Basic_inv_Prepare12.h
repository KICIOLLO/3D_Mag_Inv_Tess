#include "Basic_function_general.h"
#include "logger.h"
#include "glq.h"
#include "constants.h"
#include "geometry.h"
#include "parsers.h"
#include "linalg.h"
#include "grav_tess.h"
#include "obs_position.h"
#include "Basic_Function_inv12.h"

#define max(x,y)  ( x>y?x:y )
#define min(x,y)  ( x<y?x:y )
using namespace std;


int read_paraopen_inv(Para_inv_open * para, unsigned long long N_mod, unsigned long long N_data, 
    double * modelsmoothfun_s,double * modelsmoothfun_x,int * modelsmoothfun_xdiffdrct, 
    double * modelsmoothfun_y,int * modelsmoothfun_ydiffdrct,
    double * modelsmoothfun_z,int * modelsmoothfun_zdiffdrct,double * p1,double * p2,double * px2,
    double * py2,double * pz2,double * epsilon1,double * epsilon2,double * epsilonx2,double * epsilony2,
    double * epsilonz2,double * m_ref_n,double * m_min_n,double * m_max_n,double * m_0_n,
    double * m_alpha_s,double * m_alpha_x,double * m_alpha_y,double * m_alpha_z,
    double * m_angle_phi,double * m_angle_theta,double * m_angle_psi)
{
    
    unsigned long long i;
    int err_num_tmp = 0;

    // read spatial weighting functions from file
    double smooth1,smooth2,smooth3;
    double smooth0;
    int smooth4,smooth5,smooth6;
    if(para->withspatsmooth != 0){
        FILE * fpModelsmoothfun = fopen(para->fmodelsmoothfun_data,"rt");
        if(fpModelsmoothfun == NULL)
        {
            printf("\nSorry!\nFailed to open the fpModelsmoothfun file");
            printf( ".\n");
            err_num_tmp++;
        }
        i=0ULL;
        while(!feof(fpModelsmoothfun))
        {
            fscanf(fpModelsmoothfun, "%lf %lf %d %lf %d %lf %d\n",&smooth0,&smooth1,&smooth4,&smooth2,&smooth5,&smooth3,&smooth6);
            modelsmoothfun_s[i] = smooth0;
            modelsmoothfun_x[i] = smooth1;
            modelsmoothfun_xdiffdrct[i] = smooth4;
            modelsmoothfun_y[i] = smooth2;
            modelsmoothfun_ydiffdrct[i] = smooth5;
            modelsmoothfun_z[i] = smooth3;
            modelsmoothfun_zdiffdrct[i] = smooth6;
            i++;
        }
        fclose(fpModelsmoothfun);
    }
    else{
        for(i = 0ULL;i<N_mod;i++){
            modelsmoothfun_s[i] = 1;
            modelsmoothfun_x[i] = 1;
            modelsmoothfun_xdiffdrct[i] = 0;
            modelsmoothfun_y[i] = 1;
            modelsmoothfun_ydiffdrct[i] = 0;
            modelsmoothfun_z[i] = 1;
            modelsmoothfun_zdiffdrct[i] = 0;
        }
    }
    cout<<"Reading the spatial weighting functions File : Done"<<endl;


    // read mix-norm from file
    double normm1,normm2,normm3,normm4,normm5,normm6,norms1,norme1;
    if(para->withnormweigt == 1){
        FILE * fpNormweighfun = fopen(para->fnormweighfun_data,"rt");
        if(fpNormweighfun == NULL)
        {
            printf("\nSorry!\nFailed to open the fpNormweighfun file");
            printf( ".\n");
            err_num_tmp++;
        }
        i=0ULL;
        while(!feof(fpNormweighfun))
        {
            fscanf(fpNormweighfun, "%lf %lf %lf %lf %lf %lf %lf %lf\n",&norms1,&norme1,&normm1,&normm2,&normm3,&normm4,&normm5,&normm6);
            p1[i] = 2;
            epsilon1[i] = 0.0001;
            p2[i] = norms1;
            epsilon2[i] = norme1;
            px2[i] = normm1;
            epsilonx2[i] = normm2;
            py2[i] = normm3;
            epsilony2[i] = normm4;
            pz2[i] = normm5;
            epsilonz2[i] = normm6;

            i++;
        }
        fclose(fpNormweighfun);
    }
    else{
        for(i = 0ULL;i<N_mod;i++) {
            p1[i] = 2;
            epsilon1[i] = 0.0001;
            p2[i] = 2;
            epsilon2[i] = 0.0001;
            px2[i] = 2;
            epsilonx2[i] = 0.0001;
            py2[i] = 2;
            epsilony2[i] = 0.0001;
            pz2[i] = 2;
            epsilonz2[i] = 0.0001;
        }
    }


    // read refernce/initial/maximum/minimum models from files
    double refmdl1,orimdl1,minmdl1,maxmdl1;
    if(para->withrefmodel == 1){
        FILE * fprefmodel = fopen(para->frefmodel,"rt");
        if(fprefmodel == NULL)
        {
            printf("\nSorry!\nFailed to open the fprefmodel file");
            printf( ".\n");
            err_num_tmp++;
        }
        i=0ULL;
        while(!feof(fprefmodel))
        {
            fscanf(fprefmodel, "%lf\n",&refmdl1);
            m_ref_n[i] = refmdl1;
            i++;
        }
        fclose(fprefmodel);
    }
    else{
        #pragma omp parallel for shared(N_mod,m_ref_n) private(i)
        for (i=0ULL;i<N_mod;i++) 
            m_ref_n[i] = para->refsus;
    }

    if(para->withorimodel == 1){
        FILE * fporimodel = fopen(para->forimodel,"rt");
        if(fporimodel == NULL)
        {
            printf("\nSorry!\nFailed to open the fporimodel file");
            printf( ".\n");
            err_num_tmp++;
        }
        i=0ULL;
        while(!feof(fporimodel))
        {
            fscanf(fporimodel, "%lf\n",&orimdl1);
            m_0_n[i] = orimdl1;
            i++;
        }
        fclose(fporimodel);
    }
    else{
        #pragma omp parallel for shared(N_mod,m_0_n) private(i)
        for (i=0ULL;i<N_mod;i++) 
            m_0_n[i] = para->orisus;
    }

    if(para->withminmodel == 1){
        FILE * fpminmodel = fopen(para->fminmodel,"rt");
        if(fpminmodel == NULL)
        {
            printf("\nSorry!\nFailed to open the fpminmodel file");
            printf( ".\n");
            err_num_tmp++;
        }
        i=0ULL;
        while(!feof(fpminmodel))
        {
            fscanf(fpminmodel, "%lf\n",&minmdl1);
            m_min_n[i] = minmdl1;
            i++;
        }
        fclose(fpminmodel);
    }
    else{
        #pragma omp parallel for shared(N_mod,m_min_n) private(i)
        for (i=0ULL;i<N_mod;i++) 
            m_min_n[i] = para->minsus;
    }

    if(para->withmaxmodel == 1){
        FILE * fpmaxmodel = fopen(para->fmaxmodel,"rt");
        if(fpmaxmodel == NULL)
        {
            printf("\nSorry!\nFailed to open the fpmaxmodel file");
            printf( ".\n");
            err_num_tmp++;
        }
        i=0ULL;
        while(!feof(fpmaxmodel))
        {
            fscanf(fpmaxmodel, "%lf\n",&maxmdl1);
            m_max_n[i] = maxmdl1;
            i++;
        }
        fclose(fpmaxmodel);
    }
    else{
        #pragma omp parallel for shared(N_mod,m_max_n) private(i)
        for (i=0ULL;i<N_mod;i++) 
            m_max_n[i] = para->maxsus;
    }

    // reading the custom length scales
    double alphas1, alphax1, alphay1, alphaz1;
    if(para->withLengthScale == 1) {
        FILE * fpmixLengthscale = fopen(para->flengthscale,"rt");
        if(fpmixLengthscale == NULL)
        {
            printf("\nSorry!\nFailed to open the fpmixLengthscale file");
            printf( ".\n");
            err_num_tmp++;
        }
        i=0ULL;
        while(!feof(fpmixLengthscale))
        {
            fscanf(fpmixLengthscale, "%lf %lf %lf %lf\n",&alphas1,&alphax1,&alphay1,&alphaz1);
            m_alpha_s[i] = alphas1;
            m_alpha_x[i] = alphax1;
            m_alpha_y[i] = alphay1;
            m_alpha_z[i] = alphaz1;
            i++;
        }
        fclose(fpmixLengthscale);
    }
    else{
        #pragma omp parallel for shared(N_mod,m_alpha_s,m_alpha_x,m_alpha_y,m_alpha_z) private(i)
        for (i=0ULL;i<N_mod;i++) {
            m_alpha_s[i] = para->alpha[0];
            m_alpha_x[i] = para->alpha[1];
            m_alpha_y[i] = para->alpha[2];
            m_alpha_z[i] = para->alpha[3];
        }
    }


    // reading the custom dipping information
    double angle_phi1, angle_theta1, angle_psi1;
    if(para->withDipAngles == 1) {
        FILE * fpdipinfofile = fopen(para->fdipangles,"rt");
        if(fpdipinfofile == NULL)
        {
            printf("\nSorry!\nFailed to open the fpdipinfofile file");
            printf( ".\n");
            err_num_tmp++;
        }
        i=0ULL;
        while(!feof(fpdipinfofile))
        {
            fscanf(fpdipinfofile, "%lf %lf %lf\n",&angle_phi1,&angle_theta1,&angle_psi1);
            m_angle_phi[i] = angle_phi1;
            m_angle_theta[i] = angle_theta1;
            m_angle_psi[i] = angle_psi1;
            i++;
        }
        fclose(fpdipinfofile);
    }
    else{
        #pragma omp parallel for shared(N_mod,m_angle_phi,m_angle_theta,m_angle_psi) private(i)
        for (i=0ULL;i<N_mod;i++) {
            m_angle_phi[i] = 0;
            m_angle_theta[i] = 90;
            m_angle_psi[i] = 0;
        }
    }

    return err_num_tmp;
}

// searching the neighbor index of each cell
void Aim_difference_index(TESSEROID *model, int N_mod, int * index_lon, 
    int * index_lat, int * index_depth,
    int * modelsmoothfun_xdiffdrct, int * modelsmoothfun_ydiffdrct, int * modelsmoothfun_zdiffdrct)
{
    int i, j;
    // initilize the indices
    for (i = 0; i < N_mod; i++){
        index_lon[i] = -1;
        index_lat[i] = -1;
        index_depth[i] = -1;
    }
    
    // start to find the appropriate index in three directions
    #pragma omp parallel for \
        shared(N_mod,model,index_lon,index_lat,index_depth,modelsmoothfun_xdiffdrct,modelsmoothfun_ydiffdrct,modelsmoothfun_zdiffdrct) \
        private(i,j)
    for (i = 0; i < N_mod; i++){
        double center_lon, center_lat, center_depth, tmp_spacing_lon, tmp_spacing_lat, tmp_spacing_depth;
        center_lon = (model[i].w + model[i].e)/2;
        center_lat = (model[i].s + model[i].n)/2;
        center_depth = (model[i].top + model[i].bot)/2;

        tmp_spacing_lon = model[i].e - model[i].w;
        tmp_spacing_lat = model[i].n - model[i].s;
        tmp_spacing_depth = model[i].top - model[i].bot;


        int total_num_tmp;
        total_num_tmp = 0;
        int * tmp_index_60 = new int[60];
        double * tmp_distance_60 = new double[60];
        double tmp_tmp_min_index;
        double tmp_tmp_min;
        int i_tmp;
        for (i_tmp = 0; i_tmp < 60; i_tmp++){
            tmp_index_60[i_tmp] = -9999;
            tmp_distance_60[i_tmp] = -9999;
        }
        double center_lon_j, center_lat_j, center_depth_j; // 
        for (j = 0; j < N_mod; j++){
            if(total_num_tmp >= 60)
                break;

            if(j == i)
                continue;

            center_lon_j = (model[j].w + model[j].e)/2;
            center_lat_j = (model[j].s + model[j].n)/2;
            center_depth_j = (model[j].top + model[j].bot)/2;

            if ( fabs(center_lon_j - center_lon) < 1.8*tmp_spacing_lon ){
                if ( fabs(center_lat_j - center_lat) < 1.8*tmp_spacing_lat ) {
                    if( fabs(center_depth_j - center_depth) < 1.8*tmp_spacing_depth ){
                        tmp_index_60[total_num_tmp] = j;
                        tmp_distance_60[total_num_tmp] = sqrt((center_lon_j - center_lon)*(center_lon_j - center_lon)*100000*100000
                                                        + (center_lat_j - center_lat)*(center_lat_j - center_lat)*100000*100000
                                                        + (center_depth_j - center_depth)*(center_depth_j - center_depth));
                        total_num_tmp++;
                        
                    }
                }
            }
        }
        
        int j_tmp;
        // the index along Longitude or in the South-North direction
        tmp_tmp_min = 10000000000;
        tmp_tmp_min_index = -9999;
        if(modelsmoothfun_xdiffdrct[i] == 1){
            for (j_tmp = 0; j_tmp < total_num_tmp; j_tmp++){
                center_lat_j = (model[tmp_index_60[j_tmp]].s + model[tmp_index_60[j_tmp]].n)/2;
                if (center_lat_j < (center_lat-tmp_spacing_lat/2)){
                    if (tmp_distance_60[j_tmp] < tmp_tmp_min){
                        tmp_tmp_min = tmp_distance_60[j_tmp];
                        tmp_tmp_min_index = tmp_index_60[j_tmp];
                    }
                }
            }
            if((tmp_tmp_min/100000) < 1.2*tmp_spacing_lat){
                index_lat[i] = tmp_tmp_min_index;
            }
        }
        else{
            for (j_tmp = 0; j_tmp < total_num_tmp; j_tmp++){
                center_lat_j = (model[tmp_index_60[j_tmp]].s + model[tmp_index_60[j_tmp]].n)/2;
                if (center_lat_j > (center_lat+tmp_spacing_lat/2)){
                    if (tmp_distance_60[j_tmp] < tmp_tmp_min){
                        tmp_tmp_min = tmp_distance_60[j_tmp];
                        tmp_tmp_min_index = tmp_index_60[j_tmp];
                    }
                }
            }
            if((tmp_tmp_min/100000) < 1.2*tmp_spacing_lat){
                index_lat[i] = tmp_tmp_min_index;
            }
            
        }

        // the index along Latitude or in the West-East direction
        tmp_tmp_min = 10000000000;
        tmp_tmp_min_index = -9999;
        if(modelsmoothfun_ydiffdrct[i] == 1){
            for (j_tmp = 0; j_tmp < total_num_tmp; j_tmp++){
                center_lon_j = (model[tmp_index_60[j_tmp]].w + model[tmp_index_60[j_tmp]].e)/2;
                if (center_lon_j < (center_lon-tmp_spacing_lon/2)){
                    if (tmp_distance_60[j_tmp] < tmp_tmp_min){
                        tmp_tmp_min = tmp_distance_60[j_tmp];
                        tmp_tmp_min_index = tmp_index_60[j_tmp];
                    }
                }
            }
            if((tmp_tmp_min/100000) < 1.2*tmp_spacing_lon){
                index_lon[i] = tmp_tmp_min_index;
            }
        }
        else{
            for (j_tmp = 0; j_tmp < total_num_tmp; j_tmp++){
                center_lon_j = (model[tmp_index_60[j_tmp]].w + model[tmp_index_60[j_tmp]].e)/2;
                if (center_lon_j > (center_lon+tmp_spacing_lon/2)){
                    if (tmp_distance_60[j_tmp] < tmp_tmp_min){
                        tmp_tmp_min = tmp_distance_60[j_tmp];
                        tmp_tmp_min_index = tmp_index_60[j_tmp];
                    }
                }
            }
            if((tmp_tmp_min/100000) < 1.2*tmp_spacing_lon){
                index_lon[i] = tmp_tmp_min_index;
            }
        }

        // the index towards geocenter in radial direction
        tmp_tmp_min = 10000000000;
        tmp_tmp_min_index = -9999;
        if(modelsmoothfun_zdiffdrct[i] == 1){
            for (j_tmp = 0; j_tmp < total_num_tmp; j_tmp++){
                center_depth_j = (model[tmp_index_60[j_tmp]].top + model[tmp_index_60[j_tmp]].bot)/2;
                if (center_depth_j > (center_depth+tmp_spacing_depth/2)){
                    if (tmp_distance_60[j_tmp] < tmp_tmp_min){
                        tmp_tmp_min = tmp_distance_60[j_tmp];
                        tmp_tmp_min_index = tmp_index_60[j_tmp];
                    }
                }
            }
            if( tmp_tmp_min < 1.2*tmp_spacing_depth ){
                index_depth[i] = tmp_tmp_min_index;
            }
        }
        else{
            for (j_tmp = 0; j_tmp < total_num_tmp; j_tmp++){
                center_depth_j = (model[tmp_index_60[j_tmp]].top + model[tmp_index_60[j_tmp]].bot)/2;
                if (center_depth_j < (center_depth-tmp_spacing_depth/2)){
                    if (tmp_distance_60[j_tmp] < tmp_tmp_min){
                        tmp_tmp_min = tmp_distance_60[j_tmp];
                        tmp_tmp_min_index = tmp_index_60[j_tmp];
                    }
                }
            }
            if( tmp_tmp_min < 1.2*tmp_spacing_depth ){
                index_depth[i] = tmp_tmp_min_index;
            }
        }
    }
}