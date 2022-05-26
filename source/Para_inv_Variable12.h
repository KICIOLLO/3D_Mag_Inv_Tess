#include "Para_inv_open12.h"
#include "Basic_inv_Prepare12.h"

using namespace std;
double mu = 4*PI*1e-7;  // magnetic permeability of vacuum

class Para_inv_Variable
{ 
public: 
    Para_inv_Variable(Para_inv_open * paraOpen); 
    int Reset_Orimdl();
    int copy_mag_abm(Para_inv_open * para);

    unsigned long long N_data; // number of points of the total-field magnetic anomaly 
    unsigned long long N_mod;  // number of meshes

    // About Tessroid models
    TESSB_ARGS args;
    GLQ *glq_lon, *glq_lat, *glq_r;
    TESSEROID *model;
    OBSMAGPOS *obs_pos_mag;

    double lon, lat, height;
    double ratio1, ratio2, ratio3;

    double * obs_deltaT;
    double * obs_deltaTerr;

    double * obs_deltaT_2;
    double * obs_deltaTerr_2;
    
    double * obs_deltaTHax;
    double * obs_deltaTerrHax;
    double * obs_deltaTHay;
    double * obs_deltaTerrHay;
    double * obs_deltaTZa;
    double * obs_deltaTerrZa;
    double * obs_deltaTTa;
    double * obs_deltaTerrTa;

    double * geomag_ref_x;
    double * geomag_ref_y;
    double * geomag_ref_z;

    // about spatial weighting functions
    double * modelsmoothfun_s;

    double * modelsmoothfun_x;
    int * modelsmoothfun_xdiffdrct;
    double * modelsmoothfun_y;
    int * modelsmoothfun_ydiffdrct;
    double * modelsmoothfun_z;
    int * modelsmoothfun_zdiffdrct;

    // about norm-functions
    double * p1;
    double * p2;
    double * px2;
    double * py2;
    double * pz2;
    double * epsilon1;
    double * epsilon2;
    double * epsilonx2;
    double * epsilony2;
    double * epsilonz2;

    // about reference/minimum/maximum/original model
    double * m_ref_n;
    double * m_min_n;
    double * m_max_n;
    double * m_0_n;

    // about sensitivity matrices
    double * G_THax;
    double * G_THay;
    double * G_TZa;
    
    // about length scales 
    double * m_alpha_s;
    double * m_alpha_x;
    double * m_alpha_y;
    double * m_alpha_z;

    // about dip angles
    double * m_angle_phi;
    double * m_angle_theta;
    double * m_angle_psi;

    // records of errors when reading files
    int err_num;
    
    // the indices of neightbours along lon, lat, and in raidal directions
    int * index_lon;
    int * index_lat;
    int * index_depth;

    ~Para_inv_Variable();
};

// calculate the sensitivity kernel matrices of three components of magnetic anomalous vector
int cal_senmtx_effsto_inv(unsigned long long N_data, unsigned long long N_mod, TESSEROID *model, 
	OBSMAGPOS *obs_pos_mag,TESSB_ARGS args, double ratio1,double ratio2,double ratio3,
	double * G_THax,double * G_THay,double * G_TZa);


Para_inv_Variable::Para_inv_Variable(Para_inv_open * paraOpen)
{
    int i, j;
    err_num = 0;
    double ggt_1, ggt_2, ggt_3;

    args.verbose = 0;
    args.logtofile = 0;
    args.lon_order = 2;
    args.lat_order = 2;
    args.r_order = 2;
    args.adaptative = 1;
    args.ratio1 = 0; /* zero means use the default for the program */
    args.ratio2 = 0;
    args.ratio3 = 0;

    if (args.ratio1 != 0)
        ratio1 = args.ratio1;
    else
        ratio1 = TESSEROID_GXZ_SIZE_RATIO;

    if (args.ratio2 != 0)
        ratio2 = args.ratio2;
    else
        ratio2 = TESSEROID_GYZ_SIZE_RATIO;

    if (args.ratio3 != 0)
        ratio3 = args.ratio3;
    else
        ratio3 = TESSEROID_GZZ_SIZE_RATIO;

    glq_lon = glq_new(args.lon_order, -1, 1);
    glq_lat = glq_new(args.lat_order, -1, 1);
    glq_r   = glq_new(args.r_order, -1, 1);

    /* Read the tesseroid model file */
    log_info("Reading magnetic tesseroid model from file %s", paraOpen->fmesh_data);

    FILE *modelfile = NULL;
    modelfile = fopen(paraOpen->fmesh_data, "r");

    model = read_mag_tess_model(modelfile, &N_mod);
    fclose(modelfile);
    if (model == NULL){
        err_num++;
        cout<<endl<<endl<<" ----------  Warning ! ---------- "<<endl;
        cout<<endl<<endl<<" ----------  Warning ! ---------- "<<endl;
        cout<<"  Please check the mesh_file : "<<paraOpen->fmesh_data<<"  !"<<endl;
        cout<<"  Something is wrong when reading mesh_file."<<endl<<endl;
    }

    log_info("Total of %llu tesseroid(s) read", N_mod);

    /* Read each computation point from stdin and calculate */
    log_info("Reading the coordinates, references field components,");
    log_info("  total-field anomaly value and its standard deviation");
    log_info("  from file %s", paraOpen->fpobs_data);

    FILE *obsposfile = NULL;
    obsposfile = fopen(paraOpen->fpobs_data, "r");
    obs_pos_mag = read_mag_pos_position_magdata_err(obsposfile, &N_data);
    fclose(obsposfile);
    if (obs_pos_mag == NULL){
        err_num++;
        cout<<endl<<endl<<" ----------  Warning ! ---------- "<<endl;
        cout<<endl<<endl<<" ----------  Warning ! ---------- "<<endl;
        cout<<"  Please check the mag_anomaly_file : "<<paraOpen->fpobs_data<<"  !"<<endl;
        cout<<"  Something is wrong when reading mag_anomaly_file."<<endl<<endl;
    }
    log_info("Total of %llu observe position(s) and ... read", N_data);


    obs_deltaT = new double [N_data];
    obs_deltaTerr = new double [N_data];

    obs_deltaT_2 = new double [N_data];
    obs_deltaTerr_2 = new double [N_data];

    obs_deltaTHax = new double [N_data];
    obs_deltaTerrHax = new double [N_data];
    obs_deltaTHay = new double [N_data];
    obs_deltaTerrHay = new double [N_data];
    obs_deltaTZa = new double [N_data];
    obs_deltaTerrZa = new double [N_data];
    obs_deltaTTa = new double [N_data];
    obs_deltaTerrTa = new double [N_data];

    geomag_ref_x = new double [N_data];
    geomag_ref_y = new double [N_data];
    geomag_ref_z = new double [N_data];

    unsigned long long i_llu;
    for(i_llu=0ULL; i_llu<N_data;i_llu++){
        obs_deltaT[i_llu]  = obs_pos_mag[i_llu].deltaT;
        obs_deltaTerr[i_llu]  = obs_pos_mag[i_llu].deltaTerr;
    }

    for(i_llu=0ULL; i_llu<N_data;i_llu++){
        obs_deltaTHax[i_llu]  = obs_pos_mag[i_llu].deltaTHax;
        obs_deltaTerrHax[i_llu]  = obs_pos_mag[i_llu].deltaTerrHax;
    }

    for(i_llu=0ULL; i_llu<N_data;i_llu++){
        obs_deltaTHay[i_llu]  = obs_pos_mag[i_llu].deltaTHay;
        obs_deltaTerrHay[i_llu]  = obs_pos_mag[i_llu].deltaTerrHay;
    }
    
    for(i_llu=0ULL; i_llu<N_data;i_llu++){
        obs_deltaTZa[i_llu]  = obs_pos_mag[i_llu].deltaTZa;
        obs_deltaTerrZa[i_llu]  = obs_pos_mag[i_llu].deltaTerrZa;
    }

    for(i_llu=0ULL; i_llu<N_data;i_llu++){
        obs_deltaTTa[i_llu]  = obs_pos_mag[i_llu].deltaTTa;
        obs_deltaTerrTa[i_llu]  = obs_pos_mag[i_llu].deltaTerrTa;
    }

    for(i_llu=0ULL; i_llu<N_data;i_llu++){
        obs_deltaT_2[i_llu]  = obs_pos_mag[i_llu].deltaT_2;
        obs_deltaTerr_2[i_llu]  = obs_pos_mag[i_llu].deltaTerr_2;
    }

    for(i_llu=0ULL; i_llu<N_data; i_llu++) {
        geomag_ref_x[i_llu] = obs_pos_mag[i_llu].Bx;
        geomag_ref_y[i_llu] = obs_pos_mag[i_llu].By;
        geomag_ref_z[i_llu] = obs_pos_mag[i_llu].Bz;
    }

    // 
    modelsmoothfun_s = new double [N_mod];

    modelsmoothfun_x = new double[N_mod];
    modelsmoothfun_xdiffdrct = new int[N_mod];
    modelsmoothfun_y = new double[N_mod];
    modelsmoothfun_ydiffdrct = new int[N_mod];
    modelsmoothfun_z = new double[N_mod];
    modelsmoothfun_zdiffdrct = new int[N_mod];
    
    // 
    p1 = new double[N_mod];
    p2 = new double[N_mod];
    px2 = new double[N_mod];
    py2 = new double[N_mod];
    pz2 = new double[N_mod];
    epsilon1 = new double[N_mod];
    epsilon2 = new double[N_mod];
    epsilonx2 = new double[N_mod];
    epsilony2 = new double[N_mod];
    epsilonz2 = new double[N_mod];

    // 
    m_ref_n = new double[N_mod];
    m_min_n = new double[N_mod];
    m_max_n = new double[N_mod];
    m_0_n = new double[N_mod];

    // 
    m_alpha_s = new double [N_mod];
    m_alpha_x = new double [N_mod];
    m_alpha_y = new double [N_mod];
    m_alpha_z = new double [N_mod];

    // 
    m_angle_phi = new double [N_mod];
    m_angle_theta = new double [N_mod];
    m_angle_psi = new double [N_mod];


    int err_read_paraopen_inv;
    err_read_paraopen_inv = read_paraopen_inv(paraOpen, N_mod, N_data, 
        modelsmoothfun_s,modelsmoothfun_x,modelsmoothfun_xdiffdrct, 
        modelsmoothfun_y,modelsmoothfun_ydiffdrct,
        modelsmoothfun_z,modelsmoothfun_zdiffdrct,p1,p2,px2,py2,pz2,epsilon1,epsilon2,
        epsilonx2,epsilony2,epsilonz2,m_ref_n,m_min_n,m_max_n,m_0_n,
        m_alpha_s,m_alpha_x,m_alpha_y,m_alpha_z,m_angle_phi,m_angle_theta,m_angle_psi);


    if(err_read_paraopen_inv != 0){
        cout<<endl<<endl<<" ----------  Warning !!  ----------- "<<endl;
        cout<< "  --- Errors are found when reading the setting files."<<endl;
        cout<< "  --- Total number of errors are found: "<<err_read_paraopen_inv<<endl;
        err_num++;
    }


    index_lon = new int [N_mod];
    index_lat = new int [N_mod];
    index_depth = new int [N_mod];
    
    int N_mod_tmp = N_mod;
    Aim_difference_index(model, N_mod_tmp, index_lon, index_lat, index_depth,
        modelsmoothfun_xdiffdrct,modelsmoothfun_ydiffdrct,modelsmoothfun_zdiffdrct);
    
    cout<<endl;
    cout<<"Constructing the indices for difference matrices : Done"<<endl<<endl;
    // SaveFile_int("index_lon_variable.dat", N_mod, index_lon);
	// SaveFile_int("index_lat_variable.dat", N_mod, index_lat);
	// SaveFile_int("index_depth_variable.dat", N_mod, index_depth);

    unsigned long long N_sensitivety = N_data * N_mod;

    G_THax = new double[N_sensitivety];
    G_THay = new double[N_sensitivety];
    G_TZa = new double[N_sensitivety];

    int return_sen;
    return_sen = cal_senmtx_effsto_inv(N_data, N_mod, model, obs_pos_mag, args, 
        ratio1, ratio2, ratio3, G_THax, G_THay, G_TZa);
    if(return_sen != N_data){
        err_num++;
        cout<<endl<<endl<<" ----------  Warning ! ---------- "<<endl;
        cout<<endl<<endl<<" ----------  Warning ! ---------- "<<endl;
        cout<<" Errors exist when calculating the Sensitivity Kernal Matrices!!!"<<endl<<endl;
    }
    cout<<endl;
    cout<<"Calculating the sensitivity kernel matrix : Done"<<endl<<endl;

};


Para_inv_Variable::~Para_inv_Variable()
{
	if(model != NULL)
        delete[] model;

    if(obs_pos_mag != NULL)
        delete[] obs_pos_mag;

	if(obs_deltaT != NULL)
		delete[] obs_deltaT;

	if(obs_deltaTerr != NULL)
		delete[] obs_deltaTerr;

    if(obs_deltaT_2 != NULL)
        delete[] obs_deltaT_2;

    if(obs_deltaTerr_2 != NULL)
        delete[] obs_deltaTerr_2;

    if(geomag_ref_x != NULL)
        delete[] geomag_ref_x;

    if(geomag_ref_y != NULL)
        delete[] geomag_ref_y;

    if(geomag_ref_z != NULL)
        delete[] geomag_ref_z;

	if(obs_deltaTHax != NULL)
		delete[] obs_deltaTHax;

	if(obs_deltaTerrHax != NULL)
		delete[] obs_deltaTerrHax;

	if(obs_deltaTHay != NULL)
		delete[] obs_deltaTHay;

	if(obs_deltaTerrHay != NULL)
		delete[] obs_deltaTerrHay;

	if(obs_deltaTZa != NULL)
		delete[] obs_deltaTZa;

	if(obs_deltaTerrZa != NULL)
		delete[] obs_deltaTerrZa;

	if(obs_deltaTTa != NULL)
		delete[] obs_deltaTTa;

	if(obs_deltaTerrTa != NULL)
		delete[] obs_deltaTerrTa;


	if(modelsmoothfun_s != NULL)
		delete[] modelsmoothfun_s;

	if(modelsmoothfun_x != NULL)
		delete[] modelsmoothfun_x;

	if(modelsmoothfun_xdiffdrct != NULL)
		delete[] modelsmoothfun_xdiffdrct;

	if(modelsmoothfun_y != NULL)
		delete[] modelsmoothfun_y;

	if(modelsmoothfun_ydiffdrct != NULL)
		delete[] modelsmoothfun_ydiffdrct;

	if(modelsmoothfun_z != NULL)
		delete[] modelsmoothfun_z;

	if(modelsmoothfun_zdiffdrct != NULL)
		delete[] modelsmoothfun_zdiffdrct;

	if(p1 != NULL)
		delete[] p1;

	if(p2 != NULL)
		delete[] p2;

	if(px2 != NULL)
		delete[] px2;

	if(py2 != NULL)
		delete[] py2;

	if(pz2 != NULL)
		delete[] pz2;

	if(epsilon1 != NULL)
		delete[] epsilon1;

	if(epsilon2 != NULL)
		delete[] epsilon2;

	if(epsilonx2 != NULL)
		delete[] epsilonx2;

	if(epsilony2 != NULL)
		delete[] epsilony2;

	if(epsilonz2 != NULL)
		delete[] epsilonz2;

	if(m_ref_n != NULL)
		delete[] m_ref_n;

	if(m_min_n != NULL)
		delete[] m_min_n;

	if(m_max_n != NULL)
		delete[] m_max_n;

	if(m_0_n != NULL)
		delete[] m_0_n;

	if(G_THax != NULL)
		delete[] G_THax;

	if(G_THay != NULL)
		delete[] G_THay;

	if(G_TZa != NULL)
		delete[] G_TZa;

    if(m_alpha_s != NULL)
        delete[] m_alpha_s;
    
    if(m_alpha_x != NULL)
        delete[] m_alpha_x;

    if(m_alpha_y != NULL)
        delete[] m_alpha_y;

    if(m_alpha_z != NULL)
        delete[] m_alpha_z;

    if(m_angle_phi != NULL)
        delete[] m_angle_phi;

    if(m_angle_theta != NULL)
        delete[] m_angle_theta;

    if(m_angle_psi != NULL)
        delete[] m_angle_psi;

	cout<<"The present Para_inv_Variable object has been deleted!!!"<<endl;
};



int cal_senmtx_effsto_inv(unsigned long long N_data, unsigned long long N_mod, TESSEROID *model, 
	OBSMAGPOS *obs_pos_mag,TESSB_ARGS args, double ratio1,double ratio2,double ratio3,
	double * G_THax,double * G_THay,double * G_TZa)
{
    unsigned long long i,j;
    
    long nProcess;
    #pragma omp parallel for \
        shared(N_data,N_mod,model,obs_pos_mag,args,G_THax,G_THay,G_TZa,ratio1,ratio2,ratio3) \
        private(i,j,nProcess)
    for (i = 0ULL; i < N_data; i++) {
        nProcess=(i*100.0/N_data);
        printf("\b\b\b\b%2ld%% ", nProcess);

        for (j = 0ULL; j < N_mod; j++) {
            unsigned long long temp_num = i * N_mod + j;

            double ggt_x1, ggt_x2, ggt_x3;
            double ggt_y1, ggt_y2, ggt_y3;
            double ggt_z1, ggt_z2, ggt_z3;

            double B_to_H;
            double M_vect[3];
            double M_vect_p[3];
            
            B_to_H = 1 / (M_0);//IMPORTANT
            M_vect[0] = model[j].Bx * B_to_H;
            M_vect[1] = model[j].By * B_to_H;
            M_vect[2] = (-1)* model[j].Bz * B_to_H;
            // multiply (-1) cause the input Bz component is Down positive
            // while the programe of Eldar Baykiev is Up positive

            M_vect_p[0] = 0;
            M_vect_p[1] = 0;
            M_vect_p[2] = 0;

            conv_vect_fast(M_vect, (model[j].w + model[j].e)*0.5, (model[j].s + model[j].n)*0.5,
                obs_pos_mag[i].lon, obs_pos_mag[i].lat, M_vect_p);

            GLQ *glq_lon1, *glq_lat1, *glq_r1;

            glq_lon1 = glq_new(args.lon_order, -1, 1);
            glq_lat1 = glq_new(args.lat_order, -1, 1);
            glq_r1   = glq_new(args.r_order, -1, 1);

            if (args.adaptative)
            {
                ggt_x1 = calc_tess_model_adapt(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat, 
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gxx, ratio1);
                ggt_x2 = calc_tess_model_adapt(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat, 
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gxy, ratio2);
                ggt_x3 = calc_tess_model_adapt(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat, 
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gxz, ratio3);

                G_THax[temp_num] = M_0 * EOTVOS2SI*(ggt_x1 * M_vect_p[0] + ggt_x2 * M_vect_p[1] + ggt_x3 * M_vect_p[2]) 
                    / (G*model[j].density * 4 * PI);

                ggt_y1 = calc_tess_model_adapt(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat,
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gxy, ratio1);
                ggt_y2 = calc_tess_model_adapt(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat,
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gyy, ratio2);
                ggt_y3 = calc_tess_model_adapt(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat,
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gyz, ratio3);

                G_THay[temp_num] = M_0 * EOTVOS2SI*(ggt_y1 * M_vect_p[0] + ggt_y2 * M_vect_p[1] + ggt_y3 * M_vect_p[2]) 
                    / (G*model[j].density * 4 * PI);

                ggt_z1 = calc_tess_model_adapt(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat,
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gxz, ratio1);
                ggt_z2 = calc_tess_model_adapt(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat,
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gyz, ratio2);
                ggt_z3 = calc_tess_model_adapt(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat,
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gzz, ratio3);

                G_TZa[temp_num] = (-1) * M_0 * EOTVOS2SI*(ggt_z1 * M_vect_p[0] + ggt_z2 * M_vect_p[1] + ggt_z3 * M_vect_p[2]) 
                    / (G*model[j].density * 4 * PI);

            }
            else {
                ggt_x1 = calc_tess_model(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat, 
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gxx);
                ggt_x2 = calc_tess_model(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat, 
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gxy);
                ggt_x3 = calc_tess_model(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat, 
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gxz);

                G_THax[temp_num] = M_0 * EOTVOS2SI*(ggt_x1 * M_vect_p[0] + ggt_x2 * M_vect_p[1] + ggt_x3 * M_vect_p[2])
                    / (G*model[j].density * 4 * PI);

                ggt_y1 = calc_tess_model(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat,
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gxy);
                ggt_y2 = calc_tess_model(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat,
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gyy);
                ggt_y3 = calc_tess_model(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat,
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gyz);

                G_THay[temp_num] = M_0 * EOTVOS2SI*(ggt_y1 * M_vect_p[0] + ggt_y2 * M_vect_p[1] + ggt_y3 * M_vect_p[2])
                    / (G*model[j].density * 4 * PI);

                ggt_z1 = calc_tess_model(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat,
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gxz);
                ggt_z2 = calc_tess_model(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat,
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gyz);
                ggt_z3 = calc_tess_model(&model[j], 1, obs_pos_mag[i].lon, obs_pos_mag[i].lat,
                    obs_pos_mag[i].height + MEAN_EARTH_RADIUS, glq_lon1, glq_lat1, glq_r1, &tess_gzz);

                G_TZa[temp_num] = (-1) * M_0 * EOTVOS2SI*(ggt_z1 * M_vect_p[0] + ggt_z2 * M_vect_p[1] + ggt_z3 * M_vect_p[2])
                    / (G*model[j].density * 4 * PI);
                // multiply (-1) cause the input Bz component is Down positive
                // while the programe of Eldar Baykiev is Up positive
            }
            glq_free(glq_lon1);
            glq_free(glq_lat1);
            glq_free(glq_r1);
        }
    }
    return N_data;
}