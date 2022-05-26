#include <fstream>
#include <iostream>
#include <cstring>
using namespace std;

#define RECL 200
#define MAXINBUFF RECL+14
#define MAXREAD MAXINBUFF-2
#define PATH MAXREAD


class Para_inv_open
{
public:
    Para_inv_open(char * str_open);
    Para_inv_open(char args[13][MAXREAD], int argc);

    
    int Print_Log(char * fileLog);

    // 2022-5-26 12:30:25 This first version can only invert deltaT
    //                    The next updation would include the three components 
    //                      and other quantities like Ta and deltaT_2
    int paraInvOrNot[6];

    char fpobs_data[256];

    char fmesh_data[256];
    
    double alpha[4];

    // Using GCV to determine the optimal regularinzation parameter or not
    int withCalGCVcurve;

    // altitude for depth weighting calculation, upward positive
    double z0_deepw_real;
    // type of depth weighting function, 1 for general, 2 for sensitivity based.
    int para_deepweight_type;
    double beta;
    double para_new;

    double refsus,minsus,maxsus,orisus;

    // para for spatial weighting functions
    int withspatsmooth;
    char fmodelsmoothfun_data[256];

    int withnormweigt;
    char fnormweighfun_data[256];

    int withrefmodel;
    int withorimodel;
    int withminmodel;
    int withmaxmodel;
    
    // using which method for susceptibility bound constraints
    // its value must be the below values!
    // 1. general or compulsury constraint!
    // 2. logarithmic barrier method (LBM, e.g., Li and Oldenburg, 2003)
    // 3. logarithmic transform method (e.g., Barbosa et al., 1999)
    int whichBoundConstrainMthd;



    char frefmodel[256];
    char forimodel[256];
    char fminmodel[256];
    char fmaxmodel[256];

    // 
    int withLengthScale;
    int withDipAngles;

    char flengthscale[256];
    char fdipangles[256];

    int err_flag; // recording the errors 

    ~Para_inv_open();
};


Para_inv_open::Para_inv_open(char * str_open)
{
    FILE* fpIn = fopen(str_open, "rt");
    
    err_flag = 0;
    char temp_str[256];
    
    paraInvOrNot[0] = 1;
    paraInvOrNot[1] = 0;
    paraInvOrNot[2] = 0;
    paraInvOrNot[3] = 0;
    paraInvOrNot[4] = 0;
    paraInvOrNot[5] = 0;


    int num_invert_data = 0;
    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%s\n",fpobs_data);
    cout<<"The total-field magnetic anomaly for inversion is ** "<<fpobs_data<<" **"<<endl<<endl;

    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%s\n",fmesh_data);
    cout<<"The mesh for inversion is ** "<<fmesh_data<<" **"<<endl<<endl;

    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%lf %lf %lf %lf\n",alpha,alpha+1,alpha+2,alpha+3);
    printf("Alpha_s/x/y/z values are %lf %lf %lf %lf \n\n",alpha[0],alpha[1],alpha[2],alpha[3]);

    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%d\n",&withCalGCVcurve);
    printf("Using GCV to determine the optimal regularization parameter or not : %d\n\n",withCalGCVcurve);

    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%d %lf \n",&para_deepweight_type,&z0_deepw_real);
    cout<<"The type of depth weighting function : "<<para_deepweight_type<<endl;
    cout<<"  The z0 value when the type is 1 : "<<z0_deepw_real<<" meters (positive upward)"<<endl;

    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%lf %lf \n",&beta,&para_new);
    cout<<"  The beta value for depth weighting function : "<<beta<<endl;
    cout<<"    Default value beta is 3 when para_deepweight_type = 1; "<<endl;
    cout<<"                      and 1 when para_deepweight_type = 2. "<<endl<<endl;

    double miu = pow(10,para_new);
    cout<<"The default regularization parameter is 10^"<<para_new<<endl<<endl;

    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%d\n",&whichBoundConstrainMthd);
    if(whichBoundConstrainMthd == 1 || whichBoundConstrainMthd == 2 || whichBoundConstrainMthd == 3){
        cout<<"The method used for susceptibility bound constraints : "<<whichBoundConstrainMthd<<endl;
        cout<< "  1. general or compulsury constraint!"<<endl;
        cout<< "  2. logarithmic barrier method (LBM, e.g., Li and Oldenburg, 2003)"<<endl;
        cout<< "  3. logarithmic transform method (e.g., Barbosa et al., 1999)"<<endl<<endl;
    }
    else{
        cout<<endl<<endl<<" ----------  Warning !!  ----------- "<<endl;
        cout<< "  its value must be the below values!"<<endl;
        cout<< "  1. general or compulsury constraint!"<<endl;
        cout<< "  2. logarithmic barrier method (LBM, e.g., Li and Oldenburg, 2003)"<<endl;
        cout<< "  3. logarithmic transform method (e.g., Barbosa et al., 1999)"<<endl<<endl;
        err_flag ++;
    }

    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%lf %lf %lf %lf\n",&refsus,&orisus,&minsus,&maxsus);
    cout<<"The default reference value of susceptibility is "<<    refsus<<endl;
    cout<<"The default   initial value of susceptibility is "<<    orisus<<endl;
    cout<<"The default   minimum value of susceptibility is "<<    minsus<<endl;
    cout<<"The default   maximum value of susceptibility is "<<    maxsus<<endl<<endl;


    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%d\n",&withspatsmooth);
    fscanf(fpIn,"%s\n",fmodelsmoothfun_data);
    if(withspatsmooth != 0){
        cout<<"Using custom spatial weighting functions : ";
        cout<<fmodelsmoothfun_data<<endl<<endl;
    }
    else
        cout<<"Using default spatial weighting functions"<<endl;
    

    withnormweigt = 0;

    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%d\n",&withrefmodel);
    fscanf(fpIn,"%s\n",frefmodel);
    if(withrefmodel != 0){
        cout<<"Using custom reference model file : ";
        cout<<frefmodel<<endl;
    }
    else
        cout<<"Using default reference model."<<endl;
    

    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%d\n",&withorimodel);
    fscanf(fpIn,"%s\n",forimodel);
    if(withorimodel != 0){
        cout<<"Using custom initial model file : ";
        cout<<forimodel<<endl;
    }
    else
        cout<<"Using default initial model."<<endl;
    

    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%d\n",&withminmodel);
    fscanf(fpIn,"%s\n",fminmodel);
    if(withminmodel != 0){
        cout<<"Using custom minimum model file : ";
        cout<<fminmodel<<endl;
    }
    else
        cout<<"Using default minimum model"<<endl;
    

    fscanf(fpIn,"%s\n",temp_str);
    fscanf(fpIn,"%d\n",&withmaxmodel);
    fscanf(fpIn,"%s\n",fmaxmodel);
    
    if(withmaxmodel != 0){
        cout<<"Using custom maximum model file : ";
        cout<<fmaxmodel<<endl<<endl;
    }
    else
        cout<<"Using default maximum model"<<endl<<endl;;
    
    // the below functions would be updated in the next version
    withLengthScale = 0;
    withDipAngles = 0;

    fclose(fpIn);
};

Para_inv_open::Para_inv_open(char args[13][MAXREAD], int argc)
{
    // FILE* fpIn = fopen(str_open, "rt");
    //if(fpIn == NULL)
    //{
    //    printf("\nSorry!\nFailed to open the file: Make_mag_info_info_20131001T214803.txt");
    //    printf( ".\n");
    //    return -1;
    //}
    err_flag = 0;
    // char temp_str[256];
    
    paraInvOrNot[0] = 1;
    paraInvOrNot[1] = 0;
    paraInvOrNot[2] = 0;
    paraInvOrNot[3] = 0;
    paraInvOrNot[4] = 0;
    paraInvOrNot[5] = 0;


    int num_invert_data = 0;
    strncpy(fpobs_data,args[1],MAXREAD);

    cout<<"The total-field magnetic anomaly for inversion is ** "<<fpobs_data<<" **"<<endl<<endl;

    strncpy(fmesh_data,args[2],MAXREAD);
    cout<<"The mesh for inversion is ** "<<fmesh_data<<" **"<<endl<<endl;

    alpha[0] = 1;
    alpha[1] = 1;
    alpha[2] = 1;
    alpha[3] = 1;
    printf("Alpha_s/x/y/z values are %lf %lf %lf %lf \n\n",alpha[0],alpha[1],alpha[2],alpha[3]);

    withCalGCVcurve = 0;
    printf("Using GCV to determine the optimal regularization parameter or not : %d\n\n",withCalGCVcurve);

    para_deepweight_type = 2;
    z0_deepw_real = 4000;
    cout<<"The type of depth weighting function : "<<para_deepweight_type<<endl;
    cout<<"  The z0 value when the type is 1 : "<<z0_deepw_real<<" meters (positive upward)"<<endl;

    beta = 1;
    para_new = atof(args[3]);
    cout<<"  The beta value for depth weighting function : "<<beta<<endl;
    cout<<"    Default value beta is 3 when para_deepweight_type = 1; "<<endl;
    cout<<"                      and 1 when para_deepweight_type = 2. "<<endl<<endl;

    double miu = pow(10,para_new);
    cout<<"The default regularization parameter is 10^"<<para_new<<endl<<endl;

    whichBoundConstrainMthd = 2;
    cout<<"The method used for susceptibility bound constraints : "<<whichBoundConstrainMthd<<endl;
    cout<< "  1. general or compulsury constraint!"<<endl;
    cout<< "  2. logarithmic barrier method (LBM, e.g., Li and Oldenburg, 2003)"<<endl;
    cout<< "  3. logarithmic transform method (e.g., Barbosa et al., 1999)"<<endl<<endl;

    refsus = 0.0;
    orisus = 0.001;
    minsus = 0.0;
    maxsus = 10.0;
    cout<<"The default reference value of susceptibility is "<<    refsus<<endl;
    cout<<"The default   initial value of susceptibility is "<<    orisus<<endl;
    cout<<"The default   minimum value of susceptibility is "<<    minsus<<endl;
    cout<<"The default   maximum value of susceptibility is "<<    maxsus<<endl<<endl;

    withspatsmooth = 0;
    cout<<"Using default spatial weighting functions"<<endl<<endl;

    withnormweigt = 0;

    withrefmodel = 0;
    cout<<"Using default reference model."<<endl;

    withorimodel = 0;
    cout<<"Using default initial model."<<endl;

    withminmodel = 0;
    cout<<"Using default minimum model"<<endl;

    withmaxmodel = 0;
    cout<<"Using default maximum model"<<endl<<endl;

    withLengthScale = 0;

    withDipAngles = 0;

};


int Para_inv_open::Print_Log(char * fileLog)
{
    FILE * fRstLog;
    fRstLog = fopen(fileLog,"a");

    int i;

    if(paraInvOrNot[0] != 0){
        cout<<"The total-field magnetic anomaly for inversion is ** "<<fpobs_data<<" **"<<endl<<endl;
        fprintf(fRstLog,"The total-field magnetic anomaly for inversion ** %s **\n\n", fpobs_data);
    }

    fprintf(fRstLog,"The mesh for inversion is **  %s **\n\n", fmesh_data);

    printf("Alpha_s/x/y/z values are %lf %lf %lf %lf \n\n",alpha[0],alpha[1],alpha[2],alpha[3]);
    fprintf(fRstLog,"The aplha values in model objective function ：%lf %lf %lf %lf\n\n",alpha[0],alpha[1],alpha[2],alpha[3]);

    printf("Using GCV to determine the optimal regularization parameter or not : %d\n\n",withCalGCVcurve);
    fprintf(fRstLog,"Using GCV to determine the optimal regularization parameter or not : %d\n\n",withCalGCVcurve);
        
    cout<<"The type of depth weighting function : "<<para_deepweight_type<<endl;
    cout<<"  The z0 value when the type is 1 : "<<z0_deepw_real<<" meters (positive upward)"<<endl;
    fprintf(fRstLog,"The type of depth weighting function : %d\n\n",para_deepweight_type);
    fprintf(fRstLog,"  The z0 value when the type is 1 : %lf\n\n",z0_deepw_real);

    cout<<"  The beta value for depth weighting function : "<<beta<<endl;
    cout<<"    Default value beta is 3 when para_deepweight_type = 1; "<<endl;
    cout<<"                      and 1 when para_deepweight_type = 2. "<<endl<<endl;
    cout<<"The default regularization parameter is 10^"<<para_new<<endl<<endl;

    fprintf(fRstLog,"  The beta value for depth weighting function : %lf\n\n",beta);
    fprintf(fRstLog,"  The default regularization parameter is 10^ %lf\n\n",para_new);

    cout<<"The method used for susceptibility bound constraints : "<<whichBoundConstrainMthd<<endl;
    cout<< "  1. general or compulsury constraint!"<<endl;
    cout<< "  2. logarithmic barrier method (LBM, e.g., Li and Oldenburg, 2003)"<<endl;
    cout<< "  3. logarithmic transform method (e.g., Barbosa et al., 1999)"<<endl<<endl;
    fprintf(fRstLog,"The method used for susceptibility bound constraints : %d\n\n",whichBoundConstrainMthd);

    cout<<"The default reference value of susceptibility is "<<    refsus<<endl;
    cout<<"The default   initial value of susceptibility is "<<    orisus<<endl;
    cout<<"The default   minimum value of susceptibility is "<<    minsus<<endl;
    cout<<"The default   maximum value of susceptibility is "<<    maxsus<<endl<<endl;
    fprintf(fRstLog,"The default reference and initial value of susceptibility : %lf %lf\n\n",refsus,orisus);
    fprintf(fRstLog,"The default   minimum and maximum value of susceptibility : %lf %lf\n\n",minsus,maxsus);

    if (withspatsmooth != 0){
        cout<<"Using custom spatial weighting functions : ";
        cout<<fmodelsmoothfun_data<<endl<<endl;
    }
    else
        cout<<"Using default spatial weighting functions"<<endl;

    fprintf(fRstLog,"Using custom spatial weighting functions : %d\n",withspatsmooth);
    fprintf(fRstLog,"       The filename of the the above SPF : %s\n\n", fmodelsmoothfun_data);


    if(withrefmodel != 0){
        cout<<"Using custom reference model file : ";
        cout<<frefmodel<<endl<<endl;
    }
    else
        cout<<"Using default reference model."<<endl;

    fprintf(fRstLog,"Using custom reference model file : %d\n",withrefmodel);
    fprintf(fRstLog,"                         Filename : %s\n\n", frefmodel);

    if(withorimodel != 0){
        cout<<"Using custom initial model file : ";
        cout<<forimodel<<endl<<endl;
    }
    else
        cout<<"Using default initial model."<<endl;

    fprintf(fRstLog,"Using custom initial model file : %d\n",withorimodel);
    fprintf(fRstLog,"                       Filename : %s\n\n", forimodel);

    if(withminmodel != 0){
        cout<<"Using custom minimum model file : ";
        cout<<fminmodel<<endl<<endl;
    }
    else
        cout<<"Using default minimum model"<<endl;

    fprintf(fRstLog,"Using custom minimum model file : %d\n",withminmodel);
    fprintf(fRstLog,"                       Filename : %s\n\n", fminmodel);

    if(withmaxmodel != 0){
        cout<<"Using custom maximum model file : ";
        cout<<fmaxmodel<<endl<<endl;
    }
    else
        cout<<"Using default maximum model"<<endl<<endl;
    fprintf(fRstLog,"Using custom maximum model file : %d\n",withmaxmodel);
    fprintf(fRstLog,"                       Filename : %s\n\n", fmaxmodel);

    fclose(fRstLog);
    return 1;
}

Para_inv_open::~Para_inv_open()
{
    cout<<"The present Para_inv_open object has been deleted!!!"<<endl;
}
