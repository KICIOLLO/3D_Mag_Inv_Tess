#include "Basic_Function12.h"

int main(int argc, char *argv[])
{	
	#ifdef MAC
		ccommand(argc, argv);
	#endif

	char fpara_filename[256]; // the filename of the parameter file

	int iarg;
	char args[13][MAXREAD];

	for (iarg=0; iarg<argc; iarg++)
		if (argv[iarg] != NULL)
			strncpy(args[iarg],argv[iarg],MAXREAD);

	// printing out version number and header
	cout<<endl<<"3D_Mag_Inv_Tess.o v1.2.0 - May 24, 2022"<<endl<<endl;

	if (argc==1){
		cout<<"Please input this command for help: 3D_Mag_Inv_Tess.o h"<<endl<<endl;
		exit(2);
	}


	if ((argc==2)&&((*(args[1])=='h')||(*(args[1])=='?')||(args[1][1]=='?'))){
		cout<<endl<<endl<<"USAGE:"<<endl;
		cout<<"Please use this executable program like this:"<<endl;
		cout<<endl;
		cout<<"For help: 3D_Mag_Inv_Tess.o h"<<endl<<endl;

		cout<<"Advanced parameter_file usage (recommended): 3D_Mag_Inv_Tess.o f parameter_file"<<endl<<endl;
		cout<<"    *3D_Mag_Inv_Tess.o* : the name of executable program."<<endl;
		cout<<"     *parameter_file* : includes information about the magnetic_anomaly_file, "<<endl;
		cout<<"                        mesh_file, and parameters used for inversion."<<endl;
		cout<<endl;

		cout<<"Simple commandline usage: 3D_Mag_Inv_Tess.o mag_anomaly_file mesh_file regu_para"<<endl<<endl;
		cout<<"   *mag_anomaly_file* : filename (including its Extension) of the magnetic anomaly."<<endl;
		cout<<"          *mesh_file* : filename (including its Extension) of the mesh."<<endl;
		cout<<"          *regu_para* : log10 of the regularization parameter for the present inversion."<<endl;
		cout<<"                        e.g., input is 5.6, the regularization parameter is actually 10^5.6."<<endl;
		cout<<"     Other parameters : all default values or settings."<<endl;
		cout<<endl;

		exit(2);
	}

	if ((argc==3)&&(*(args[1])=='f')) {
		strncpy(fpara_filename,args[2],MAXREAD);
		Para_inv_open * paraOpen = new Para_inv_open(fpara_filename);
		if (paraOpen->err_flag > 0){
			cout<< " --- Error(s) exist in the parameter files:  "<<fpara_filename<<endl;
			cout<< " --- Details were shown in the therminal."<<endl;

			return 0;
		}
		Para_inv_Variable * ParaVariable = new Para_inv_Variable(paraOpen);
		if (ParaVariable->err_num > 0){
			cout<< " --- Erros(s) exist when reading files for inversion."<<endl;
			cout<< " --- Details were shown in the therminal."<<endl;
			return 0;
		}
		Inversion(paraOpen, ParaVariable);
	}

	if (argc==4){
		Para_inv_open * paraOpen = new Para_inv_open(args, argc);
		
		Para_inv_Variable * ParaVariable = new Para_inv_Variable(paraOpen);
		if (ParaVariable->err_num > 0){
			cout<< " --- Erros(s) exist when reading files for inversion."<<endl;
			cout<< " --- Details were shown in the therminal."<<endl;
			return 0;
		}
		Inversion(paraOpen, ParaVariable);
	}

	// cin.get();
	return 1;
}