/*
Input and output parsing tools.
*/


#ifndef _TESSEROIDS_PARSERS_H_
#define _TESSEROIDS_PARSERS_H_

/* Needed for definition of TESSEROID and PRISM */
#include "geometry.h"
// needed for definition of OBSPOS
#include "obs_position.h"
/* Need for the definition of FILE */
#include <stdio.h>
#include "logger.h"
// #include "version.h"
#include "constants.h"
/** Store basic input arguments and option flags */
typedef struct basic_args
{
	char *inputfname; /**< name of the input file */
	int verbose; /**< flag to indicate if verbose printing is enabled */
	int logtofile; /**< flag to indicate if logging to a file is enabled */
	char *logfname; /**< name of the log file */
} BASIC_ARGS;

typedef struct tessh_args
{
	int lon_order; /**< glq order in longitude integration */
	int lat_order; /**< glq order in latitude integration */
	int r_order; /**< glq order in radial integration */
	char *modelfname; /**< name of the file with the tesseroid model */
	int verbose; /**< flag to indicate if verbose printing is enabled */
	int logtofile; /**< flag to indicate if logging to a file is enabled */
	char *logfname; /**< name of the log file */
	int adaptative; /**< flat to indicate wether to use the adaptative size
                         of tesseroid algorithm */
	double ratio1; /**< distance-size ratio used for recusive division */
	double ratio2; /**< distance-size ratio used for recusive division */
	double ratio3; /**< distance-size ratio used for recusive division */
} TESSB_ARGS;


typedef struct gradcalc_args
{
	int gridbx_set;
	int gridby_set;
	int gridbz_set;

	char* gridbx_fn;
	char* gridby_fn;
	char* gridbz_fn;

	int out_set;


	int bz_NEU_NED;
	int bz_NEU_NED_set;

	int verbose; /**< flag to indicate if verbose printing is enabled */
	int logtofile; /**< flag to indicate if logging to a file is enabled */

} GRADCALC_ARGS;


/* Strip trailing spaces and newlines from the end of a string */
void strstrip(char *str)
{
	int i;
	for (i = strlen(str) - 1; i >= 0; i--)
	{
		if (str[i] != ' ' && str[i] != '\n' && str[i] != '\r' && str[i] != '\0')
			break;
	}
	str[i + 1] = '\0';
}


/* Read a single tesseroid from a string */
int gets_mag_tess(const char *str, TESSEROID *tess)
{
	double w, e, s, n, top, bot, dens, suscept, Bx, By, Bz, Rx, Ry, Rz;
	int nread, nchars;

	nread = sscanf(str, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf%n", &w, &e, &s,
		&n, &top, &bot, &dens, &suscept, &Bx, &By, &Bz, &nchars);
	if (nread != 11 || str[nchars] != '\0')
	{
		return 1;
	}
	tess->w = w;
	tess->e = e;
	tess->s = s;
	tess->n = n;
	tess->top = top;
	tess->bot = bot;
	tess->r1 = MEAN_EARTH_RADIUS + bot;
	tess->r2 = MEAN_EARTH_RADIUS + top;
	tess->density = dens;
	tess->suscept = suscept;
	tess->Bx = Bx;
	tess->By = By;
	tess->Bz = Bz;      
	//tess->Rx = Rx;
	//tess->Ry = Ry;
	//tess->Rz = Rz;
	return 0;
}



//ELDAR BAYKIEV////////////////////////////////
TESSEROID * read_mag_tess_model(FILE *modelfile, unsigned long long *size)
{
	TESSEROID *model, *tmp;
	unsigned long long buffsize = 300ULL;
	int line, badinput = 0, error_exit = 0;
	char sbuff[10000];

	/* Start with a single buffer allocation and expand later if necessary */
	model = (TESSEROID *)malloc(buffsize * sizeof(TESSEROID));
	if (model == NULL)
	{
		log_error("problem allocating initial memory to load tesseroid model.");
		return NULL;
	}
	*size = 0ULL;
	for (line = 1; !feof(modelfile); line++)
	{
		if (fgets(sbuff, 10000, modelfile) == NULL)
		{
			if (ferror(modelfile))
			{
				log_error("problem encountered reading line %d.", line);
				error_exit = 1;
				break;
			}
		}
		else
		{
			/* Check for comments and blank lines */
			if (sbuff[0] == '#' || sbuff[0] == '\r' || sbuff[0] == '\n')
			{
				continue;
			}
			if (*size == buffsize)
			{
				buffsize += buffsize;
				tmp = (TESSEROID *)realloc(model, buffsize * sizeof(TESSEROID));
				if (tmp == NULL)
				{
					/* Need to free because realloc leaves unchanged in case of
					error */
					free(model);
					log_error("problem expanding memory for tesseroid model.\nModel is too big.");
					return NULL;
				}
				model = tmp;
			}
			/* Remove any trailing spaces or newlines */
			strstrip(sbuff);
			if (gets_mag_tess(sbuff, &model[*size]))
			{
				log_warning("bad/invalid tesseroid at line %d.", line);
				badinput = 1;
				continue;
			}
			(*size)++;
		}
	}
	if (badinput || error_exit)
	{
		free(model);
		return NULL;
	}
	/* Adjust the size of the model */
	if (*size != 0)
	{
		tmp = (TESSEROID *)realloc(model, (*size) * sizeof(TESSEROID));
		if (tmp == NULL)
		{
			/* Need to free because realloc leaves unchanged in case of
			error */
			free(model);
			log_error("problem freeing excess memory for tesseroid model.");
			return NULL;
		}
		model = tmp;
		// free(tmp);
	}
	return model;
}

/* Read a single rectangular prism from a string */

/* Read a single observe position from a string */
int gets_mag_obspos(const char *str, OBSPOS *obs_pos)
{
	int nread, nchars;
	double lon, lat, height;
	double Bx, By, Bz;
	nread = sscanf(str, "%lf %lf %lf %lf %lf %lf%n", &lon, &lat, &height, 
		&Bx, &By, &Bz, &nchars);
	if (nread != 6 || str[nchars] != '\0')
	{
		return 1;
	}
	obs_pos->lon = lon;
	obs_pos->lat = lat;
	obs_pos->height = height;
	obs_pos->Bx = Bx;
	obs_pos->By = By;
	obs_pos->Bz = Bz;
	return 0;
}

// SHIDA SUN ////////////////////////////////
OBSPOS * read_mag_pos_position(FILE *obspos_file, unsigned long long *size)
{
	OBSPOS *obs_pos, *tmp;
	unsigned long long buffsize = 300ULL;
	int line, badinput = 0, error_exit = 0;
	char sbuff[10000];

	/* Start with a single buffer allocation and expand later if necessary */
	obs_pos = (OBSPOS *)malloc(buffsize * sizeof(OBSPOS));
	if (obs_pos == NULL)
	{
		log_error("problem allocating initial memory to load observe positions.");
		return NULL;
	}
	*size = 0ULL;
	for (line = 1; !feof(obspos_file); line++)
	{
		if (fgets(sbuff, 10000, obspos_file) == NULL)
		{
			if (ferror(obspos_file))
			{
				log_error("problem encountered reading line %d.", line);
				error_exit = 1;
				break;
			}
		}
		else
		{
			/* Check for comments and blank lines */
			if (sbuff[0] == '#' || sbuff[0] == '\r' || sbuff[0] == '\n')
			{
				continue;
			}
			if (*size == buffsize)
			{
				buffsize += buffsize;
				tmp = (OBSPOS *)realloc(obs_pos, buffsize * sizeof(OBSPOS));
				if (tmp == NULL)
				{
					/* Need to free because realloc leaves unchanged in case of
					error */
					free(obs_pos);
					log_error("problem expanding memory for observe positions.\nobs_pos is too big.");
					return NULL;
				}
				obs_pos = tmp;
			}
			/* Remove any trailing spaces or newlines */
			strstrip(sbuff);
			if (gets_mag_obspos(sbuff, &obs_pos[*size]))
			{
				log_warning("bad/invalid OBSPOS at line %d.", line);
				badinput = 1;
				continue;
			}
			(*size)++;
		}
	}
	if (badinput || error_exit)
	{
		free(obs_pos);
		return NULL;
	}
	/* Adjust the size of the obs_pos */
	if (*size != 0)
	{
		tmp = (OBSPOS *)realloc(obs_pos, (*size) * sizeof(OBSPOS));
		if (tmp == NULL)
		{
			/* Need to free because realloc leaves unchanged in case of
			error */
			free(obs_pos);
			log_error("problem freeing excess memory for OBSPOS obs_pos.");
			return NULL;
		}
		obs_pos = tmp;
	}
	return obs_pos;
}


/* Read a single observe position and magnetic data from a string */
int gets_mag_obspos_mag_err(const char *str, OBSMAGPOS *obs_pos_mag)
{
	int nread, nchars;
	double lon, lat, height;
	double Bx, By, Bz;
	double tmp1,tmp1_err;
	double tmp2,tmp2_err;
	double tmp3,tmp3_err;
	double tmp4,tmp4_err;
	double tmp5,tmp5_err;
	double tmp6,tmp6_err;
	
	nread = sscanf(str, "%lf %lf %lf %lf %lf %lf %lf %lf%n", 
		&lon, &lat, &height, &Bx, &By, &Bz, 
		&tmp1, &tmp1_err, 
		// &tmp2, &tmp2_err, &tmp3, &tmp3_err, 
		// &tmp4, &tmp4_err, &tmp5, &tmp5_err, &tmp6, &tmp6_err, 
		&nchars);

	if (nread != 8 || str[nchars] != '\0')
	{
		return 1;
	}

	tmp2 = 0; 
	tmp2_err = 0; 
	tmp3 = 0; 
	tmp3_err = 0; 
	
	tmp4 = 0; 
	tmp4_err = 0; 
	tmp5 = 0; 
	tmp5_err = 0; 
	tmp6 = 0; 
	tmp6_err = 0; 


	obs_pos_mag->lon = lon;
	obs_pos_mag->lat = lat;
	obs_pos_mag->height = height;
	obs_pos_mag->Bx = Bx;
	obs_pos_mag->By = By;
	obs_pos_mag->Bz = Bz;
	obs_pos_mag->deltaT = tmp1;
    obs_pos_mag->deltaTerr = tmp1_err;
    obs_pos_mag->deltaTHax = tmp2;
    obs_pos_mag->deltaTerrHax = tmp2_err;
    obs_pos_mag->deltaTHay = tmp3;
    obs_pos_mag->deltaTerrHay = tmp3_err;
    obs_pos_mag->deltaTZa = tmp4;
    obs_pos_mag->deltaTerrZa = tmp4_err;
    obs_pos_mag->deltaTTa = tmp5;
    obs_pos_mag->deltaTerrTa = tmp5_err;
	obs_pos_mag->deltaT_2 = tmp6;
    obs_pos_mag->deltaTerr_2 = tmp6_err;
	return 0;
}


// SHIDA SUN ////////////////////////////////
OBSMAGPOS * read_mag_pos_position_magdata_err(FILE *OBSMAGPOS_file, unsigned long long *size)
{
	OBSMAGPOS *obs_pos_mag, *tmp;
	unsigned long long buffsize = 300ULL;
	int line, badinput = 0, error_exit = 0;
	char sbuff[10000];

	/* Start with a single buffer allocation and expand later if necessary */
	obs_pos_mag = (OBSMAGPOS *)malloc(buffsize * sizeof(OBSMAGPOS));
	if (obs_pos_mag == NULL)
	{
		log_error("problem allocating initial memory to load observe positions.");
		return NULL;
	}
	*size = 0ULL;
	for (line = 1; !feof(OBSMAGPOS_file); line++)
	{
		if (fgets(sbuff, 10000, OBSMAGPOS_file) == NULL)
		{
			if (ferror(OBSMAGPOS_file))
			{
				log_error("problem encountered reading line %d.", line);
				error_exit = 1;
				break;
			}
		}
		else
		{
			/* Check for comments and blank lines */
			if (sbuff[0] == '#' || sbuff[0] == '\r' || sbuff[0] == '\n')
			{
				continue;
			}
			if (*size == buffsize)
			{
				buffsize += buffsize;
				tmp = (OBSMAGPOS *)realloc(obs_pos_mag, buffsize * sizeof(OBSMAGPOS));
				if (tmp == NULL)
				{
					/* Need to free because realloc leaves unchanged in case of
					error */
					free(obs_pos_mag);
					log_error("problem expanding memory for observe positions.\nobs_pos_mag is too big.");
					return NULL;
				}
				obs_pos_mag = tmp;
			}
			/* Remove any trailing spaces or newlines */
			strstrip(sbuff);
			if (gets_mag_obspos_mag_err(sbuff, &obs_pos_mag[*size]))
			{
				log_warning("bad/invalid OBSMAGPOS at line %d.", line);
				badinput = 1;
				continue;
			}
			(*size)++;
		}
	}
	if (badinput || error_exit)
	{
		free(obs_pos_mag);
		return NULL;
	}
	/* Adjust the size of the obs_pos_mag */
	if (*size != 0)
	{
		tmp = (OBSMAGPOS *)realloc(obs_pos_mag, (*size) * sizeof(OBSMAGPOS));
		if (tmp == NULL)
		{
			/* Need to free because realloc leaves unchanged in case of
			error */
			free(obs_pos_mag);
			log_error("problem freeing excess memory for OBSMAGPOS obs_pos_mag.");
			return NULL;
		}
		obs_pos_mag = tmp;
	}
	return obs_pos_mag;
}

#endif
