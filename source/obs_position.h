#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
/*
Data structures for observation positions and functions that operate on them.
*/

#ifndef _COMPUTATIONAL_POSITION_H_
#define _COMPUTATIONAL_POSITION_H_


/* Store information on a observe position */
typedef struct obs_position_struct {
	/* lon, lat in degrees (WGS84 ellipsoid). height are the atitude above the Geoid */
	double lon;
	double lat;
	double height;
	double Bx; /* x-component of ambient magnetic field */
	double By; /* y-component of ambient magnetic field */
	double Bz; /* z-component of ambient magnetic field */
} OBSPOS;

typedef struct obs_position_mag_struct {
	/* lon, lat in degrees (WGS84 ellipsoid). height are the atitude above the Geoid */
	double lon;
	double lat;
	double height;

	double deltaT;
    double deltaTerr;
	double deltaT_2;
    double deltaTerr_2;
    double deltaTHax;
    double deltaTerrHax;
    double deltaTHay;
    double deltaTerrHay;
    double deltaTZa;
    double deltaTerrZa;
    double deltaTTa;
    double deltaTerrTa;

	double Bx; /* x-component of ambient magnetic field */
	double By; /* y-component of ambient magnetic field */
	double Bz; /* z-component of ambient magnetic field */
} OBSMAGPOS;


#endif
