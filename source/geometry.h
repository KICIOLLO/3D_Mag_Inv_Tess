#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "constants.h"
#include "logger.h"
/*
Data structures for geometric elements and functions that operate on them.
Defines the TESSEROID, SPHERE, and PRISM structures.
*/

#ifndef _TESSEROIDS_GEOMETRY_H_
#define _TESSEROIDS_GEOMETRY_H_


/* Store information on a tesseroid */
typedef struct tess_struct {
    /* s, n, w, e in degrees. r1 and r2 are the smaller and larger radius */
    double density; /* in SI units */
    double w; /* western longitude border in degrees */
    double e; /* eastern longitude border in degrees */
    double s; /* southern latitude border in degrees */
    double n; /* northern latitude border in degrees */
	double top; /* top depth in SI units */
	double bot; /* bottom depth in SI units */
    double r1; /* smallest radius border in SI units */
    double r2; /* largest radius border in SI units */
	double suscept; /* magnetic susceptibility */
	double Bx; /* x-component of ambient magnetic field */
	double By; /* y-component of ambient magnetic field */
	double Bz; /* z-component of ambient magnetic field */
	//double Rx;
	//double Ry;
	//double Rz;
} TESSEROID;


/* Split a tesseroid into 8. */
void split_tess(TESSEROID tess, TESSEROID *split)
{
	double dlon = 0.5*(tess.e - tess.w),
		dlat = 0.5*(tess.n - tess.s),
		dr = 0.5*(tess.r2 - tess.r1),
		ws[2], ss[2], r1s[2];
	int i, j, k, t = 0;

	ws[0] = tess.w;
	ws[1] = tess.w + dlon;
	ss[0] = tess.s;
	ss[1] = tess.s + dlat;
	r1s[0] = tess.r1;
	r1s[1] = tess.r1 + dr;
	for (k = 0; k < 2; k++)
	{
		for (j = 0; j < 2; j++)
		{
			for (i = 0; i < 2; i++)
			{
				split[t].w = ws[i];
				split[t].e = ws[i] + dlon;
				split[t].s = ss[j];
				split[t].n = ss[j] + dlat;
				split[t].r1 = r1s[k];
				split[t].r2 = r1s[k] + dr;
				split[t].density = tess.density;
				t++;
			}
		}
	}
}

#endif
