#include <math.h>
#include "logger.h"
#include "geometry.h"
#include "glq.h"
#include "constants.h"
/*
Functions that calculate the gravitational potential and its first and second
derivatives for the tesseroid.

The gravity gradients can be calculated using the general formula of
Grombein et al. (2010).
The integrals are solved using the Gauss-Legendre Quadrature rule
(Asgharzadeh et al., 2007).

The derivatives of the potential are made with respect to the local coordinate
system x->North, y->East, z->Up (away from center of the Earth).

To maintain the standard convention, only for component gz the z axis is
inverted, so a positive density results in positive gz.

Example
-------

To calculate the gzz component due to a tesseroid on a regular grid:

    #include <stdio.h>
    #include "glq.h"r
    #include "constants.h"
    #include "grav_tess.h"

    int main()
    {
        TESSEROID tess = {1000, 44, 46, -1, 1, MEAN_EARTH_RADIUS - 100000,
                          MEAN_EARTH_RADIUS};
        GLQ *glqlon, *glqlat, *glqr;
        double lon, lat, r = MEAN_EARTH_RADIUS + 1500000, res;
        int order = 8;

        glqlon = glq_new(order, tess.w, tess.e);
        glqlat = glq_new(order, tess.s, tess.n);
        glqr = glq_new(order, tess.r1, tess.r2);

        for(lat = 20; lat <= 70; lat += 0.5)
        {
            for(lon = -25; lon <= 25; lon += 0.5)
            {
                res = tess_gzz(tess, lon, lat, r, *glqlon, *glqlat, *glqr);
                printf("%g %g %g\n", lon, lat, res);
            }
        }

        glq_free(glqlon);
        glq_free(glqlat);
        glq_free(glqr);

        return 0;
    }

References
----------

Asgharzadeh, M.F., von Frese, R.R.B., Kim, H.R., Leftwich, T.E. & Kim, J.W.
(2007): Spherical prism gravity effects by Gauss-Legendre quadrature integration.
Geophysical Journal International, 169, 1-11.

Grombein, T.; Seitz, K.; Heck, B. (2010): Untersuchungen zur effizienten
Berechnung topographischer Effekte auf den Gradiententensor am Fallbeispiel der
Satellitengradiometriemission GOCE.
KIT Scientific Reports 7547, ISBN 978-3-86644-510-9, KIT Scientific Publishing,
Karlsruhe, Germany.
*/

#ifndef _TESSEROIDS_GRAV_TESS_H_
#define _TESSEROIDS_GRAV_TESS_H_


/* Needed for definition of TESSEROID */
#include "geometry.h"
/* Needed for definition of GLQ */
#include "glq.h"


/* Calculates the field of a tesseroid model at a given point. */
double calc_tess_model(TESSEROID *model, int size, double lonp, double latp, double rp, GLQ *glq_lon, GLQ *glq_lat, GLQ *glq_r, double (*field)(TESSEROID, double, double, double, GLQ, GLQ, GLQ))
{
    double res;
    int tess;

    res = 0;
    for(tess = 0; tess < size; tess++)
    {
        if(lonp >= model[tess].w && lonp <= model[tess].e &&
           latp >= model[tess].s && latp <= model[tess].n &&
           rp >= model[tess].r1 && rp <= model[tess].r2)
        {
            log_warning("Point (%g %g %g) is on tesseroid %d: %g %g %g %g %g %g %g. Can't guarantee accuracy.",
                        lonp, latp, rp - MEAN_EARTH_RADIUS, tess,
                        model[tess].w, model[tess].e, model[tess].s,
                        model[tess].n, model[tess].r2 - MEAN_EARTH_RADIUS,
                        model[tess].r1 - MEAN_EARTH_RADIUS,
                        model[tess].density);
        }
        glq_set_limits(model[tess].w, model[tess].e, glq_lon);
        glq_set_limits(model[tess].s, model[tess].n, glq_lat);
        glq_set_limits(model[tess].r1, model[tess].r2, glq_r);
        res += field(model[tess], lonp, latp, rp, *glq_lon, *glq_lat, *glq_r);
    }
    return res;
}


/* Adaptatively calculate the field of a tesseroid model at a given point */
// 2018-4-12 11:52:39 ???????????????????????? Tesseroid 
double calc_tess_model_adapt(TESSEROID *model, int size, double lonp, double latp, double rp, GLQ *glq_lon, GLQ *glq_lat, GLQ *glq_r, double (*field)(TESSEROID, double, double, double, GLQ, GLQ, GLQ), double ratio)
{
    double res, dist, lont, latt, rt, d2r = PI/180.;
    int tess;
    TESSEROID split[8];

    res = 0;
    for(tess = 0; tess < size; tess++)
    {
        rt = model[tess].r2;
        lont = 0.5*(model[tess].w + model[tess].e);
        latt = 0.5*(model[tess].s + model[tess].n);
        dist = sqrt(rp*rp + rt*rt - 2*rp*rt*(sin(d2r*latp)*sin(d2r*latt) +
                    cos(d2r*latp)*cos(d2r*latt)*cos(d2r*(lonp - lont))));

        /* Would get stuck in infinite loop if dist = 0 and get wrong results if
           inside de tesseroid. Still do the calculation but warn user that it's
           probably wrong. */
        if(lonp >= model[tess].w && lonp <= model[tess].e &&
           latp >= model[tess].s && latp <= model[tess].n &&
           rp >= model[tess].r1 && rp <= model[tess].r2)
        {
            log_warning("Point (%g %g %g) is on top of tesseroid %d: %g %g %g %g %g %g %g. Can't guarantee accuracy.",
                        lonp, latp, rp - MEAN_EARTH_RADIUS, tess,
                        model[tess].w, model[tess].e, model[tess].s,
                        model[tess].n, model[tess].r2 - MEAN_EARTH_RADIUS,
                        model[tess].r1 - MEAN_EARTH_RADIUS,
                        model[tess].density);
            glq_set_limits(model[tess].w, model[tess].e, glq_lon);
            glq_set_limits(model[tess].s, model[tess].n, glq_lat);
            glq_set_limits(model[tess].r1, model[tess].r2, glq_r);
            res += field(model[tess], lonp, latp, rp, *glq_lon, *glq_lat,
                         *glq_r);
        }
        /* Check if the computation point is at an acceptable distance. If not
           split the tesseroid using the given ratio */
        else if(
            dist < ratio*MEAN_EARTH_RADIUS*d2r*(model[tess].e - model[tess].w) ||
            dist < ratio*MEAN_EARTH_RADIUS*d2r*(model[tess].n - model[tess].s) ||
            dist < ratio*(model[tess].r2 - model[tess].r1))
        {
            log_debug("Splitting tesseroid %d (%g %g %g %g %g %g %g) at point (%g %g %g) using ratio %g",
                        tess, model[tess].w, model[tess].e, model[tess].s,
                        model[tess].n, model[tess].r2 - MEAN_EARTH_RADIUS,
                        model[tess].r1 - MEAN_EARTH_RADIUS, model[tess].density,
                        lonp, latp, rp - MEAN_EARTH_RADIUS, ratio);
            /* Do it recursively until ratio*size is smaller than distance */
            split_tess(model[tess], split);
            res += calc_tess_model_adapt(split, 8, lonp, latp, rp, glq_lon,
                                        glq_lat, glq_r, field, ratio);
        }
        else
        {
            glq_set_limits(model[tess].w, model[tess].e, glq_lon);
            glq_set_limits(model[tess].s, model[tess].n, glq_lat);
            glq_set_limits(model[tess].r1, model[tess].r2, glq_r);
            res += field(model[tess], lonp, latp, rp, *glq_lon, *glq_lat,
                         *glq_r);
        }
    }
    return res;
}


double tess_gxx(TESSEROID tess, double lonp, double latp, double rp, GLQ glq_lon,
                GLQ glq_lat, GLQ glq_r)
{
    double d2r = PI/180., l_sqr, kphi, coslatp, coslatc, sinlatp, sinlatc,
           coslon, rc, kappa, res;
    register int i, j, k;

    coslatp = cos(d2r*latp);
    sinlatp = sin(d2r*latp);

    res = 0;

    for(k = 0; k < glq_lon.order; k++)
    {
        for(j = 0; j < glq_lat.order; j++)
        {
            for(i = 0; i < glq_r.order; i++)
            {
                rc = glq_r.nodes[i];
                sinlatc = sin(d2r*glq_lat.nodes[j]);
                coslatc = cos(d2r*glq_lat.nodes[j]);
                coslon = cos(d2r*(lonp - glq_lon.nodes[k]));

                l_sqr = rp*rp + rc*rc - 2*rp*rc*(sinlatp*sinlatc +
                                                 coslatp*coslatc*coslon);

                kphi = coslatp*sinlatc - sinlatp*coslatc*coslon;

                kappa = rc*rc*coslatc;

                res += glq_lon.weights[k]*glq_lat.weights[j]*glq_r.weights[i]*
                       kappa*(3*rc*kphi*rc*kphi - l_sqr)/pow(l_sqr, 2.5);
            }
        }
    }

    res *= SI2EOTVOS*G*tess.density*d2r*(tess.e - tess.w)*d2r*(tess.n - tess.s)*
           (tess.r2 - tess.r1)*0.125;

    return res;
}


/* Calculates gxy caused by a tesseroid. */
double tess_gxy(TESSEROID tess, double lonp, double latp, double rp, GLQ glq_lon,
                GLQ glq_lat, GLQ glq_r)
{
    double d2r = PI/180., l_sqr, kphi, coslatp, coslatc, sinlatp, sinlatc,
           coslon, sinlon, rc, kappa, deltax, deltay, res;
    register int i, j, k;

    coslatp = cos(d2r*latp);
    sinlatp = sin(d2r*latp);

    res = 0;

    for(k = 0; k < glq_lon.order; k++)
    {
        for(j = 0; j < glq_lat.order; j++)
        {
            for(i = 0; i < glq_r.order; i++)
            {
                rc = glq_r.nodes[i];
                sinlatc = sin(d2r*glq_lat.nodes[j]);
                coslatc = cos(d2r*glq_lat.nodes[j]);
                coslon = cos(d2r*(lonp - glq_lon.nodes[k]));
                sinlon = sin(d2r*(glq_lon.nodes[k] - lonp));

                l_sqr = rp*rp + rc*rc - 2*rp*rc*(sinlatp*sinlatc +
                                                 coslatp*coslatc*coslon);

                kphi = coslatp*sinlatc - sinlatp*coslatc*coslon;

                kappa = rc*rc*coslatc;

                deltax = rc*kphi;

                deltay = rc*coslatc*sinlon;

                res += glq_lon.weights[k]*glq_lat.weights[j]*glq_r.weights[i]*
                       kappa*(3*deltax*deltay)/pow(l_sqr, 2.5);
            }
        }
    }

    res *= SI2EOTVOS*G*tess.density*d2r*(tess.e - tess.w)*d2r*(tess.n - tess.s)*
           (tess.r2 - tess.r1)*0.125;

    return res;
}


/* Calculates gxz caused by a tesseroid. */
double tess_gxz(TESSEROID tess, double lonp, double latp, double rp, GLQ glq_lon,
                GLQ glq_lat, GLQ glq_r)
{
    double d2r = PI/180., l_sqr, kphi, coslatp, coslatc, sinlatp, sinlatc,
           coslon, cospsi, rc, kappa, deltax, deltaz, res;
    register int i, j, k;

    coslatp = cos(d2r*latp);
    sinlatp = sin(d2r*latp);

    res = 0;

    for(k = 0; k < glq_lon.order; k++)
    {
        for(j = 0; j < glq_lat.order; j++)
        {
            for(i = 0; i < glq_r.order; i++)
            {
                rc = glq_r.nodes[i];
                sinlatc = sin(d2r*glq_lat.nodes[j]);
                coslatc = cos(d2r*glq_lat.nodes[j]);
                coslon = cos(d2r*(lonp - glq_lon.nodes[k]));

                cospsi = sinlatp*sinlatc + coslatp*coslatc*coslon;

                l_sqr = rp*rp + rc*rc - 2*rp*rc*cospsi;

                kphi = coslatp*sinlatc - sinlatp*coslatc*coslon;

                kappa = rc*rc*coslatc;

                deltax = rc*kphi;

                deltaz = rc*cospsi - rp;

                res += glq_lon.weights[k]*glq_lat.weights[j]*glq_r.weights[i]*
                       kappa*(3*deltax*deltaz)/pow(l_sqr, 2.5);
            }
        }
    }

    res *= SI2EOTVOS*G*tess.density*d2r*(tess.e - tess.w)*d2r*(tess.n - tess.s)*
           (tess.r2 - tess.r1)*0.125;

    return res;
}


/* Calculates gyy caused by a tesseroid. */
double tess_gyy(TESSEROID tess, double lonp, double latp, double rp, GLQ glq_lon,
                GLQ glq_lat, GLQ glq_r)
{
    double d2r = PI/180., l_sqr, coslatp, coslatc, sinlatp, sinlatc,
           coslon, sinlon, rc, kappa, deltay, res;
    register int i, j, k;

    coslatp = cos(d2r*latp);
    sinlatp = sin(d2r*latp);

    res = 0;

    for(k = 0; k < glq_lon.order; k++)
    {
        for(j = 0; j < glq_lat.order; j++)
        {
            for(i = 0; i < glq_r.order; i++)
            {
                rc = glq_r.nodes[i];
                sinlatc = sin(d2r*glq_lat.nodes[j]);
                coslatc = cos(d2r*glq_lat.nodes[j]);
                coslon = cos(d2r*(lonp - glq_lon.nodes[k]));
                sinlon = sin(d2r*(glq_lon.nodes[k] - lonp));

                l_sqr = rp*rp + rc*rc - 2*rp*rc*(sinlatp*sinlatc +
                                                 coslatp*coslatc*coslon);

                kappa = rc*rc*coslatc;

                deltay = rc*coslatc*sinlon;

                res += glq_lon.weights[k]*glq_lat.weights[j]*glq_r.weights[i]*
                       kappa*(3*deltay*deltay - l_sqr)/pow(l_sqr, 2.5);
            }
        }
    }

    res *= SI2EOTVOS*G*tess.density*d2r*(tess.e - tess.w)*d2r*(tess.n - tess.s)*
           (tess.r2 - tess.r1)*0.125;

    return res;
}


/* Calculates gyz caused by a tesseroid. */
double tess_gyz(TESSEROID tess, double lonp, double latp, double rp, GLQ glq_lon,
                GLQ glq_lat, GLQ glq_r)
{
    double d2r = PI/180., l_sqr, coslatp, coslatc, sinlatp, sinlatc,
           coslon, sinlon, cospsi, rc, kappa, deltay, deltaz, res;
    register int i, j, k;

    coslatp = cos(d2r*latp);
    sinlatp = sin(d2r*latp);

    res = 0;

    for(k = 0; k < glq_lon.order; k++)
    {
        for(j = 0; j < glq_lat.order; j++)
        {
            for(i = 0; i < glq_r.order; i++)
            {
                rc = glq_r.nodes[i];
                sinlatc = sin(d2r*glq_lat.nodes[j]);
                coslatc = cos(d2r*glq_lat.nodes[j]);
                coslon = cos(d2r*(lonp - glq_lon.nodes[k]));
                sinlon = sin(d2r*(glq_lon.nodes[k] - lonp));

                cospsi = sinlatp*sinlatc + coslatp*coslatc*coslon;

                l_sqr = rp*rp + rc*rc - 2*rp*rc*cospsi;

                kappa = rc*rc*coslatc;

                deltay = rc*coslatc*sinlon;

                deltaz = rc*cospsi - rp;

                res += glq_lon.weights[k]*glq_lat.weights[j]*glq_r.weights[i]*
                       kappa*(3*deltay*deltaz)/pow(l_sqr, 2.5);
            }
        }
    }

    res *= SI2EOTVOS*G*tess.density*d2r*(tess.e - tess.w)*d2r*(tess.n - tess.s)*
           (tess.r2 - tess.r1)*0.125;

    return res;
}


/* Calculates gzz caused by a tesseroid. */
double tess_gzz(TESSEROID tess, double lonp, double latp, double rp, GLQ glq_lon,
                GLQ glq_lat, GLQ glq_r)
{
    double d2r = PI/180., l_sqr, coslatp, coslatc, sinlatp, sinlatc,
           coslon, cospsi, rc, kappa, deltaz, res;
    register int i, j, k;

    coslatp = cos(d2r*latp);
    sinlatp = sin(d2r*latp);

    res = 0;

    for(k = 0; k < glq_lon.order; k++)
    {
        for(j = 0; j < glq_lat.order; j++)
        {
            for(i = 0; i < glq_r.order; i++)
            {
                rc = glq_r.nodes[i];
                sinlatc = sin(d2r*glq_lat.nodes[j]);
                coslatc = cos(d2r*glq_lat.nodes[j]);
                coslon = cos(d2r*(lonp - glq_lon.nodes[k]));

                cospsi = sinlatp*sinlatc + coslatp*coslatc*coslon;

                l_sqr = rp*rp + rc*rc - 2*rp*rc*cospsi;

                kappa = rc*rc*coslatc;

                deltaz = rc*cospsi - rp;

                res += glq_lon.weights[k]*glq_lat.weights[j]*glq_r.weights[i]*
                       kappa*(3*deltaz*deltaz - l_sqr)/pow(l_sqr, 2.5);
            }
        }
    }

    res *= SI2EOTVOS*G*tess.density*d2r*(tess.e - tess.w)*d2r*(tess.n - tess.s)*
           (tess.r2 - tess.r1)*0.125;

    return res;
}


#endif


