#include <stdlib.h>
#include <math.h>
#include "constants.h"
#include "logger.h"
/*
Functions for implementing a Gauss-Legendre Quadrature numerical integration
(Hildebrand, 1987).

Usage example
-------------

To integrate the cossine function from 0 to 90 degrees:

    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include "src/c/glq.h"

    int main(){
        // Create a new glq structure
        GLQ *glq;
        double result = 0, a = 0, b = 0.5*3.14;
        int i;

        glq = glq_new(5, a, b);

        if(glq == NULL){
            printf("malloc error");
            return 1;
        }

        // Calculate the integral
        for(i = 0; i < glq->order; i++)
            result += glq->weights[i]*cos(glq->nodes[i]);

        // Need to multiply by a scale factor of the integration limits
        result *= 0.5*(b - a);

        printf("Integral of cossine from 0 to 90 degrees = %lf\n", result);

        // Free allocated memory
        glq_free(glq);

        return 0;
    }

References
----------

* Hildebrand, F.B (1987): Introduction to numerical analysis.
  Courier Dover Publications, 2. ed.
*/

#ifndef _TESSEROIDS_GLQ_H_
#define _TESSEROIDS_GLQ_H_


/** \var GLQ_MAXIT
Max iterations of the root-finder algorithm */
const int GLQ_MAXIT = 1000;


/** \var GLQ_MAXERROR
Max error allowed for the root-finder algorithm */
const double GLQ_MAXERROR = 0.000000000000001;


/** Store the nodes and weights needed for a GLQ integration */
typedef struct glq_struct
{
    int order; /**< order of the quadrature, ie number of nodes */
    double *nodes; /**< abscissas or discretization points of the quadrature */
    double *weights; /**< weighting coefficients of the quadrature */
    double *nodes_unscaled; /**< nodes in [-1,1] interval */
} GLQ;


/** Make a new GLQ structure and set all the parameters needed

<b>WARNING</b>: Don't forget to free the memory malloced by this function using
glq_free()!

Prints error and warning messages using the logging.h module.

@param order order of the quadrature, ie number of nodes
@param lower lower integration limit
@param upper upper integration limit

@return GLQ data structure with the nodes and weights calculated. NULL if there
    was an error with allocation.
*/
GLQ * glq_new(int order, double lower, double upper);


/** Free the memory allocated to make a GLQ structure

@param glq pointer to the allocated memory
*/
void glq_free(GLQ *glq);


/** Put the GLQ nodes to the integration limits <b>IN PLACE</b>.

Will replace the values of glq.nodes with ones in the specified integration
limits.

In case the GLQ structure was created with glq_new(), the integration limits can
be reset using this function.

@param lower lower integration limit
@param upper upper integration limit
@param glq pointer to a GLQ structure created with glq_new() and with all
           necessary memory allocated

@return Return code:
    - 0: if everything went OK
    - 1: if invalid order
    - 2: if NULL pointer for nodes or nodes_unscaled
*/
int glq_set_limits(double lower, double upper, GLQ *glq);


/** Calculates the GLQ nodes using glq_next_root.

Nodes will be in the [-1,1] interval. To convert them to the integration limits
use glq_scale_nodes

@param order order of the quadrature, ie how many nodes. Must be >= 2.
@param nodes pre-allocated array to return the nodes.

@return Return code:
    - 0: if everything went OK
    - 1: if invalid order
    - 2: if NULL pointer for nodes
    - 3: if number of maximum iterations was reached when calculating the root.
         This usually means that the desired accuracy was not achieved. Default
         desired accuracy is GLQ_MAXERROR. Default maximum iterations is
         GLQ_MAXIT.
*/
int glq_nodes(int order, double *nodes);


/** Calculate the next Legendre polynomial root given the previous root found.

Uses the root-finder algorithm of:

  Barrera-Figueroa, V., Sosa-Pedroza, J. and López-Bonilla, J., 2006,
  "Multiple root finder algorithm for Legendre and Chebyshev polynomials via
  Newton's method", 2006, Annales mathematicae et Informaticae, 33, pp 3-13

@param initial initial estimate of the next root. I recommend the use of
               \f$ \cos\left(\pi\frac{(N - i - 0.25)}{N + 0.5}\right) \f$,
               where \f$ i \f$ is the index of the desired root
@param root_index index of the desired root, starting from 0
@param order order of the Legendre polynomial, ie number of roots.
@param roots array with the roots found so far. Will return the next root in
             roots[root_index], so make sure to malloc enough space.

@return Return code:
    - 0: if everything went OK
    - 1: if order is not valid
    - 2: if root_index is not valid (negative)
    - 3: if number of maximum iterations was reached when calculating the root.
         This usually means that the desired accuracy was not achieved. Default
         desired accuracy is GLQ_MAXERROR. Default maximum iterations is
         GLQ_MAXIT.
*/
int glq_next_root(double initial, int root_index, int order,
                         double *roots);


/** Calculates the weighting coefficients for the GLQ integration.

@param order order of the quadrature, ie number of nodes and weights.
@param nodes array containing the GLQ nodes calculated by glq_nodes.
             <b>IMPORTANT</b>: needs the nodes in [-1,1] interval! Scaled nodes
             will result in wrong weights!
@param weights pre-allocated array to return the weights

@return Return code:
    - 0: if everything went OK
    - 1: if order is not valid
    - 2: if nodes is a NULL pointer
    - 3: if weights is a NULL pointer
*/
int glq_weights(int order, double *nodes, double *weights);



/* Make a new GLQ structure and set all the parameters needed */
GLQ * glq_new(int order, double lower, double upper)
{
	GLQ *glq;
	int rc;

	glq = (GLQ *)malloc(sizeof(GLQ));
	if (glq == NULL)
	{
		return NULL;
	}
	glq->order = order;
	glq->nodes = (double *)malloc(sizeof(double)*order);
	if (glq->nodes == NULL)
	{
		free(glq);
		return NULL;
	}
	glq->nodes_unscaled = (double *)malloc(sizeof(double)*order);
	if (glq->nodes_unscaled == NULL)
	{
		free(glq);
		free(glq->nodes);
		return NULL;
	}
	glq->weights = (double *)malloc(sizeof(double)*order);
	if (glq->weights == NULL)
	{
		free(glq);
		free(glq->nodes);
		free(glq->nodes_unscaled);
		return NULL;
	}
	rc = glq_nodes(order, glq->nodes_unscaled);
	if (rc != 0 && rc != 3)
	{
		switch (rc)
		{
		case 1:
			log_error("glq_nodes invalid GLQ order %d. Should be >= 2.",
				order);
			break;
		case 2:
			log_error("glq_nodes NULL pointer for nodes");
			break;
		default:
			log_error("glq_nodes unknown error code %g", rc);
			break;
		}
		glq_free(glq);
		return NULL;
	}
	else if (rc == 3)
	{
		log_warning("glq_nodes max iterations reached in root finder");
		log_warning("nodes might not have desired accuracy %g", GLQ_MAXERROR);
	}
	rc = glq_weights(order, glq->nodes_unscaled, glq->weights);
	if (rc != 0)
	{
		switch (rc)
		{
		case 1:
			log_error("glq_weights invalid GLQ order %d. Should be >= 2.",
				order);
			break;
		case 2:
			log_error("glq_weights NULL pointer for nodes");
			break;
		case 3:
			log_error("glq_weights NULL pointer for weights");
			break;
		default:
			log_error("glq_weights unknown error code %d\n", rc);
			break;
		}
		glq_free(glq);
		return NULL;
	}
	if (glq_set_limits(lower, upper, glq) != 0)
	{
		glq_free(glq);
		return NULL;
	}
	return glq;
}


/* Free the memory allocated to make a GLQ structure */
void glq_free(GLQ *glq)
{
	free(glq->nodes);
	free(glq->nodes_unscaled);
	free(glq->weights);
	free(glq);
}


/* Calculates the GLQ nodes using glq_next_root. */
int glq_nodes(int order, double *nodes)
{
	register int i;
	int rc = 0;
	double initial;

	if (order < 2)
	{
		return 1;
	}
	if (nodes == NULL)
	{
		return 2;
	}
	for (i = 0; i < order; i++)
	{
		initial = cos(PI*(order - i - 0.25) / (order + 0.5));
		if (glq_next_root(initial, i, order, nodes) == 3)
		{
			rc = 3;
		}
	}
	return rc;
}


/* Put the GLQ nodes to the integration limits IN PLACE. */
int glq_set_limits(double lower, double upper, GLQ *glq)
{
	/* Only calculate once to optimize the code */
	double tmpplus = 0.5*(upper + lower), tmpminus = 0.5*(upper - lower);
	register int i;

	if (glq->order < 2)
	{
		return 1;
	}
	if (glq->nodes == NULL)
	{
		return 2;
	}
	if (glq->nodes_unscaled == NULL)
	{
		return 2;
	}
	for (i = 0; i < glq->order; i++)
	{
		glq->nodes[i] = tmpminus * glq->nodes_unscaled[i] + tmpplus;
	}
	return 0;
}


/* Calculate the next Legendre polynomial root given the previous root found. */
int glq_next_root(double initial, int root_index, int order, double *roots)
{
	double x1, x0, pn, pn_2, pn_1, pn_line, sum;
	int it = 0;
	register int n;

	if (order < 2)
	{
		return 1;
	}
	if (root_index < 0 || root_index >= order)
	{
		return 2;
	}
	x1 = initial;
	do
	{
		x0 = x1;

		/* Calculate Pn(x0) */
		/* Starting from P0(x) and P1(x), */
		/* find the others using the recursive relation: */
		/*     Pn(x)=(2n-1)xPn_1(x)/n - (n-1)Pn_2(x)/n   */
		pn_1 = 1.;   /* This is Po(x) */
		pn = x0;    /* and this P1(x) */
		for (n = 2; n <= order; n++)
		{
			pn_2 = pn_1;
			pn_1 = pn;
			pn = (((2 * n - 1)*x0*pn_1) - ((n - 1)*pn_2)) / n;
		}
		/* Now calculate Pn'(x0) using another recursive relation: */
		/*     Pn'(x)=n(xPn(x)-Pn_1(x))/(x*x-1)                    */
		pn_line = order * (x0*pn - pn_1) / (x0*x0 - 1);
		/* Sum the roots found so far */
		for (n = 0, sum = 0; n < root_index; n++)
		{
			sum += 1. / (x0 - roots[n]);
		}
		/* Update the estimate for the root */
		x1 = x0 - (double)pn / (pn_line - pn * sum);

		/** Compute the absolute value of x */
#define GLQ_ABS(x) ((x) < 0 ? -1*(x) : (x))
	} while (GLQ_ABS(x1 - x0) > GLQ_MAXERROR && ++it <= GLQ_MAXIT);
#undef GLQ_ABS

	roots[root_index] = x1;

	/* Tell the user if stagnation occurred */
	if (it > GLQ_MAXIT)
	{
		return 3;
	}
	return 0;
}


/* Calculates the weighting coefficients for the GLQ integration. */
int glq_weights(int order, double *nodes, double *weights)
{
	register int i, n;
	double xi, pn, pn_2, pn_1, pn_line;

	if (order < 2)
	{
		return 1;
	}
	if (nodes == NULL)
	{
		return 2;
	}
	if (weights == NULL)
	{
		return 3;
	}
	for (i = 0; i < order; i++) {

		xi = nodes[i];

		/* Find Pn'(xi) with the recursive relation to find Pn and Pn-1: */
		/*   Pn(x)=(2n-1)xPn_1(x)/n - (n-1)Pn_2(x)/n   */
		/* Then use:   Pn'(x)=n(xPn(x)-Pn_1(x))/(x*x-1) */

		/* Find Pn and Pn-1 stating from P0 and P1 */
		pn_1 = 1;   /* This is Po(x) */
		pn = xi;    /* and this P1(x) */
		for (n = 2; n <= order; n++)
		{
			pn_2 = pn_1;
			pn_1 = pn;
			pn = ((2 * n - 1)*xi*pn_1 - (n - 1)*pn_2) / n;
		}
		pn_line = order * (xi*pn - pn_1) / (xi*xi - 1.);
		/* ith weight is: wi = 2/(1 - xi^2)(Pn'(xi)^2) */
		weights[i] = 2. / ((1 - xi * xi)*pn_line*pn_line);
	}
	return 0;
}

#endif
