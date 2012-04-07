#ifndef MOTMOT_EIGEN_H
#define MOTMOT_EIGEN_H

#ifdef __cplusplus
extern "C" {
#endif

int eigen_2x2_real( double A, double B,
		    double C, double D,
		    double *evalA, double *evecA1,
		    double *evalB, double *evecB1 );

#ifdef __cplusplus
}
#endif

#endif
