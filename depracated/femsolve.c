#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define L 1.0
#define T 0.1
#define ALPHA 0.00001
#define NE 50
#define NT 500





int main() {
    // Parameters
    double dx = L / NE;  // Element size
    double dt = T / NT;  // Time step size
    int nodes = NE + 1;

    // Allocate arrays for matrices and solution
    double *x = (double *)malloc(nodes * sizeof(double));
    double *u = (double *)malloc(nodes * sizeof(double));

    // Initialize node coordinates
    for (int i = 0; i < nodes; i++) {
        x[i] = i * dx;
    }

    // Initial condition
    for (int i = 0; i < nodes; i++) {
        u[i] = (i > 0.4 * nodes && i < 0.6 * nodes) ? 1.0 : 0.0;
    }


    // Print initial condition
    for (int i = 0; i < nodes; i++) {
        printf("%lf, %lf\n", x[i], u[i]);
    }

    free(x);
    free(u);
    return 0;
}