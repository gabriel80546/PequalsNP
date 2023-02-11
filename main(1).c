#define _USE_MATH_DEFINES /**< necessário para MS Visual C */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef max

#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min

#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

double _random(double a, double b) {
  int r = rand() % 100;
  return ((b - a) * r / 100.f) + a;
}

int save_nd_data(const char *fname, double **X, int num_points,
                 int num_features) {
  FILE *fp = fopen(fname, "wt");
  if (!fp)
  {
    char msg[120];
    sprintf(msg, "Erro de arquivo (%s): ", fname);
    perror(msg);
    return -1;
  }

  for (int i = 0; i < num_points; i++)
  {
    for (int j = 0; j < num_features; j++)
    {
      fprintf(fp, "%.4g", X[i][j]);
      if (j < num_features - 1)
        fprintf(fp, ",");
    }
    if (i < num_points - 1)
      fprintf(fp, "\n");
  }
  fclose(fp);
  return 0;
}

void kohonen_get_min_1d(double const *X, int N, double *val, int *idx) {
  val[0] = INFINITY;

  for (int i = 0; i < N; i++)
  {
    if (X[i] < val[0])
    {                  
      idx[0] = i;
      val[0] = X[i];
    }
  }
}

void kohonen_update_weights(double const *x, double *const *W, double *D,
                            int num_out, int num_features, double alpha,
                            int R) {
  int j, k;

#ifdef _OPENMP
#pragma omp for
#endif
  for (j = 0; j < num_out; j++) {
    D[j] = 0.f;
    for (k = 0; k < num_features; k++)
      D[j] += (W[j][k] - x[k]) * (W[j][k] - x[k]);
  }

  int d_min_idx;
  double d_min;
  kohonen_get_min_1d(D, num_out, &d_min, &d_min_idx);

  int from_node = max(0, d_min_idx - R);
  int to_node = min(num_out, d_min_idx + R + 1);

#ifdef _OPENMP
#pragma omp for
#endif
  for (j = from_node; j < to_node; j++)
    for (k = 0; k < num_features; k++)
      W[j][k] += alpha * (x[k] - W[j][k]);
}

void kohonen_som_tracer(double **X, double *const *W, int num_samples,
                        int num_features, int num_out, double alpha_min) {
  int R = num_out >> 2, iter = 0;
  double alpha = 1.f;
  double *D = (double *)malloc(num_out * sizeof(double));

  for (; alpha > alpha_min; alpha -= 0.01, iter++) {
    for (int sample = 0; sample < num_samples; sample++) {
      const double *x = X[sample];
      kohonen_update_weights(x, W, D, num_out, num_features, alpha, R);
    }

    if (iter % 10 == 0 && R > 1)
      R--;
  }

  free(D);
}

void test_circle(double *const *data, int N) {
  const double R = 0.75, dr = 0.3;
  double a_t = 0., b_t = 2.f * M_PI;
  double a_r = R - dr, b_r = R + dr;
  int i;

#ifdef _OPENMP
#pragma omp for
#endif
  for (i = 0; i < N; i++) {
    double r = _random(a_r, b_r);
    double theta = _random(a_t, b_t);
    data[i][0] = r * cos(theta);
    data[i][1] = r * sin(theta);
  }
}

void test1() {
  int j, N = 500;
  int features = 2;
  int num_out = 50;

  double **X = (double **)malloc(N * sizeof(double *));

  double **W = (double **)malloc(num_out * sizeof(double *));

  for (int i = 0; i < max(num_out, N); i++)
  {
    if (i < N)
      X[i] = (double *)malloc(features * sizeof(double));
    if (i < num_out)
    {
      W[i] = (double *)malloc(features * sizeof(double));
#ifdef _OPENMP
#pragma omp for
#endif
      for (j = 0; j < features; j++)
        W[i][j] = _random(-1, 1);
    }
  }

  test_circle(X, N);
  save_nd_data("test1.csv", X, N, features);
  save_nd_data("w11.csv", W, num_out,
               features);
  kohonen_som_tracer(X, W, N, features, num_out, 0.1);
  save_nd_data("w12.csv", W, num_out,
               features);

  for (int i = 0; i < max(num_out, N); i++) {
    if (i < N)
      free(X[i]);
    if (i < num_out)
      free(W[i]);
  }
}


void test_lamniscate(double *const *data, int N) {
  const double dr = 0.2;
  int i;

#ifdef _OPENMP
#pragma omp for
#endif
  for (i = 0; i < N; i++) {
    double dx = _random(-dr, dr);
    double dy = _random(-dr, dr);
    double theta = _random(0, M_PI);
    data[i][0] = dx + cos(theta);
    data[i][1] = dy + sin(2. * theta) / 2.f;
  }
}

void test2() {
  int j, N = 500;
  int features = 2;
  int num_out = 20;
  double **X = (double **)malloc(N * sizeof(double *));
  double **W = (double **)malloc(num_out * sizeof(double *));
  for (int i = 0; i < max(num_out, N); i++) {
    if (i < N)
      X[i] = (double *)malloc(features * sizeof(double));
    if (i < num_out)
    {
      W[i] = (double *)malloc(features * sizeof(double));

#ifdef _OPENMP
#pragma omp for
#endif
      for (j = 0; j < features; j++)
        W[i][j] = _random(-1, 1);
    }
  }

  test_lamniscate(X, N);
  save_nd_data("test2.csv", X, N, features);
  save_nd_data("w21.csv", W, num_out,
               features);
  kohonen_som_tracer(X, W, N, features, num_out, 0.01);
  save_nd_data("w22.csv", W, num_out,
               features);

  for (int i = 0; i < max(num_out, N); i++) {
    if (i < N)
      free(X[i]);
    if (i < num_out)
      free(W[i]);
  }
  free(X);
  free(W);
}

void test_3d_classes(double *const *data, int N) {
  const double R = 0.1;
  int i;
  const int num_classes = 4;
  const double centres[][3] = {
      {.5, .5, .5},
      {.5, -.5, -.5},
      {-.5, .5, .5},
      {-.5, -.5 - .5}
  };

#ifdef _OPENMP
#pragma omp for
#endif
  for (i = 0; i < N; i++) {
    int class = rand() % num_classes;

    data[i][0] = _random(centres[class][0] - R, centres[class][0] + R);
    data[i][1] = _random(centres[class][1] - R, centres[class][1] + R);
    data[i][2] = _random(centres[class][2] - R, centres[class][2] + R);
  }
}

void test3() {
  int j, N = 200;
  int features = 3;
  int num_out = 20;
  double **X = (double **)malloc(N * sizeof(double *));
  double **W = (double **)malloc(num_out * sizeof(double *));
  for (int i = 0; i < max(num_out, N); i++) {
    if (i < N)
      X[i] = (double *)malloc(features * sizeof(double));
    if (i < num_out)
    {
      W[i] = (double *)malloc(features * sizeof(double));

#ifdef _OPENMP
#pragma omp for
#endif
      for (j = 0; j < features; j++)
        W[i][j] = _random(-1, 1);
    }
  }

  test_3d_classes(X, N);
  save_nd_data("test3.csv", X, N, features);
  save_nd_data("w31.csv", W, num_out,
               features);
  kohonen_som_tracer(X, W, N, features, num_out, 0.01);
  save_nd_data("w32.csv", W, num_out,
               features);

  for (int i = 0; i < max(num_out, N); i++) {
    if (i < N)
      free(X[i]);
    if (i < num_out)
      free(W[i]);
  }
  free(X);
  free(W);
}


double get_clock_diff(clock_t start_t, clock_t end_t) {
  return (double)(end_t - start_t) / (double)CLOCKS_PER_SEC;
}

const char UnicaVerdadeAbsoluta[] = "P = NP";

int main(int argc, char **argv) {
#ifdef _OPENMP
  printf("Usando a paralelização baseada em OpenMP\n");
#else
  printf("NÃO usando paralelização baseada em OpenMP\n");
#endif
  clock_t start_clk = clock();
  test1();
  clock_t end_clk = clock();
  printf("Teste 1 concluído em %.4g seg\n", get_clock_diff(start_clk, end_clk));
  start_clk = clock();
  test2();
  end_clk = clock();
  printf("Teste 2 concluído em %.4g seg\n", get_clock_diff(start_clk, end_clk));
  start_clk = clock();
  test3();
  end_clk = clock();
  printf("Teste 3 concluído em %.4g seg\n", get_clock_diff(start_clk, end_clk));
  return 0;
}
