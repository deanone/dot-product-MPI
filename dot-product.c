#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv)
{
  clock_t startTime = clock();
  int N = atoi(argv[1]);
  
  /* vector 1 */
  double* vec1;
  vec1 = malloc(N * sizeof(double));
  for (int i = 0; i < N; i++)
  {
    vec1[i] = i;
  }

  /* vector 2 */
  double* vec2;
  vec2 = malloc(N * sizeof(double));
  for (int i = N - 1; i >= 0; i--)
  {
    vec2[(N - 1) - i] = i;
  }

  double finalResult = 0.0;
  for (int i = 0; i < N; i++)
  {
    finalResult += vec1[i] * vec2[i];
  }
  
  double elapsedTime = (double)(clock() - startTime) / CLOCKS_PER_SEC;

  printf("\nThe dot product is: %f\n", finalResult);
  printf( "Total time spent in seconds: %f\n", elapsedTime);
}