#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void initializeMPI(int argc, char** argv, int* mpiRank, int* numProcs)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, numProcs);
}

void finalizeMPI()
{
  MPI_Finalize();  
}

int main(int argc, char** argv)
{
  int mpiRank;
  int numProcs;
  initializeMPI(argc, argv, &mpiRank, &numProcs);
  double startTime = MPI_Wtime();

  int N = atoi(argv[1]); 
  int partOfVecSize = N / (numProcs - 1); // no part of the actual dot product computation takes place at node 0

  if (mpiRank == 0)
  {
    double* vec1;
    vec1 = malloc(N * sizeof(double));
/*    printf("\nVector 1 in node %d:\n", mpiRank);
    printf("[");*/
    for (int i = 0; i < N; i++)
    {
      vec1[i] = i;

/*      if (i != (N - 1))
      {
        char str[10];
        sprintf(str, "%f", vec1[i]);
        printf("%s", str);
        printf(", ");
      }
      else
      {
        char str[10];
        sprintf(str, "%f", vec1[i]);
        printf("%s", str);
        printf("]");
      }*/
    }

    double* vec2;
    vec2 = malloc(N * sizeof(double));
/*    printf("\nVector 2 in node %d:\n", mpiRank);
    printf("[");*/
    for (int i = N - 1; i >= 0; i--)
    {
      vec2[(N - 1) - i] = i;

/*      if (i != 0)
      {
        char str[10];
        sprintf(str, "%f", vec2[(N - 1) - i]);
        printf("%s", str);
        printf(", ");
      }
      else
      {
        char str[10];
        sprintf(str, "%f", vec2[(N - 1) - i]);
        printf("%s", str);
        printf("]");
      }*/
    }
    
    /* Send a part of each of the vectors vec1 and vec2 to each node */
    for (int otherRank = 1; otherRank < numProcs; otherRank++)
    {
      int start = (otherRank - 1) * partOfVecSize;
      int end = start + partOfVecSize;

      /* vector 1*/
      double* partOfVec1 = malloc(partOfVecSize * sizeof(double));
      for (int i = start; i < end; i++)
      {
        partOfVec1[i - start] = vec1[i];
      }

      /* vector 2*/
      double* partOfVec2 = malloc(partOfVecSize * sizeof(double));
      for (int i = start; i < end; i++)
      {
        partOfVec2[i - start] = vec2[i];
      }

      /* send vector 1*/
      MPI_Send(partOfVec1, partOfVecSize, MPI_DOUBLE, otherRank, 0, MPI_COMM_WORLD);

      /* send vector 2*/
      MPI_Send(partOfVec2, partOfVecSize, MPI_DOUBLE, otherRank, 0, MPI_COMM_WORLD);

      free(partOfVec1);
      free(partOfVec2);
    }

    /* receive partial resutls and calculate the final dot product value */
    double finalResult = 0.0;
    for (int otherRank = 1; otherRank < numProcs; otherRank++)
    {
      double partialResult = 0.0;
      MPI_Recv(&partialResult, 1, MPI_DOUBLE, otherRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      finalResult += partialResult;
    }

    printf("\nThe dot product is: %f\n", finalResult);

    free(vec1);
    free(vec2);

  }
  else
  {
    /* receive vector 1 */
    double* partOfVec1 = malloc(partOfVecSize * sizeof(double));
    MPI_Recv(partOfVec1, partOfVecSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
/*    printf("\nVector 1 part received by node %d:\n", mpiRank);
    printf("[");
    for (int i = 0; i < partOfVecSize; i++)
    {
      if (i != (partOfVecSize - 1))
      {
        char str[10];
        sprintf(str, "%f", partOfVec1[i]);
        printf("%s", str);
        printf(", ");
      }
      else
      {
        char str[10];
        sprintf(str, "%f", partOfVec1[i]);
        printf("%s", str);
        printf("]");
      }
    }*/

    /* receive vector 2 */
    double* partOfVec2 = malloc(partOfVecSize * sizeof(double));
    MPI_Recv(partOfVec2, partOfVecSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
/*    printf("\nVector 2 part received by node %d:\n", mpiRank);
    printf("[");
    for (int i = 0; i < partOfVecSize; i++)
    {
      if (i != (partOfVecSize - 1))
      {
        char str[10];
        sprintf(str, "%f", partOfVec2[i]);
        printf("%s", str);
        printf(", ");
      }
      else
      {
        char str[10];
        sprintf(str, "%f", partOfVec2[i]);
        printf("%s", str);
        printf("]");
      }
    }
    printf("\n");*/

    double partialResult = 0.0;
    for (int i = 0; i < partOfVecSize; i++)
    {
      partialResult += partOfVec1[i] * partOfVec2[i];
    }

    /*printf("The partial result from node %d is %f.\n", mpiRank, partialResult);*/

    /* send partial result */
    MPI_Send(&partialResult, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    free(partOfVec1);
    free(partOfVec2);

  }

  double elapsedTime = MPI_Wtime() - startTime;
  double totalTime;
  MPI_Reduce( &elapsedTime, &totalTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
  if (mpiRank == 0)
  {
    printf( "Total time spent in seconds: %f\n", totalTime);
  }

  finalizeMPI();
  return 0;
}