#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv)
{
  if (argc > 2)
  {
    int N = atoi(argv[1]);
    int VERBOSE = atoi(argv[2]);
    int mpiRank, numProcs, partOfVecSize;
    double start, end;
    
    // Initialize MPI section
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    partOfVecSize = N / (numProcs - 1); // -1 means that no part of the actual dot product computation takes place at node 0
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    if (mpiRank == 0)
    {
      if (VERBOSE)
      {
        printf("Num. processes: %d\n", numProcs);
        printf("Size of input vectors: %d\n", N);
        printf("Size of parts of input vectors: %d\n", partOfVecSize);
      }

      double* vec1;
      vec1 = malloc(N * sizeof(double));
      if (VERBOSE)
      {
        printf("\nVector 1 in node %d:\n", mpiRank);
        printf("[");
      }
      for (int i = 0; i < N; i++)
      {
        vec1[i] = i;
        if (VERBOSE)
        {
          if (i != (N - 1))
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
          }
        }
      }

      double* vec2;
      vec2 = malloc(N * sizeof(double));
      if (VERBOSE)
      {
        printf("\nVector 2 in node %d:\n", mpiRank);
        printf("[");
      }
      for (int i = N - 1; i >= 0; i--)
      {
        vec2[(N - 1) - i] = i;
        if (VERBOSE)
        {
          if (i != 0)
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
          }
        }
      }
      
      // Send a part of each of the vectors vec1 and vec2 to each node
      for (int otherRank = 1; otherRank < numProcs; otherRank++)
      {
        int start = (otherRank - 1) * partOfVecSize;
        int end = start + partOfVecSize;

        // vector 1
        double* partOfVec1 = malloc(partOfVecSize * sizeof(double));
        for (int i = start; i < end; i++)
        {
          partOfVec1[i - start] = vec1[i];
        }

        // vector 2
        double* partOfVec2 = malloc(partOfVecSize * sizeof(double));
        for (int i = start; i < end; i++)
        {
          partOfVec2[i - start] = vec2[i];
        }

        // send vector 1
        MPI_Send(partOfVec1, partOfVecSize, MPI_DOUBLE, otherRank, 0, MPI_COMM_WORLD);

        // send vector 2
        MPI_Send(partOfVec2, partOfVecSize, MPI_DOUBLE, otherRank, 0, MPI_COMM_WORLD);

        free(partOfVec1);
        free(partOfVec2);
      }

      // receive partial resutls and calculate the final dot product value
      double finalResult = 0.0;
      for (int otherRank = 1; otherRank < numProcs; otherRank++)
      {
        double partialResult = 0.0;
        MPI_Recv(&partialResult, 1, MPI_DOUBLE, otherRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        finalResult += partialResult;
      }

      printf("The dot product is: %f\n", finalResult);

      free(vec1);
      free(vec2);
    }
    else
    {
      // receive vector 1
      double* partOfVec1 = malloc(partOfVecSize * sizeof(double));
      MPI_Recv(partOfVec1, partOfVecSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (VERBOSE)
      {
        printf("\nVector 1 part received by node %d:\n", mpiRank);
        printf("[");
      }
      for (int i = 0; i < partOfVecSize; i++)
      {
        if (VERBOSE)
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
        }
      }

      // receive vector 2
      double* partOfVec2 = malloc(partOfVecSize * sizeof(double));
      MPI_Recv(partOfVec2, partOfVecSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (VERBOSE)
      {
        printf("\nVector 2 part received by node %d:\n", mpiRank);
        printf("[");
      }
      for (int i = 0; i < partOfVecSize; i++)
      {
        if (VERBOSE)
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
      }
      printf("\n");

      double partialResult = 0.0;
      for (int i = 0; i < partOfVecSize; i++)
      {
        partialResult += partOfVec1[i] * partOfVec2[i];
      }
      
      if (VERBOSE)
      {
        printf("The partial result from node %d is %f.\n", mpiRank, partialResult);
      }

      // send partial result to node 0
      MPI_Send(&partialResult, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

      free(partOfVec1);
      free(partOfVec2);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    MPI_Finalize(); // clen up MPI environment
    
    if (mpiRank == 0)
    {
      printf("Elapsed time (sec.): %f\n", end - start);
    }

    return 0;
  }
  else
  {
    printf("\nThe program is executed as follows:\n");
    printf("\n./doc-product-mpi.out N 1|0\n");
    printf("\nwhere:\n");
    printf("argv[0]: the name of the program\n");
    printf("argv[1]: the size of input vectors\n");
    printf("argv[2]: 1 for verbose execution, 0 for non-verbose\n\n");
  }
}