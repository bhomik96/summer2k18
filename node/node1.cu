#include<string.h> 
#include<stdio.h>
#include<iostream>
#include<math.h>
#include<fstream>
#include<sys/time.h>
#include<cuda.h>
#define INF 10000;
#define MAX_THREADS_PER_BLOCK 1024


using namespace std;

dim3  gridDimension;
dim3  blockDimension;

bool readInput(char *fileName, int &vCount, int &eCount, int *&vertex, int *&edge, int *&departure, int *&duration, int &source)
{

	ifstream fin;

	fin.open(fileName);	

	fin>>vCount>>eCount>>source;

	vertex = new int[vCount+1];
	edge = new int[eCount];
	departure = new int[eCount];
	duration = new int[eCount];

	for(int i=0; i<=vCount; i++)
		fin>>vertex[i];

	for(int i=0; i<=eCount-1; i++)
		fin>>edge[i]; 

	for(int i=0; i<=eCount-1; i++)
		fin>>departure[i]; 

	for(int i=0; i<=eCount-1; i++)
		fin>>duration[i]; 

	//cout<<"reading the input is over"<<endl;

return true;
}

bool printInput(int vCount, int eCount, int *vertex, int *edge, int *departure, int *duration)
{
	ofstream fout;

	fout.open("csr2.txt");	


	for(int i=0; i<=vCount; i++)
		fout<<vertex[i]<<" ";
	fout<<endl;

	for(int i=0; i<=eCount-1; i++)
		fout<<edge[i]<<" "; 
	fout<<endl;

	for(int i=0; i<=eCount-1; i++)
		fout<<departure[i]<<" "; 
	fout<<endl;

	for(int i=0; i<=eCount-1; i++)
		fout<<duration[i]<<" "; 
	fout<<endl;

return true;
}

void initConfiguration(dim3 &grid, dim3 &block, int n)
{
	int num_of_blocks = 1;
	int num_of_threads_per_block = n;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(n>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(n/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}
	grid.x = num_of_blocks; grid.y=1; grid.z=1;
	block.x = num_of_threads_per_block; block.y=1; block.z=1;

}


//**should be initialized with specified time instead of zero
void initArray(int *&X, int n)
{
	X = new int[n];

	for(int i=0; i<=n-1; i++)
	{
		X[i] = INF;
	}

}

void cudaCopyToDevice(int *X, int *&cX, int n)
{
	cudaMalloc((void**)&cX, n*sizeof(int));
	cudaMemcpy( cX, X, n*sizeof(int), cudaMemcpyHostToDevice);
}

__global__
void processVertex(int *vertex, int *edge, int *departure, int *duration, int *earliestTime,int *level)
{
	int i,u,v,t, lambda;

			i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
			if(i>=1024*100) return;
			u = 0;
			v = edge[i];
			t = departure[i];
			lambda = duration[i];
			if(earliestTime[u]<=t && t+lambda < earliestTime[v])
				{			//if(i==0){printf("first thread updating:after \n"); }
					earliestTime[v]= t + lambda;
					level[v]=1;
				}
}
__global__
void processVertices(int iterations, int vCount, int eCount, int *vertex, int *edge, int *departure, int *duration, int *earliestTime, bool *dContinue, int *level)
{
		int i,j,u,v,t,lambda,degree;		

		i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
		if(i >= vCount) return; 
		//if(iterations==0 && i!=0) return;
		//if(level[i] != iterations) return;
 

		u = i;
		degree = vertex[u+1] - vertex[u];
		for(j=1; j<=degree; j++)
		{
			v = edge[vertex[u]+j-1];
			t = departure[vertex[u]+j-1];
			lambda = duration[vertex[u]+j-1];
			if(earliestTime[u]<=t && t+lambda < earliestTime[v])
				{			//if(i==0){printf("first thread updating:after \n"); }
					earliestTime[v]= t + lambda;
					*dContinue=true;
					//level[v]=iterations+1;
				}
		}
}

void computeEarliestTimes(int vCount, int eCount, int *vertex, int *edge, int *departure, int *duration, int *earliestTime, int *level)
{
	int iterations=0;

	bool hContinue;
	bool *dContinue;
	cudaMalloc( (void**) &dContinue, sizeof(bool));

//processVertex<<< 100, 1024>>>(vertex, edge, departure, duration, earliestTime,level);
	iterations=1;

	//Call the Kernel untill all the elements of Frontier are not false
	do
	{
		//if no thread changes this value then the loop stops
		hContinue=false;
		
		cudaMemcpy(dContinue, &hContinue, sizeof(bool), cudaMemcpyHostToDevice) ;
		processVertices<<< gridDimension, blockDimension, 0 >>>(iterations,vCount, eCount, vertex, edge, departure, duration, earliestTime,dContinue,level);
		// check if kernel execution generated and error
		//Kernel2<<< grid, threads, 0 >>>( d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);
		// check if kernel execution generated and error
		cudaMemcpy( &hContinue, dContinue, sizeof(bool), cudaMemcpyDeviceToHost) ;
		iterations++;
	}
	while(hContinue);

}

int main(int argc, char *argv[])
{
	int vCount, eCount, source;
	int *edge, *vertex, *departure, *duration, *earliestTime, *level;
	int *cEdge, *cVertex, *cDeparture, *cDuration, *cEarliestTime, *cLevel;
	char fileName[100];

	struct timeval start,stop;
	double time;

	strcpy(fileName, argv[1]);
	readInput(fileName,vCount, eCount, vertex, edge, departure, duration, source);
	initConfiguration(gridDimension,blockDimension, vCount);
	cudaCopyToDevice(vertex,cVertex,vCount);
	cudaCopyToDevice(edge,cEdge,eCount);
	cudaCopyToDevice(departure,cDeparture,eCount);
	cudaCopyToDevice(duration,cDuration,eCount);

	initArray(earliestTime,vCount);
	earliestTime[source]=0; // starting time
	cudaCopyToDevice(earliestTime,cEarliestTime,vCount);
	//initArray(level,vCount);
//	level[source]=0;
//	cudaCopyToDevice(level,cLevel,vCount);


	gettimeofday(&start,0);
	computeEarliestTimes(vCount,eCount,cVertex,cEdge,cDeparture,cDuration,cEarliestTime,cLevel);
	gettimeofday(&stop,0);
	time = (1000000.0*(stop.tv_sec-start.tv_sec) + stop.tv_usec-start.tv_usec)/1000.0;

	cudaMemcpy(earliestTime, cEarliestTime, vCount*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(cEarliestTime);
	cudaFree(cEdge);
	cudaFree(cVertex);
	cudaFree(cDeparture);
	cudaFree(cDuration);

	//cout<<"Memory copied"<<endl;
	for(int i=0;i<=vCount-1;i++)
	{
		cout<<i<<" "<<earliestTime[i]<<endl;
		//fprintf(fp1,"Earliest time for %d is %d\n",i,earliest[i]); 
	}
	cout<<"Time is "<<time<<endl;

	
return 0;
}
 
