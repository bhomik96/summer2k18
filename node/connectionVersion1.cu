// Edge Version: Input Edge Stream
// 3 Kernels: Process only active edges in every iteration

#include<string.h> 
#include<stdio.h>
#include<iostream>
#include<math.h>
#include<fstream>
#include<sys/time.h>
#include<cuda.h>
//#include"common.h"

#define INF INT_MAX;
#define MAX_THREADS_PER_BLOCK 1024
#define PRINTFLAG true



using namespace std;

dim3  gridDimension;
dim3  blockDimension;

void printEarliestTimes(int earliestTime[], int vCount,	double runningTime, bool print)
{
	if(print == true)
	{//cout<<"Memory copied"<<endl;
		for(int i=0;i<=vCount-1;i++)
		{
			cout<<i<<" "<<earliestTime[i]<<endl;
		//fprintf(fp1,"Earliest time for %d is %d\n",i,earliest[i]); 
		}
	}
	cout<<"Time is "<<runningTime<<endl;
}


void createHostMemoryBool(bool *&X, int n)
{
	X = new bool[n];
}

bool readInput(char *fileName, int &vCount, int &cCount, int *&from, int *&to, int *&departure, int *&duration)
{

	ifstream fin;

	fin.open(fileName);	

	fin>>vCount>>cCount;

	from = new int[cCount];
	to = new int[cCount];
	departure = new int[cCount];
	duration = new int[cCount];

	for(int i=0; i<=cCount-1; i++)
		fin>>from[i]>>to[i]>>departure[i]>>duration[i]; 

	//cout<<"reading the input is over"<<endl;

return true;
}

bool printInput(int vCount, int cCount, int *vertex, int *edge, int *departure, int *duration)
{
	ofstream fout;

	fout.open("csr2.txt");	


	for(int i=0; i<=vCount; i++)
		fout<<vertex[i]<<" ";
	fout<<endl;

	for(int i=0; i<=cCount-1; i++)
		fout<<edge[i]<<" "; 
	fout<<endl;

	for(int i=0; i<=cCount-1; i++)
		fout<<departure[i]<<" "; 
	fout<<endl;

	for(int i=0; i<=cCount-1; i++)
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

void initEarlistTime(int *earliestTime, int vCount, int source, int departureTime)
{
	for(int i=0; i<=vCount-1; i++)
	{
		earliestTime[i] = INF;
	}

	earliestTime[source]=departureTime;
}

void initActiveVertices(bool *active, int vCount, int source)
{
	for(int i=0; i<=vCount-1; i++)
	{
		active[i] = false;
	}

	active[source]=true;
}

void cudaCopyToDevice(int *X, int *&cX, int n)
{
	cudaMalloc((void**)&cX, n*sizeof(int));
	cudaMemcpy( cX, X, n*sizeof(int), cudaMemcpyHostToDevice);
}

void cudaCopyToDeviceBool(bool *X, bool *&cX, int n)
{
	cudaMalloc((void**)&cX, n*sizeof(bool));
	cudaMemcpy(cX, X, n*sizeof(bool), cudaMemcpyHostToDevice);
}

__global__
void processEdges2(int n, int *from, int *to, int *departure, int *duration, int *earliestTime, bool *dContinue, bool *active, bool *nextActive)
{
		int i,u,v,t,lambda;		

		i = blockIdx.x * blockDim.x + threadIdx.x;
		if(i>= n || active[from[i]] == false) return;
		//**** LOOP 
		//** n 
		
			u = from[i];
			v = to[i];
			t = departure[i];
			lambda = duration[i];
			if((active[u]==true))
			{
				if((earliestTime[u]<=t) && (t+lambda < earliestTime[v]))
				{			
				//if(i==0){printf("first thread updating:after \n"); }
					//earliestTime[v]= t + lambda;
					atomicMin(&earliestTime[v], t+lambda);
					//nextActive[v] = true;
					nextActive[v] = true;
					*dContinue=true;
				}				
			}
		
}

__global__
void processEdges1(int n, int *from, int *to, int *departure, int *duration, int *earliestTime, bool *dContinue, bool *active, bool *nextActive)
{
		int i,u,v,t,lambda;		

		i = blockIdx.x * blockDim.x + threadIdx.x;
		if(i>= n || active[from[i]] == false) return;
		//**** LOOP 
		//** n 
		
			u = from[i];
			v = to[i];
			t = departure[i];
			lambda = duration[i];
			if((active[u]==true))
			{
				if((earliestTime[u]<=t) && (t+lambda < earliestTime[v]))
				{			
				//if(i==0){printf("first thread updating:after \n"); }
					//earliestTime[v]= t + lambda;
					atomicMin(&earliestTime[v], t+lambda);
					//nextActive[v] = true;
					active[v] = true;
					*dContinue=true;
				}				
			}
		
}

__global__
void processEdges(int n, int *from, int *to, int *departure, int *duration, int *earliestTime, bool *dContinue, int *active, int *nextActive)
{
		int i,u,v,t,lambda;		

		i = blockIdx.x * blockDim.x + threadIdx.x;
		if(i>= n) return;
		//**** LOOP 
		//** n 
		
			u = from[i];
			v = to[i];
			t = departure[i];
			lambda = duration[i];
			//if((active[u]==true))
			{
				if((earliestTime[u]<=t) && (t+lambda < earliestTime[v]))
				{			
				//if(i==0){printf("first thread updating:after \n"); }
					//earliestTime[v]= t + lambda;
					atomicMin(&earliestTime[v], t+lambda);
					nextActive[v] = true;
					*dContinue=true;
				}
			}
		
}

__global__
void passiveToActive(int n, int *active, int *nextActive)
{
	int v,i;

//****LOOP
			i = blockIdx.x * blockDim.x + threadIdx.x;
			v = i;

			if( i< n && nextActive[v]==true)
			{
				nextActive[v] = false;
				active[v] = true;
			}

}

__global__
void activeToPassive(int n, int *active)
{
	int u,i;

//***LOOP
			i = blockIdx.x * blockDim.x + threadIdx.x;
			u = i;

			if( i< n && active[u]==true)
			{
				active[u] = false;
			}
}
//		computeEarliestTimes(vCount,cCount,cFrom,cTo,cDeparture,cDuration,cEarliestTime,cActive, cNextActive);
//		computeEarliestTimes(vCount,cCount,cFrom,cTo,cDeparture,cDuration,cEarliestTime,cActive, cNextActive);
		//computeEarliestTime(vCount,cCount,cFrom,cTo,cDeparture,cDuration,cEarliestTime,cActive, cNextActive);

void computeEarliestTime(int vCount, int cCount, int *from, int *to, int *departure, int *duration, int *earliestTime, bool *active, bool *nextActive)
{
	int iterations=0;

	bool hContinue;
	bool *dContinue;
	cudaMalloc( (void**) &dContinue, sizeof(bool));

	//Call the Kernel untill all the elements of Frontier are not false
	do
	{
		//if no thread changes this value then the loop stops
		hContinue=false;
		
		cudaMemcpy(dContinue, &hContinue, sizeof(bool), cudaMemcpyHostToDevice) ;
		processEdges1<<< ceil(cCount/1024.0), 1024>>>(cCount ,from, to, departure, duration, earliestTime,dContinue,active,nextActive);
//		updateActiveVertices<<<ceil(vCount/1024.0), 1024>>>(vCount,active,nextActive);

		//activeToPassive<<<ceil(vCount/1024.0), 1024>>>(vCount,active);
		//passiveToActive<<<ceil(vCount/1024.0), 1024>>>(vCount,active, nextActive);


		cudaMemcpy( &hContinue, dContinue, sizeof(bool), cudaMemcpyDeviceToHost) ;
		cudaDeviceSynchronize();
		iterations++;
	}
	while(hContinue);

}

//computeEarliestTimesSetUP(vCount,cCount,from,to,departure,duration,source, departureTime,earliestTime,active, nextActive, runTime);
void computeEarliestTimesSetUP(int vCount, int cCount,int from[],int to[],int departure[], int duration[], int source[], int departureTime[], int earliestTime[],int qCount)
{
	int *cFrom, *cTo, *cDeparture, *cDuration, *cEarliestTime;
	bool *cActive, *cNextActive;
	struct timeval start,stop;
	bool *active, *nextActive;
	double time;


	initConfiguration(gridDimension,blockDimension, cCount);
	createHostMemoryBool(active,vCount);
	createHostMemoryBool(nextActive,vCount);

	cudaCopyToDevice(from,cFrom,cCount);
	cudaCopyToDevice(to,cTo,cCount);
	cudaCopyToDevice(departure,cDeparture,cCount);
	cudaCopyToDevice(duration,cDuration,cCount);


	//cudaCopyToDevice(earliestTime,cEarliestTime,vCount);
	//cudaCopyToDevice(active,cActive,vCount);
	//cudaCopyToDevice(nextActive,cNextActive,vCount);
	
	for(int i=0; i<=qCount-1; i++)
	{
		initEarlistTime(earliestTime,vCount,source[i],departureTime[i]);
		initActiveVertices(active,vCount,source[i]);

		cudaCopyToDevice(earliestTime,cEarliestTime,vCount);
		cudaCopyToDeviceBool(active,cActive,vCount);
	
		gettimeofday(&start,0);
		computeEarliestTime(vCount,cCount,cFrom,cTo,cDeparture,cDuration,cEarliestTime,cActive, cNextActive);
		gettimeofday(&stop,0);
		time = (1000000.0*(stop.tv_sec-start.tv_sec) + stop.tv_usec-start.tv_usec)/1000.0;

		cudaMemcpy(earliestTime, cEarliestTime, vCount * sizeof(int), cudaMemcpyDeviceToHost);
		printEarliestTimes(earliestTime,vCount,time,PRINTFLAG);

	}
	
	cudaFree(cEarliestTime);
	cudaFree(cFrom);
	cudaFree(cTo);
	cudaFree(cDeparture);
	cudaFree(cDuration);
	cudaFree(cActive);
	cudaFree(cNextActive);
}

void createHostMemory(int *&X, int n)
{
	X = new int[n];
}

void readQuery(char *fileName, int *&source, int *&departureTime, int &qCount)
{
	ifstream fin;

	fin.open(fileName);	

	fin>>qCount;

	source = new int[qCount];
	departureTime = new int[qCount];
	
	for(int i=0; i<=qCount-1; i++)
		fin>>source[i]>>departureTime[i]; 
}

int main(int argc, char *argv[])
{
	int vCount, cCount, qCount;
	int *from, *to, *departure, *duration, *earliestTime,  *source, *departureTime;
	char queryFile[100];
	char fileName[100];


	strcpy(fileName, argv[1]);
	strcpy(queryFile,"query.txt");

	readInput(fileName,vCount, cCount, from, to, departure, duration);
	readQuery(queryFile, source, departureTime, qCount);
	createHostMemory(earliestTime,vCount);

	//departureTime = 180; //Change this ****************************************
	computeEarliestTimesSetUP(vCount,cCount,from,to,departure,duration,source, departureTime,earliestTime,qCount);

	
return 0;
}
 
