#include<string.h> 
#include<stdio.h>
#include<iostream>
#include<math.h>
#include<fstream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;



int totalVertices,noOfConnections,src;

void read(string filename,int* &vertexOffset,int* &connection,int* &departureTime, int* &duration){

    ifstream scan;
    scan.open(filename.c_str());
    scan>>totalVertices>>noOfConnections>>src;

  vertexOffset = new int[totalVertices+1];
	connection = new int[noOfConnections];
	departureTime = new int[noOfConnections];
	duration = new int[noOfConnections];

    for(int i=0; i<totalVertices+1; i++)
		scan>>vertexOffset[i]; 

    for(int i=0; i<noOfConnections; i++)
        scan>>connection[i]; 
        
    for(int i=0; i<noOfConnections; i++)
		scan>>departureTime[i]; 

    for(int i=0; i<noOfConnections; i++)
		scan>>duration[i]; 
}

void cudaCopyToDevice(int *X, int *&cX, int n){

	cudaMalloc((void**)&cX, n*sizeof(int));
  cudaMemcpy( cX, X, n*sizeof(int), cudaMemcpyHostToDevice);
  
}
__global__
void processVertices(int totalVertices,int* dvertexOffset,int* dconnection,int* ddepartureTime,int* dduration,int* dtimes, bool *dContinue)
{
		int i,j,u,v,t,lambda,degree;		

		i =threadIdx.x;
		if(i >=totalVertices) return; 
		//if(iterations==0 && i!=0) return;
		//if(level[i] != iterations) return;
 
		u = i;
    degree = dvertexOffset[u+1] - dvertexOffset[u];
    
		for(j=1; j<=degree; j++){
      
			v = dconnection[dvertexOffset[u]+j-1];
			t = ddepartureTime[dvertexOffset[u]+j-1];
			lambda = dduration[dvertexOffset[u]+j-1];
			if(dtimes[u]<=t && t+lambda < dtimes[v])
				{			//if(i==0){printf("first thread updating:after \n"); }
          dtimes[v]= t + lambda;
					dContinue=true;
					//level[v]=iterations+1;
				}
		}
}

void computeEarliestTimes(int* dvertexOffset,int* dconnection,int* ddepartureTime,int* dduration,int* dtimes)
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
		processVertices<<< 1,3 >>>(totalVertices, dvertexOffset, dconnection, dvertexOffset, ddepartureTime, dduration,dContinue);
		// check if kernel execution generated and error
		//Kernel2<<< grid, threads, 0 >>>( d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);
		// check if kernel execution generated and error
		cudaMemcpy( &hContinue, dContinue, sizeof(bool), cudaMemcpyDeviceToHost) ;
		iterations++;
	}
	while(hContinue);

}

int main(int argc,char* argv[]){

  int* vertexOffset, connection, departureTime, duration, times;
  int* dvertexOffset, dconnection, ddepartureTime, dduration, dtimes;
  string filename;
  struct timeval start,stop;
	double time;

  filename=argv[1];
  read(filename,vertexOffset,connection,departureTime,duration);
  cudaCopyToDevice(vertexOffset,dvertexOffset,totalVertices);

  cudaCopyToDevice(connection,dconnection,noOfConnections);
  
  cudaCopyToDevice(departureTime,ddepartureTime,noOfConnections);
  
	cudaCopyToDevice(duration,dduration,noOfConnections);


  times=new int[totalVertices];

  for(int i=0;i<totalVertices;i++)
    times[i]=INT_MAX;
    times[src]=0;

    cudaCopyToDevice(times,dtimes,totalVertices);
   
    gettimeofday(&start,0);
    computeEarliestTimes(dvertexOffset,dconnection,ddepartureTime,dduration,dtimes);
    gettimeofday(&stop,0);
    time = (1000000.0*(stop.tv_sec-start.tv_sec) + stop.tv_usec-start.tv_usec)/1000.0;

    cudaMemcpy(times, dtimes, totalVertices*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dvertexOffset);
    cudaFree(dconnection);
    cudaFree(ddepartureTime);
    cudaFree(dduration);
    cudaFree(dtimes);
  

    for(int i=0;i<totalVertices;i++)
       cout<<i<<" "<<times[i]<<endl;

       cout<<"Time is "<<time<<endl;
    return 0;
}