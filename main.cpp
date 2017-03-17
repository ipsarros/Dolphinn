#include <iostream>
#include <vector>

#include "IO.h"
#include "hypercube.h"

#include <ctime>
#include <ratio>
#include <chrono>

#define N 10000
#define D 128
#define K floor(log2(N))
#define M 1
#define Q 100
#define T double
#define bitT char
#define MAX_PNTS_TO_SEARCH N/10
#define THREADS_NO 1
#define RADIUS 1

//     /usr/bin/time -l ./dolphinn
int main()
{
	if(MAX_PNTS_TO_SEARCH > N)
	{
    	std::cerr << "MAX_PNTS_TO_SEARCH > N" << std::endl;
        return -1;
  	}

  	// vector is actually 1D, emulating a 2D vector
  	std::vector<T> pointset(N * D);

  	std::cout << "N = " << N << ", D = " << D << ", K = " << K << ", MAX_PNTS_TO_SEARCH = " << MAX_PNTS_TO_SEARCH << std::endl;

  	//read_points_IDX_format<T>(pointset, N, D, "/Users/gsamaras/Code/C++/create_pointset/MNIST/train-images-idx3-ubyte");
  	readfvecs(pointset, N, D, "./siftsmall/siftsmall_base.fvecs");
	//print_2D_vector<T>(pointset, N, D);



  	using namespace std::chrono;
  	high_resolution_clock::time_point t1 = high_resolution_clock::now();
  	double r=300;
  	Dolphinn::Hypercube<T, bitT> hypercube(pointset, N, D, K, THREADS_NO, r);

  	high_resolution_clock::time_point t2 = high_resolution_clock::now();
 
  	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
 
  	std::cout << "Build: " << time_span.count() << " seconds.\n";


  	//hypercube.print_no_of_assigned_points_per_vertex();
  
  	//print_2D_vector<bitT>(mapped_pointset, N, K, true);

	
  	// QUERY
  	std::vector<T> query(Q * D);
  	//read_points_IDX_format<T>(query, Q, D, "/Users/gsamaras/Code/C++/create_pointset/MNIST/t10k-images-idx3-ubyte");
  	readfvecs(query, Q, D, "./siftsmall/siftsmall_query.fvecs");
    std::vector< std::vector<std::pair<int, float>> > results_idxs(Q, std::vector<std::pair<int, float>>(M));

  	t1 = high_resolution_clock::now();

        //hypercube.radius_query(query, Q, RADIUS, MAX_PNTS_TO_SEARCH, results_idxs, THREADS_NO);
    hypercube.m_nearest_neighbors_query(query, Q, M,  MAX_PNTS_TO_SEARCH, results_idxs, THREADS_NO);
    //nearest_neighbor_query(std::vector<std::pair<int, float>>& results_idxs_dists, const int threads_no = std::thread::hardware_concurrency())




  	t2 = high_resolution_clock::now();
  	time_span = duration_cast<duration<double>>(t2 - t1);
 
  	std::cout << "Search: " << time_span.count()/(double)Q << " seconds.\n";

  	t1 = high_resolution_clock::now();

  	//int squared_radius = RADIUS * RADIUS;
  	double cur_mind;
  	int cur_mini;
  	std::vector<int> brute_results_idxs(Q);
  	for(int q = 0; q < Q; ++q)
  	{
  			cur_mind=squared_Eucl_distance(query.begin() + q * D, (query.begin() + q * D) + D, pointset.begin() );
  			cur_mini=0;
    	    for(int n = 1; n < N; ++n)
    	    {
      		if(squared_Eucl_distance(query.begin() + q * D, (query.begin() + q * D) + D, pointset.begin() + n * D) <= cur_mind)
      		{
      			cur_mind=squared_Eucl_distance(query.begin() + q * D, (query.begin() + q * D) + D, pointset.begin() + n * D);
      			cur_mini=n;
        		//brute_results_idxs[q] = q;
        		//break;
      		}
    	    }
    	    brute_results_idxs[q] = cur_mini;
  	}

  	t2 = high_resolution_clock::now();
  	time_span = duration_cast<duration<double>>(t2 - t1);
 
  	std::cout << "Brute force: " << time_span.count()/(double)Q << " seconds.\n";


  	//print_2D_vector(results_idxs, Q, M);
  	//print_1D_vector(brute_results_idxs);
  	int correct = 0;
  	for(int q = 0; q < Q; ++q){
  		for(int j=0; j<M; ++j){
			if (results_idxs[q][j].first== brute_results_idxs[q] )
				correct++;
  		}
  	}
  	std::cout << "Correct = " << (correct * 100)/(double)Q << "% , correct = " << correct << ", Q = " << Q << std::endl;

  	return 0;
}
