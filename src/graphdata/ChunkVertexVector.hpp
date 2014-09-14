#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <set>
#include <string>
//#include <string.h>
#include <sstream>
#include <map>

class ChunkVertexVector{
public:
	std::map<unsigned int, unsigned int> cvmap;//vertex, index
	std::set<unsigned int> vv;
	int *vvArray;
	int *vvPartition;
	int *vvSize;

private:
	std::map<unsigned int, unsigned int> vcmap;//index, vertex
	unsigned int vertex_num;
	unsigned int chunk_size;
	
public:
	ChunkVertexVector(){}
	~ChunkVertexVector()
	{
		if(vvArray != NULL)
		{
			free(vvArray);
			free(vvSize);
			free(vvPartition);
			//vv.clear();
			//cvmap.clear();
			//vcmap.clear();
		}
	}

	unsigned int get_chunk_size()
	{
		return chunk_size;
	}

	unsigned int get_vertex_num()
	{
		return vertex_num;
	}

	void set_chunk_size(unsigned int _size)
	{
		chunk_size = _size;
	}

	void set_vertex_num(unsigned int _num)
	{
		vertex_num = _num;
	}

	int initChunkVertexVector(int _chunk_size, unsigned int v_num)
	{
		chunk_size = _chunk_size;
		vertex_num = v_num;

		if(vertex_num == 0 || chunk_size == 0)
		{
			std::cout << "No Vertex Records Or Not set chunk size." << std::endl;	
			exit(-1);
		}

		vvArray = (int *)malloc(sizeof(int)*vertex_num*chunk_size);
		vvPartition = (int *)malloc(sizeof(int)*vertex_num);
		vvSize = (int *)malloc(sizeof(int)*chunk_size);
		if(vvSize == NULL ||  vvPartition == NULL || vvArray == NULL)
		{
			std::cout << "Malloc vvAray or VVPartition or vvSize Error." << std::endl;
		}
		memset(vvArray, 0, vertex_num * chunk_size * (sizeof(int)/ sizeof(char)));
		memset(vvPartition, 0, vertex_num * (sizeof(int)/ sizeof(char)));
		memset(vvSize, 0, chunk_size * (sizeof(int)/ sizeof(char)));
		return 0;
	}

	int get_map_size()
	{
		return cvmap.size();
	}

	void addVertex(unsigned int src, unsigned int dst, int _both)
	{
		int i = (src) * chunk_size, j = (dst) * chunk_size;//index
		//std::cout << "addVertex: " << src << " , " << dst << ", index:" << i << ", " << j << std::endl;
		//src not processed.
		if(vv.count(src) == 0)
		{
			int ii;
			for(ii = 0; ii < chunk_size; ii++)
			{
				if(vvArray[i+ii] == 0)
				{
					vvArray[i+ii] = 1;
					vvPartition[src] = ii;
					vvSize[ii]++;
					if(vv.count(dst) == 0 && _both == 1 && vvArray[j+ii] != 1)
					{
						vvArray[j+ii] += -1;
					}
					//std::cout << "adding...... table: "<< ii << ", vvSize = " << vvSize[ii] << std::endl;
					//<<", arrary: "<< vvArray[i+ii] <<" "<< vvArray[j+ii] <<std::endl;
					break;
				}
			}
			if(ii == chunk_size)
			{	
				ii--;
				vvArray[i+ii] = 1;
				vvPartition[src] = ii;
				vvSize[ii]++;
				if(vv.count(dst) == 0 && _both == 1 && vvArray[j+ii] != 1)
				{
					vvArray[j+ii] += -1;
				}
				//std::cout << "adding...... table: "<< ii << ", vvSize = " << vvSize[ii] << std::endl;
				//std::cout << "adding...... table: "<< ii <<", arrary: "<< vvArray[i+ii] <<" "<< vvArray[j+ii] <<std::endl;
			}
                        vv.insert(src);
			return ;
		}

		if(vv.count(src) == 1 && _both == 1 && vv.count(dst) == 0)
		{
			int ii;
                        for(ii = 0; ii < chunk_size; ii++)
                        {
                                if(vvArray[i+ii] == 1)
                                {
                                        if(vv.count(dst) == 0 && vvArray[j+ii] != 1)    vvArray[j+ii] += -1;
                                        break;
                                }
                        }
			//std::cout << "adding...... table: "<< ii <<", arrary: "<< vvArray[i+ii] <<" "<<vvArray[j+ii]<<std::endl;
                        return ;
		}
	}

	// add the vertex just as the dst in the file
	void after_add_vertex()
	{
		for(int i = 0; i < vertex_num; i++)
		{
			//if( isInChunkVV(i) && vv.count(i) == 0)
			if(vv.count(i) == 0)
			{
				int ii = i * chunk_size, iii;
                        	for(iii = 0; iii < chunk_size; iii++)
                        	{
                                	if(vvArray[ii+iii] == 0)
                                	{
						vvArray[ii+iii] = 1;
						vvPartition[i] = iii;
						vvSize[iii]++;
						//std::cout << "adding...... table: "<< iii << ", vvSize = " << vvSize[iii] << std::endl;
                                        	break;
                                	}        
                        	}
				if(iii == chunk_size)	
				{
					iii--;
					vvArray[ii+iii] = 1;
					vvPartition[i] = iii;
					vvSize[iii]++;
					//std::cout << "adding...... table: "<< iii << ", vvSize = " << vvSize[iii] << std::endl;
				}
				vv.insert(i);
			}
		}
/*
		if(vv.size() == vertex_num)
		{
			for(int i = 0; i < vertex_num * chunk_size; i++)
			{
				if(vvArray[i] == 1)
				{
					vvPartition[i/chunk_size] = i%chunk_size;
				}
			}
		}
*/
	}

	int get_rate_collision(int chunk_i, int chunk_j)
	{
		if(chunk_i == chunk_j)
		{
			//if(chunk_i == (chunk_size -1))
			{
				int rate = 0;
				for(int index = 0; index < vertex_num * chunk_size; )
                		{
                        		if(vvArray[index+chunk_i] != 1)
                        		{
                                		rate += vvArray[index+chunk_i];
                        		}
                        		index += chunk_size;
                		}
				std::cout << "Pocessing the rate of colision between " << chunk_i << " && " << chunk_j << "; "<< "the rate number is " << -rate << std::endl;
				return -rate;
			}

			//std::cout << "input error , chunk should be different..." << std::endl;
			//return 0;
		}
		if(chunk_i > chunk_size || chunk_j > chunk_size)
		{
			std::cout<< "input error , chunk number is too large..." << std::endl;
			return 0;
		}
		//i = min, j = max
		int i = chunk_i < chunk_j ? chunk_i : chunk_j, j = chunk_i > chunk_j ? chunk_i : chunk_j;
		int rate = 0;


/*
 * //rate 0.1v
		for(int index = 0; index < vertex_num * chunk_size; )
		{
			if((vvArray[index+i] + vvArray[index+j]) == 0)
			{
				if(vvArray[index+i])
				{
					rate++;
				}
			}
			index += chunk_size;
		}
*/

		for(int index = 0; index < vertex_num * chunk_size; )
                {
                        if(vvArray[index+j] == 1)
                        {
				rate += (- vvArray[index+i]);
                        }
			if(vvArray[index+i] == 1)
			{
				rate += (-vvArray[index+j]);
			}

                        index += chunk_size;
                }
		std::cout << "Pocessing the rate of colision between " << i << " && " << j << "; "<< "the rate number is " << rate << std::endl;
		return rate;
	}


	int get_src_partition(unsigned int src, unsigned int dst)
	{
		if(src > vertex_num)
		{
			std::cout << "Error , out the vertex!" << std::endl;
			exit(-1);
		}
		return vvPartition[src];
	}
/*
	int releaseVertexVector()
        {
                if(vvArray != NULL)
                {
                        free(vvArray);
			free(vvPartition);
                        free(vvSize);
                        vv.clear();
                }
		return 1;
        }
*/
	int saveVertexVector(std::string file_name)
	{
		std::string save_file = file_name + ".vv";
		FILE * fv = fopen(save_file.c_str(), "w");

		if(fv == NULL){
			std::cout << "file open error in vv..." << std::endl;
		}

		fprintf(fv, "vv %u %u\n", vertex_num, chunk_size);
		for(unsigned int i = 0; i < vertex_num; i++)
		{
			fprintf(fv, "%u", i);
			int index = i * chunk_size;
			for(int j = 0; j < chunk_size; j++)
			{
				fprintf(fv, " %d", vvArray[index + j]);
			}
			fprintf(fv, "\n");
		}
		fclose(fv);
		return 1;
	}

	int saveVVPartition(std::string file_name)
	{
		std::string save_file = file_name + ".vp";
                FILE * fv = fopen(save_file.c_str(), "w");

                if(fv == NULL){
                        std::cout << "file open error in vp..." << std::endl;
                }

                fprintf(fv, "vp %u %u\n", vertex_num, chunk_size);
                for(unsigned int i = 0; i < vertex_num; i++)
                {
                        fprintf(fv, "%u %d\n", i, vvPartition[i]);
                }
                fclose(fv);
                return 1;
	}


	unsigned int insert2map(unsigned int vertex, unsigned int index)
	{
		cvmap.insert( std::pair<unsigned int, unsigned int>(vertex, index) );
		vcmap.insert( std::pair<unsigned int, unsigned int>(index, vertex) );
		return index;
	}

	unsigned int get_id_by_index(unsigned int index)
	{
		return vcmap[index];
	}

	int isInChunkVV(unsigned int id)
	{
		return cvmap.count(id);
	}

	void after_add_chunk_vertex()
	{
		after_add_vertex();
	}

	int saveChunkVertexVector(std::string file_name, int _chunk_num, int _chunk_size)
        {
                std::string save_file = file_name + ".cv";
                FILE * fv = fopen(save_file.c_str(), "w");

                if(fv == NULL){
                        std::cout << "file open error in vv..." << std::endl;
                }
		unsigned int _vertex_num = get_vertex_num();

                fprintf(fv, "cv v%u c%d s%d\n", _vertex_num, _chunk_num, _chunk_size);
                for(unsigned int i = 0; i < _vertex_num; i++)
                {
                        fprintf(fv, "%u", get_id_by_index(i));
                        int index = i * _chunk_size;
                        for(int j = 0; j < _chunk_size; j++)
                        {
                                fprintf(fv, " %d", vvArray[index + j]);
                        }
                        fprintf(fv, "\n");
                }
                fclose(fv);
                return 1;
        }

	int saveChunkVPartition(std::string file_name, int chunk_num)
        {
		std::stringstream str;
		str<< chunk_num;
                std::string save_file = file_name + "." + str.str() + ".cvp";
                FILE * fv = fopen(save_file.c_str(), "w");

                if(fv == NULL){
                        std::cout << "file open error in cvp..." << std::endl;
                }

                fprintf(fv, "vp %u %u\n", vertex_num, chunk_size);
                for(unsigned int i = 0; i < vertex_num; i++)
                {
                        fprintf(fv, "%u %d\n", get_id_by_index(i), vvPartition[i]);
                }
                fclose(fv);
                return 1;
        }

};
