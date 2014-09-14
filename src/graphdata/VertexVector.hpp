#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <string.h>

#include "../graphdata/VertexData.hpp"

class VertexVector {
private:
	unsigned int vertex_num;
	unsigned int chunk_size;

public:
	std::set<unsigned int> vv;
	int *vvArray;
	int *vvPartition;
	unsigned int *vvSize;
	unsigned int divided;
	
public:
	VertexVector()
	{
		divided = 0;
		vvArray = NULL;
		vvPartition = NULL;
		vvSize = NULL;
	}

	~VertexVector()
	{
//		if(vvArray != NULL)
//			free(vvArray);

        if(vvSize != NULL)
			free(vvSize);

        if(vvPartition != NULL)
			free(vvPartition);
        
        if(vv.size())
			vv.clear();
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

	void initVertexVector()
	{
		if(vertex_num == 0 || chunk_size == 0)
		{
			std::cout << "No Vertex Records Or Not set chunk size." << std::endl;	
			exit(-1);
		}

		vvArray = (int *)malloc(sizeof(int)*vertex_num*chunk_size);
		vvPartition = (int *)malloc(sizeof(int)*vertex_num);
		vvSize = (unsigned int *)malloc(sizeof(unsigned int)*chunk_size);
		if(vvSize == NULL ||  vvPartition == NULL || vvArray == NULL)
		{
			std::cout << "Malloc vvAray or VVPartition or vvSize Error." << std::endl;
		}
		memset(vvArray, 0, vertex_num * chunk_size * (sizeof(int)/ sizeof(char)));
		memset(vvPartition, 0, vertex_num * (sizeof(int)/ sizeof(char)));
		memset(vvSize, 0, chunk_size * (sizeof(unsigned int)/ sizeof(char)));
        std::cout << "vertex vector init success ..." <<std::endl;
    }

	void addVertexOne(unsigned int src, unsigned int dst)
	{
		int i = (src) * chunk_size, j = (dst) * chunk_size;
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
                                        if(vv.count(dst) == 0 && vvArray[j+ii] != 1)
                                        {
                                                vvArray[j+ii] += -1;
                                        }
                                        break;
                                }
                        }
                        if(ii == chunk_size)
                        {
                                ii--;
                                vvArray[i+ii] = 1;
                                vvPartition[src] = ii;
                                vvSize[ii]++;
                                if(vv.count(dst) == 0 && vvArray[j+ii] != 1)
                                {
                                        vvArray[j+ii] += -1;
                                }
                        }
                        vv.insert(src);
                        return ;
                }

		 if(vv.count(src) == 1 && vv.count(dst) == 0)
                {
                        int ii;
                        for(ii = 0; ii < chunk_size; ii++)
                        {
                                if(vvArray[i+ii] == 1)
                                {
                                        if(vvArray[j+ii] != 1)    vvArray[j+ii] += -1;
                                        break;
                                }
                        }
                        return ;
                }
	}

	void addVertex(unsigned int src, std::vector<unsigned int>& dsts)
	{
		unsigned int len = dsts.size(), is = (src) * chunk_size;
		//std::cout << "len = " << len << std::endl;
		for(unsigned int i = 0; i < len; i++)
		{
			unsigned int tmp = dsts[i];
			if(vv.count(tmp) == 1)
			{
				int dp = vvPartition[tmp];
				vvArray[is + dp] += -1;
				//dsts.erase(dsts.begin() + i);
			}
		}
	
		int ii = 0;
		for(; ii < chunk_size; ii++)
		{
			if(vvArray[is+ii] == 0)
                       	{
                                vvArray[is+ii] = 1;
                                vvPartition[src] = ii;
                                vvSize[ii]++;
                                break;
                        }
		}	
		if(ii == chunk_size)
                {
                        ii--;
                        vvArray[is+ii] = 1;
                        vvPartition[src] = ii;
                        vvSize[ii]++;
                }

                vv.insert(src);

		len = dsts.size();
		for(unsigned int i = 0; i < len; i++)
		{
			unsigned int tmp = dsts[i];
			//std::cout << tmp << " ";
			if(vv.count(tmp) == 0)
                        {
				unsigned int j = tmp * chunk_size;
                                vvArray[j+ii] += -1;
                        }
		}
		//std::cout << std::endl;	
	}

	void add_vertex_direct()
	{
		
	}
	// add the vertex just as the dst in the file
	void after_add_vertex()
	{
		for(int i = 0; i < vertex_num; i++)
		{
			if( vv.count(i) == 0)
			{
				int ii = i * chunk_size, iii;
                        	for(iii = 0; iii < chunk_size; iii++)
                        	{
                                	if(vvArray[ii+iii] == 0)
                                	{
						vvArray[ii+iii] = 1;
						vvPartition[i] = iii;
						vvSize[iii]++;
                                        	break;
                                	}        
                        	}
				if(iii == chunk_size)	
				{
					iii--;
					vvArray[ii+iii] = 1;
					vvPartition[i] = iii;
					vvSize[iii]++;
				}
				vv.insert(i);
			}
		}

		if(vv.size() > 0)
		{
			{
				std::set<unsigned int> temp;
				vv.swap(temp);
			}

			std::cout << "release set vv ... size() = " << vv.size() << std::endl;
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

	int saveVertexVector(std::string file_name, unsigned int e_num)
	{
		std::string save_file = file_name + ".vv";
		FILE * fv = fopen(save_file.c_str(), "w");

		if(fv == NULL){
			std::cout << "file open error in vv..." << std::endl;
		}

		fprintf(fv, "vv %u %u %u\n", vertex_num, e_num, chunk_size);

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

	void set_bakery(unsigned int src, unsigned int dst)
    	{
         	int i = (src) * chunk_size, j = (dst) * chunk_size, last = chunk_size - 1;
         	vvArray[i + last] +=1;
         	vvArray[j + last] +=1; 
    	}

	int get_bakery(std::string file, int chunk)
	{
		std::cout << "Start to processing input file again and creat lock values: " << file << std::endl;
		std::ifstream gt_file(file.c_str());
		if(!gt_file.good())
		{
			std::cout << "Failed Open File " << file << std::endl;
			exit(-1);
		}

		unsigned int src_id, dst_id, e_num;
		float edge_weight;
		char line[1024];
		char first_ch;
		int cmp = chunk - 1;
		
		for(unsigned int i = 0; i < vertex_num; i++)
        	{
             		if(vvPartition[i] == cmp)
             		{
                  		vvArray[i * chunk_size + cmp] = 1;                 
             		}             
        	}
		
		while(gt_file.get(first_ch))
		{
			if(first_ch == 'p')
			{
				std::string temp;
				gt_file>>temp>>vertex_num>>e_num;
				std::cout << "v_num = " << vertex_num <<" , e_num = " << e_num <<std::endl;
				gt_file.getline(line, 1024);//eat the line break
				break;
			}
			gt_file.getline(line, 1024);//eat the line break
		}

		

		while(gt_file.get(first_ch))
		{
			if(first_ch == 'a')
			{
				gt_file>>src_id>>dst_id>>edge_weight;
				if(vvPartition[src_id] == cmp && vvPartition[dst_id] == cmp)
                		{
                    			set_bakery(src_id, dst_id);
                		}
			}
		}
     
		gt_file.close();
       		saveVertexVector(file, e_num);
		saveVVPartition(file);	
		
		return vv.size();
	}

};
