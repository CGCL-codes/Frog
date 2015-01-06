#pragma once
#include <fstream>
#include <algorithm>
#include "../graphdata/VertexVector.hpp"
#include "../graphdata/EdgeVector.hpp"
#include "../graphdata/ChunkVertexVector.hpp"
#include "../device/memCPUGPU.h"

class PreProcessing {
public:
	VertexVector vv;
	EdgeVector ev;
	ChunkVertexVector *chunk_vv;
	unsigned int v_num, e_num;

private:
	int chunk_vv_init;//count the init-ed number
	std::string file_save;

public:
	PreProcessing():chunk_vv_init(-1){}

	~PreProcessing()
	{
		std::cout << "delete PreProcessing ...................."<< std::endl;
		if(file_save.empty() == false)
		{
			std::cout << "delete PreProcessing ...................."<< std::endl;
		}
	}

	int check_incremental_edges(std::string file_name, int chunk, unsigned int o_v, unsigned int o_e)
	{
		std::ifstream gt_file(file_name.c_str());
                if(!gt_file.good())
                {
                        std::cout << "Failed Open File " << file_name << std::endl;
                        exit(-1);
                }

                char line[1024];
                char first_ch;

                while(gt_file.get(first_ch))
                {
                        if(first_ch == 'p')
                        {
                                std::string temp;
                                gt_file>>temp>>v_num>>e_num;
                                std::cout << "v_num = " << v_num <<" , e_num = " << e_num <<std::endl;
                                gt_file.getline(line, 1024);//eat the line break
                                break;
                        }
                        gt_file.getline(line, 1024);//eat the line break
                }

		if(v_num == o_v)
		{
			std::cout << "No vertices incremental ......" << std::endl;
		}

		if(e_num == o_e)
		{
			std::cout << "No edges incremental ......" << std::endl;
			return 0;
		}

		if(v_num > o_v && e_num > o_e)
		{
			std::cout << "vertices incremental ... total " << (v_num - o_e) <<std::endl;
			std::cout << "edges incremental ... total " << (e_num - o_e) <<std::endl;
		}

		if(e_num > o_e)
		{
			std::cout << "edges incremental ... total " << (e_num - o_e) <<std::endl;
		}
		
		unsigned int src_id, dst_id, max_id, total = e_num - o_e;
                float edge_weight;

		total = 0;
		while(gt_file.get(first_ch))
        {
            if(first_ch == 'a')
            {
                gt_file>>src_id>>dst_id>>edge_weight;
				if(vv.vvPartition[src_id] == vv.vvPartition[dst_id])
				{
					//std::cout << src_id << " " << dst_id << " , " <<vv.vvPartition[src_id] << " " << vv.vvPartition[dst_id] << std::endl;
					if(vv.vvPartition[src_id] != (chunk - 1))
					{
						total++;
						max_id = src_id > dst_id ? src_id :dst_id;
						unsigned int i = (max_id) * chunk, j = vv.vvPartition[max_id];
						vv.vvSize[j]--;
						vv.vvArray[i + j] = -1;
						j = chunk - 1;
						vv.vvSize[j]++;
                        vv.vvArray[i + j] = 1;
                        vv.vvPartition[max_id] = j;
					}
				}
            }
        }

		std::cout << "check incremental total = " << total << std::endl;
		return 0;
	}

	int if_exist_vv(std::string file_name, int partition_num)
	{
		std::cout << "Start to processing input file: " << file_name << " , partition_num is: " << partition_num << std::endl;
		std::string file_vv = file_name + ".vv", file_vp = file_name + ".vp";
                std::ifstream gt_file_vv(file_vv.c_str()), gt_file_vp(file_vp.c_str());
		file_save = file_name;

                if(!gt_file_vv.good() || !gt_file_vp.good())
                {
                        std::cout << "Failed Open File " << file_vv << " Or " << file_vp << std::endl;
                        gt_file_vv.close();
                	gt_file_vp.close();
			return 0;
                }

		unsigned int v_num, e_num, v_chunk;
                char line[1024];
                char first_ch, second_ch;
		//int v_case;

                while(gt_file_vv.get(first_ch))
                {
                        if(first_ch == 'v')
                        {
				gt_file_vv.get(second_ch);
				if(second_ch == 'v')
				{
                                	gt_file_vv>> v_num >> e_num >> v_chunk;
                                	std::cout << "Processing the file VV : v_num = " << v_num << " , e_num = " << e_num <<" , v_chunk = " << v_chunk <<std::endl;
                                	gt_file_vv.getline(line, 1024);//eat the line break
                                	break;
				}
                        }
                        gt_file_vv.getline(line, 1024);//eat the line break
                }
		
		while(gt_file_vp.get(first_ch))
                {
                        if(first_ch == 'v')
                        {
                                gt_file_vp.get(second_ch);
                                if(second_ch == 'p')
                                {
                                        gt_file_vp>>v_num>>v_chunk;
                                        std::cout << "Processing the file VP : v_num = " << v_num <<" , v_chunk = " << v_chunk <<std::endl;
                                        gt_file_vp.getline(line, 1024);//eat the line break
                                        break;
                                }
                        }
                        gt_file_vp.getline(line, 1024);//eat the line break
                }
		
		vv.set_vertex_num(v_num);
                vv.set_chunk_size(v_chunk);
		if(v_chunk != partition_num)
		{
			if((v_chunk + 1) == partition_num)
			{
				std::cout << "divide the first partition from " << v_chunk << " to " << partition_num << std::endl;
				vv.set_chunk_size(partition_num);	
			}
			else{
				std::cout << "exist vv & vp files is old, processing the file again......" << std::endl;
				return 0;
			}
		}

                vv.initVertexVector();

		unsigned int tmp = 0, src_id, src_vp;

                while(tmp < v_num)
                {
			gt_file_vv>>src_id;
			int index = src_id * v_chunk;
			for(int i = 0; i < v_chunk; i++)
			{
				gt_file_vv>>vv.vvArray[index + i];
			}
			tmp++;
		}

		tmp = 0;
		while(tmp < v_num)
                {
                        gt_file_vp>>src_id>>src_vp;
			vv.vvPartition[src_id] = src_vp;
			vv.vvSize[src_vp]++;
                        tmp++;
                }

		if((v_chunk + 1) == partition_num)
		{
			int maxi = 0, tmp = 0;
			for(int i = 0; i < (v_chunk-1); i++)
			{
				if(vv.vvSize[i] > tmp)
				{
					tmp = vv.vvSize[i];
					maxi = i;
				}
			}

			std::cout << "divide partition is " << maxi << std::endl;
			tmp = 0;
			while(tmp < v_num)
                	{
                        	if(vv.vvPartition[tmp] >= maxi)
				{
					vv.vvPartition[tmp]++;
				}
				tmp++;
                	}
			tmp = vv.vvSize[maxi] / 2;
			for(int i = v_chunk; i > maxi; i--)
			{
				vv.vvSize[i] = vv.vvSize[i-1];
			}
			vv.vvSize[maxi] = tmp;
			vv.vvSize[maxi+1] -= tmp;
			for(unsigned int i = 0; i < v_num && tmp > 0; i++)
			{
				if(vv.vvPartition[i] == (maxi+1))
				{
					vv.vvPartition[i] = maxi;
					tmp--;
				}
			}
			std::cout << "divide success ......" << std::endl;
		}
		gt_file_vv.close();
		gt_file_vp.close();

		//vv.saveVertexVector(file_vv);
		//vv.saveVVPartition(file_vp);

		check_incremental_edges(file_name, partition_num, v_num, e_num);

		return v_num;
	}

	int get_partition(std::string file_name, int partition_num)
	{
		file_save = file_name;
		if( if_exist_vv(file_name, partition_num))
        	{       
                	std::cout << "processed already, now start reading the files......" << std::endl;
                	return vv.get_vertex_num();
       		}

		std::cout << "Start to processing input file: " << file_name << " , partition_num is: " << partition_num << std::endl;
		std::ifstream gt_file(file_name.c_str());
		if(!gt_file.good())
		{
			std::cout << "Failed Open File " << file_name << std::endl;
			exit(-1);
		}

		unsigned int src_id, dst_id, tmp_id;
		float edge_weight;
		char line[1024];
		char first_ch;

		while(gt_file.get(first_ch))
		{
			if(first_ch == 'p')
			{
				std::string temp;
				gt_file>>temp>>v_num>>e_num;
				std::cout << "v_num = " << v_num <<" , e_num = " << e_num <<std::endl;
				gt_file.getline(line, 1024);//eat the line break
				break;
			}
			gt_file.getline(line, 1024);//eat the line break
		}

		vv.set_vertex_num(v_num);	
		vv.set_chunk_size(partition_num);
		vv.initVertexVector();

		std::vector<unsigned int> dsts;

		gt_file.get(first_ch);
		if(first_ch == 'a')
                {
                        gt_file>>tmp_id>>dst_id>>edge_weight;
			//std::cout << tmp_id << "," << dst_id << std::endl;
                }
		src_id = tmp_id;
		//std::cout << "add dsts " << dst_id << std::endl;
		dsts.push_back(dst_id);

		while(gt_file.get(first_ch))
		{
			if(first_ch == 'a')
			{
				gt_file>>tmp_id>>dst_id>>edge_weight;
				//std::cout << tmp_id << " " << dst_id << std::endl;

				if(src_id == tmp_id)
                        	{
                                	//std::cout << "add dsts " << dst_id << std::endl;
                                	dsts.push_back(dst_id);
                        	}

                        	if(src_id != tmp_id)
                        	{
                                	vv.addVertex(src_id, dsts);
                                	src_id = tmp_id;
                                	dsts.clear();
                                	//std::cout << "add dsts " << dst_id << std::endl;
                                	dsts.push_back(dst_id);
                        	}

                if(src_id % 1000000 == 0)
                {
                    std::cout << "get partition processing id: " << src_id << std::endl;
                }
			}
		}

		if(dsts.size())
		{
			vv.addVertex(src_id, dsts);
		}

		vv.after_add_vertex();

		gt_file.close();

		std::cout << "save processed results in get_partition ..." << std::endl;
		vv.saveVertexVector(file_name, e_num);
		vv.saveVVPartition(file_name);

		return vv.vv.size();
	}

	int get_partition_adj_list(std::string file_name, int partition_num)
	{
		file_save = file_name;
                if( if_exist_vv(file_name, partition_num))
                {       
                        std::cout << "processed already, now start reading the files......" << std::endl;
                        return vv.get_vertex_num();
                }

		std::cout << "Start to processing input file: " << file_name << " , partition_num is: " << partition_num << std::endl;
                std::ifstream gt_file(file_name.c_str());
		if(!gt_file.good())
                {
                        std::cout << "Failed Open File " << file_name << std::endl;
                        exit(-1);
                }
		
		unsigned int src_id, dst_id, tmp_id;
		float edge_weight;
		char line[2014];
		char first_ch;

		while(gt_file.get(first_ch))
		{
			if(first_ch == 'p')
                        {
                                std::string temp;
                                gt_file>>temp>>v_num>>e_num;
                                std::cout << "v_num = " << v_num <<" , e_num = " << e_num <<std::endl;
                                gt_file.getline(line, 1024);//eat the line break
                                break;
                        }
                        gt_file.getline(line, 1024);//eat the line break
		}
		
		vv.set_vertex_num(v_num);
                vv.set_chunk_size(partition_num);
                vv.initVertexVector();

                std::vector<unsigned int> dsts;

		std::cout << "processing lists......" << std::endl;

		while(gt_file.get(first_ch))
		{
			std::cout << first_ch;
			if(first_ch == 'a')
			{
				gt_file >> src_id;
				std::cout << " lists: " << src_id;
				while(gt_file >> dst_id)
				{
					std::cout << " " << dst_id;
					dsts.push_back(dst_id);
				}
				std::cout << " ..." <<std::endl;
				vv.addVertex(src_id, dsts);
				dsts.clear();
			}
		}

		if(dsts.size())
                {
                        vv.addVertex(src_id, dsts);
                }

                vv.after_add_vertex();

		gt_file.close();

		//vv.saveVertexVector(file_name, e_num);
		vv.saveVVPartition(file_name);

		return vv.vv.size();
	}

	unsigned int get_all_edges(std::string file_name, int partition_num)
	{
		std::cout << "Start to processing input file: " << file_name << std::endl;
                std::ifstream gt_file(file_name.c_str());
                if(!gt_file.good())
                {
                        std::cout << "Failed Open File " << file_name << std::endl;
                        exit(-1);
                }

                unsigned int src_id, dst_id;
                float edge_weight;
                char line[1024];
                char first_ch;
		
		while(gt_file.get(first_ch))
                {
                        if(first_ch == 'p')
                        {
                                std::string temp;
                                gt_file>>temp>>v_num>>e_num;
                                std::cout << "v_num = " << v_num <<" , e_num = " << e_num <<std::endl;
                                gt_file.getline(line, 1024);//eat the line break
                                break;
                        }
                        gt_file.getline(line, 1024);//eat the line break
                }
		
		ev.set_edge_num(0);
		ev.initEdgeVector(e_num);
		//set_n_edge(partition_num);
		std::cout << "init EdgeVector sucess..." << std::endl;

                while(gt_file.get(first_ch))
                {
                        if(first_ch == 'a')
                        {
                                gt_file>>src_id>>dst_id>>edge_weight;
				if(src_id == dst_id) continue;
                                ev.add_edge(src_id, dst_id, edge_weight);
				m_edge[vv.vvPartition[src_id]]++; //TODO
				n_edge[vv.vvPartition[dst_id]]++;
                        }
                }

		std::cout << "processing left edges ..." << std::endl;
		ev.after_add_edges();

                gt_file.close();

                return ev.get_edge_num();
	}

	int get_chunk_edges(int _chunk, std::string file_name)
	{
		std::cout << "Start to get chunk edges, partition_num is: " << _chunk << std::endl;
                std::ifstream gt_file(file_name.c_str());
                if(!gt_file.good())
                {
                        std::cout << "Failed Open File " << file_name << std::endl;
                        exit(-1);
                }

                unsigned int src_id, dst_id;
                float edge_weight;
                char line[1024];
                char first_ch;

                while(gt_file.get(first_ch))
                {
                        if(first_ch == 'p')
                        {
                                std::string temp;
				int v_num, e_num;
                                gt_file>>temp>>v_num>>e_num;
                                std::cout << "v_num = " << v_num <<" , e_num = " << e_num <<std::endl;
                                gt_file.getline(line, 1024);//eat the line break
                                break;
                        }
                        gt_file.getline(line, 1024);//eat the line break
                }

		ev.set_chunk_num(_chunk);
		ev.set_edge_num(0);
		ev.initEdgeVector(e_num);
	
		while(gt_file.get(first_ch))
                {
                        if(first_ch == 'a')
                        {
                                gt_file>>src_id>>dst_id>>edge_weight;
                                ev.add_edge_data(src_id, dst_id, edge_weight, vv.get_src_partition(src_id, dst_id));
                        }
                }

                ev.after_add_edges();

                gt_file.close();

		return ev.get_edge_num();

	}
	

	int initChunkVV(int chunk_num, int chunk_size)
	{
		//if(chunk_vv_init == -1)
		{
			chunk_vv = new ChunkVertexVector;
			chunk_vv_init = chunk_num;

			//int total = vv.vvSize[chunk_num];
			unsigned int index = 0;
			for(unsigned int i = 0; i < v_num; i++)
			{
				if(vv.vvPartition[i] == chunk_num)
				{
					//if(index < 20 || index > 15600)std::cout<< "insert to map id : " << i << " ,  index : " << index << std::endl;
					chunk_vv->insert2map(i, index);
					index++;
				}
			}
			std::cout << "map vertex number is " << chunk_vv->get_map_size() << std::endl;

			//if(index == total)
			if(index)
			{
				chunk_vv->initChunkVertexVector(chunk_size, index);
				std::cout << "init Chunk VertexVector success......" << std::endl;
				return 1;
			}
		}
		std::cout << "Do nothing......PreProcessing.initChunkVV()......" << std::endl;
		return 0;
	}

	void release_chunk_vv()
	{
		if(chunk_vv_init != -1 && chunk_vv)
                {
                        //chunk_vv->releaseVertexVector();
                        delete chunk_vv;
                }
	}
	int get_chunk_vv(std::string file_name, int chunk_num, int chunk_size)
	{
		initChunkVV(chunk_num, chunk_size);
/*	
		unsigned int sum = 0, p_num = chunk_vv->get_chunk_size();
                for(int i = 0; i < p_num; i++)
                {
                        sum += chunk_vv->vvSize[i];
                        std::cout << "The " << i << "chunk has vertex num is " << chunk_vv->vvSize[i] << std::endl;
                }
                std::cout << "The sum is " << sum << std::endl;
*/
	
		std::cout << "Start to get chunk vv iteration, partition_num is: " << chunk_num << std::endl;
                std::ifstream gt_file(file_name.c_str());
                if(!gt_file.good())
                {
                        std::cout << "Failed Open File " << file_name << std::endl;
                        exit(-1);
                }

                unsigned int src_id, dst_id, map_src, map_dst;
                float edge_weight;
		char line[1024];
                char first_ch;

		while(gt_file.get(first_ch))
                {
                        if(first_ch == 'p')
                        {
                                std::string temp;
                                int v_num, e_num;
                                gt_file>>temp>>v_num>>e_num;
                                gt_file.getline(line, 1024);//eat the line break
                                break;
                        }
                        gt_file.getline(line, 1024);//eat the line break
                }
		
		int count = 0;
		while(gt_file.get(first_ch))
                {
                        if(first_ch == 'a')
                        {
                                gt_file>>src_id>>dst_id>>edge_weight;

                                //if(vv.vvPartition[src_id] == chunk_num && vv.vvPartition[dst_id] == chunk_num)
				if(chunk_vv->isInChunkVV(src_id))
				{
					int _both = 0;
					if(chunk_vv->isInChunkVV(dst_id))
					{
						_both = 1;
					}
					map_src = chunk_vv->cvmap[src_id];
					map_dst = chunk_vv->cvmap[dst_id];
					//if(_both) std::cout << "Processing the chunk , edge: " << src_id << " , " << dst_id << "; map id :" << map_src << " + "<< map_dst << std::endl;
					chunk_vv->addVertex(map_src, map_dst, _both);
					count++;
				}
                        }
                }
		std::cout << "get_chunk_vv add Vertex to map : " << count << std::endl;

		chunk_vv->after_add_chunk_vertex();

		chunk_vv->saveChunkVertexVector(file_name, chunk_num, chunk_size);
		chunk_vv->saveChunkVPartition(file_name, chunk_num);

		gt_file.close();

		return chunk_vv->vv.size();
	}

	void debug_print_vertex(int partition_num)
	{
	/*
		for(int i = v_num; i < v_num; i++)
		{
			if( i < 100)
			{	
				std::cout << "v_id = " << i << "table : ";
				for(int j = 0; j < partition_num; j++)
				{
					std::cout << vv.vvArray[i*partition_num +j] <<" ";
				}
				std::cout << std::endl;
			}
		}
	*/
		unsigned int sum = 0;
		for(int i = 0; i < partition_num; i++)
		{
			sum += vv.vvSize[i];
			std::cout << "The " << i << " chunk has vertex num is " << vv.vvSize[i] << std::endl;
		}
		std::cout << "The sum is " << sum << std::endl;
/*
		std::cout << "The Tables are :" << std::endl;

		for(int i = 0; i < vv.get_chunk_size() * vv.get_vertex_num(); i++)
		{
			//if((i % vv.get_vertex_num()) == 0)  std::cout << std::endl;
			std::cout << vv.vvArray[i] << " ";
		}

		int i;
		for(i = 1; i < vv.get_chunk_size();i++)
		{
			vv.get_rate_collision(i-1, i);
		}
		//vv.get_rate_collision(i-1, i-1);


		for(int i = 0, j = 0; j < vv.get_chunk_size(); i++)
		{
			if(i == 0)	std::cout << std::endl << "Table " << j << ": ";
			if(vv.vvPartition[i] == j)	
			{
				std::cout << i << "\t";
			}
			if(i == v_num) i = -1, j++;
		}
*/
		std::cout << std::endl;
	}

	int debug_print_chunk(int chunk_num)
	{
		if(chunk_num != chunk_vv_init)
		{
			std::cout<< "Input the chunk num ERROR......" << "chunk_num : " << chunk_num << " , chunk_vv_init : " <<chunk_vv_init <<std::endl;
			return 0;
		}

		std::cout << "The Chunk Vertex Num is " << chunk_vv->get_vertex_num() << " , Chunk Size is " << chunk_vv->get_chunk_size()<< std::endl;

		unsigned int sum = 0, p_num = chunk_vv->get_chunk_size();
                for(int i = 0; i < p_num; i++)
                {
                        sum += chunk_vv->vvSize[i];
                        std::cout << "The " << i << "chunk has vertex num is " << chunk_vv->vvSize[i] << std::endl;
                }
                std::cout << "The sum is " << sum << std::endl;

		//chunk_vv->saveChunkVPartition("test.txt", chunk_num);

		int i;
                for(i = 1; i < p_num;i++)
                {
                        chunk_vv->get_rate_collision(i-1, i);
                }

		std::cout<< std::endl;

		return 1;
	}

	int get_n_vertex_partition(int partition_num)
	{
		std::cout << "set vertex partition ..." << std::endl;
		set_n_vertex(partition_num);
		for(int n = 0; n < partition_num; n++)
		{
			n_vertex[n] = vv.vvSize[n];
			std::cout<< "n_vertex[ " << n << " ] = " << n_vertex[n] << std::endl;
		}
		return partition_num;
	}

	int set_partition_edges_dst(int chunk, int _e_num)
	{
		printf("coming into set_partition_edges 1......\n");
		if(m_all_edge == NULL)
                {
                        return 0;
                }

		unsigned int *c_size;
        	c_size = (unsigned int *) malloc(sizeof(unsigned int) * chunk);
        	memset(c_size, 0, sizeof(unsigned int)/sizeof(char) * chunk);
		
		printf("coming into set_partition_edges 2...... e_num = %d\n", _e_num);
/*		
		for(unsigned int n = 0; n < _e_num ; n++)
                {
                        unsigned int j = vv.vvPartition[ev.ev[n].dst];
                        unsigned int index = c_size[j]++;
			std::cout << "j = " << j << " , index = " << index << "......";
			m_all_edge[j][index].src = ev.ev[n].src;
                        m_all_edge[j][index].dst = ev.ev[n].dst; 
                        m_all_edge[j][index].val = 0.0;
			std::cout << m_all_edge[j][index].src << " " << m_all_edge[j][index].dst << " " << m_all_edge[j][index].val << std::endl;
		}

*/

		const unsigned int chunk_e = 4, edgeNumberofChunk = _e_num / chunk_e, leftNumber = _e_num % chunk_e;
		unsigned int lastChunk = edgeNumberofChunk + leftNumber, total_e = 0;

		std::cout << "processing edges dst ... edgeNumber = " << edgeNumberofChunk << "... lastChunk =" << lastChunk << std::endl;
		for(unsigned int i = 0; i < (chunk_e - 1); i++)
		{
			for(unsigned int n = 0; n < edgeNumberofChunk; n++)
        		{			
                		unsigned int j = vv.vvPartition[ev.ev[n].dst];
                		unsigned int index = c_size[j]++;
				if(n == edgeNumberofChunk / 10)
					std::cout << "j = " << j << " , index = " << index << "......";
                		m_all_edge[j][index].src = ev.ev[n].src;
                		m_all_edge[j][index].dst = ev.ev[n].dst;
                		//m_all_edge[j][index].val = 0.0;
				if(n == edgeNumberofChunk / 10)
					std::cout << m_all_edge[j][index].src << " " << m_all_edge[j][index].dst << std::endl;
                	}
			std::cout << "processing edge chunk " << i << "... left number is  "<< ev.ev.size() <<std::endl;
			ev.ev.erase(ev.ev.begin(), ev.ev.begin() + edgeNumberofChunk);
		}

		for(unsigned int n = 0; n < lastChunk; n++)
		{
			unsigned int j = vv.vvPartition[ev.ev[n].dst];
			unsigned int index = c_size[j]++;
			if(n == edgeNumberofChunk / 10)
				std::cout << "j = " << j << " , index = " << index << "......";
			m_all_edge[j][index].src = ev.ev[n].src;
			m_all_edge[j][index].dst = ev.ev[n].dst;
			//m_all_edge[j][index].val = 0.0;

			if(n == edgeNumberofChunk / 10)
				std::cout << m_all_edge[j][index].src << " " << m_all_edge[j][index].dst << std::endl;
		}
		std::cout << "processing edge chunk " << chunk_e << "... left number is " << ev.ev.size() <<std::endl;
		ev.ev.erase(ev.ev.begin(), ev.ev.begin() + lastChunk);

		free(c_size);

		printf("coming into set_partition_edges 3 sort......\n");

		for(int n = 0; n < chunk; n++)
		{
			std::qsort(m_all_edge[n], n_edge[n], sizeof(struct EdgeData), cmpEdgeData);
		}

		printf("leveling out set_partition_edges ......\n");
		return chunk;
	}

	int set_partition_edges_src(int chunk, int _e_num)
        {
                if(m_all_edge == NULL)
                {
                        return 0;
                }

                unsigned int *c_size;
                c_size = (unsigned int *) malloc(sizeof(unsigned int) * chunk);
                memset(c_size, 0, sizeof(unsigned int)/sizeof(char) * chunk);

                for(unsigned int n = 0; n < _e_num ; n++)
                {
                        unsigned int j = vv.vvPartition[ev.ev[n].src];
                        unsigned int index = c_size[j]++;
                        m_all_edge[j][index].src = ev.ev[n].src;
                        m_all_edge[j][index].dst = ev.ev[n].dst;
                        m_all_edge[j][index].weight = ev.ev[n].weight;
                }
                free(c_size);

                return chunk;
        }
	
        int set_partition_edges_hybrid(int chunk, int _e_num)
        {
                if(m_all_edge == NULL)
                {
                        return 0;
                }

                unsigned int *c_size;
                c_size = (unsigned int *) malloc(sizeof(unsigned int) * chunk);
                memset(c_size, 0, sizeof(unsigned int)/sizeof(char) * chunk);

                for(unsigned int n = 0; n < _e_num ; n++)
                {
                        unsigned int j = vv.vvPartition[ev.ev[n].src];
                        unsigned int index = c_size[j]++;
                        m_all_edge[j][index].src = ev.ev[n].src;
                        m_all_edge[j][index].dst = ev.ev[n].dst;
                        //m_all_edge[j][index].val = 0.0;
                }
                free(c_size);

		int _chunk = chunk - 1;
                for(int n = 0; n < _chunk; n++)
                {
                        std::qsort(m_all_edge[n], m_edge[n], sizeof(struct EdgeData), cmpEdgeData);
                }
                return chunk;
        }

	int init_vertex_lock(int chunk)
	{
		int cmp = chunk - 1;
		unsigned int size = n_vertex[cmp], id = 0, index = 0;
		for(unsigned int i = 0; i < size; i++)
                {
			id = m_all_vertex[cmp][index].id;
                	v_lock[index] = vv.vvArray[id * chunk + cmp];
			m_all_vertex[cmp][index].lock = index++;// = index of v_lock
			//m_all_vertex[cmp][index].lock = v_lock[index++] - 1; // = lock val, -1 to start from 0; lock start from 1
                }
		return chunk;
	}

	int get_bakery_lock(int chunk)
	{
		int cmp = chunk - 1;
		unsigned int size = num[cmp], src = 0, dst = 0;

		for(unsigned int i = 0; i < v_num; i++)                                   
                {                                                                              
                        if(vv.vvPartition[i] == cmp)                                              
                        {                                                                      
                                vv.vvArray[i * chunk + cmp] = 1;                             
                        }                                                                      
                }

		for(unsigned int n = 0; n < size; n++)
		{
			src = m_all_edge[cmp][n].src;
			dst = m_all_edge[cmp][n].dst;
			if(vv.vvPartition[src] == vv.vvPartition[dst])
			{
				unsigned int i = (src) * chunk, j = (dst) * chunk;
                		//vv.vvArray[i + cmp] +=1;
                		//vv.vvArray[j + cmp] +=1;
                		if(vv.vvArray[i + cmp] == vv.vvArray[j + cmp])
				{
					vv.vvArray[i + cmp] = vv.vvArray[j + cmp] + 1;
				}
				//vv.vvArray[i + cmp] = src + 1;
                		//vv.vvArray[j + cmp] = dst + 1;	
			}
		}
		
		return 0;
	}
};
