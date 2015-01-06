//#ifndef EDGEV_H
//#define EDGAV_H

//#endif
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <string.h>

#include "../graphdata/EdgeData.hpp"

class EdgeVector{
private:
	int chunk_num;
	unsigned int edge_num;

public:
	std::vector<struct EdgeData> ev;

public:
	EdgeVector():chunk_num(0), edge_num(0){}
	~EdgeVector()
	{
		if(edge_num)
		{
			//ev.clear();
			edge_num = 0;
		}
	}
	
	void set_chunk_num(int _num)
	{
		chunk_num = _num;
	}

	int get_chunk_num()
	{
		return chunk_num;
	}

	void set_edge_num(int _edge)
	{
		edge_num = _edge;
	}

	int get_edge_num()
	{
		return edge_num;
	}

	void initEdgeVector(unsigned int _init)
	{
		if(edge_num)
		{
			ev.clear();
			edge_num = 0;
			//TODO
		}
		std::cout << "reserve edgevector number is " << _init << std::endl;
		ev.reserve(_init);
		std::cout << "init EdgeVector sucess!" <<std::endl;
	}

	int add_edge_data(unsigned int _src, unsigned int _dst, float _val, int src_partition)
	{
		if(src_partition == chunk_num)
		{
			add_edge(_src, _dst, _val);
		}
		return ev.size();
	}

	void add_edge(unsigned int _src, unsigned int _dst, float _val)
	{
		struct EdgeData tmp;
		tmp.src = _src;
		tmp.dst = _dst;
		//tmp.val = _val;
		tmp.weight = 2;
		ev.push_back(tmp);
		edge_num++;
/*
		if(edge_num < 100)
		{
			std::cout << "edges weight " << tmp.src << " ," << tmp.dst << " ," << tmp.weight << std::endl;
		}
		if(edge_num % (10000000) == 0)
			std::cout << "add edge number is already " << edge_num << std::endl;
*/
	}
/*
	float get_edge_value(unsigned int _src, unsigned int _dst)
	{
		for(int i = 0; i < ev.size(); i++)
		{
			if(ev[i].src == _src && ev[i].dst == _dst)
			{
				return ev[i].val;
			}
		}

		return 0.0;	
	}

	void set_edge_value(unsigned int _src, unsigned int _dst, float _val)
	{
		for(int i = 0; i < ev.size(); i++)
                {
                        if(ev[i].src == _src && ev[i].dst == _dst)
                        {
                                ev[i].val = _val;
                        }
                }
	}
*/
	void after_add_edges()
	{
		std::cout << "The total edge num is " << edge_num << std::endl;
		std::cout << "add edges complete!" << std::endl << std::endl;
	}

};
