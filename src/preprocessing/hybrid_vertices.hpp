#pragma once
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <set>
#include <algorithm>
#include "../device/memCPUGPU.h"

using namespace std;

int *vvPar;

int mem_vv_partition(unsigned int v_num)
{
	vvPar = (int *) malloc(sizeof(int) * v_num);
	memset(vvPar, 0, sizeof(int)/sizeof(char) * v_num);
	return v_num;
}

unsigned int n_vertices_partition(int a)
{
	std::set<unsigned int> v_a;
	for(unsigned int i = 0; i < num[a]; i++)
        {
                v_a.insert(m_all_edge[a][i].src);
                v_a.insert(m_all_edge[a][i].dst);
        }
	std::cout << "vertices[" << a "] = " << v_a.size() < std::endl;
	return v_a.size();
}

unsigned int n_hybird_vertices(int a, int b, unsigned int v_num)
{
	unsigned int sum = 0;
	std::set<unsigned int> v_a, v_b;
//	std::cout << "hyvrid num = " << num[a] <<" && "<< num[b] << std::endl;

	for(unsigned int i = 0; i < num[a]; i++)
	{
		v_a.insert(m_all_edge[a][i].src);
		v_a.insert(m_all_edge[a][i].dst);
	}
	for(unsigned int i = 0; i < num[b]; i++)
	{
		v_b.insert(m_all_edge[b][i].src);
                v_b.insert(m_all_edge[b][i].dst);
	}

	for(unsigned int n = 0; n < v_num; n++)
	{
		if(v_a.count(n) == 1 && v_b.count(n) == 1)
		{
			sum++;
		}
	}
/*
	for(unsigned int i = 0; i < num[b]; i++)
	{
		if(vvPar[m_all_edge[b][i].src] == a)
		{
			sum++;
		}
	}
*/
	return sum;
}
