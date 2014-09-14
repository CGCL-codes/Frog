#pragma once
#include "../graphdata/EdgeData.hpp"
#include "../device/memCPUGPU.h"

int set_partition_vertex_src(int chunk)
{
        if(m_all_vertex == NULL)
        {
                return 0;
        }
        unsigned int pre = 0, cur = 0, index = 0;
        for(int i = 0; i < chunk; i++)
        {
                index = 0;
                pre = m_all_edge[i][0].src;
                m_all_vertex[i][index].end = 0;
                m_all_vertex[i][index].id = pre;
                m_all_vertex[i][index].begin = 0;
                for(unsigned int n = 0; n < m_edge[i]; n++)
		{
                        cur = m_all_edge[i][n].src;
                        if(cur != pre)
                        {
                                pre = cur;
                                m_all_vertex[i][index++].end = n;
                                m_all_vertex[i][index].id = pre;
                                m_all_vertex[i][index].begin = n;
                        }
                }
                m_all_vertex[i][index].end = m_edge[i];
        }
        return chunk;
}

int set_partition_vertex_dst(int chunk)
{
        if(m_all_vertex == NULL)
        {
                return 0;
        }
        unsigned int pre = 0, cur = 0, index = 0;
        for(int i = 0; i < chunk; i++)
        {
                index = 0;
                pre = m_all_edge[i][0].dst;
                m_all_vertex[i][index].end = 0;
                m_all_vertex[i][index].id = pre;
                m_all_vertex[i][index].begin = 0;
                for(unsigned int n = 0; n < n_edge[i]; n++)
                {
                        cur = m_all_edge[i][n].dst;
                        if(cur != pre)
                        {
                                pre = cur;
                                m_all_vertex[i][index++].end = n;
				//std::cout << "partition dst " << m_all_vertex[i][index-1].id << " " <<  m_all_vertex[i][index-1].begin << " " <<m_all_vertex[i][index-1].end << std::endl;
                                m_all_vertex[i][index].id = pre;
                                m_all_vertex[i][index].begin = n;
                        }
                }
                m_all_vertex[i][index].end = n_edge[i];
		//std::cout << "partition dst " << m_all_vertex[i][index].id << " " <<  m_all_vertex[i][index].begin << " " <<m_all_vertex[i][index].end << std::endl;
        }
        return chunk;
}

int set_partition_vertex_hybrid(int chunk)
{
        if(m_all_vertex == NULL)
        {
                return 0;
        }
        unsigned int pre = 0, cur = 0, index = 0;

	unsigned int _chunk = chunk - 1;	
	for(int i = 0; i < _chunk; i++)
        {
                index = 0;
                pre = m_all_edge[i][0].dst;
                m_all_vertex[i][index].end = 0;
                m_all_vertex[i][index].id = pre;
                m_all_vertex[i][index].begin = 0;
                for(unsigned int n = 0; n < m_edge[i]; n++)
                {
                        cur = m_all_edge[i][n].dst;
                        if(cur != pre)
                        {
                                pre = cur;
                                m_all_vertex[i][index++].end = n;
                                m_all_vertex[i][index].id = pre;
                                m_all_vertex[i][index].begin = n;
                        }
                }
                m_all_vertex[i][index].end = m_edge[i];
        }

        for(int i = _chunk; i < chunk; i++)
        {
                index = 0;
                pre = m_all_edge[i][0].src;
                m_all_vertex[i][index].end = 0;
                m_all_vertex[i][index].id = pre;
                m_all_vertex[i][index].begin = 0;
                for(unsigned int n = 0; n < n_edge[i]; n++)
                {
                        cur = m_all_edge[i][n].src;
                        if(cur != pre)
                        {
                                pre = cur;
                                m_all_vertex[i][index++].end = n;
                                m_all_vertex[i][index].id = pre;
                                m_all_vertex[i][index].begin = n;
                        }
                }
                m_all_vertex[i][index].end = n_edge[i];
        }
        return chunk;

}


/*
void swap(struct EdgeData &a, struct EdgeData &b)
{
	struct EdgeData tmp;
	std::cout << "before swap " <<a.dst << " " << b.dst << std::endl;
	tmp.src = a.src, tmp.dst = a.dst, tmp.val = a.val;
	a.src = b.src, a.dst = b.dst, a.val = b.val;
	b.src = tmp.src, b.dst = tmp.dst, b.val = tmp.val;
	std::cout << "after swap "<<a.dst << " " << b.dst << std::endl;
}

void quickSort(struct EdgeData *edges, unsigned int left, unsigned int right)
{
	unsigned int key = edges[left].dst;
	unsigned int m =left, n =right + 1;
	if(m > n)	return;

	std::cout << "Start Sort..."<<std::endl;
	while(m < n)
	{
		if(edges[n].dst > key)
		{
			n--;
		}
		else
		{
			if(edges[m].dst <= key)	m++;
		}
		swap(edges[m], edges[n]);
	}

	swap(edges[left], edges[n]);

	quickSort(edges, left, n-1);
	quickSort(edges, n+1, right);
}
*/
