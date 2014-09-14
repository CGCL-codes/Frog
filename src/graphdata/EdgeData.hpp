#pragma once
#include <stdio.h>
#include <string>
#include <iostream>

struct EdgeData{
	unsigned int src;
	unsigned int dst;
//	float val;
	bool operator < (struct EdgeData &p);
};

bool EdgeData::operator < (struct EdgeData &p)
{
	if(dst < p.dst)
	{
		return true;
	}
	if(dst == p.dst)
	{
		return src < p.src;
	}
	return false;
}

int cmpEdgeData(const void * a, const void * b)
{
	if(*(struct EdgeData *)a < *(struct EdgeData *)b)
	{
		return -1;
	}
	return 1;
}

