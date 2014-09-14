#pragma once
#include <stdio.h>
#include <string>
#include <iostream>

struct VertexData{
	unsigned int id;
	unsigned int begin;
	unsigned int end;
	unsigned int lock;
	float val;
	//bool operator == (struct EdgeData &p);
};
/*
bool EdgeData::operator == (struct EdgeData &p)
{
	if(src == p.src && dst == p.dst)
	{
		return true;
	}
	return false;
}
*/
