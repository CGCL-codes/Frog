#ifndef PARTITION_H_INCLUDED
#define PARTITION_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

/* This file defines the functions for graph partition
 * Note here is not a full graph partition
 * vertices in a graph is divided into specified number of chunks
 */

/* structure for partition result */
struct part_table {

	/* vertex number of the graph */
	int vertex_num;

	/* edge number of the graph */
	int edge_num;

	/* number of chunks in the table */
	int part_num;

	/* number of vertices in each partition */
	int * part_vertex_num;

	/* number of edges in each partition */
	int * part_edge_num;

	/* partition number for each vertex [1, part_num] */
	int * vertex_part;

	/* vertex indices in each partition [1, vertex_num] */
	int ** part_vertex;

};

/* allocate & free memory for the table */
struct part_table * allocate_table (const int vertex_num, const int edge_num, const int part_num);
void release_table (struct part_table * t);

/* save & write table to / from disk file */
void write_table_bin(const struct part_table * const t, const char * const filename);
struct part_table * read_table_bin(const char * const filename);

/* do the partition on graph */
struct part_table * partition(const int vertex_num, const int edge_num, const int *const vertex_begin, const int *const edge_dest, const int part_num);

#include "graph.h"
/* a graph after partition was cut into several graphs */
Graph ** get_cut_graphs(const Graph * const g, const struct part_table * const t);

#ifdef __cplusplus
}
#endif

#endif // #ifndef PARTITION_H_INCLUDED
