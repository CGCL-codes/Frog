#ifndef GRAPH_H_INCLUDED
#define GRAPH_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

/*
This file defines the data structures for Graph
NOTE:
	IDs for Vertex & Edge start from 0
	Partition IDs start from 1
	ASCII file has IDs for Vertex & Edge start from 1
	ASCII file format Could be found in code of write_graph_ascii()
*/

/* The Graph Data Structure is an EDGE LIST combined with index for vertices */
struct graph_t
{
    /* number of vertices & edges */
    int vertex_num;
    int edge_num;
    /* Vertex v has outgoing edge from
        index vertex_begin[v] (inclusive) to
        index vertex_begin[v + 1] (exclusive) */
    int * vertex_begin;
    /* edge_dest[e] is the destination vertex of Edge e */
    int * edge_dest;
    /* edge_src[e] is the source vertex of Edge e */
    int * edge_src;
    /* vertex_begin[v] == vertex_begin[v + 1]) */
    int * vertex_end;
};
typedef struct graph_t Graph;
/* write & read graph data from ASCII file */
void write_graph_ascii(const Graph * const g, const char * const filename);
Graph * read_graph_ascii(const char * const filename);
/* write & read graph data from & to binary file */
void write_graph_bin(const Graph * const g, const char * const filename);
Graph * read_graph_bin(const char * const filename);
/* allocate & release memory */
Graph * allocate_graph(const int vertex_num, const int edge_num);
void release_graph(Graph * g);

/* CSR graph storage struct */
struct graph_csr_t
{
    int vertex_num;
    int edge_num;
    int * vertex_begin;
    int * edge_dest;
};
/* read graph data from ASCII file, returns an compressed data struct */
struct graph_csr_t * read_csr_ascii(const char * const filename);
/* write & read CSR graph data from & to binary file */
void write_csr_bin(const struct graph_csr_t * const g, const char * const filename);
struct graph_csr_t * read_csr_bin(const char * const filename);
/* allocate & release memory */
struct graph_csr_t * allocate_csr_t(const int vertex_num, const int edge_num);
void release_csr_t(struct graph_csr_t * g);
/* transform from graph_csr_t to graph_t, this is NOT deep copy */
Graph * get_graph(struct graph_csr_t * g);

/* graph with all edges reversed, should Not be called on Partition */
Graph * get_reverse_graph(const Graph * const g);

#ifdef __cplusplus
}
#endif

#endif // #ifndef GRAPH_H_INCLUDED
