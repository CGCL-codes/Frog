#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#include "graph.h"

void write_graph_ascii(const Graph * const g, const char * const filename)
{
    int i = 0;
    FILE * f = fopen(filename, "w");
    if (f != NULL) {
        fprintf(f, "p sp %d %d\n", g->vertex_num, g->edge_num);
        for (i = 0; i < g->edge_num; i++) {
            /* weight is ignored, and IDs are added with 1 to restore original value */
            fprintf(f, "a %d %d 1\n", g->edge_src[i] + 1, g->edge_dest[i] + 1);
        }
    }
    fclose(f);
}

void write_graph_bin(const Graph * const g, const char * const filename)
{
    int s1 = sizeof(Graph);
    int s2 = sizeof(int);
    FILE * f = fopen(filename, "wb");
    if (f != NULL) {
        /* size of Graph & int */
        fwrite(&s1, sizeof(int), 1, f);
        fwrite(&s2, sizeof(int), 1, f);
        /* number of Vertex & Edge */
        fwrite(&g->vertex_num, sizeof(int), 1, f);
        fwrite(&g->edge_num, sizeof(int), 1, f);
        /* Vertex & Edge data */
        fwrite(g->vertex_begin, sizeof(int), g->vertex_num + 1, f);
        fwrite(g->edge_src, sizeof(int), g->edge_num, f);
        fwrite(g->edge_dest, sizeof(int), g->edge_num, f);
    }
    fclose(f);
}

Graph * read_graph_ascii(const char * const filename)
{
    /* declarations */
    int counter = 0;
    int last = 0;
    Graph * graph = NULL;
    int * vertex_begin = NULL;
    int * edge_dest = NULL;
    int * edge_src = NULL;
    int vertex_num = 0;
    int edge_num = 0;
    int c = 0;
    int i = 0;
    FILE * f = NULL;

    /* try to open the file */
    f = fopen(filename, "r");
    if (f == NULL) {
        fprintf(stderr, "File open failed: %s ", filename);
        perror("");
        /* exit program when fail */
        exit(1);
    }

    /* locate to the first character 'p' */
    if ((c = fgetc(f)) != 'p')
        while(c != EOF && c != 'p') c = fgetc(f);
    /* locate to the numbers after character 'p' */
    while(c != EOF && !isdigit(c)) c = fgetc(f);
    if (c == EOF) {
        fprintf(stderr, "Unknown file format: %s\n", filename);
        fclose(f);
        exit(1);
    } else {
        /* put back the digit, and read vertex_num and edge_num */
        ungetc(c, f);
        fscanf(f, "%d %d", &vertex_num, &edge_num);

        /* allocate memory */
        if (vertex_num > 0 && edge_num > 0) {
            //printf("v = %d, e = %d, file: %s\n", vertex_num, edge_num, filename);
            graph = (Graph *) malloc(sizeof(Graph));
            vertex_begin = (int *) calloc(vertex_num, sizeof(int));
            edge_dest = (int *) malloc(edge_num * sizeof(int));
            edge_src = (int *) malloc(edge_num * sizeof(int));
            if (graph == NULL || vertex_begin == NULL || edge_dest == NULL || edge_src == NULL) {
                perror("Out of Memory for graph");
                /* exit program when fail */
                fclose(f);
                exit(1);
            }
            graph->vertex_num = vertex_num;
            graph->edge_num = edge_num;
            graph->vertex_begin = vertex_begin;
            graph->vertex_end = vertex_begin + 1;
            graph->edge_dest = edge_dest;
            graph->edge_src = edge_src;
        } else {
            /* wrong numbers */
            fprintf(stderr, "WRONG numbers: vertex_num = %d, edge_num = %d, file: %s\n", vertex_num, edge_num, filename);
            fclose(f);
            exit(1);
        }
    }

    /* read all the edges, each begins with character 'a' followed with three numbers */
    last = 0; // assumed that vertex id start from 1
    vertex_begin[0] = 0;
    for (counter = 0; counter < edge_num; counter++) {
        c = fgetc(f);
        /* Note the file should NOT have other contents such as comments ('a' without numbers followed) */
        while (c != EOF && c != 'a') c = fgetc(f);
        if (c == EOF) {
            fprintf(stderr, "File contains only %d / %d edge information: %s\n", counter, edge_num, filename);
            graph->edge_num = counter;
            break;
        } else {
            /* vertex ids are decreased by 1 */
            fscanf(f, "%d", &c);
            c--;
            edge_src[counter] = c;
            if (c != last) {
                /* e is an out edge from Vertex other than last */
                for (i = last + 1; i <= c; i++) {
                    vertex_begin[i] = counter;
                }
                last = c;
            }
            fscanf(f, "%d", &c);
            c--;
            edge_dest[counter] = c;
        }
    }
    fclose(f);
    for (i = last + 1; i <= vertex_num; i++) {
        vertex_begin[i] = edge_num;
    }
    return graph;

}

Graph * read_graph_bin(const char * const filename)
{
    /* declarations */
    int s1 = 0;
    int s2 = 0;
    Graph * g = NULL;
    int * vertex_begin = NULL;
    int * edge_src = NULL;
    int * edge_dest = NULL;
    int vertex_num = 0;
    int edge_num = 0;

    FILE * f = fopen(filename, "rb");
    if (f == NULL) return NULL;

    /* size of vertex & edge structure */
    fread(&s1, sizeof(int), 1, f);
    fread(&s2, sizeof(int), 1, f);
    if (s1 != sizeof(Graph) || s2 != sizeof(int)) {
        fprintf(stderr, "Unknown File Format\n");
        fclose(f);
        return NULL;
    }
    /* number of Vertex & Edge in the graph */
    fread(&vertex_num, sizeof(int), 1, f);
    fread(&edge_num, sizeof(int), 1, f);
    if(edge_num > 0 && vertex_num > 0) {
        /* allocate memory */
        g = (Graph *) calloc(1, sizeof(Graph));
        vertex_begin = (int *) malloc(sizeof(int) * (vertex_num + 1));
        edge_src = (int *) malloc(sizeof(int) * edge_num);
        edge_dest = (int *) malloc(sizeof(int) * edge_num);
        if (g == NULL || vertex_begin == NULL || edge_src == NULL || edge_dest == NULL) {
            perror("Out of memory reading graph file");
        } else if (
            fread(vertex_begin, sizeof(int), vertex_num + 1, f) != vertex_num + 1 ||
            fread(edge_src, sizeof(int), edge_num, f) != edge_num ||
            fread(edge_dest, sizeof(int), edge_num, f) != edge_num) {
            perror("Damaged file content of graph file data");
        } else {
            // read the binary file successfully
            g->vertex_num = vertex_num;
            g->edge_num = edge_num;
            g->vertex_begin = vertex_begin;
            g->vertex_end = vertex_begin + 1;
            g->edge_src = edge_src;
            g->edge_dest = edge_dest;
            fclose(f);
            return g;
        }
        // failed reading the file, free memory
        release_graph(g);
    } else {
        printf("Bad Graph Size");
    }
    fclose(f);
    return NULL;
}

Graph * allocate_graph(const int vertex_num, const int edge_num) {
    Graph * g = (Graph *) calloc(1, sizeof(Graph));
    int * vertex_begin = (int *) malloc(sizeof(int) * (vertex_num + 1));
    int * edge_src = (int *) malloc(sizeof(int) * edge_num);
    int * edge_dest = (int *) malloc(sizeof(int) * edge_num);
    if (g == NULL || vertex_begin == NULL || edge_src == NULL || edge_dest == NULL) {
        perror("Out of memory for graph");
        exit(1);
    } else {
        g->vertex_num = vertex_num;
        g->edge_num = edge_num;
        g->vertex_begin = vertex_begin;
        g->vertex_end = vertex_begin + 1;
        g->edge_src = edge_src;
        g->edge_dest = edge_dest;
    }
    return g;
}

void release_graph(Graph * g)
{
    free(g->vertex_begin);
    free(g->edge_src);
    free(g->edge_dest);
    g->vertex_num = 0;
    g->vertex_begin = NULL;
    g->vertex_end = NULL;
    g->edge_num = 0;
    g->edge_src = NULL;
    g->edge_dest = NULL;
    free(g);
    //g = NULL;
}

struct graph_csr_t * read_csr_ascii(const char * const filename)
{
	/* declarations */
	int counter = 0;
    int last = 0;
	struct graph_csr_t * graph = NULL;
    int * vertex_begin = NULL;
    int * edge_dest = NULL;
    int vertex_num = 0;
	int edge_num = 0;
	int c = 0;
	int i = 0;
	FILE * f = NULL;

    /* try to open the file */
    f = fopen(filename, "r");
    if (f == NULL)
    {
        fprintf(stderr, "File open failed: %s ", filename);
        perror("");
        /* exit program when fail */
        exit(1);
    }

	/* locate to the first character 'p' */
	if ((c = fgetc(f)) != 'p')
		while(c != EOF && c != 'p') c = fgetc(f);
	/* locate to the numbers after character 'p' */
    while(c != EOF && !isdigit(c)) c = fgetc(f);
    if (c == EOF)
    {
        fprintf(stderr, "Unknown file format: %s\n", filename);
        fclose(f);
        exit(1);
    }
    else
    {
        /* put back the digit, and read vertex_num and edge_num */
        ungetc(c, f);
        fscanf(f, "%d %d", &vertex_num, &edge_num);

        /* allocate memory */
        if (vertex_num > 0 && edge_num > 0)
        {
            printf("v = %d, e = %d, file: %s\n", vertex_num, edge_num, filename);
			graph = (struct graph_csr_t *) malloc(sizeof(struct graph_csr_t));
            vertex_begin = (int *) calloc(vertex_num, sizeof(int));
            edge_dest = (int *) malloc(edge_num * sizeof(int));
            if (graph == NULL || vertex_begin == NULL || edge_dest == NULL)
            {
                perror("Out of Memory for graph");
                /* exit program when fail */
				fclose(f);
                exit(1);
            }
			graph->vertex_num = vertex_num;
			graph->edge_num = edge_num;
			graph->vertex_begin = vertex_begin;
			graph->edge_dest = edge_dest;
        }
        else
        {
            /* wrong numbers */
            fprintf(stderr, "WRONG numbers: vertex_num = %d, edge_num = %d, file: %s\n", vertex_num, edge_num, filename);
            fclose(f);
            exit(1);
        }
    }

    /* read all the edges, each begins with character 'a' followed with three numbers */
    last = 0;
	vertex_begin[0] = 0;
    for (counter = 0; counter < edge_num; counter++)
    {
        c = fgetc(f);
		/* Note the file should NOT have other contents such as comments ('a' without numbers followed) */
        while (c != EOF && c != 'a') c = fgetc(f);
        if (c == EOF)
        {
            fprintf(stderr, "File contains only %d / %d edge information: %s\n", counter, edge_num, filename);
            graph->edge_num = counter;
            break;
        }
        else
        {
            /* vertex ids are decreased by 1 */
            fscanf(f, "%d", &c);
            c--;
            if (c != last) {
                /* e is an out edge from Vertex other than last */
                for (i = last + 1; i <= c; i++) {
                    vertex_begin[i] = counter;
                }
                last = c;
            }
            fscanf(f, "%d", &c);
            c--;
            edge_dest[counter] = c;
        }
    }
    fclose(f);
    for (i = last + 1; i <= vertex_num; i++)
    {
        vertex_begin[i] = edge_num;
    }
    return graph;
}

void write_csr_bin(const struct graph_csr_t * const g, const char * const filename)
{
	int s1 = sizeof(struct graph_csr_t);
	int s2 = sizeof(int);
	FILE * f = fopen(filename, "wb");
    if (f != NULL)
    {
        /* size of struct graph_csr_t & int */
        fwrite(&s1, sizeof(int), 1, f);
        fwrite(&s2, sizeof(int), 1, f);
        /* number of Vertex & Edge */
        fwrite(&g->vertex_num, sizeof(int), 1, f);
        fwrite(&g->edge_num, sizeof(int), 1, f);
        /* Vertex & Edge data */
        fwrite(g->vertex_begin, sizeof(int), g->vertex_num + 1, f);
        fwrite(g->edge_dest, sizeof(int), g->edge_num, f);
    }
	fclose(f);
}

struct graph_csr_t * read_csr_bin(const char * const filename)
{
	/* declarations */
    int s1 = 0;
    int s2 = 0;
	struct graph_csr_t * g = NULL;
	int * vertex_begin = NULL;
	int * edge_dest = NULL;
    int vertex_num = 0;
    int edge_num = 0;

    FILE * f = fopen(filename, "rb");
    if (f == NULL) return NULL;

    /* size of struct graph_csr_t & int */
    fread(&s1, sizeof(int), 1, f);
    fread(&s2, sizeof(int), 1, f);
    if (s1 != sizeof(struct graph_csr_t) || s2 != sizeof(int))
    {
        fprintf(stderr, "Unknown File Format\n");
		fclose(f);
        return NULL;
    }
    /* number of Vertex & Edge in the graph */
    fread(&vertex_num, sizeof(int), 1, f);
    fread(&edge_num, sizeof(int), 1, f);
    if(edge_num > 0 && vertex_num > 0)
    {
		/* allocate memory */
		g = (struct graph_csr_t *) calloc(1, sizeof(struct graph_csr_t));
		vertex_begin = (int *) malloc(sizeof(int) * (vertex_num + 1));
		edge_dest = (int *)malloc(sizeof(int) * edge_num);
		if (g == NULL || vertex_begin == NULL || edge_dest == NULL)
		{
			perror("Out of memory reading graph file");
		} else if (
             fread(vertex_begin, sizeof(int), vertex_num + 1, f) != vertex_num + 1 ||
             fread(edge_dest, sizeof(int), edge_num, f) != edge_num)
        {
            perror("Damaged file content of graph file data");
			release_csr_t(g);
        } else {
			g->vertex_num = vertex_num;
			g->edge_num = edge_num;
			g->vertex_begin = vertex_begin;
			g->edge_dest = edge_dest;
			fclose(f);
			return g;
		}
    } else {
		printf("Bad Graph Size");
	}
	fclose(f);
	return NULL;
}

struct graph_csr_t * allocate_csr_t(const int vertex_num, const int edge_num)
{
    struct graph_csr_t * g = (struct graph_csr_t *) calloc(1, sizeof(struct graph_csr_t));
    int * vertex_begin = (int *) malloc(sizeof(int) * (vertex_num + 1));
    int * edge_dest = (int *)malloc(sizeof(int) * edge_num);
    if (g == NULL || vertex_begin == NULL || edge_dest == NULL)
    {
        perror("Out of memory for graph");
        exit(1);
    } else {
        g->vertex_num = vertex_num;
        g->edge_num = edge_num;
        g->vertex_begin = vertex_begin;
        g->edge_dest = edge_dest;
    }
    return g;
}

void release_csr_t(struct graph_csr_t * g)
{
    free(g->vertex_begin);
    free(g->edge_dest);
    g->vertex_num = 0;
    g->vertex_begin = NULL;
    g->edge_num = 0;
    g->edge_dest = NULL;
    free(g);
    //g = NULL;
}

Graph * get_graph(struct graph_csr_t * g) {
    int i, k;
    Graph * ret = calloc(1, sizeof(Graph));
    ret->vertex_num = g->vertex_num;
    ret->edge_num = g->edge_num;
    ret->vertex_begin = g->vertex_begin;
    ret->vertex_end = g->vertex_begin + 1;
    ret->edge_dest = g->edge_dest;
    ret->edge_src = malloc(g->edge_num * sizeof(int));
    // assign source vertex IDs
    for (i = 0; i < ret->vertex_num; i++)
        for (k = ret->vertex_begin[i]; k < ret->vertex_end[i]; k++)
            ret->edge_dest[k] = i;
    return ret;
}

Graph * get_reverse_graph(const Graph * const g) {
    int i, k, tem, dest, index;
    int vertex_num = g->vertex_num;
    int edge_num = g->edge_num;
    int * vertex_begin = g->vertex_begin;
    int * edge_dest = g->edge_dest;
    // allocate memory
    Graph * ret = allocate_graph(vertex_num, edge_num);
    int * rev_begin = ret->vertex_begin;
    int * rev_dest = ret->edge_dest;
    int * rev_src = ret->edge_src;
    int * ind = (int *) calloc(vertex_num, sizeof(int));
    if (ind == NULL) {
        perror("Out of Memory to reverse Graph");
        exit(1);
    }
    // count degrees
    for (i = 0; i < vertex_num; i++)
        for (k = vertex_begin[i]; k < vertex_begin[i + 1]; k++)
            ind[edge_dest[k]]++;
    // calculate edge offset numbers
    tem = 0;
    for (i = 0; i < vertex_num; i++) {
        rev_begin[i] = tem;
        tem += ind[i];
        ind[i] = rev_begin[i];
    }
    rev_begin[vertex_num] = tem;
    // reorder edges
    for (i = 0; i < vertex_num; i++)
        for (k = vertex_begin[i]; k < vertex_begin[i + 1]; k++) {
            dest = edge_dest[k];
            index = ind[dest];
            // rev_dest = source, rev_src = destination
            rev_dest[index] = i;
            rev_src[index] = dest;
            ind[dest] = index + 1;
        }
    // finish
    free(ind);
    return ret;
}
