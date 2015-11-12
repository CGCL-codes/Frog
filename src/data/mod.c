#include <stdio.h>
#include <stdlib.h>

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

int main(int argc, char ** argv) {
	if (argc > 2) {
		printf("reading file '%s' ... \n", argv[1]);
		Graph * g = read_graph_ascii(argv[1]);
		if (g != NULL) {
			printf("saving file '%s' ... \n", argv[2]);
			write_graph_bin(g, argv[2]);
		}
	} else {
		printf("need file names !\n");
	}
	return 0;
}