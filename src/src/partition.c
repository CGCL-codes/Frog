#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#include "partition.h"

struct part_table * allocate_table (const int vertex_num, const int edge_num, const int part_num) {
    int i;
    struct part_table *ret = (struct part_table *) calloc (1, sizeof (struct part_table));
    int *vertex_part = (int *) calloc (vertex_num, sizeof (int));
    int *part_vertex_num = (int *) calloc (part_num, sizeof (int));
    int *part_edge_num = (int *) calloc (part_num, sizeof (int));
    int **part_vertex = (int **) calloc (part_num, sizeof (int *));
    int *vertex_id = (int *) calloc (vertex_num, sizeof (int));
    if (ret == NULL || vertex_part == NULL
            || part_vertex_num == NULL || part_edge_num == NULL
            || part_vertex == NULL || vertex_id == NULL) {
        // exit program when fail
        perror ("Out of Memory for Partition Table");
        exit (1);
    }
    for (i = 0; i < part_num; i++) {
        part_vertex[i] = vertex_id;
    }
    ret->part_num = part_num;
    ret->vertex_num = vertex_num;
    ret->edge_num = edge_num;
    ret->part_vertex_num = part_vertex_num;
    ret->part_edge_num = part_edge_num;
    ret->part_vertex = part_vertex;
    ret->vertex_part = vertex_part;
    return ret;
}

void release_table (struct part_table * t) {
    free (t->part_vertex[0]);
    free (t->part_vertex);
    free (t->part_vertex_num);
    free (t->part_edge_num);
    free (t->vertex_part);
    t->part_num = 0;
    t->vertex_num = 0;
    t->edge_num = 0;
    t->part_vertex_num = NULL;
    t->part_edge_num = NULL;
    t->part_vertex = NULL;
    t->vertex_part = NULL;
    free (t);
}

struct part_table * partition (
    const int vertex_num,
    const int edge_num,
    const int *const vertex_begin,
    const int *const edge_dest,
    const int part_num
) {
    /* variable declaration */
    int i, k, p, flag;
    struct part_table * ret = allocate_table(vertex_num, edge_num, part_num);
    int * part_vertex_num = ret->part_vertex_num;
    int * part_edge_num = ret->part_edge_num;
    int * vertex_part = ret->vertex_part;
    int * part_vertex = ret->part_vertex[0];
    int n = 0;
    for (p = 1; p < part_num; p++) {
        // value of vertex_part[v] means:
            // 0 = unvisited
            // p = included by partition p
            // -p = excluded by partition p
        // SO, vertex_part[v] < 1 means not partitioned yet
        for (i = 0; i < vertex_num; i++) {
            if (vertex_part[i] < 1 && vertex_part[i] > -p) {
                // Vertex i possibly can be included by partition p
                flag = 1;
                for (k = vertex_begin[i]; k < vertex_begin[i + 1]; k++) {
                    if (vertex_part[edge_dest[k]] == p) {
                        // detect a conflict, Vertex i can Not be included by partition p
                        vertex_part[i] = -p;
                        flag = 0;
                        break;
                    }
                }
                if (flag) {
                    // assign Vertex i to partition p
                    vertex_part[i] = p;
                    (*part_vertex_num)++;
                    (*part_edge_num) += vertex_begin[i + 1] - vertex_begin[i];
                    part_vertex[n++] = i;
                    for (k = vertex_begin[i]; k < vertex_begin[i + 1]; k++) {
                        if (vertex_part[edge_dest[k]] < 1) {
                            // mark the adjacent vertex as conflicted
                            vertex_part[edge_dest[k]] = -p;
                        }
                    }
                }
            }
        }
        // move pointers for partition
        part_vertex_num++;
        part_edge_num++;
        ret->part_vertex[p] = part_vertex + n;
    }
    // the last partition contains all un-partitioned vertices
    for (i = 0; i < vertex_num; i++) {
        if (vertex_part[i] < 1) {
            vertex_part[i] = part_num;
            (*part_vertex_num)++;
            (*part_edge_num) += vertex_begin[i + 1] - vertex_begin[i];
            part_vertex[n++] = i;
        }
    }
    return ret;
}


void write_table_bin(const struct part_table * const t, const char * const filename) {
    int s1 = sizeof(struct part_table);
    int s2 = sizeof(int);
    FILE * f = fopen(filename, "wb");
    if (f != NULL) {
        /* size of table & int */
        fwrite(&s1, sizeof(int), 1, f);
        fwrite(&s2, sizeof(int), 1, f);
        /* numbers */
        fwrite(&t->vertex_num, sizeof(int), 1, f);
        fwrite(&t->edge_num, sizeof(int), 1, f);
        fwrite(&t->part_num, sizeof(int), 1, f);
        /* partition data */
        fwrite(t->part_vertex_num, sizeof(int), t->part_num, f);
        fwrite(t->part_edge_num, sizeof(int), t->part_num, f);
        fwrite(t->vertex_part, sizeof(int), t->vertex_num, f);
        fwrite(t->part_vertex[0], sizeof(int), t->vertex_num, f);
    } else {
        perror("Can NOT write table file");
    }
    fclose(f);
}

struct part_table * read_table_bin(const char * const filename) {
    /* declarations */
    int s1 = 0;
    int s2 = 0;
    struct part_table * ret = NULL;
    int vertex_num = 0;
    int edge_num = 0;
    int part_num = 0;
    int i;
    FILE * f = fopen(filename, "rb");
    if (f == NULL) return NULL;
    /* read size numbers */
    fread(&s1, sizeof(int), 1, f);
    fread(&s2, sizeof(int), 1, f);
    if (s1 != sizeof(struct part_table) || s2 != sizeof(int)) {
        fprintf(stderr, "Unknown File Format\n");
        fclose(f);
        return NULL;
    }
    /* read numbers */
    fread(&vertex_num, sizeof(int), 1, f);
    fread(&edge_num, sizeof(int), 1, f);
    fread(&part_num, sizeof(int), 1, f);
    if(edge_num > 0 && vertex_num > 0 && part_num > 0) {
        ret = allocate_table(vertex_num, edge_num, part_num);
        /* read data */
        if (fread(ret->part_vertex_num, sizeof(int), part_num, f) != part_num ||
            fread(ret->part_edge_num, sizeof(int), part_num, f) != part_num ||
            fread(ret->vertex_part, sizeof(int), vertex_num, f) != vertex_num ||
            fread(ret->part_vertex[0], sizeof(int), vertex_num, f) != vertex_num) {
            perror("Damaged file content of Partition Table");
        } else {
            // read the binary file successfully
            for (i = 1; i < part_num; i++) {
                // adjust the pointers
                ret->part_vertex[i] = ret->part_vertex[i - 1] + ret->part_vertex_num[i];
            }
            fclose(f);
            return ret;
        }
        // failed reading the file, free memory
        release_table(ret);
    } else {
        printf("Bad Graph or Partition Size");
    }
    fclose(f);
    return NULL;
}

Graph ** get_cut_graphs(const Graph * const g, const struct part_table * const t) {
    int i, k, p;
    int part_num = t->part_num;
    Graph ** ret = (Graph **) calloc(part_num, sizeof(Graph *));
    if (ret == NULL) {
        perror("Out of Memory for Partitioned Graph");
        exit(1);
    }
    // Copy Data
    for (p = 0; p < part_num; p++) {
        int vertex_num = t->part_vertex_num[p];
        int edge_num = t->part_edge_num[p];
        ret[p] = allocate_graph(vertex_num, edge_num);
        int * vertex_begin = ret[p]->vertex_begin;
        int * edge_src = ret[p]->edge_src;
        int * edge_dest = ret[p]->edge_dest;
        int count = 0;
        for (i = 0; i < vertex_num; i ++) {
            vertex_begin[i] = count;
            int src = t->part_vertex[p][i];
            for (k = g->vertex_begin[src]; k < g->vertex_begin[src + 1]; k++) {
                int dest = g->edge_dest[k];
                edge_src[count] = src;
                edge_dest[count] = dest;
                count++;
            }
        }
        vertex_begin[vertex_num] = edge_num;
    }
    return ret;
}
