#ifndef OWN_MODIFICATONS_H
#define OWN_MODIFICATONS_H


//extern "C" 
//{
//#include "ccv.h"
//}

#ifdef __cplusplus
extern "C" {
#endif

// all of your legacy C code here

int ccv_is_equal_modified(ccv_dense_matrix_t *a, ccv_dense_matrix_t *b, int ch);

int ccv_read_modified(const char *file_name, const char *dataset_name, ccv_dense_matrix_t **probmaps, int ch);

int ccv_write_modified(ccv_dense_matrix_t *a, const char *file_name, const char *dataset_name, int ch);

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //OWN_MODIFICATIONS
