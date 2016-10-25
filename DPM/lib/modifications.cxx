#include <vigra/multi_array.hxx>  //C++ header
#include <vigra/hdf5impex.hxx>

extern "C" //C header
{
#include "ccv.h"
#include <stdio.h>
#include "modifications.h"
}



/*test function
proofs if a is equal to b for each value
return: return 1 if matrices are equal
*/
extern "C" int ccv_is_equal_modified(ccv_dense_matrix_t *a, ccv_dense_matrix_t *b, int ch)
{
    if( !(a->step==b->step) )
        return 0;
    if( !(a->rows==a->cols) )
        return 0;
    int idx = 0;
    for(int row=0; row<a->rows; row++)
    {
        for(int col=0; col<a->cols; col++)
        {
            for(int c=0; c<ch; c++)
            {
                if ( !(a->data.f32[idx] == b->data.f32[idx]) )
                    return 0;
                idx++; 
            }
        }
    }
    return 1;
}

/*
input: hdf5 dataset file specified with
file_name: name of file that should be read in
dataset_name: name of dataset in hdf5 file
ch: number of channels/classes in probmaps for checking/assertions
output:
probmaps: probmaps is ccv_dense_matrix_t format with cols, rows and ch channels
*/

extern "C" int ccv_read_modified(const char *file_name, const char *dataset_name, ccv_dense_matrix_t **probmaps, int ch)
{
    vigra::HDF5ImportInfo info(file_name, dataset_name);
    vigra_precondition(info.numDimensions() == 3, "Dataset must be 3-dimensional.");
    //create MultiArray
    vigra::MultiArrayShape<3>::type shape(info.shape().begin());
    vigra::MultiArray<3, float> array(shape);
    vigra::readHDF5(info, array);
    //printf("ch: %d shape[0]: %d", ch, (int)shape[0]);
    //check if channel number is correct
    assert(ch == shape[0]);
    int cols = shape[1], rows = shape[2]; //or other way round
    // CCV_32F|4 32 bit float with ch channels: head, upper body, lower body, wings; 0,0: allocate new memory 
    *probmaps = ccv_dense_matrix_new(rows, cols, CCV_32F|ch, 0, 0); 
    float *dptr = (*probmaps)->data.f32;
    //flatten MultiArray in vector for ccv_dense_matrix_t
    for(int row=0; row<rows; row++)
    {
        for(int col=0; col<cols; col++)
        {
            for(int c=0; c<ch; c++)
            {
                dptr[c] = static_cast<float>(array(c,col,row)); 
            }
            dptr += ch;
        }
    }
    //printf("attributes of ccv_dense_matrix_t: \n step: %d, rows = %d, cols = %d \n" ,(*probmaps)->step, (*probmaps)->rows, (*probmaps)->cols);
    return 1;     
}

extern "C" int ccv_write_modified(ccv_dense_matrix_t *a, const char *file_name, const char *dataset_name, int ch)
{
    //create MultiArray
    vigra::MultiArrayShape<3>::type shape(ch, a->cols, a->rows);
    vigra::MultiArray<3, float> array(shape);
    float *dptr = a->data.f32;
    for(int row=0; row<a->rows; row++)
    {
        for(int col=0; col<a->cols; col++)
        {
            for(int c=0; c<ch; c++)
            {
                array(c,col,row) = dptr[c]; 
            }
            dptr += ch;
        }
    }
    vigra::writeHDF5(file_name, dataset_name, array);
    return 1;
}


