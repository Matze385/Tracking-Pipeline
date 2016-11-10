#include "ccv.h"
#include "modifications.h"
#include <sys/time.h>
#include <ctype.h>

#define CCV_NUMBER_CHANNELS (3)

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

//test if string end on some suffix such as .txt, if true, from stackoverflo, from stackoverflow
static int string_ends_with(const char * str, const char * suffix)
{
  int str_len = strlen(str);
  int suffix_len = strlen(suffix);

  return 
    (str_len >= suffix_len) &&
    (0 == strcmp(str + (str_len-suffix_len), suffix));
}

int main(int argc, char** argv)
{
	assert(argc >= 3);
	int i, j, n_pred = 1;
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	//ccv_read(argv[1], &image, CCV_IO_ANY_FILE);
	ccv_dpm_mixture_model_t* model = ccv_dpm_read_mixture_model(argv[2]);
        int is_txtfile = string_ends_with(argv[1], ".txt");
        printf("value of is_txtfile should be 1 and is: %d", is_txtfile);
	if (is_txtfile != 1)
	{
	        ccv_read_modified(argv[1], "data", &image, CCV_NUMBER_CHANNELS);
	        //test_detect(image, &model, 1, ccv_dpm_default_params);
		unsigned int elapsed_time = get_current_time();
		ccv_array_t* seq = ccv_dpm_detect_objects(image, &model, 1, ccv_dpm_default_params);
		elapsed_time = get_current_time() - elapsed_time;
		if (seq)
		{
			for (i = 0; i < seq->rnum; i++)
			{
				ccv_root_comp_t* comp = (ccv_root_comp_t*)ccv_array_get(seq, i);
				printf("%d %d %d %d %f %d\n", comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->classification.confidence, comp->pnum);
				for (j = 0; j < comp->pnum; j++)
					printf("| %d %d %d %d %f\n", comp->part[j].rect.x, comp->part[j].rect.y, comp->part[j].rect.width, comp->part[j].rect.height, comp->part[j].classification.confidence);
			}
			printf("total : %d in time %dms\n", seq->rnum, elapsed_time);
			ccv_array_free(seq);
		} else {
			printf("elapsed time %dms\n", elapsed_time);
		}
                
		ccv_matrix_free(image);
	} else {
		FILE* r = fopen(argv[1], "rt");
		if (argc == 4)
			chdir(argv[3]);
		if(r)
		{
			size_t len = 1024;
			char* file = (char*)malloc(len);
			ssize_t read;
			while((read = getline(&file, &len, r)) != -1)
			{
				while(read > 1 && isspace(file[read - 1]))
					read--;
				file[read] = 0;
				image = 0;
                                
				ccv_read_modified(file, "data", &image, CCV_NUMBER_CHANNELS);
				//ccv_read_modified(file, &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
				test_detect(image, &model, 1, ccv_dpm_default_params, file);
				if (n_pred%8==1)
                                {
                                    printf("progress %d", n_pred);
                                    n_pred++;
                                }
                                assert(image != 0);
                                
                                /*
				ccv_array_t* seq = ccv_dpm_detect_objects(image, &model, 1, ccv_dpm_default_params);
				if (seq != 0)
				{
					for (i = 0; i < seq->rnum; i++)
					{
						ccv_root_comp_t* comp = (ccv_root_comp_t*)ccv_array_get(seq, i);
						printf("%s %d %d %d %d %f %d\n", file, comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->classification.confidence, comp->pnum);
						for (j = 0; j < comp->pnum; j++)
							printf("| %d %d %d %d %f\n", comp->part[j].rect.x, comp->part[j].rect.y, comp->part[j].rect.width, comp->part[j].rect.height, comp->part[j].classification.confidence);
					}
					ccv_array_free(seq);
				}
                                */
				ccv_matrix_free(image);
			}
			free(file);
			fclose(r);
		}
	}
	ccv_drain_cache();
	ccv_dpm_mixture_model_free(model);
	return 0;
}
