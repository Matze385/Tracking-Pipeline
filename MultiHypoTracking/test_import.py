import sys

sys.path.append('.')

from grid_search import*

if __name__ == '__main__':
    grid_search_helper = grid_search_helper()
    print grid_search_helper.get_idx_range()
    print grid_search_helper.get_weights_out_of_idx(84)

