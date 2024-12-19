import cProfile
import pstats
import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from EffortSharingTools.class_allocation import allocation

def profile_allocation():
    '''
    This function profiles the allocation class to identify the slowest functions
    '''
    region = input("Choose a focus country or region: ")
    allocator = allocation(region)
    allocator.gf()
    allocator.pc()
    allocator.pcc()
    allocator.pcb()
    allocator.ecpc()
    allocator.ap()
    allocator.gdr()
    allocator.save()

cProfile.run('profile_allocation()', 'profile_output')

# Print the profiling results
p = pstats.Stats('profile_output')
p.sort_stats('cumulative').print_stats(40) # Print top 20 slowest functions
