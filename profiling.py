import cProfile
import pstats
from class_allocation import allocation

def profile_allocation():
    region = input("Choose a focus country or region: ")
    allocator = allocation(region)
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
p.sort_stats('cumulative').print_stats(20) # Print top 20 slowest functions