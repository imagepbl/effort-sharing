import cProfile
import pstats

import effortsharing as es
from effortsharing.allocation import determine_allocations, save_allocations


def profile_allocation():
    """
    This function profiles the allocation class to identify the slowest functions
    """
    # Load configuration
    config = es.Config.from_file("config.yml")
    region = input("Choose a focus country or region: ")
    allocations = determine_allocations(config, region)
    save_allocations(config, region, allocations)


cProfile.run("profile_allocation()", "profile_output")

# Print the profiling results
p = pstats.Stats("profile_output")
p.sort_stats("cumulative").print_stats(40)  # Print top 20 slowest functions
