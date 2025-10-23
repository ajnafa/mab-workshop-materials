from mabworkshoppy.thompson import (
    thompson_sampling_bernoulli, 
    plot_cumulative_regret, 
    plot_selection_proportions, 
    plot_posterior_distributions,
    ggsave
)
from numpy.random import default_rng

seed = 666
rng = default_rng(seed)
true_probabilities = [0.25, 0.18, 0.13, 0.22, 0.02]
number_of_trials = 2000

# Run the simulation
simulation_results = thompson_sampling_bernoulli(
    true_probs = true_probabilities,
    n_trials = number_of_trials,
    rng = rng
)

# Generate and print the plots
p1 = plot_cumulative_regret(simulation_results)
ggsave(
    p1, 
    filename=".output/cumulative_regret.png",
    height=7,
    width=9
)

p2 = plot_selection_proportions(simulation_results)
ggsave(
    p2, 
    filename=".output/selection_proportions.png",
    height=7,
    width=9
)

p3 = plot_posterior_distributions(simulation_results)
ggsave(
    p3, 
    filename=".output/posterior_distributions.png",
    height=7,
    width=9
)
