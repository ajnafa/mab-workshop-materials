from numpy import (
    argmax, 
    zeros, 
    ones, 
    cumsum, 
    linspace, 
    unique, 
    array,
    arange
)
from scipy.stats import beta as beta_dist

from pandas import (
    DataFrame, 
    melt,
    Categorical,
    concat
)

from plotnine import (
    ggplot, 
    aes, 
    geom_line, 
    labs, 
    theme_minimal, 
    geom_bar, 
    geom_vline, 
    ggsave, 
    scale_color_manual
)

def thompson_sampling_bernoulli(true_probs, n_trials, rng):
    """
    Simulate Thompson Sampling for a Bernoulli Bandit.

    :param true_probs: A list or numpy array of true success probabilities for each arm.
    :param n_trials: The number of trials to simulate.
    :param rng: A numpy random number generator.
    :return: A dictionary containing the simulation results.
    """
    n_arms = len(true_probs)
    # Initialize posterior parameters (Beta distribution)
    alpha = ones(n_arms)
    beta = ones(n_arms)
    
    # Store results
    choices = zeros(n_trials, dtype=int)
    rewards = zeros(n_trials, dtype=int)
    
    # Regret tracking
    best_prob = max(true_probs)
    realized_regret = zeros(n_trials)
    
    for t in range(n_trials):
        # Sample from the posterior distribution of each arm
        sampled_probs = rng.beta(alpha, beta)
        
        # Choose the arm with the highest sampled probability
        chosen_arm = argmax(sampled_probs)
        choices[t] = chosen_arm
        
        # Simulate pulling the arm and observe the reward
        reward = rng.binomial(1, true_probs[chosen_arm])
        rewards[t] = reward
        
        # Update the posterior distribution for the chosen arm
        if reward == 1:
            alpha[chosen_arm] += 1
        else:
            beta[chosen_arm] += 1
            
        # Calculate realized regret for this trial
        realized_regret[t] = best_prob - true_probs[chosen_arm]
        
    cumulative_regret = cumsum(realized_regret)
    
    # Calculate regret for a static A/B test (random assignment)
    static_choices = rng.integers(0, n_arms, n_trials)
    static_realized_regret = best_prob - array(true_probs)[static_choices]
    static_cumulative_regret = cumsum(static_realized_regret)
    
    return {
        "choices": choices,
        "rewards": rewards,
        "cumulative_regret": cumulative_regret,
        "static_cumulative_regret": static_cumulative_regret,
        "alpha": alpha,
        "beta": beta,
        "true_probs": true_probs
    }

def plot_cumulative_regret(sim_results):
    """Plot Cumulative Regret."""
    regret_data = DataFrame({
        'Trial': arange(len(sim_results['cumulative_regret'])),
        'TSRegret': sim_results['cumulative_regret'],
        'StaticRegret': sim_results['static_cumulative_regret']
    }).melt(id_vars='Trial', var_name='Method', value_name='CumulativeRegret')
    
    p = (
        ggplot(regret_data, aes(x='Trial', y='CumulativeRegret', color='Method')) +
        geom_line() +
        labs(
            title="Cumulative Regret: Thompson Sampling vs. Static A/B Test",
            x="Trial Number",
            y="Cumulative Regret"
        ) +
        theme_minimal() +
        scale_color_manual(values={"TSRegret": "firebrick", "StaticRegret": "steelblue"})
    )
    return p

def plot_selection_proportions(sim_results):
    """Plot Arm Selection Proportions."""
    n_arms = len(sim_results['true_probs'])
    
    choices, counts = unique(sim_results['choices'], return_counts=True)
    selection_data = DataFrame({
        'Arm': [str(c) for c in choices],
        'Count': counts
    })
    selection_data['Proportion'] = selection_data['Count'] / len(sim_results['choices'])
    selection_data['Arm'] = Categorical(selection_data['Arm'], categories=[str(i) for i in range(n_arms)])

    p = (
        ggplot(selection_data, aes(x='Arm', y='Proportion', fill='Arm')) +
        geom_bar(stat="identity") +
        labs(
            title="Arm Selection Proportions",
            x="Arm",
            y="Proportion of Times Selected"
        ) +
        theme_minimal()
    )
    return p

def plot_posterior_distributions(sim_results):
    """Plot Posterior Distributions."""
    n_arms = len(sim_results['true_probs'])
    x_vals = linspace(0, 1, 1000)
    
    plot_data = DataFrame()
    for i in range(n_arms):
        posterior_density = beta_dist.pdf(x_vals, sim_results['alpha'][i], sim_results['beta'][i])
        arm_data = DataFrame({
            'Arm': str(i),
            'Probability': x_vals,
            'Density': posterior_density
        })
        plot_data = concat([plot_data, arm_data])
        
    true_probs_data = DataFrame({
        'Arm': [str(i) for i in range(n_arms)],
        'TrueProb': sim_results['true_probs']
    })
    
    p = (
        ggplot(plot_data, aes(x='Probability', y='Density', color='Arm')) +
        geom_line(size=1) +
        geom_vline(data=true_probs_data, mapping=aes(xintercept='TrueProb', color='Arm'), linetype="dashed", size=1) +
        labs(
            title="Final Posterior Distributions of Arm Probabilities",
            x="Success Probability (theta)",
            y="Density"
        ) +
        theme_minimal()
    )
    return p
