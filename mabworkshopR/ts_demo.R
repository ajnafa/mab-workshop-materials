# Load required libraries
library(ggplot2)
library(tidyr)

#' Simulate Thompson Sampling for a Bernoulli Bandit
#'
#' @param true_probs A numeric vector of true success probabilities for each arm.
#' @param n_trials The number of trials to simulate.
#' @return A list containing the simulation results: choices, rewards,
#'         cumulative_regret, and final posterior parameters (alpha, beta).
thompson_sampling_bernoulli <- function(true_probs, n_trials) {
  
  n_arms <- length(true_probs)
  
  # Initialize posterior parameters (Beta distribution)
  # Start with a uniform prior Beta(1, 1)
  alpha <- rep(1, n_arms)
  beta <- rep(1, n_arms)
  
  # Store results
  choices <- integer(n_trials)
  rewards <- integer(n_trials)
  
  # Regret tracking
  best_prob <- max(true_probs)
  realized_regret <- numeric(n_trials)
  
  for (t in 1:n_trials) {
    # Sample from the posterior distribution of each arm
    sampled_probs <- rbeta(n_arms, alpha, beta)
    
    # Choose the arm with the highest sampled probability
    chosen_arm <- which.max(sampled_probs)
    choices[t] <- chosen_arm
    
    # Simulate pulling the arm and observe the reward
    reward <- rbinom(1, 1, true_probs[chosen_arm])
    rewards[t] <- reward
    
    # Update the posterior distribution for the chosen arm
    if (reward == 1) {
      alpha[chosen_arm] <- alpha[chosen_arm] + 1
    } else {
      beta[chosen_arm] <- beta[chosen_arm] + 1
    }
    
    # Calculate realized regret for this trial
    realized_regret[t] <- best_prob - true_probs[chosen_arm]
  }
  
  cumulative_regret <- cumsum(realized_regret)
  
  # Calculate regret for a static A/B test (random assignment)
  static_choices <- sample(1:n_arms, n_trials, replace = TRUE)
  static_realized_regret <- best_prob - true_probs[static_choices]
  static_cumulative_regret <- cumsum(static_realized_regret)
  
  return(list(
    choices = choices,
    rewards = rewards,
    cumulative_regret = cumulative_regret,
    static_cumulative_regret = static_cumulative_regret,
    alpha = alpha,
    beta = beta,
    true_probs = true_probs
  ))
}

#' Plot Cumulative Regret
#'
#' @param sim_results The results from the thompson_sampling_bernoulli simulation.
plot_cumulative_regret <- function(sim_results) {
  
  regret_data <- data.frame(
    Trial = seq_along(sim_results$cumulative_regret),
    ThompsonRegret = sim_results$cumulative_regret,
    StaticRegret = sim_results$static_cumulative_regret
  ) %>%
  pivot_longer(cols = c("ThompsonRegret", "StaticRegret"), 
               names_to = "Method", 
               values_to = "CumulativeRegret")
  
  ggplot(regret_data, aes(x = .data$Trial, y = .data$CumulativeRegret, color = .data$Method)) +
    geom_line() +
    labs(
      title = "Cumulative Regret: Thompson Sampling vs. Static A/B Test",
      x = "Trial Number",
      y = "Cumulative Regret",
      color = "Method"
    ) +
    theme_minimal() +
    scale_color_manual(values = c("ThompsonRegret" = "firebrick", "StaticRegret" = "steelblue"))
}

#' Plot Arm Selection Proportions
#'
#' @param sim_results The results from the thompson_sampling_bernoulli simulation.
plot_selection_proportions <- function(sim_results) {
  
  n_arms <- length(sim_results$true_probs)
  
  selection_data <- as.data.frame(table(sim_results$choices))
  names(selection_data) <- c("Arm", "Count")
  selection_data$Proportion <- selection_data$Count / length(sim_results$choices)
  selection_data$Arm <- factor(selection_data$Arm, levels = 1:n_arms)
  
  ggplot(selection_data, aes(x = .data$Arm, y = .data$Proportion, fill = .data$Arm)) +
    geom_bar(stat = "identity") +
    labs(
      title = "Arm Selection Proportions",
      x = "Arm",
      y = "Proportion of Times Selected"
    ) +
    theme_minimal()
}

#' Plot Posterior Distributions
#'
#' @param sim_results The results from the thompson_sampling_bernoulli simulation.
plot_posterior_distributions <- function(sim_results) {
  
  n_arms <- length(sim_results$true_probs)
  
  # Create a data frame for plotting posteriors
  plot_data <- data.frame()
  x_vals <- seq(0, 1, length.out = 1000)
  
  for (i in 1:n_arms) {
    posterior_density <- dbeta(x_vals, sim_results$alpha[i], sim_results$beta[i])
    arm_data <- data.frame(
      Arm = as.factor(i),
      Probability = x_vals,
      Density = posterior_density
    )
    plot_data <- rbind(plot_data, arm_data)
  }
  
  # Create a data frame for true probabilities
  true_probs_data <- data.frame(
      Arm = as.factor(1:n_arms),
      TrueProb = sim_results$true_probs
  )
  
  ggplot(plot_data, aes(x = .data$Probability, y = .data$Density, color = .data$Arm)) +
    geom_line(linewidth = 1) +
    geom_vline(data = true_probs_data, aes(xintercept = .data$TrueProb, color = .data$Arm), linetype = "dashed", linewidth = 1) +
    labs(
      title = "Final Posterior Distributions of Arm Probabilities",
      x = "Success Probability (theta)",
      y = "Density"
    ) +
    theme_minimal()
}

# Define the true probabilities of success for each arm
set.seed(666)
true_probabilities <- c(0.25, 0.18, 0.13, 0.22, 0.02)
number_of_trials <- 2000

# Run the simulation
simulation_results <- thompson_sampling_bernoulli(
  true_probs = true_probabilities,
  n_trials = number_of_trials
)

# Generate and print the plots
p1 <- plot_cumulative_regret(simulation_results)
ggsave("mabworkshopR/.output/cumulative_regret.png", plot = p1, bg = "white")

p2 <- plot_selection_proportions(simulation_results)
ggsave("mabworkshopR/.output/selection_proportions.png", plot = p2, bg = "white")

p3 <- plot_posterior_distributions(simulation_results)
ggsave("mabworkshopR/.output/posterior_distributions.png", plot = p3, bg = "white")

