library(ggdist)

get_beta_hdci = function(df,
                         observations,
                         n_samples, # number of samples to draw from Beta
                         groups=c(), # applies group_by
                         width=.95,
                         seed=NA) {
  if (!is.na(seed)) {
    set.seed(seed)
  }
  
  if (length(groups) > 0) {
    df = df %>% 
      group_by(!!!syms(groups))
  }
  df.params = df %>% 
    summarize(count = n(), 
              alpha = sum(!!enquo(observations)),
              beta = count - alpha) 
  df = df.params %>% 
    mutate(sample = map2(alpha, beta, ~rbeta(n_samples, .x, .y))) %>% 
    unnest(cols = c(sample))
  if (length(groups) > 0) {
    df = df %>% 
      group_by(!!!syms(groups))
  }
  df = df %>%
    mean_hdci(sample, .width = width) %>%
    left_join(df.params) %>% 
    mutate(mean = alpha / count) %>% 
    select(!!enquo(groups),
           count, 
           mean,
           alpha,
           beta,
           estimate = sample,
           hdci_l = .lower,
           hdci_h = .upper)
  return (df)
}


get_counts_with_zeros = function(df, groups=c()) {
  # Used to get group counts where some groups may have 0 occurrences
  # groups must be a vector of strings
  df %>% 
    group_by(!!!syms(groups)) %>%
    tally() %>%
    ungroup() %>%
    complete(!!!syms(groups), fill = list(n = 0))
}


sample_dirichlet = function(n, alphas, names, longer=TRUE, eps=.000001) {
  df = rdirichlet(n, alphas + eps) %>% 
    as_tibble()
  for (i in 1:ncol(df)) {
    df = df %>%
      rename(!!quo_name(names[i]) := glue("V{i}"))
  }
  if (longer) {
    df = df %>%
      pivot_longer(cols = everything(),
                   names_to = 'category',
                   values_to = 'p')
  }
  return (df)
}


get_dirichlet_hdci = function(df,
                              groups=c(), # applies group_by, last value should be category
                              n_samples, # number of samples to draw from Dirichlet
                              width=.95,
                              seed=NA) {
  assertthat::assert_that(length(groups) >= 1)
  
  if (!is.na(seed)) {
    set.seed(seed)
  }
  
  df = df %>% 
    get_counts_with_zeros(groups)
  
  category_name = groups[length(groups)]
  groups = groups[1:length(groups)-1]
  categories = levels(df %>% pull(category_name))
  
  df = df %>%
    group_by(!!!syms(groups)) %>%
    nest() %>%
    mutate(lst = map(data, ~ pull(., n)),
           samples = map(lst, ~ sample_dirichlet(n_samples, ., categories))) %>% 
    unnest(cols = c(samples)) %>%
    group_by(!!!syms(c(groups, 'category'))) %>% 
    mean_hdci(p, .width = width) %>% 
    select(!!enquo(groups), !!enquo(category_name) := category, hdci_l = .lower, hdci_h = .upper)
  return (df)
}