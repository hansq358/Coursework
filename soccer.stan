data {
  int<lower=0> ng ;//ngames
  int<lower=0> nt ;//nteams
  int<lower=0> yh[ng];//observed number of goals for each home team
  int<lower=0> yv[ng];//observed number of goals for each visiting team
  int<lower=0> ht[ng]; //each home team index 
  int<lower=0> vt[ng]; //each visiting team index
  int<lower=0> np; //number of predicted games
  int<lower=0> htpred[np];// prediction of ht
  int<lower=0> vtpred[np];// prediciton of vt
}
parameters {
  real home; //home advantage
  vector[nt] att_star; //"sum to zero" constraint
  vector[nt] def_star; //"sum to zero" constraint
  //hyperparameters
  real mu_att; // mean of attack ability
  real mu_def; // mean of defense ability
  real<lower=0> tau_att; // tau of attack ability
  real<lower=0> tau_def; // tau of defense ability
}
transformed parameters {
  vector[nt] att; //attack ability of each team
  vector[nt] def; //defense ability of each team
  vector[ng] theta1; //probability of completing goals for home teams
  vector[ng] theta2; //probablitiy of completing goals for visiting teams
  real<lower=0> sig_att; // transform tau of attack to sigma attack
  real<lower=0> sig_def; // transform tau of defense to sigma defense
  att = att_star - mean(att_star[]); //using att_star to calculate att
  def = def_star - mean(def_star[]); //using def_star to calculate def
  theta1 = exp(home + att[ht] + def[vt]); 
  theta2 = exp(att[vt] + def[ht]);
  sig_att=sqrt(1/tau_att);
  sig_def=sqrt(1/tau_def);
}
model {
  // hyper priors
  mu_att ~ normal(0,1);
  mu_def ~ normal(0,1);
  tau_att ~ gamma(0.01,0.01);
  tau_def ~ gamma(0.01,0.01);
  // priors
  home ~ normal(0,0.5);
  att_star ~ normal(mu_att,sig_att);
  def_star ~ normal(mu_def,sig_def);
  // likelihood
  yh ~ poisson(theta1);
  yv ~ poisson(theta2);
}
generated quantities{
//generate predictions
  vector[np] theta1pred; //prediction of probability of completing goals for home teams
  vector[np] theta2pred; //prediction of probability of completing goals for visiting teams
  real yhpred[np]; //prediction of goals for home teams
  real yvpred[np]; //prediction of goals for visiting teams
  theta1pred = exp(home+att[htpred]+def[vtpred]);
  theta2pred = exp(att[vtpred]+def[htpred]);
  yhpred = poisson_rng(theta1pred);
  yvpred = poisson_rng(theta2pred);
}
