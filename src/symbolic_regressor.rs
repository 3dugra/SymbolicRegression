use std::sync::{Arc, RwLock};
use ndarray::{Array1, Array2};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::expr::{Expr, Dimension};
use crate::config::Config;

// Remove duplicate constants since they're now in Config

pub struct SymbolicRegressor {
    config: Config,
    pub population: Vec<Expr>,
    pub inputs: Array2<f64>,
    pub outputs: Array1<f64>,
    pub pop_size: usize,
    pub adaptive_weights: (f64, f64),
    pub prev_best_rmse: Option<f64>,
    pub prev_best_rse: Option<f64>,
    pub previous_best_rse: f64,
    pub stagnation_count: usize,
    pub generation_times: Vec<Duration>,
    // Replace RefCell with Arc<RwLock>>
    pub fitness_cache: Arc<RwLock<HashMap<String, (f64, f64, f64)>>>,
}

impl SymbolicRegressor {
    pub fn new(inputs: Array2<f64>, outputs: Array1<f64>, config: Config) -> Self {
        let mut pop = Vec::with_capacity(config.initial_pop);
        let physics_count = (config.initial_pop as f64 * config.initial_physics_ratio) as usize;
        
        for i in 0..config.initial_pop {
            let mut expr;
            let mut attempts = 0;
            loop {
                expr = if i < physics_count {
                    Expr::prototype().mutate(2)
                } else {
                    Expr::random(3)
                };
                expr = expr.simplify();
                if expr.dimensionality() == Dimension::Dimensionless {
                    break;
                }
                attempts += 1;
                if attempts >= 10 {
                    expr = Expr::Constant(0.0);
                    break;
                }
            }
            pop.push(expr);
        }

        SymbolicRegressor {
            population: pop,
            inputs,
            outputs,
            pop_size: config.initial_pop,
            adaptive_weights: (config.error_weight, config.physics_weight),
            prev_best_rmse: None,
            prev_best_rse: None,
            previous_best_rse: f64::MAX,
            stagnation_count: 0,
            generation_times: Vec::with_capacity(config.max_generations),
            // Initialize fitness cache
            fitness_cache: Arc::new(RwLock::new(HashMap::new())),
            config: config.clone(), // Clone the config here
        }
    }

    fn update_adaptive_weights(&mut self, rmse: f64, physics_score: f64) {
        // If we have previous values to compare
        if let (Some(prev_rmse), _) = (self.prev_best_rmse, self.prev_best_rse) {
            let rmse_change = (prev_rmse - rmse) / prev_rmse;
            
            // If RMSE is improving rapidly, we can relax physics constraints a bit
            if rmse_change > 0.1 {  // Good improvement in RMSE
                self.adaptive_weights.0 = (self.adaptive_weights.0 + 0.05).min(0.8);  // Increase RMSE weight
                self.adaptive_weights.1 = (self.adaptive_weights.1 - 0.05).max(0.2);  // Decrease physics weight
            } 
            // If RMSE is stagnating, increase physics constraints to explore new areas
            else if rmse_change < 0.01 {  // Little improvement in RMSE
                self.adaptive_weights.0 = (self.adaptive_weights.0 - 0.05).max(0.2);  // Decrease RMSE weight
                self.adaptive_weights.1 = (self.adaptive_weights.1 + 0.05).min(0.8);  // Increase physics weight
            }
            
            // If physics violations are high, prioritize physics more
            if physics_score > 0.5 {
                self.adaptive_weights.1 = (self.adaptive_weights.1 + 0.1).min(0.8);  // Increase physics weight
            }
        }
    }

    pub fn update_stagnation(&mut self, current_rse: f64) {
        let improvement = (self.previous_best_rse - current_rse).abs();
        if improvement < self.config.stagnation_threshold {
            self.stagnation_count += 1;
        } else {
            self.stagnation_count = 0;
            self.previous_best_rse = current_rse;
        }
    }


    fn calculate_mutation_rate(&self, elapsed: Duration) -> f64 {
        let time_ratio = elapsed.as_secs_f64() / self.config.target_time().as_secs_f64();
        let time_factor = time_ratio.clamp(0.5, 2.0);
        
        let stagnation_factor = self.config.stagnation_response.powi(self.stagnation_count as i32);
        
        (self.config.base_mutation_rate * time_factor * stagnation_factor)
            .clamp(self.config.min_mutation, self.config.max_mutation)
    }

    pub fn evolve(&mut self) -> Duration {
        let start = Instant::now();
        let mut rng = thread_rng();
        
        // Calculate current mutation rate
        let elapsed_time = self.generation_times.last()
            .copied()
            .unwrap_or(Duration::from_secs(1));
        let mutation_rate = self.calculate_mutation_rate(elapsed_time);
        
        // Get current best for weights adaptation
        let best = self.best_expression().clone();
        let (rmse, rse, physics_score) = self.fitness(&best);
        
        // Update adaptive weights based on performance
        self.update_adaptive_weights(rmse, physics_score);
        
        // Use single population adaptation method
        self.adaptive_population(elapsed_time, rmse, rse);
        
        // Ensure population size stays within bounds
        self.pop_size = self.pop_size.clamp(self.config.min_pop, self.config.max_pop);
        
        // Tournament selection and breeding with adaptive weights
        let mut new_population = Vec::with_capacity(self.pop_size);
        
        while new_population.len() < self.pop_size {
            // Tournament selection
            let idx1 = rng.gen_range(0..self.population.len());
            let idx2 = rng.gen_range(0..self.population.len());
            
            // Use evaluate instead of fitness for faster comparison
            let score1 = self.population[idx1].evaluate(self.inputs.row(0).to_slice().unwrap());
            let score2 = self.population[idx2].evaluate(self.inputs.row(0).to_slice().unwrap());
            
            let parent = if score1 < score2 {
                &self.population[idx1]
            } else {
                &self.population[idx2]
            };
            
            // Mutation with new mutation rate
            if rng.gen_bool(mutation_rate) {
                new_population.push(parent.mutate(2).simplify());
            } else {
                new_population.push(parent.clone());
            }
        }
        
        // Replace population
        self.population = new_population;
        
        // Record generation time
        let duration = start.elapsed();
        self.generation_times.push(duration);
        
        duration
    }

    pub fn adaptive_population(&mut self, last_time: Duration, current_rmse: f64, current_rse: f64) {
        let time_ratio = last_time.as_secs_f64() / self.config.target_time().as_secs_f64();
    
        // Time-based adaptation
        let time_factor = time_ratio.clamp(0.5, 2.0);
    
        // RMSE-driven adaptation
        let rmse_factor = match self.prev_best_rmse {
            Some(prev) => {
                let improvement = (prev - current_rmse) / prev;
                match improvement {
                    i if i > 0.05 => 1.2,  // Significant improvement → expand population
                    i if i > 0.0  => 1.0,   // Neutral
                    _             => 0.8,   // Degradation → shrink population
                }
            }
            None => 1.0,
        };
    
        // RSE-driven adaptation
        let rse_factor = match self.prev_best_rse {
            Some(prev) => {
                let improvement = (prev - current_rse) / prev;
                match improvement {
                    i if i > 0.1  => 1.5,
                    i if i > 0.05 => 1.2,
                    _             => 0.9,
                }
            }
            None => 1.0,
        };
    
        // Combine factors with weights
        let combined_factor = (time_factor * 0.4) + 
                            (rmse_factor * 0.4) + 
                            (rse_factor * 0.2);
    
        // Update population size
        self.pop_size = (self.pop_size as f64 * combined_factor)
            .clamp(self.config.min_pop as f64, self.config.max_pop as f64) as usize;
    
        // Store historical values
        self.prev_best_rmse = Some(current_rmse);
        self.prev_best_rse = Some(current_rse);
    }

    pub fn best_expression(&self) -> &Expr {
        self.population
            .par_iter()
            .max_by(|a, b| {
                let (rmse_a, rse_a, physics_a) = self.fitness(a);
                let (rmse_b, rse_b, physics_b) = self.fitness(b);
                
                let a_score = rmse_a * self.adaptive_weights.0 + 
                             physics_a * self.adaptive_weights.1 + 
                             rse_a * self.config.rse_weight;
                
                let b_score = rmse_b * self.adaptive_weights.0 + 
                             physics_b * self.adaptive_weights.1 + 
                             rse_b * self.config.rse_weight;
                
                b_score.partial_cmp(&a_score).unwrap()
            })
            .unwrap()
    }
    
    pub fn fitness(&self, expr: &Expr) -> (f64, f64, f64) {
        let expr_key = format!("{}", expr);
        
        // Try to get cached result
        if let Some(&result) = self.fitness_cache.read().unwrap().get(&expr_key) {
            return result;
        }
        
        // Calculate fitness if not in cache
        let result = self.calculate_fitness(expr);
        
        // Cache the result using RwLock
        let mut cache = self.fitness_cache.write().unwrap();
        if cache.len() < 10000 {
            cache.insert(expr_key, result);
        }
        
        result
    }
    // Improvement 4: Implement batch evaluation for fitness calculations
    fn calculate_fitness(&self, expr: &Expr) -> (f64, f64, f64) {
        let n_rows = self.inputs.nrows();
        let mut total_error = 0.0;
        let mut physics_violations = 0.0;
        let mut y_mean = 0.0;
        
        // First pass - calculate mean
        for i in 0..n_rows {
            y_mean += self.outputs[i];
        }
        y_mean /= n_rows as f64;
        
        // Second pass - calculate errors
        let mut total_ss = 0.0;  // Total sum of squares
        
        for batch_start in (0..n_rows).step_by(self.config.batch_size) {
            let batch_end = std::cmp::min(batch_start + self.config.batch_size, n_rows);
            
            let (batch_error, batch_physics, batch_ss) = (batch_start..batch_end)
                .into_par_iter()
                .map(|i| {
                    let row = self.inputs.row(i);
                    let prediction = expr.evaluate_with_depth(row.to_slice().unwrap(), 0).clamp(0.0, 1.0);
                    let actual = self.outputs[i];
                    
                    let error = (prediction - actual).powi(2);
                    let mut violations = 0.0;
                    
                    // Physics violations
                    if expr.dimensionality() != Dimension::Dimensionless {
                        violations += 1.0;
                    }
                    if !prediction.is_finite() || prediction.abs() > 10.0 {
                        violations += 100.0;  // Heavily penalize numerical instability
                    }
                    
                    // Sum of squares for R²
                    let ss = (actual - y_mean).powi(2);
                    
                    (error, violations, ss)
                })
                .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
            
            total_error += batch_error;
            physics_violations += batch_physics;
            total_ss += batch_ss;
        }
        
        let n = n_rows as f64;
        let rmse = (total_error / n).sqrt();
        let rse = if total_ss > 0.0 { total_error / total_ss } else { f64::MAX };
        let physics_score = physics_violations / n;
        
        (rmse, rse, physics_score)
    }

    pub fn test_fitness(&self, expr: &Expr, test_inputs: &Array2<f64>, test_outputs: &Array1<f64>) -> (f64, f64, f64) {
        let n_rows = test_inputs.nrows();
        let mut total_error = 0.0;
        let mut physics_violations = 0.0;
        
        // Use the BATCH_SIZE constant here as well
        for batch_start in (0..n_rows).step_by(self.config.batch_size) {
            let batch_end = std::cmp::min(batch_start + self.config.batch_size, n_rows);
            
            let (batch_error, batch_violations) = (batch_start..batch_end)
                .into_par_iter()
                .map(|i| {
                    let row = test_inputs.row(i);
                    let prediction = expr.evaluate_with_depth(row.to_slice().unwrap(), 0).clamp(0.0, 1.0);
                    let actual = test_outputs[i];
                    
                    let error = (prediction - actual).powi(2);
                    let mut violations = 0.0;
                    
                    if expr.dimensionality() != Dimension::Dimensionless {
                        violations += 1.0;
                    }
                    violations += (prediction - prediction.clamp(0.0, 1.0)).abs();
                    
                    (error, violations)
                })
                .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
            
            total_error += batch_error;
            physics_violations += batch_violations;
        }
        
        let n = n_rows as f64;
        let rmse = (total_error / n).sqrt();
        let rse = total_error;
        let physics_score = physics_violations / n;
        
        (rmse, rse, physics_score)
    }
    
}
