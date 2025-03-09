mod expr;
mod symbolic_regressor;
mod config;

use symbolic_regressor::SymbolicRegressor;
use config::Config;
use std::time::Instant;
use ndarray::{Array1, Array2};
use std::time::Duration;

fn parse_data_file(path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let data_str = content.split('[').nth(1).and_then(|s| s.split(']').next()).ok_or("Invalid format")?;
    
    let mut rows = Vec::new();
    for line in data_str.split(';') {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        
        let values: Vec<f64> = trimmed.split_whitespace()
            .map(|s| s.parse())
            .collect::<Result<_, _>>()?;
        
        if values.len() == 3 {
            rows.push(values);
        }
    }

    let mut inputs = Array2::zeros((rows.len(), 2));
    let mut outputs = Array1::zeros(rows.len());
    
    for (i, row) in rows.iter().enumerate() {
        // Convert millivolts to volts for input voltage
        inputs[[i, 0]] = row[0] / 1000.0;  // Now in volts
        // Resistance remains in ohms
        inputs[[i, 1]] = row[1];          // Already in ohms
        // Efficiency is dimensionless (0-1 range)
        outputs[i] = row[2];              // Efficiency remains unchanged
    }

    Ok((inputs, outputs))
}

// Modified main function with reduced printing frequency
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    
    // Load configuration
    let config = Config::load("config.toml").unwrap_or_default();
    
    // Load data
    let (train_inputs, train_outputs) = parse_data_file("training_points.m")?;
    let (test_inputs, test_outputs) = parse_data_file("testing_points.m")?;
    
    // Initialize regressor with config
    let mut regressor = SymbolicRegressor::new(train_inputs, train_outputs, config.clone());
    let mut gen_times = Vec::new();

    // Use config.max_generations instead of MAX_GENERATIONS
    for generation in 0..config.max_generations {
        let elapsed = regressor.evolve();
        gen_times.push(elapsed);
        
        let best = regressor.best_expression().clone();
        let (train_rmse, train_rse, _) = regressor.fitness(&best);
        let (test_rmse, test_rse, _) = regressor.test_fitness(&best, &test_inputs, &test_outputs);
        
        // Update stagnation and adaptive parameters
        regressor.update_stagnation(train_rse);
        regressor.adaptive_population(elapsed, train_rmse, train_rse);
        
        // Only calculate complexity when needed for logging
        if generation % 5 == 0 || generation == config.max_generations - 1 {
            let complexity = best.expr_complexity();
            println!("Generation {:3} | Pop: {:5} | Time: {:5.2?} | Stagnation: {} | Complexity: {}", 
                generation, regressor.pop_size, elapsed, regressor.stagnation_count, complexity);
            println!("Train RMSE: {:.4} | R2: {:.2} | Test RMSE: {:.4} | R2: {:.2}", 
                train_rmse, 1.0-train_rse, test_rmse, 1.0-test_rse);
            println!("Best Expression: {}", best);
        }

        // Early stopping check
        if !gen_times.is_empty() && (
            gen_times.len() > config.target_time_secs as usize || 
            regressor.stagnation_count > (config.max_generations / 10)
        ) {
            let avg_time = gen_times.iter().sum::<Duration>() / gen_times.len() as u32;
            if avg_time > config.target_time() * 5 {
                println!("\nStopping early due to {}", 
                    if regressor.stagnation_count > (config.max_generations / 10) {
                        "stagnation"
                    } else {
                        "time limit"
                    });
                break;
            }
        }
    }

    let total_time = start_time.elapsed();
    println!("\nTotal execution: {:.2?} | Avg gen time: {:.2?}", 
        total_time, total_time / gen_times.len() as u32);
    
    Ok(())
}