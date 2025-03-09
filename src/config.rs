use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::fs;
use std::path::Path;
use toml;

#[derive(Debug, Serialize, Deserialize, Clone)]  // Add Clone here
pub struct Config {
    // Population settings
    pub initial_pop: usize,
    pub min_pop: usize,
    pub max_pop: usize,
    pub max_generations: usize,
    
    // Time settings
    pub target_time_secs: u64,
    
    // Mutation settings
    pub base_mutation_rate: f64,
    pub min_mutation: f64,
    pub max_mutation: f64,
    
    // Weight settings
    pub error_weight: f64,
    pub physics_weight: f64,
    pub rse_weight: f64,
    pub rmse_weight: f64,
    
    // Other parameters
    pub initial_physics_ratio: f64,
    pub stagnation_threshold: f64,
    pub stagnation_response: f64,
    pub batch_size: usize,
}

impl Config {
    pub fn target_time(&self) -> Duration {
        Duration::from_secs(self.target_time_secs)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            initial_pop: 2000,      // Increase population size
            min_pop: 1000,
            max_pop: 5000,          // Reduced to focus on quality
            max_generations: 100,    // Reduce total generations but make them count
            target_time_secs: 100,  // Increase from 250 to allow ~100s per generation
            base_mutation_rate: 0.1,
            min_mutation: 0.05,
            max_mutation: 0.1,
            error_weight: 0.5,
            physics_weight: 0.75,
            rse_weight: 0.25,
            rmse_weight: 0.25,
            initial_physics_ratio: 0.8,
            stagnation_threshold: 0.2,
            stagnation_response: 0.5,
            batch_size: 64,
        }
    }
}