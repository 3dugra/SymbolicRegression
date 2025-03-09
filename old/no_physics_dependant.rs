use ndarray::{Array1, Array2};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::fmt;
use std::time::Instant;


const POPULATION_SIZE: usize = 5000;
const MAX_GENERATIONS: usize = 100;
const TOURNAMENT_SIZE: usize = 7;

#[derive(Debug, Clone)]
enum Expr {
    Constant(f64),
    Variable(usize),  // 0 = offset voltage, 1 = resistance
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, i32),
}

impl Expr {
    fn evaluate(&self, inputs: &[f64]) -> f64 {
        match self {
            Expr::Constant(val) => *val,
            Expr::Variable(index) => inputs[*index],
            Expr::Add(left, right) => left.evaluate(inputs) + right.evaluate(inputs),
            Expr::Sub(left, right) => left.evaluate(inputs) - right.evaluate(inputs),
            Expr::Mul(left, right) => left.evaluate(inputs) * right.evaluate(inputs),
            Expr::Div(left, right) => {
                let denom = right.evaluate(inputs);
                if denom == 0.0 { 1.0 } else { left.evaluate(inputs) / denom }
            }
            Expr::Pow(base, exp) => base.evaluate(inputs).powi(*exp),
        }
    }

    fn random(depth: usize) -> Self {
        let mut rng = thread_rng();
        if depth == 0 {
            match rng.gen_range(0..3) {
                0 => Expr::Variable(rng.gen_range(0..2)),
                1 => Expr::Constant(rng.gen_range(-10.0..10.0)),
                _ => Expr::Pow(Box::new(Expr::Variable(0)), 2),
            }
        } else {
            match rng.gen_range(0..5) {
                0 => Expr::Add(Box::new(Expr::random(depth-1)), Box::new(Expr::random(depth-1))),
                1 => Expr::Sub(Box::new(Expr::random(depth-1)), Box::new(Expr::random(depth-1))),
                2 => Expr::Mul(Box::new(Expr::random(depth-1)), Box::new(Expr::random(depth-1))),
                3 => Expr::Div(Box::new(Expr::random(depth-1)), Box::new(Expr::random(depth-1))),
                4 => Expr::Pow(Box::new(Expr::random(depth-1)), rng.gen_range(2..4)),
                _ => unreachable!(),
            }
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Constant(val) => write!(f, "{:.4}", val),
            Expr::Variable(0) => write!(f, "V_offset"),
            Expr::Variable(1) => write!(f, "R"),
            Expr::Variable(_) => write!(f, "?"),
            Expr::Add(left, right) => write!(f, "({} + {})", left, right),
            Expr::Sub(left, right) => write!(f, "({} - {})", left, right),
            Expr::Mul(left, right) => write!(f, "({} * {})", left, right),
            Expr::Div(left, right) => write!(f, "({} / {})", left, right),
            Expr::Pow(base, exp) => write!(f, "({})^{}", base, exp),
        }
    }
}

struct SymbolicRegressor {
    population: Vec<Expr>,
    inputs: Array2<f64>,
    outputs: Array1<f64>,
}

impl SymbolicRegressor {
    fn new(inputs: Array2<f64>, outputs: Array1<f64>) -> Self {
        let population = (0..POPULATION_SIZE)
            .into_par_iter()
            .map(|_| Expr::random(4))
            .collect();
        
        SymbolicRegressor { population, inputs, outputs }
    }

    fn fitness(&self, expr: &Expr) -> f64 {
        (0..self.inputs.nrows())
            .into_par_iter()
            .map(|i| {
                let row = self.inputs.row(i);
                let prediction = expr.evaluate(row.to_slice().unwrap());
                let error = prediction - self.outputs[i];
                error * error
            })
            .sum::<f64>().sqrt() * -1.0
    }

    fn test_fitness(&self, expr: &Expr, test_inputs: &Array2<f64>, test_outputs: &Array1<f64>) -> f64 {
        (0..test_inputs.nrows())
            .into_par_iter()
            .map(|i| {
                let row = test_inputs.row(i);
                let prediction = expr.evaluate(row.to_slice().unwrap());
                let error = prediction - test_outputs[i];
                error * error
            })
            .sum::<f64>().sqrt() * -1.0
    }

    fn evolve(&mut self) {
        let new_population: Vec<Expr> = (0..POPULATION_SIZE)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let candidates: Vec<&Expr> = (0..TOURNAMENT_SIZE)
                    .map(|_| &self.population[rng.gen_range(0..self.population.len())])
                    .collect();
                
                candidates.into_iter()
                    .max_by(|a, b| self.fitness(a).partial_cmp(&self.fitness(b)).unwrap())
                    .unwrap()
                    .clone()
            })
            .collect();

        self.population = new_population
            .into_par_iter()
            .map(|mut expr| {
                let mut rng = thread_rng();
                if rng.gen_bool(0.1) {
                    expr = Expr::random(3);
                }
                expr
            })
            .collect();
    }

    fn best_expression(&self) -> &Expr {
        self.population
            .par_iter()
            .max_by(|a, b| self.fitness(a).partial_cmp(&self.fitness(b)).unwrap())
            .unwrap()
    }
}

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
        inputs[[i, 0]] = row[0];  // Offset voltage
        inputs[[i, 1]] = row[1];  // Resistance
        outputs[i] = row[2];      // Efficiency
    }

    Ok((inputs, outputs))
}



fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    println!("Current directory: {:?}", std::env::current_dir()?);
    println!("Current time: {:?}", start_time);
    let train_path = "training_points.m";
    let test_path = "testing_points.m";
    println!("Checking for: {}", train_path);
    println!("Exists: {}", std::path::Path::new(train_path).exists());
    
    println!("Checking for: {}", test_path);
    println!("Exists: {}", std::path::Path::new(test_path).exists());
    
    let (train_inputs, train_outputs) = parse_data_file("training_points.m")?;
    let (test_inputs, test_outputs) = parse_data_file("testing_points.m")?;
    
    let mut regressor = SymbolicRegressor::new(train_inputs, train_outputs);

    for generation in 0..MAX_GENERATIONS {
        let gen_start = Instant::now();
        regressor.evolve();
        
        let best = regressor.best_expression();
        let train_rmse = -regressor.fitness(best);
        let test_rmse = -regressor.test_fitness(best, &test_inputs, &test_outputs);
        
        println!("Generation {:3} | Train RMSE: {:.4} | Test RMSE: {:.4}", generation, train_rmse, test_rmse);
        println!("Best expression: {}", best);
        println!("Generation time: {:.2?}\n", gen_start.elapsed());
    }

    let best = regressor.best_expression();
    println!("\nFinal Best Expression: {}", best);
    println!("Final Training RMSE: {:.4}", -regressor.fitness(best));
    println!("Final Testing RMSE: {:.4}", -regressor.test_fitness(best, &test_inputs, &test_outputs));
    let end_time = Instant::now();
    println!("Test went from {:.2?} to {:.2?}; Total execution time: {:.2?}", start_time, end_time, end_time.duration_since(start_time));
    
    Ok(())
}