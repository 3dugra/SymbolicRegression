use ndarray::{Array1, Array2};
use rand::Rng;
use rayon::prelude::*;
use std::fmt;

const BATCH_SIZE: usize = 150;

#[derive(Debug, Clone)]
enum Expr {
    Const(f64),
    Var(usize), // 0 = x1, 1 = x2
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
}

impl Expr {
    fn evaluate(&self, inputs: &[f64]) -> f64 {
        match self {
            Expr::Const(c) => *c,
            Expr::Var(i) => inputs[*i],
            Expr::Add(a, b) => a.evaluate(inputs) + b.evaluate(inputs),
            Expr::Sub(a, b) => a.evaluate(inputs) - b.evaluate(inputs),
            Expr::Mul(a, b) => a.evaluate(inputs) * b.evaluate(inputs),
            Expr::Div(a, b) => {
                let denom = b.evaluate(inputs);
                if denom.abs() > 1e-6 {
                    a.evaluate(inputs) / denom
                } else {
                    1.0
                }
            }
            Expr::Sin(a) => a.evaluate(inputs).sin(),
            Expr::Cos(a) => a.evaluate(inputs).cos(),
        }
    }

    fn random(depth: usize, rng: &mut impl Rng) -> Expr {
        if depth == 0 || rng.gen_bool(0.3) {
            return if rng.gen_bool(0.5) {
                Expr::Var(rng.gen_range(0..2))
            } else {
                Expr::Const(rng.gen_range(-2.0..=2.0))
            };
        }

        match rng.gen_range(0..6) {
            0 => Expr::Add(
                Box::new(Expr::random(depth - 1, rng)),
                Box::new(Expr::random(depth - 1, rng)),
            ),
            1 => Expr::Sub(
                Box::new(Expr::random(depth - 1, rng)),
                Box::new(Expr::random(depth - 1, rng)),
            ),
            2 => Expr::Mul(
                Box::new(Expr::random(depth - 1, rng)),
                Box::new(Expr::random(depth - 1, rng)),
            ),
            3 => Expr::Div(
                Box::new(Expr::random(depth - 1, rng)),
                Box::new(Expr::random(depth - 1, rng)),
            ),
            4 => Expr::Sin(Box::new(Expr::random(depth - 1, rng))),
            5 => Expr::Cos(Box::new(Expr::random(depth - 1, rng))),
            _ => unreachable!(),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Const(c) => write!(f, "{:.2}", c),
            Expr::Var(0) => write!(f, "x1"),
            Expr::Var(1) => write!(f, "x2"),
            Expr::Var(_) => write!(f, "?"),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Sin(a) => write!(f, "sin({})", a),
            Expr::Cos(a) => write!(f, "cos({})", a),
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
        let mut rng = rand::thread_rng();
        let population = (0..1000).map(|_| Expr::random(4, &mut rng)).collect();
        SymbolicRegressor { population, inputs, outputs }
    }

    fn fitness(&self, expr: &Expr) -> f64 {
        let total_samples = self.inputs.nrows();
        let mut total_error = 0.0;
        
        for chunk in 0..(total_samples / BATCH_SIZE + 1) {
            let start = chunk * BATCH_SIZE;
            let end = (chunk + 1) * BATCH_SIZE;
            let end = end.min(total_samples);
            
            let error = (start..end)
                .into_par_iter()
                .map(|i| {
                    let inputs = self.inputs.row(i).to_vec();
                    let prediction = expr.evaluate(&inputs);
                    let error = prediction - self.outputs[i];
                    error * error
                })
                .sum::<f64>();
                
            total_error += error;
        }
        
        -total_error / total_samples as f64
    }

    fn evolve(&mut self) {
        let mut rng = rand::thread_rng();
        
        // Tournament selection
        let mut new_population = Vec::with_capacity(self.population.len());
        for _ in 0..self.population.len() {
            let a_idx = rng.gen_range(0..self.population.len());
            let b_idx = rng.gen_range(0..self.population.len());
            let winner = if self.fitness(&self.population[a_idx]) > self.fitness(&self.population[b_idx]) {
                self.population[a_idx].clone()
            } else {
                self.population[b_idx].clone()
            };
            new_population.push(winner);
        }

        // Mutation
        for expr in &mut new_population {
            if rng.gen_bool(0.1) {
                *expr = Expr::random(3, &mut rng);
            }
        }

        self.population = new_population;
    }

    fn best_expression(&self) -> &Expr {
        self.population
            .iter()
            .max_by(|a, b| self.fitness(a).partial_cmp(&self.fitness(b)).unwrap())
            .unwrap()
    }
}

fn parse_data_file(path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let start = content.find('[').ok_or("Missing '['")? + 1;
    let end = content.find(']').ok_or("Missing ']'")?;
    let data_str = &content[start..end];

    let mut rows = Vec::new();
    for row in data_str.split(';') {
        let trimmed = row.trim();
        if trimmed.is_empty() {
            continue;
        }
        let values: Vec<f64> = trimmed
            .split_whitespace()
            .map(|s| s.parse())
            .collect::<Result<_, _>>()?;
        rows.push(values);
    }

    let n_samples = rows.len();
    let mut inputs = Array2::zeros((n_samples, 2));
    let mut outputs = Array1::zeros(n_samples);

    for (i, row) in rows.iter().enumerate() {
        inputs[[i, 0]] = row[0];
        inputs[[i, 1]] = row[1];
        outputs[i] = row[2];
    }

    Ok((inputs, outputs))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (inputs, outputs) = parse_data_file("all_data_points.m")?;
    let mut regressor = SymbolicRegressor::new(inputs, outputs);
    
    for generation in 0..100 {
        regressor.evolve();
        let best = regressor.best_expression();
        println!("Generation {:3} | MSE: {:8.4} | Best Function: {}",
               generation, 
               -regressor.fitness(best), 
               best);
    }

    Ok(())
}