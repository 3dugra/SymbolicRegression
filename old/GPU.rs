/*
Still under development

*/

use arrayfire::{Array, Dim4};
use rand::Rng;
use std::fmt;
use std::time::Instant;

const POPULATION_SIZE: usize = 10_000;
const MAX_GENERATIONS: usize = 100;

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
    fn evaluate(&self, x1: &Array<f64>, x2: &Array<f64>) -> Array<f64> {
        match self {
            Expr::Const(c) => {
                // Create constant array with same dimensions as inputs
                Array::new(&[*c], Dim4::new(&[1, 1, 1, 1])).tile(x1.dims())
            },
            Expr::Var(0) => x1.clone(),
            Expr::Var(1) => x2.clone(),
            Expr::Var(_) => panic!("Unexpected variable index, only 0 (x1) and 1 (x2) are allowed"),
            Expr::Add(a, b) => a.evaluate(x1, x2) + b.evaluate(x1, x2),
            Expr::Sub(a, b) => a.evaluate(x1, x2) - b.evaluate(x1, x2),
            Expr::Mul(a, b) => a.evaluate(x1, x2) * b.evaluate(x1, x2),
            Expr::Div(a, b) => a.evaluate(x1, x2) / (b.evaluate(x1, x2) + 1e-6),
            Expr::Sin(a) => arrayfire::sin(&a.evaluate(x1, x2)),
            Expr::Cos(a) => arrayfire::cos(&a.evaluate(x1, x2)),
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
            0 => Expr::Add(Box::new(Expr::random(depth-1, rng)), Box::new(Expr::random(depth-1, rng))),
            1 => Expr::Sub(Box::new(Expr::random(depth-1, rng)), Box::new(Expr::random(depth-1, rng))),
            2 => Expr::Mul(Box::new(Expr::random(depth-1, rng)), Box::new(Expr::random(depth-1, rng))),
            3 => Expr::Div(Box::new(Expr::random(depth-1, rng)), Box::new(Expr::random(depth-1, rng))),
            4 => Expr::Sin(Box::new(Expr::random(depth-1, rng))),
            5 => Expr::Cos(Box::new(Expr::random(depth-1, rng))),
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
    x1: Array<f64>,
    x2: Array<f64>,
    y: Array<f64>,
}

impl SymbolicRegressor {
    fn new(x1: Array<f64>, x2: Array<f64>, y: Array<f64>) -> Self {
        let mut rng = rand::thread_rng();
        let population = (0..POPULATION_SIZE)
            .map(|_| Expr::random(4, &mut rng))
            .collect();
        
        SymbolicRegressor { population, x1, x2, y }
    }

    fn fitness(&self, expr: &Expr) -> f64 {
        let pred = expr.evaluate(&self.x1, &self.x2);
        let diff = &pred - &self.y;
        let squared_error = &diff * &diff;
        let error_sum = arrayfire::sum(&squared_error, 0);
        let error_scalar = arrayfire::scalar_value(&error_sum).unwrap();
        -error_scalar
    }

    fn evolve(&mut self) {
        let mut rng = rand::thread_rng();
        let mut new_population = Vec::with_capacity(POPULATION_SIZE);

        // Tournament selection
        for _ in 0..POPULATION_SIZE {
            let a = rng.gen_range(0..POPULATION_SIZE);
            let b = rng.gen_range(0..POPULATION_SIZE);
            let winner = if self.fitness(&self.population[a]) > self.fitness(&self.population[b]) {
                self.population[a].clone()
            } else {
                self.population[b].clone()
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

fn parse_data_file(path: &str) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let start = content.find('[').ok_or("Missing '['")? + 1;
    let end = content.find(']').ok_or("Missing ']'")?;
    let data_str = &content[start..end];

    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    let mut y = Vec::new();

    for row in data_str.split(';') {
        let trimmed = row.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut cols = trimmed.split_whitespace();
        x1.push(cols.next().ok_or("Missing column")?.parse()?);
        x2.push(cols.next().ok_or("Missing column")?.parse()?);
        y.push(cols.next().ok_or("Missing column")?.parse()?);
    }

    Ok((x1, x2, y))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    
    // Initialize ArrayFire with CUDA
    arrayfire::set_backend(arrayfire::Backend::CUDA);
    arrayfire::set_device(0);

    // Load and prepare data
    let (x1_vec, x2_vec, y_vec) = parse_data_file("all_data_points.m")?;
    let dims = Dim4::new(&[x1_vec.len() as u64, 1, 1, 1]);
    
    let x1 = Array::new(&x1_vec, dims);
    let x2 = Array::new(&x2_vec, dims);
    let y = Array::new(&y_vec, dims);

    let mut regressor = SymbolicRegressor::new(x1, x2, y);

    // Run evolution
    for generation in 0..MAX_GENERATIONS {
        let gen_start = Instant::now();
        regressor.evolve();
        let best = regressor.best_expression();
        println!("Generation {:3} | MSE: {:8.4} | {}",
               generation,
               -regressor.fitness(best),
               best);
        println!("Generation time: {:.2?}", gen_start.elapsed());
    }

    let total_time = total_start.elapsed();
    println!("\nTotal execution time: {:.2?}", total_time);
    
    Ok(())
}