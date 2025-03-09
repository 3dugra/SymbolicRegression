use ndarray::{Array1, Array2};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::fmt;
use std::time::Instant;

const POPULATION_SIZE: usize = 100;
const MAX_GENERATIONS: usize = 10;
const TOURNAMENT_SIZE: usize = 15;
const PHYSICS_WEIGHT: f64 = 0.3;
const ERROR_WEIGHT: f64 = 0.7;

#[derive(Debug, Clone, PartialEq)]
enum Dimension {
    Voltage,
    Resistance,
    Power,
    Dimensionless,
    Invalid,
}

#[derive(Debug, Clone, PartialEq)]
enum Expr {
    Constant(f64),
    Variable(usize),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, i32),
}

impl Expr {
    fn prototype() -> Self {
        Expr::Div(
            Box::new(Expr::Div(
                Box::new(Expr::Pow(Box::new(Expr::Variable(0)), 2)),
                Box::new(Expr::Variable(1))
            )),
            Box::new(Expr::Add(
                Box::new(Expr::Div(
                    Box::new(Expr::Pow(Box::new(Expr::Variable(0)), 2)),
                    Box::new(Expr::Constant(980.0))
                )),
                Box::new(Expr::Div(
                    Box::new(Expr::Pow(Box::new(Expr::Variable(0)), 2)),
                    Box::new(Expr::Variable(1))
                ))
            ))
        )
    }
    

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

    fn dimensionality(&self) -> Dimension {
        match self {
            Expr::Variable(0) => Dimension::Voltage,
            Expr::Variable(1) => Dimension::Resistance,
            Expr::Constant(_) => Dimension::Dimensionless,
            Expr::Variable(_) => Dimension::Invalid,
            
            Expr::Add(a, b) | Expr::Sub(a, b) => {
                if a.dimensionality() == b.dimensionality() {
                    a.dimensionality()
                } else {
                    Dimension::Invalid
                }
            }
            
            Expr::Mul(a, b) => match (a.dimensionality(), b.dimensionality()) {
                (Dimension::Voltage, Dimension::Voltage) => Dimension::Power,
                (Dimension::Power, Dimension::Resistance) => Dimension::Dimensionless,
                (a, b) if a == b => a,
                _ => Dimension::Invalid,
            },
            
            Expr::Div(a, b) => match (a.dimensionality(), b.dimensionality()) {
                (Dimension::Power, Dimension::Resistance) => Dimension::Voltage,
                (a, b) if a == b => Dimension::Dimensionless,
                _ => Dimension::Invalid,
            },
            
            Expr::Pow(base, exp) => match base.dimensionality() {
                Dimension::Voltage if *exp == 2 => Dimension::Power,
                Dimension::Dimensionless => Dimension::Dimensionless,
                _ => Dimension::Invalid,
            },
        }
    }

    fn complexity(&self) -> usize {
        match self {
            Expr::Constant(_) | Expr::Variable(_) => 1,
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                1 + a.complexity() + b.complexity()
            }
            Expr::Pow(b, _) => 1 + b.complexity(),
        }
    }

    fn simplify(self) -> Self {
        match self {
            Expr::Add(a, b) if *a == *b => Expr::Mul(a, Box::new(Expr::Constant(2.0))),
            Expr::Mul(a, b) if *a == Expr::Constant(1.0) => *b,
            Expr::Mul(a, b) if *b == Expr::Constant(1.0) => *a,
            Expr::Pow(b, 1) => *b,
            Expr::Add(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                if sa == sb {
                    Expr::Mul(Box::new(sa), Box::new(Expr::Constant(2.0)))
                } else {
                    Expr::Add(Box::new(sa), Box::new(sb))
                }
            }
            Expr::Sub(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                if sa == sb {
                    Expr::Constant(0.0)
                } else {
                    Expr::Sub(Box::new(sa), Box::new(sb))
                }
            }
            expr => expr,
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
        let mut population: Vec<Expr> = (0..POPULATION_SIZE)
            .into_par_iter()
            .map(|i| {
                let mut _rng = thread_rng();
                if i < POPULATION_SIZE / 10 {
                    Expr::prototype().simplify()
                } else {
                    Expr::random(4).simplify()
                }
            })
            .collect();

            population.par_iter_mut().for_each(|expr| {
                *expr = expr.clone().simplify();
            });

        SymbolicRegressor { population, inputs, outputs }
    }

    fn fitness(&self, expr: &Expr) -> (f64, f64) {
        let (total_error, physics_violations) = (0..self.inputs.nrows())
            .into_par_iter()
            .map(|i| {
                let row = self.inputs.row(i);
                let prediction = expr.evaluate(row.to_slice().unwrap()).clamp(0.0, 1.0);
                let actual = self.outputs[i];
                
                let error = (prediction - actual).powi(2);
                let mut violations = 0.0;

                if expr.dimensionality() != Dimension::Dimensionless {
                    violations += 1.0;
                }
                
                violations += (prediction - prediction.clamp(0.0, 1.0)).abs();

                (error, violations)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        let rmse = total_error.sqrt();
        let physics_score = physics_violations / self.inputs.nrows() as f64;
        
        (rmse, physics_score)
    }

    fn test_fitness(&self, expr: &Expr, test_inputs: &Array2<f64>, test_outputs: &Array1<f64>) -> (f64, f64) {
        let (total_error, physics_violations) = (0..test_inputs.nrows())
            .into_par_iter()
            .map(|i| {
                let row = test_inputs.row(i);
                let prediction = expr.evaluate(row.to_slice().unwrap()).clamp(0.0, 1.0);
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

        let rmse = total_error.sqrt();
        let physics_score = physics_violations / test_inputs.nrows() as f64;
        
        (rmse, physics_score)
    }

    fn evolve(&mut self) {
        let best_fitness = self.pareto_front().first().map(|e| self.fitness(e).0).unwrap_or(1.0);
        
        let new_population: Vec<Expr> = (0..POPULATION_SIZE)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let candidates: Vec<&Expr> = (0..TOURNAMENT_SIZE)
                    .map(|_| &self.population[rng.gen_range(0..self.population.len())])
                    .collect();

                candidates.into_iter()
                    .max_by(|a, b| {
                        let fa = self.fitness(a);
                        let fb = self.fitness(b);
                        (fa.0 * ERROR_WEIGHT + fa.1 * PHYSICS_WEIGHT)
                            .partial_cmp(&(fb.0 * ERROR_WEIGHT + fb.1 * PHYSICS_WEIGHT))
                            .unwrap()
                    })
                    .unwrap()
                    .clone()
            })
            .collect();

        self.population = new_population
            .into_par_iter()
            .map(|mut expr| {
                let mut rng = thread_rng();
                let current_fitness = self.fitness(&expr).0;
                let mutation_rate = if current_fitness > best_fitness * 1.1 {
                    0.4
                } else if current_fitness > best_fitness * 0.9 {
                    0.2
                } else {
                    0.05
                };

                if rng.gen_bool(mutation_rate) {
                    expr = Expr::random(2).simplify();
                }
                expr.simplify()
            })
            .collect();
    }

    fn pareto_front(&self) -> Vec<&Expr> {
        self.population
            .par_iter()
            .filter(|e1| {
                !self.population.par_iter().any(|e2| {
                    let c1 = e1.complexity();
                    let c2 = e2.complexity();
                    let (e1_rmse, e1_phys) = self.fitness(e1);
                    let (e2_rmse, e2_phys) = self.fitness(e2);
                    
                    c2 <= c1 && 
                    e2_rmse <= e1_rmse && 
                    e2_phys <= e1_phys &&
                    (c2 < c1 || e2_rmse < e1_rmse || e2_phys < e1_phys)
                })
            })
            .collect()
    }

    fn best_expression(&self) -> &Expr {
        self.population
            .par_iter()
            .max_by(|a, b| {
                let fa = self.fitness(a);
                let fb = self.fitness(b);
                fa.partial_cmp(&fb).unwrap()
            })
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
        inputs[[i, 0]] = row[0];
        inputs[[i, 1]] = row[1];
        outputs[i] = row[2];
    }

    Ok((inputs, outputs))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    
    let (train_inputs, train_outputs) = parse_data_file("training_points.m")?;
    let (test_inputs, test_outputs) = parse_data_file("testing_points.m")?;
    
    let mut regressor = SymbolicRegressor::new(train_inputs, train_outputs);

    for generation in 0..MAX_GENERATIONS {
        let gen_start = Instant::now();
        regressor.evolve();
        
        let best = regressor.best_expression();
        let (train_rmse, train_phys) = regressor.fitness(best);
        let (test_rmse, test_phys) = regressor.test_fitness(best, &test_inputs, &test_outputs);
        
        println!("Generation {:3}", generation);
        println!("Train RMSE: {:.4} | Physics Violations: {:.4}", train_rmse, train_phys);
        println!("Test RMSE:  {:.4} | Physics Violations: {:.4}", test_rmse, test_phys);
        println!("Best expression: {}", best);
        println!("Generation time: {:.2?}\n", gen_start.elapsed());
    }

    let best = regressor.best_expression();
    let (final_train_rmse, _) = regressor.fitness(best);
    let (final_test_rmse, _) = regressor.test_fitness(best, &test_inputs, &test_outputs);
    
    println!("\nFinal Best Expression: {}", best);
    println!("Final Training RMSE: {:.4}", final_train_rmse);
    println!("Final Testing RMSE:  {:.4}", final_test_rmse);
    println!("Total execution time: {:.2?}", start_time.elapsed());
    
    Ok(())
}