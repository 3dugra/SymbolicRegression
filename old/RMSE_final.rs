use ndarray::{Array1, Array2};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::fmt;
use std::time::{Duration, Instant};

// Constants
const INITIAL_POP: usize = 1000;
const MIN_POP: usize = 1000;
const MAX_POP: usize = 20000;
const MAX_GENERATIONS: usize = 500;
const TARGET_TIME: Duration = Duration::from_secs(250);
const BASE_MUTATION_RATE: f64 = 0.1;
const ERROR_WEIGHT: f64 = 0.5;
const PHYSICS_WEIGHT: f64 = 0.75;
const INITIAL_PHYSICS_RATIO: f64 = 0.8;
const MIN_MUTATION: f64 = 0.05;
const MAX_MUTATION: f64 = 0.1;
const STAGNATION_THRESHOLD: f64 = 0.2;
const STAGNATION_RESPONSE: f64 = 0.5;
// Constants for physics-based expressions
const I_DSS: f64 = 0.438;  // Saturation current (A)
const VP: f64 = 0.4;       // Pinch-off voltage (V)

#[derive(Debug, Clone, PartialEq)]
enum Dimension {
    Voltage,
    Resistance,
    Conductance,
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
    GValue(Box<Expr>),  // Conductance expression
    RValue(Box<Expr>),  // Resistance expression
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Constant(val) => write!(f, "{:.4}", val),
            Expr::Variable(0) => write!(f, "V_offset"),
            Expr::Variable(1) => write!(f, "R"),
            Expr::Variable(_) => write!(f, "?"),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Pow(b, e) => write!(f, "({})^{}", b, e),
            // Add these new match arms
            Expr::GValue(inner) => write!(f, "G({})", inner),
            Expr::RValue(inner) => write!(f, "R({})", inner),
        }
    }
}

impl Expr {
    fn prototype() -> Self {
        const S: f64 = 0.0001;
        
        Expr::Div(
            Box::new(Expr::Div(
                Box::new(Expr::Pow(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Variable(0)),
                        Box::new(Expr::Div(
                            Box::new(Expr::Mul(
                                Box::new(Expr::GValue(Box::new(Expr::Constant(1.0)))),
                                Box::new(Expr::Mul(
                                    Box::new(Expr::Variable(1)),
                                    Box::new(Expr::Pow(Box::new(Expr::Constant(S)), 4))
                                ))
                            )),
                            Box::new(Expr::Add(
                                Box::new(Expr::Mul(
                                    Box::new(Expr::Constant(470.0)),
                                    Box::new(Expr::Constant(S))
                                )),
                                Box::new(Expr::Add(
                                    Box::new(Expr::Mul(
                                        Box::new(Expr::Constant(330.0)),
                                        Box::new(Expr::Mul(
                                            Box::new(Expr::Pow(Box::new(Expr::Constant(S)), 2)),
                                            Box::new(Expr::Add(
                                                Box::new(Expr::GValue(Box::new(Expr::Constant(1.0)))),
                                                Box::new(Expr::Div(
                                                    Box::new(Expr::Constant(1.0)),
                                                    Box::new(Expr::RValue(Box::new(Expr::Constant(1.0))))
                                                ))
                                            ))
                                        ))
                                    )),
                                    Box::new(Expr::Add(
                                        Box::new(Expr::Mul(
                                            Box::new(Expr::Constant(680.0)),
                                            Box::new(Expr::Pow(Box::new(Expr::Constant(S)), 3))
                                        )),
                                        Box::new(Expr::Add(
                                            Box::new(Expr::Mul(
                                                Box::new(Expr::Constant(220.0)),
                                                Box::new(Expr::Mul(
                                                    Box::new(Expr::Pow(Box::new(Expr::Constant(S)), 4)),
                                                    Box::new(Expr::Add(
                                                        Box::new(Expr::GValue(Box::new(Expr::Constant(1.0)))),
                                                        Box::new(Expr::Div(
                                                            Box::new(Expr::Constant(1.0)),
                                                            Box::new(Expr::RValue(Box::new(Expr::Constant(1.0))))
                                                        ))
                                                    ))
                                                ))
                                            )),
                                            Box::new(Expr::Mul(
                                                Box::new(Expr::Constant(100.0)),
                                                Box::new(Expr::Pow(Box::new(Expr::Constant(S)), 5))
                                            ))
                                        ))
                                    ))
                                ))
                            ))
                        ))
                    )),
                    2
                )),
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
                0 => Expr::Add(Box::new(Self::random(depth-1)), Box::new(Self::random(depth-1))),
                1 => Expr::Sub(Box::new(Self::random(depth-1)), Box::new(Self::random(depth-1))),
                2 => Expr::Mul(Box::new(Self::random(depth-1)), Box::new(Self::random(depth-1))),
                3 => Expr::Div(Box::new(Self::random(depth-1)), Box::new(Self::random(depth-1))),
                4 => Expr::Pow(Box::new(Self::random(depth-1)), rng.gen_range(2..4)),
                _ => unreachable!(),
            }
        }
    }

    fn mutate(&self, depth: usize) -> Self {
        let mut rng = thread_rng();
        match rng.gen_range(0..4) {
            0 => Self::random(depth),
            1 => {
                let mut cloned = self.clone();
                if let Some(sub_expr) = cloned.get_random_subtree() {
                    *sub_expr = Box::new(Self::random(depth));
                }
                cloned
            }
            2 => Expr::Add(Box::new(self.clone()), Box::new(Self::random(depth))),
            _ => Expr::Mul(Box::new(self.clone()), Box::new(Self::random(depth))),
        }
    }

    fn get_random_subtree(&mut self) -> Option<&mut Box<Expr>> {
        let mut rng = thread_rng();
        match self {
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                if rng.gen_bool(0.5) { Some(a) } else { Some(b) }
            }
            Expr::Pow(b, _) => Some(b),
            _ => None,
        }
    }

    fn simplify(self) -> Self {
        match self {
            Expr::Add(a, b) if *a == *b => Expr::Mul(a, Box::new(Expr::Constant(2.0))),
            Expr::Mul(a, b) if *a == Expr::Constant(1.0) => *b,
            Expr::Mul(a, b) if *b == Expr::Constant(1.0) => *a,
            Expr::Pow(b, 1) => *b,
            Expr::Add(a, b) => Expr::Add(Box::new(a.simplify()), Box::new(b.simplify())),
            Expr::Sub(a, b) => Expr::Sub(Box::new(a.simplify()), Box::new(b.simplify())),
            _ => self,
        }
    }

    fn evaluate(&self, inputs: &[f64]) -> f64 {
        match self {
            Expr::GValue(v_ds) => {
                let v = v_ds.evaluate(inputs);
                2.0 * I_DSS / VP.powi(2) * v
            }
            Expr::RValue(v_ds) => {
                let v = v_ds.evaluate(inputs);
                VP.powi(2) / (2.0 * I_DSS * (VP + v))
            }
            Expr::Constant(val) => *val,
            Expr::Variable(idx) => inputs[*idx],
            Expr::Add(a, b) => a.evaluate(inputs) + b.evaluate(inputs),
            Expr::Sub(a, b) => a.evaluate(inputs) - b.evaluate(inputs),
            Expr::Mul(a, b) => a.evaluate(inputs) * b.evaluate(inputs),
            Expr::Div(a, b) => {
                let denom = b.evaluate(inputs);
                if denom.abs() < 1e-6 { 1.0 } else { a.evaluate(inputs) / denom }
            }
            Expr::Pow(b, e) => b.evaluate(inputs).powi(*e),
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
                (Dimension::Power, Dimension::Resistance) => Dimension::Voltage,
                (Dimension::Conductance, Dimension::Voltage) => Dimension::Dimensionless,
                (a, b) if a == b => a,
                _ => Dimension::Invalid,
            },
            Expr::Div(a, b) => match (a.dimensionality(), b.dimensionality()) {
                (Dimension::Power, Dimension::Resistance) => Dimension::Voltage,
                (Dimension::Voltage, Dimension::Voltage) => Dimension::Dimensionless,
                (Dimension::Power, Dimension::Voltage) => Dimension::Resistance,
                (a, b) if a == b => Dimension::Dimensionless,
                _ => Dimension::Invalid,
            },
            Expr::Pow(b, e) => match (b.dimensionality(), *e) {
                (Dimension::Voltage, 2) => Dimension::Power,
                (Dimension::Dimensionless, _) => Dimension::Dimensionless,
                _ => Dimension::Invalid,
            },
            Expr::GValue(v_ds) => {
                if v_ds.dimensionality() == Dimension::Voltage {
                    Dimension::Conductance
                } else {
                    Dimension::Invalid
                }
            }
            Expr::RValue(v_ds) => {
                if v_ds.dimensionality() == Dimension::Voltage {
                    Dimension::Resistance
                } else {
                    Dimension::Invalid
                }
            }
        }
    }
}

struct SymbolicRegressor {
    population: Vec<Expr>,
    inputs: Array2<f64>,
    outputs: Array1<f64>,
    pop_size: usize,
    prev_best_rmse: Option<f64>,
    adaptive_weights: (f64, f64),
    previous_best_rmnse: f64,
    stagnation_count: usize,
    generation_times: Vec<Duration>,
}

impl SymbolicRegressor {
    fn new(inputs: Array2<f64>, outputs: Array1<f64>) -> Self {
        let mut pop = Vec::with_capacity(INITIAL_POP);
        let physics_count = (INITIAL_POP as f64 * INITIAL_PHYSICS_RATIO) as usize;
        
        for i in 0..INITIAL_POP {
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
            pop_size: INITIAL_POP,
            prev_best_rmse: None,
            adaptive_weights: (0.7, 0.3),
            previous_best_rmnse: f64::MIN,
            stagnation_count: 0,
            generation_times: Vec::with_capacity(MAX_GENERATIONS),
        }
    }

    fn update_stagnation(&mut self, current_rmse: f64) {
        let improvement = (self.previous_best_rmnse - current_rmse).abs();
        
        if improvement < STAGNATION_THRESHOLD {
            self.stagnation_count += 1;
        } else {
            self.stagnation_count = 0;
            self.previous_best_rmnse = current_rmse;
        }
    }

    fn calculate_mutation_rate(&self, elapsed: Duration) -> f64 {
        let time_ratio = elapsed.as_secs_f64() / TARGET_TIME.as_secs_f64();
        let time_factor = time_ratio.clamp(0.5, 2.0);
        
        let stagnation_factor = STAGNATION_RESPONSE.powi(self.stagnation_count as i32);
        
        (BASE_MUTATION_RATE * time_factor * stagnation_factor)
            .clamp(MIN_MUTATION, MAX_MUTATION)
    }

    
    fn evolve(&mut self) -> Duration {
        let start = Instant::now();
        
        let new_pop: Vec<Expr> = (0..self.pop_size)
            .into_par_iter()
            .map(|_| {
            let mut rng = thread_rng();
            let candidates: Vec<&Expr> = (0..5)
                .map(|_| &self.population[rng.gen_range(0..self.population.len())])
                .collect();
            
            let parent = candidates.iter()
                .max_by(|a, b| {
                    let fa = self.fitness(a);
                    let fb = self.fitness(b);
                    let a_score = fa.0 * ERROR_WEIGHT + fa.1 * PHYSICS_WEIGHT;
                    let b_score = fb.0 * ERROR_WEIGHT + fb.1 * PHYSICS_WEIGHT;
                    b_score.partial_cmp(&a_score).unwrap()
                })
                .unwrap();
            
                let mutation_rate = self.calculate_mutation_rate(start.elapsed());
    
                // Generate candidate with physics-based expressions
                let mut candidate = if rng.gen_bool(mutation_rate) {  // Use mutation_rate here
                    match rng.gen_range(0..2) {
                        0 => Expr::GValue(Box::new(Expr::Variable(0))),
                        _ => Expr::RValue(Box::new(Expr::Variable(0))),
                    }
                } else {
                    (**parent).clone()
                };

            if candidate.dimensionality() != Dimension::Dimensionless {
                let mut attempts = 0;
                while attempts < 5 {
                    candidate = match rng.gen_range(0..3) {
                        0 => Expr::GValue(Box::new(Expr::Variable(0))),
                        1 => Expr::RValue(Box::new(Expr::Variable(0))),
                        _ => Expr::random(3),
                    };
                    if candidate.dimensionality() == Dimension::Dimensionless {
                        break;
                    }
                    attempts += 1;
                }
                if candidate.dimensionality() != Dimension::Dimensionless {
                    candidate = Expr::Constant(0.0);
                }
            }
            
            candidate
        })
        .collect();

        self.population = new_pop;
        let elapsed = start.elapsed();
        self.generation_times.push(elapsed);

        let best = self.best_expression();
        let (rmse, _) = self.fitness(best);
        self.update_stagnation(rmse);

        println!("Current mutation rate: {:.2}%", self.calculate_mutation_rate(elapsed) * 100.0);
        elapsed
    }

    fn adaptive_population(&mut self, last_time: Duration, current_rmse: f64) {
        let time_ratio = last_time.as_secs_f64() / TARGET_TIME.as_secs_f64();
        let time_factor = match time_ratio {
            r if r > 1.0 => r/time_ratio,
            r if r < 1.0 => r*time_ratio,
            _ => 1.0,
        };

        let rmse_factor = match self.prev_best_rmse {
            Some(prev) => {
                let improvement = (prev - current_rmse) / prev;
                match improvement {
                    i if i > 0.05 => 1.2,
                    i if i > 0.0 => 1.0,
                    _ => 0.8,
                }
            }
            None => 1.0,
        };

        let combined_factor = (time_factor * self.adaptive_weights.0) + 
                            (rmse_factor * self.adaptive_weights.1);
        
        self.pop_size = (self.pop_size as f64 * combined_factor)
            .clamp(MIN_POP as f64, MAX_POP as f64) as usize;

        if let Some(prev) = self.prev_best_rmse {
            let trend = (prev - current_rmse).abs() / prev;
            self.adaptive_weights = if trend < 0.01 {
                (0.9, 0.1)
            } else {
                (0.6, 0.4)
            };
        }

        self.prev_best_rmse = Some(current_rmse);
    }

    fn best_expression(&self) -> &Expr {
        self.population
            .par_iter()
            .max_by(|a, b| {
                let fa = self.fitness(a);
                let fb = self.fitness(b);
                let a_score = fa.0 * ERROR_WEIGHT + fa.1 * PHYSICS_WEIGHT;
                let b_score = fb.0 * ERROR_WEIGHT + fb.1 * PHYSICS_WEIGHT;
                b_score.partial_cmp(&a_score).unwrap()
            })
            .unwrap()
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

        let rmse = if total_error > 0.0 { 
            (total_error / self.inputs.nrows() as f64).sqrt() 
        } else { 
            0.0 
        };
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
        // Convert millivolts to volts for input voltage
        inputs[[i, 0]] = row[0] / 1000.0;  // Now in volts
        // Resistance remains in ohms
        inputs[[i, 1]] = row[1];          // Already in ohms
        // Efficiency is dimensionless (0-1 range)
        outputs[i] = row[2];              // Efficiency remains unchanged
    }

    Ok((inputs, outputs))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let (train_inputs, train_outputs) = parse_data_file("training_points.m")?;
    let (test_inputs, test_outputs) = parse_data_file("testing_points.m")?;
    
    let mut regressor = SymbolicRegressor::new(train_inputs, train_outputs);
    let mut gen_times = Vec::new();

    for generation in 0..MAX_GENERATIONS {
        let elapsed = regressor.evolve();
        gen_times.push(elapsed);
        
        // Get best expression and metrics
        let best = regressor.best_expression().clone();
        let (train_rmse, train_phys) = regressor.fitness(&best);
        let (test_rmse, test_phys) = regressor.test_fitness(&best, &test_inputs, &test_outputs);

        // Update adaptive parameters
        regressor.adaptive_population(elapsed, train_rmse);
        regressor.update_stagnation(train_rmse);

        // Performance reporting
        println!("Generation {:3} | Pop: {:5} | Time: {:5.2?}", 
            generation, regressor.pop_size, elapsed);
        println!("Train RMSE: {:.4} | Phys: {:.2} | Test RMSE: {:.4} | Phys: {:.2}", 
            train_rmse, train_phys, test_rmse, test_phys);
        println!("Best Expression: {}", best);

        // Early stopping check
        if gen_times.len() > 10 {
            let avg_time: Duration = gen_times.iter().sum::<Duration>() / gen_times.len() as u32;
            if avg_time > TARGET_TIME * 2 {
                eprintln!("Timeout: Average generation time exceeded");
                break;
            }
        }

        // Print timing statistics
        let avg_gen_time = regressor.generation_times.iter()
                                    .sum::<Duration>() / regressor.generation_times.len() as u32;
        println!("Average generation time: {:.2?}", avg_gen_time);
    }

    // Final reporting
    let total_time = start_time.elapsed();
    println!("\nTotal execution: {:.2?} | Avg gen time: {:.2?}", 
        total_time, total_time / gen_times.len() as u32);
    
    Ok(())
}