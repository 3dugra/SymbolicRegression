use std::fmt;
use rand::{thread_rng, Rng};

// Constants for physics-based expressions
pub const I_DSS: f64 = 0.438;  // Saturation current (A)
pub const VP: f64 = 0.4;       // Pinch-off voltage (V)
pub const MAX_RECURSION_DEPTH: usize = 100;

#[derive(Debug, Clone, PartialEq)]
pub enum Dimension {
    Voltage,
    Resistance,
    Conductance,
    Power,
    Dimensionless,
    Invalid,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Constant(f64),
    Variable(usize),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, i32),
    GValue(Box<Expr>),
    RValue(Box<Expr>),
}

impl Expr {
    pub fn prototype() -> Self {
        Expr::Div(
            Box::new(Expr::Mul(
                Box::new(Expr::Constant(2.0)),  // First constant
                Box::new(Expr::Pow(
                    Box::new(Expr::Variable(0)),  // V_offset
                    2  // Square power
                ))
            )),
            Box::new(Expr::Mul(
                Box::new(Expr::Variable(1)),  // R
                Box::new(Expr::Constant(1.0))  // Second constant
            ))
        )
    }

    pub fn random(depth: usize) -> Self {
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

    pub fn mutate(&self, depth: usize) -> Self {
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

    pub fn get_random_subtree(&mut self) -> Option<&mut Box<Expr>> {
        let mut rng = thread_rng();
        match self {
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                if rng.gen_bool(0.5) { Some(a) } else { Some(b) }
            }
            Expr::Pow(b, _) => Some(b),
            _ => None,
        }
    }

    pub fn simplify(self) -> Self {
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

    // Improvement 2: Implement early stopping for expression evaluation
    pub fn evaluate(&self, inputs: &[f64]) -> f64 {
        self.evaluate_with_depth(inputs, 0)
    }

    pub fn evaluate_with_depth(&self, inputs: &[f64], depth: usize) -> f64 {
        if depth > MAX_RECURSION_DEPTH {
            return 0.0;
        }
        
        let result = match self {
            Expr::GValue(v_ds) => {
                let v = v_ds.evaluate_with_depth(inputs, depth + 1);
                2.0 * I_DSS / VP.powi(2) * v
            },
            Expr::RValue(v_ds) => {
                let v = v_ds.evaluate_with_depth(inputs, depth + 1);
                VP.powi(2) / (2.0 * I_DSS * (VP + v))
            },
            Expr::Constant(val) => *val,
            Expr::Variable(idx) => inputs[*idx],
            Expr::Add(a, b) => a.evaluate_with_depth(inputs, depth + 1) + b.evaluate_with_depth(inputs, depth + 1),
            Expr::Sub(a, b) => a.evaluate_with_depth(inputs, depth + 1) - b.evaluate_with_depth(inputs, depth + 1),
            Expr::Mul(a, b) => a.evaluate_with_depth(inputs, depth + 1) * b.evaluate_with_depth(inputs, depth + 1),
            Expr::Div(a, b) => {
                let denom = b.evaluate_with_depth(inputs, depth + 1);
                if denom.abs() < 1e-10 {
                    0.0  // Return 0 instead of infinity
                } else {
                    a.evaluate_with_depth(inputs, depth + 1) / denom
                }
            },
            Expr::Pow(b, e) => b.evaluate_with_depth(inputs, depth + 1).powi(*e),
        };
        
        // Add numerical stability checks
        if !result.is_finite() {
            0.0
        } else {
            result.clamp(-1e6, 1e6)  // Prevent extreme values
        }
    }

    pub fn dimensionality(&self) -> Dimension {
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

    pub fn expr_complexity(&self) -> usize {
        match self {
            Expr::Constant(_) | Expr::Variable(_) => 1,
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => 
                1 + a.expr_complexity() + b.expr_complexity(),
            Expr::Pow(b, _) => 1 + b.expr_complexity(),
            Expr::GValue(v) | Expr::RValue(v) => 1 + v.expr_complexity(),
        }
    }
}

// Add Display implementation for Expr
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
            Expr::GValue(inner) => write!(f, "G({})", inner),
            Expr::RValue(inner) => write!(f, "R({})", inner),
        }
    }
}


