use ndarray::ArrayD;
use std::rc::Rc;

use crate::ast::Expr;

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Tensor(Rc<ArrayD<f64>>),
    Closure(Closure),
    Builtin(BuiltinFn),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BuiltinFn {
    Sum,
    Mean,
    Exp,
    Log,
    Tanh,
    Sqrt,
    Zeros,
    Ones,
    Transpose,
    Reshape,
}

#[derive(Debug, Clone)]
pub struct Closure {
    pub param: String,
    pub body: Expr,
    pub env: Env,
}

/// 永続連結リストによる環境（Rc で安価にクローン）
#[derive(Debug, Clone)]
pub struct Env(pub Option<Rc<EnvNode>>);

#[derive(Debug)]
pub struct EnvNode {
    pub name: String,
    pub value: Value,
    pub parent: Env,
}

impl Env {
    pub fn empty() -> Self {
        Env(None)
    }

    pub fn extend(&self, name: String, value: Value) -> Self {
        Env(Some(Rc::new(EnvNode {
            name,
            value,
            parent: self.clone(),
        })))
    }

    pub fn lookup(&self, name: &str) -> Option<Value> {
        let mut cur = &self.0;
        while let Some(node) = cur {
            if node.name == name {
                return Some(node.value.clone());
            }
            cur = &node.parent.0;
        }
        None
    }
}

#[derive(Debug)]
pub enum EvalError {
    UnboundVariable(String),
    TypeMismatch {
        expected: &'static str,
        got: &'static str,
    },
    DivisionByZero,
    TensorShapeMismatch {
        op: &'static str,
        a: Vec<usize>,
        b: Vec<usize>,
    },
    TensorNonUniform,
    TensorWrongRank {
        op: &'static str,
        expected: usize,
        got: usize,
    },
    ArityMismatch,
    NoMain,
    InvalidArgument(String),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::UnboundVariable(n) => write!(f, "未定義の変数: {}", n),
            EvalError::TypeMismatch { expected, got } => {
                write!(f, "型不一致: {} が必要ですが {} でした", expected, got)
            }
            EvalError::DivisionByZero => write!(f, "ゼロ除算"),
            EvalError::TensorShapeMismatch { op, a, b } => {
                write!(
                    f,
                    "演算子 `{}` の shape 不一致: {:?} と {:?}",
                    op, a, b
                )
            }
            EvalError::TensorNonUniform => {
                write!(f, "テンソルリテラルの行の長さが揃っていません")
            }
            EvalError::TensorWrongRank { op, expected, got } => {
                write!(
                    f,
                    "演算子 `{}`: {}次元テンソルが必要ですが {}次元でした",
                    op, expected, got
                )
            }
            EvalError::ArityMismatch => write!(f, "関数でない値を適用しようとしました"),
            EvalError::NoMain => write!(f, "`main` の定義が見つかりません"),
            EvalError::InvalidArgument(msg) => write!(f, "引数エラー: {}", msg),
        }
    }
}

fn value_type_name(v: &Value) -> &'static str {
    match v {
        Value::Int(_) => "Int",
        Value::Float(_) => "Float",
        Value::Bool(_) => "Bool",
        Value::Tensor(_) => "Tensor",
        Value::Closure(_) => "Closure",
        Value::Builtin(_) => "Builtin",
    }
}

pub fn coerce_to_float(v: Value) -> Result<f64, EvalError> {
    match v {
        Value::Float(x) => Ok(x),
        Value::Int(n) => Ok(n as f64),
        other => Err(EvalError::TypeMismatch {
            expected: "Float または Int",
            got: value_type_name(&other),
        }),
    }
}

pub fn coerce_to_tensor(v: Value) -> Result<Rc<ArrayD<f64>>, EvalError> {
    match v {
        Value::Tensor(t) => Ok(t),
        other => Err(EvalError::TypeMismatch {
            expected: "Tensor",
            got: value_type_name(&other),
        }),
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(x) => {
                if x.fract() == 0.0 && x.abs() < 1e15 {
                    write!(f, "{:.1}", x)
                } else {
                    write!(f, "{}", x)
                }
            }
            Value::Bool(b) => write!(f, "{}", b),
            Value::Tensor(t) => write!(f, "{}", t),
            Value::Closure(_) => write!(f, "<closure>"),
            Value::Builtin(b) => write!(f, "<builtin:{:?}>", b),
        }
    }
}
