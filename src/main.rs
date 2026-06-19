mod ast;
mod interpreter;
mod lexer;
mod parser;
mod value;

use value::Value;

#[derive(Debug)]
enum HaploError {
    Lex(lexer::LexError),
    Parse(parser::ParseError),
    Eval(value::EvalError),
    Io(std::io::Error),
}

impl std::fmt::Display for HaploError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HaploError::Lex(e) => write!(f, "字句解析エラー: {}", e),
            HaploError::Parse(e) => write!(f, "構文解析エラー: {}", e),
            HaploError::Eval(e) => write!(f, "評価エラー: {}", e),
            HaploError::Io(e) => write!(f, "IO エラー: {}", e),
        }
    }
}

impl From<lexer::LexError> for HaploError {
    fn from(e: lexer::LexError) -> Self {
        HaploError::Lex(e)
    }
}
impl From<parser::ParseError> for HaploError {
    fn from(e: parser::ParseError) -> Self {
        HaploError::Parse(e)
    }
}
impl From<value::EvalError> for HaploError {
    fn from(e: value::EvalError) -> Self {
        HaploError::Eval(e)
    }
}
impl From<std::io::Error> for HaploError {
    fn from(e: std::io::Error) -> Self {
        HaploError::Io(e)
    }
}

pub fn run(source: &str) -> Result<Value, HaploError> {
    let tokens = lexer::lex(source)?;
    let program = parser::parse(&tokens)?;
    let val = interpreter::eval_program(&program)?;
    Ok(val)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = if args.len() >= 2 {
        let source = std::fs::read_to_string(&args[1]).map_err(HaploError::from);
        source.and_then(|s| run(&s))
    } else {
        eprintln!("使い方: haplo <file.hpl>");
        std::process::exit(1);
    };

    match result {
        Ok(val) => println!("{}", val),
        Err(e) => {
            eprintln!("エラー: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integration_g0_canonical() {
        let val = run("main = 2 + 3 * 4").unwrap();
        assert_eq!(val.to_string(), "14");
    }

    #[test]
    fn integration_g0_fn_chain() {
        let src = "
f x = x + 1
g x = x * 2
main = g (f 3)
";
        assert_eq!(run(src).unwrap().to_string(), "8");
    }

    #[test]
    fn integration_g1_matmul_identity() {
        let src = "
a = [1.0, 0.0; 0.0, 1.0]
main = a @ a
";
        let val = run(src).unwrap();
        match val {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                assert!((t[[0, 0]] - 1.0).abs() < 1e-9);
                assert!((t[[1, 1]] - 1.0).abs() < 1e-9);
            }
            _ => panic!("expected tensor"),
        }
    }

    #[test]
    fn integration_g1_sum_pipe() {
        let val = run("main = [1.0, 2.0, 3.0] |> sum").unwrap();
        match val {
            Value::Float(x) => assert!((x - 6.0).abs() < 1e-9),
            _ => panic!(),
        }
    }
}
