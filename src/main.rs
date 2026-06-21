// Haplo の CLI エントリポイントとエラー統合。
// run() 関数がテスト可能な純粋関数として字句解析→構文解析→評価のパイプラインを担い、
// main() はファイル読み込みと標準出力への表示のみを担当する。
mod ast;
mod autodiff;
mod interpreter;
mod lexer;
mod parser;
mod shape_stage;
mod value;

use value::Value;

// パイプライン全体のエラーを一つの型に統合する。
// From トレイトを実装することで ? 演算子でエラーを自動変換できる。
// 代替: anyhow や thiserror クレートを使う方法があるが、
// 外部依存を最小にするために手書きを選んだ。P0 のエラー種類は少ないので手間は小さい。
#[derive(Debug)]
enum HaploError {
    Lex(lexer::LexError),
    Parse(parser::ParseError),
    Shape(shape_stage::ShapeError),
    Eval(value::EvalError),
    Io(std::io::Error),
}

impl std::fmt::Display for HaploError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HaploError::Lex(e) => write!(f, "字句解析エラー: {}", e),
            HaploError::Parse(e) => write!(f, "構文解析エラー: {}", e),
            HaploError::Shape(e) => write!(f, "shape 検査エラー: {}", e),
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
impl From<shape_stage::ShapeError> for HaploError {
    fn from(e: shape_stage::ShapeError) -> Self {
        HaploError::Shape(e)
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

// ソース文字列を受け取り、評価結果の Value を返す純粋な関数。
// ファイル読み込みや標準出力を行わないため、テストから直接呼べる。
// パイプラインは: lex() → parse() → shape_eval_program() → eval_program() の4段。
// shape_eval_program は P2 で追加した「実行前ゲート」。eval の前に shape 不整合
// （行列積の内次元不一致・要素ごと演算の shape 不一致）を静的に検出して弾く。
// shape を推論できない箇所は Unknown を伝播させ、正しいプログラムは素通しする
// （偽陽性ゼロ方針）。? 演算子で各段のエラーを HaploError に変換しながら伝播させる。
pub fn run(source: &str) -> Result<Value, HaploError> {
    let tokens = lexer::lex(source)?;
    let program = parser::parse(&tokens)?;
    shape_stage::shape_eval_program(&program)?;
    let val = interpreter::eval_program(&program)?;
    Ok(val)
}

// 実行ファイルのエントリポイント。
// コマンドライン引数からファイルパスを取得し、読み込んで run() に渡す。
// 成功時は結果を println! で表示（IO はここだけで行う）。
// エラー時は eprintln! で標準エラーに出力し、終了コード 1 で終了する。
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

    // P2: shape 検査が run() のパイプラインに組み込まれ、不整合を「評価前」に弾くことを確認する。
    // a(2×3) @ b(2×2) は内次元 3≠2 で行列積できない。eval まで進めば実行時エラーになるが、
    // shape_eval_program が先に Shape エラーで止めるはず（G4 = 実行前検出）。
    #[test]
    fn integration_g4_shape_mismatch_rejected_before_eval() {
        let src = "
a = [1.0, 2.0, 3.0; 4.0, 5.0, 6.0]
b = [1.0, 2.0; 3.0, 4.0]
main = a @ b
";
        match run(src) {
            Err(HaploError::Shape(_)) => {} // 期待どおり shape 段で弾かれた
            other => panic!("shape エラーで弾かれるはず: {:?}", other),
        }
    }

    // P2: 北極星プログラム（学習ループ）が shape 検査を偽陽性なく通過し、最後まで実行できること。
    // zeros 由来の Unknown が随所に伝播するが確定した矛盾は無いので、shape 段を素通りして
    // eval が学習後の重み Tensor[3] を返す。staging が正しいプログラムを壊さない最重要回帰。
    #[test]
    fn integration_g4_linreg_passes_shape_check() {
        let src = std::fs::read_to_string("examples/linreg_train.hpl")
            .expect("examples/linreg_train.hpl が読めません");
        match run(&src) {
            Ok(Value::Tensor(t)) => assert_eq!(t.shape(), &[3]),
            other => panic!("Tensor[3] を期待: {:?}", other),
        }
    }

    // 北極星プログラム（examples/linreg_train.hpl）を、ソース文字列ではなく
    // 「実ファイルから読み込んで」end-to-end で実行する統合テスト。
    // interpreter.rs 側の g3_* テストはインライン文字列だが、こちらは配布する
    // サンプルファイルが実際に壊れていないことまで保証する（例が腐らないようにする）。
    // run() は lex→parse→eval の全段を通すので、ファイル1本で P1 全機能の通し確認になる。
    #[test]
    fn integration_g3_linreg_example_file() {
        // CARGO のテストはクレートルートが作業ディレクトリなので相対パスで読める。
        let src = std::fs::read_to_string("examples/linreg_train.hpl")
            .expect("examples/linreg_train.hpl が読めません");
        let val = run(&src).unwrap();
        match val {
            // main は学習後の重み Tensor[3]。形と有限性だけ確認する
            // （具体的な数値は g3_linreg_converges 側で損失減少として検証済み）。
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[3]);
                assert!(t.iter().all(|x| x.is_finite()));
            }
            other => panic!("Tensor[3] を期待: {:?}", other),
        }
    }
}
