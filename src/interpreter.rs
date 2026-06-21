// Haplo のツリーウォーキングインタプリタ（tree-walking interpreter）。
// AST（抽象構文木）のノードを再帰的に評価して Value を返す。
//
// 「ツリーウォーキング」を選んだ理由:
// バイトコードコンパイラ方式（AST → 命令列 → VM）は高速だが複雑になる。
// P1 で追加予定の autodiff テープ（計算グラフ）は、eval() の呼び出し順に
// 演算を記録することで実装できるため、ツリーウォーキングの方が自然に統合できる。
// バイトコード方式では eval の中間状態をテープに記録するフックが難しくなる。
use ndarray::{ArrayD, IxDyn};
use std::rc::Rc;

use crate::ast::*;
use crate::value::*;

// プログラム全体を評価するエントリポイント。
// トップレベル定義からグローバル環境を構築し、main を評価して返す。
// main がなければ EvalError::NoMain を返す。
pub fn eval_program(program: &Program) -> Result<Value, EvalError> {
    let env = build_global_env(program)?;
    env.lookup("main").ok_or(EvalError::NoMain)
}

// トップレベルの Binding を上から順に評価し、グローバル環境を構築する。
// 評価順: ファイルの記述順（前から後）。
// 各束縛は、それより前に定義された束縛だけが見える環境の中で評価される。
// これは「順序依存の線形スコープ」であり、前方参照はできない。
//
// 相互再帰（例: f が g を呼び、g が f を呼ぶ）は P0 では未対応。
// 対応するには2パス（まず名前だけ登録 → 次に本体を評価）が必要で、P1 で実装予定。
//
// TypeAnnotation はここで読み飛ばす（P0 では型検査をしないため）。
fn build_global_env(program: &Program) -> Result<Env, EvalError> {
    let mut env = load_builtins(Env::empty());
    for item in program {
        if let TopLevel::Binding { name, params, body } = item {
            let val = if params.is_empty() {
                eval(body, &env)?
            } else {
                let lambda = desugar_lambda(params, body);
                eval(&lambda, &env)?
            };
            env = env.extend(name.clone(), val);
        }
        // TypeAnnotation は P0 では無視
    }
    Ok(env)
}

// 組み込み関数を Value::Builtin として環境に注入する。
// ユーザ定義関数と同じ仕組み（環境内の名前束縛）で提供することで、
// eval() の App ブランチで特別扱いせずに済む。
// 組み込みもユーザ定義関数と同じように |> で使えるのはこの設計のおかげ。
// 代替: eval() 内で組み込み名を特別にマッチする方法もあるが、
// 組み込みを追加するたびに eval() を修正する必要が生じ、拡張性が低い。
fn load_builtins(env: Env) -> Env {
    let builtins: &[(&str, BuiltinFn)] = &[
        ("sum", BuiltinFn::Sum),
        ("mean", BuiltinFn::Mean),
        ("exp", BuiltinFn::Exp),
        ("log", BuiltinFn::Log),
        ("tanh", BuiltinFn::Tanh),
        ("sqrt", BuiltinFn::Sqrt),
        ("zeros", BuiltinFn::Zeros),
        ("ones", BuiltinFn::Ones),
        ("transpose", BuiltinFn::Transpose),
        ("reshape", BuiltinFn::Reshape),
    ];
    let mut e = env;
    for (name, f) in builtins {
        e = e.extend(name.to_string(), Value::Builtin(*f));
    }
    e
}

// 多引数関数定義をネストしたラムダ式に変換する（カリー化）。
// "f x y = body" → Lambda{x, Lambda{y, body}}
//
// 実装: params を逆順（rev()）にしてから fold する。
// fold は左から畳み込むため、params = [x, y] の場合:
//   rev → [y, x]
//   fold 初期値 body:
//     1回目: Lambda{y, body}
//     2回目: Lambda{x, Lambda{y, body}}  ← これが欲しい形
// rev() なしで fold すると Lambda{y, Lambda{x, body}} になってしまい、
// 引数の順序が逆になる。
fn desugar_lambda(params: &[String], body: &Expr) -> Expr {
    params.iter().rev().fold(body.clone(), |acc, param| {
        Expr::Lambda {
            param: param.clone(),
            body: Box::new(acc),
        }
    })
}

pub fn eval(expr: &Expr, env: &Env) -> Result<Value, EvalError> {
    match expr {
        Expr::Lit(Literal::Int(n)) => Ok(Value::Int(*n)),
        Expr::Lit(Literal::Float(x)) => Ok(Value::Float(*x)),
        Expr::Lit(Literal::Bool(b)) => Ok(Value::Bool(*b)),

        Expr::Var(name) => env
            .lookup(name)
            .ok_or_else(|| EvalError::UnboundVariable(name.clone())),

        Expr::UnaryMinus(e) => match eval(e, env)? {
            Value::Int(n) => Ok(Value::Int(-n)),
            Value::Float(x) => Ok(Value::Float(-x)),
            Value::Tensor(t) => Ok(Value::Tensor(Rc::new(t.mapv(|x| -x)))),
            other => Err(EvalError::TypeMismatch {
                expected: "数値",
                got: value_type_name(&other),
            }),
        },

        Expr::BinOp { op, left, right } => {
            let l = eval(left, env)?;
            let r = eval(right, env)?;
            eval_binop(op, l, r)
        }

        Expr::App(func, arg) => {
            let fval = eval(func, env)?;
            let aval = eval(arg, env)?;
            apply(fval, aval)
        }

        // ラムダ式の評価: AST ノードをクロージャ（Value::Closure）に変換する。
        // 実行はしない。現在の環境を env フィールドにキャプチャするだけ。
        // 呼び出し（apply）は Expr::App の評価時に行われる。
        Expr::Lambda { param, body } => Ok(Value::Closure(Closure {
            param: param.clone(),
            body: *body.clone(),
            env: env.clone(),
        })),

        // let 式の評価。value を先に評価して val を得てから、
        // name を val に束縛した新しい環境 new_env を作り、body を評価する。
        // params がある場合（`let f x = ...`）は desugar_lambda でラムダに変換してから評価。
        // 元の環境 env は変更されないため、body の外では name は見えない（レキシカルスコープ）。
        Expr::Let {
            name,
            params,
            value,
            body,
        } => {
            let val = if params.is_empty() {
                eval(value, env)?
            } else {
                let lambda = desugar_lambda(params, value);
                eval(&lambda, env)?
            };
            let new_env = env.extend(name.clone(), val);
            eval(body, &new_env)
        }

        Expr::If { cond, then, else_ } => match eval(cond, env)? {
            Value::Bool(true) => eval(then, env),
            Value::Bool(false) => eval(else_, env),
            other => Err(EvalError::TypeMismatch {
                expected: "Bool",
                got: value_type_name(&other),
            }),
        },

        Expr::TensorLit(rows) => eval_tensor_lit(rows, env),

        Expr::Pipe(left, right) => {
            // a |> f  ≡  f a
            let aval = eval(left, env)?;
            let fval = eval(right, env)?;
            apply(fval, aval)
        }
    }
}

// 関数値 f を引数 arg に適用する。
// Closure の場合: param を arg に束縛した新しい環境で body を評価する。
// Builtin の場合: apply_builtin に委譲する。
// それ以外の値（Int, Float 等）を適用しようとした場合はエラー。
fn apply(f: Value, arg: Value) -> Result<Value, EvalError> {
    match f {
        Value::Closure(c) => {
            let new_env = c.env.extend(c.param.clone(), arg);
            eval(&c.body, &new_env)
        }
        Value::Builtin(b) => apply_builtin(b, arg),
        other => Err(EvalError::TypeMismatch {
            expected: "関数",
            got: value_type_name(&other),
        }),
    }
}

// 組み込み関数の実装。各関数は Rust の標準ライブラリや ndarray に委譲する。
// テンソルに対する mapv() は要素ごとに関数を適用して新しいテンソルを返す。
// Rc を使っているため、元のテンソルはコピーされる（不変性を保つ）。
fn apply_builtin(b: BuiltinFn, arg: Value) -> Result<Value, EvalError> {
    match b {
        BuiltinFn::Sum => {
            let t = coerce_to_tensor(arg)?;
            Ok(Value::Float(t.sum()))
        }
        BuiltinFn::Mean => {
            let t = coerce_to_tensor(arg)?;
            let n = t.len() as f64;
            if n == 0.0 {
                return Err(EvalError::InvalidArgument(
                    "mean: 空のテンソル".to_string(),
                ));
            }
            Ok(Value::Float(t.sum() / n))
        }
        BuiltinFn::Exp => match arg {
            Value::Float(x) => Ok(Value::Float(x.exp())),
            Value::Int(n) => Ok(Value::Float((n as f64).exp())),
            Value::Tensor(t) => Ok(Value::Tensor(Rc::new(t.mapv(f64::exp)))),
            other => Err(EvalError::TypeMismatch {
                expected: "数値またはテンソル",
                got: value_type_name(&other),
            }),
        },
        BuiltinFn::Log => match arg {
            Value::Float(x) => Ok(Value::Float(x.ln())),
            Value::Int(n) => Ok(Value::Float((n as f64).ln())),
            Value::Tensor(t) => Ok(Value::Tensor(Rc::new(t.mapv(f64::ln)))),
            other => Err(EvalError::TypeMismatch {
                expected: "数値またはテンソル",
                got: value_type_name(&other),
            }),
        },
        BuiltinFn::Tanh => match arg {
            Value::Float(x) => Ok(Value::Float(x.tanh())),
            Value::Int(n) => Ok(Value::Float((n as f64).tanh())),
            Value::Tensor(t) => Ok(Value::Tensor(Rc::new(t.mapv(f64::tanh)))),
            other => Err(EvalError::TypeMismatch {
                expected: "数値またはテンソル",
                got: value_type_name(&other),
            }),
        },
        BuiltinFn::Sqrt => match arg {
            Value::Float(x) => Ok(Value::Float(x.sqrt())),
            Value::Int(n) => Ok(Value::Float((n as f64).sqrt())),
            Value::Tensor(t) => Ok(Value::Tensor(Rc::new(t.mapv(f64::sqrt)))),
            other => Err(EvalError::TypeMismatch {
                expected: "数値またはテンソル",
                got: value_type_name(&other),
            }),
        },
        BuiltinFn::Zeros => {
            let shape = extract_shape(arg)?;
            let arr = ArrayD::<f64>::zeros(IxDyn(&shape));
            Ok(Value::Tensor(Rc::new(arr)))
        }
        BuiltinFn::Ones => {
            let shape = extract_shape(arg)?;
            let arr = ArrayD::<f64>::ones(IxDyn(&shape));
            Ok(Value::Tensor(Rc::new(arr)))
        }
        BuiltinFn::Transpose => {
            let t = coerce_to_tensor(arg)?;
            if t.ndim() != 2 {
                return Err(EvalError::TensorWrongRank {
                    op: "transpose",
                    expected: 2,
                    got: t.ndim(),
                });
            }
            let transposed = t.t().to_owned();
            Ok(Value::Tensor(Rc::new(transposed.into_dyn())))
        }
        BuiltinFn::Reshape => {
            // reshape はカリー化（1引数ずつ）: reshape tensor shape
            // 最初の引数（テンソル）をクロージャの環境にキャプチャして返すが、
            // 2番目の引数（shape）を受け取る処理はダミー実装。
            // __reshape_applied__ は環境に存在しないため呼び出すと UnboundVariable エラー。
            // P0 では reshape は未完成。P1 以降で正式対応する。
            Ok(Value::Closure(Closure {
                param: "__shape__".to_string(),
                body: Expr::Var("__reshape_applied__".to_string()), // dummy
                env: Env::empty().extend("__tensor__".to_string(), arg),
            }))
        }
    }
}

fn extract_shape(v: Value) -> Result<Vec<usize>, EvalError> {
    match v {
        Value::Tensor(t) => {
            // shape を1Dテンソルの整数値から取り出す
            let flat: Vec<usize> = t
                .iter()
                .map(|&x| x as usize)
                .collect();
            Ok(flat)
        }
        Value::Int(n) => Ok(vec![n as usize]),
        other => Err(EvalError::InvalidArgument(format!(
            "shape は Int またはテンソルである必要があります、got: {:?}",
            value_type_name(&other)
        ))),
    }
}

// 二項演算子の評価。左辺・右辺の型の組み合わせで振る舞いが変わる。
// パターンマッチの順序が重要: 具体的なケース（Int×Int）を先に書き、
// 汎用的なフォールバック（Float 昇格）を後に書く。
//
// 型昇格のルール:
//   Int × Int → Int（整数演算。5/2=2）
//   Tensor × Tensor → Tensor（shape が一致する場合のみ）
//   Tensor × スカラー → Tensor（全要素にスカラーを適用）
//   それ以外 → Float に昇格して演算
fn eval_binop(op: &BinOpKind, l: Value, r: Value) -> Result<Value, EvalError> {
    match (op, &l, &r) {
        // Int × Int 算術
        (BinOpKind::Add, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
        (BinOpKind::Sub, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
        (BinOpKind::Mul, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
        (BinOpKind::Div, Value::Int(a), Value::Int(b)) => {
            if *b == 0 {
                Err(EvalError::DivisionByZero)
            } else {
                Ok(Value::Int(a / b))
            }
        }
        (BinOpKind::Pow, Value::Int(a), Value::Int(b)) if *b >= 0 => {
            Ok(Value::Int(a.pow(*b as u32)))
        }

        // @ は行列積演算子（Python の PEP 465 由来）。
        // ndarray の dot() を使うが、dot() は型が静的に決まっている（Array2.dot(Array2) 等）。
        // ArrayD（動的ランク）から必要なランクの view を取り出すには into_dimensionality() を使う。
        // 対応ケース:
        //   2D × 2D → 2D（行列 × 行列）
        //   2D × 1D → 1D（行列 × ベクトル）: 線形回帰の feats @ w に必要
        // テンソル演算
        (BinOpKind::MatMul, Value::Tensor(a), Value::Tensor(b)) => {
            use ndarray::{Ix1, Ix2};
            match (a.ndim(), b.ndim()) {
                (2, 2) => {
                    let a2 = a.view().into_dimensionality::<Ix2>().unwrap();
                    let b2 = b.view().into_dimensionality::<Ix2>().unwrap();
                    if a2.shape()[1] != b2.shape()[0] {
                        return Err(EvalError::TensorShapeMismatch {
                            op: "@",
                            a: a2.shape().to_vec(),
                            b: b2.shape().to_vec(),
                        });
                    }
                    Ok(Value::Tensor(Rc::new(a2.dot(&b2).into_dyn())))
                }
                (2, 1) => {
                    // 行列 × ベクトル → ベクトル
                    let a2 = a.view().into_dimensionality::<Ix2>().unwrap();
                    let b1 = b.view().into_dimensionality::<Ix1>().unwrap();
                    if a2.shape()[1] != b1.len() {
                        return Err(EvalError::TensorShapeMismatch {
                            op: "@",
                            a: a2.shape().to_vec(),
                            b: b1.shape().to_vec(),
                        });
                    }
                    Ok(Value::Tensor(Rc::new(a2.dot(&b1).into_dyn())))
                }
                (da, db) => Err(EvalError::InvalidArgument(format!(
                    "@ は 2D×2D または 2D×1D のみ対応（{da}D × {db}D は未対応）"
                ))),
            }
        }

        // テンソル要素ごと演算
        (BinOpKind::Add, Value::Tensor(a), Value::Tensor(b)) => {
            check_shape_match("+", a, b)?;
            Ok(Value::Tensor(Rc::new((**a).clone() + &**b)))
        }
        (BinOpKind::Sub, Value::Tensor(a), Value::Tensor(b)) => {
            check_shape_match("-", a, b)?;
            Ok(Value::Tensor(Rc::new((**a).clone() - &**b)))
        }
        (BinOpKind::Mul, Value::Tensor(a), Value::Tensor(b)) => {
            check_shape_match("*", a, b)?;
            Ok(Value::Tensor(Rc::new((**a).clone() * &**b)))
        }
        (BinOpKind::Div, Value::Tensor(a), Value::Tensor(b)) => {
            check_shape_match("/", a, b)?;
            Ok(Value::Tensor(Rc::new((**a).clone() / &**b)))
        }
        (BinOpKind::Pow, Value::Tensor(a), Value::Tensor(b)) => {
            check_shape_match("^", a, b)?;
            let c = ndarray::Zip::from(a.as_ref())
                .and(b.as_ref())
                .map_collect(|&x, &y| x.powf(y));
            Ok(Value::Tensor(Rc::new(c)))
        }

        // テンソルとスカラーの演算は、ndarray の ScalarOperand トレイトで処理する。
        // Rust はアドホック多相を持たないため、Tensor×Float と Tensor×Int を
        // 別々のパターンとして列挙する必要がある（マクロで簡略化も可能だが可読性優先）。
        // s - tensor（スカラーが左）の場合は ndarray が直接サポートしないため、
        // ArrayD::from_elem でスカラーをブロードキャストしたテンソルを作成して減算する。
        // テンソル × スカラー ブロードキャスト
        (BinOpKind::Add, Value::Tensor(a), Value::Float(s)) => {
            Ok(Value::Tensor(Rc::new((**a).clone() + *s)))
        }
        (BinOpKind::Add, Value::Float(s), Value::Tensor(a)) => {
            Ok(Value::Tensor(Rc::new(*s + (**a).clone())))
        }
        (BinOpKind::Sub, Value::Tensor(a), Value::Float(s)) => {
            Ok(Value::Tensor(Rc::new((**a).clone() - *s)))
        }
        (BinOpKind::Sub, Value::Float(s), Value::Tensor(a)) => {
            Ok(Value::Tensor(Rc::new(ArrayD::from_elem(a.shape(), *s) - &**a)))
        }
        (BinOpKind::Mul, Value::Tensor(a), Value::Float(s)) => {
            Ok(Value::Tensor(Rc::new((**a).clone() * *s)))
        }
        (BinOpKind::Mul, Value::Float(s), Value::Tensor(a)) => {
            Ok(Value::Tensor(Rc::new(*s * (**a).clone())))
        }
        (BinOpKind::Div, Value::Tensor(a), Value::Float(s)) => {
            Ok(Value::Tensor(Rc::new((**a).clone() / *s)))
        }
        (BinOpKind::Pow, Value::Tensor(a), Value::Float(s)) => {
            Ok(Value::Tensor(Rc::new(a.mapv(|x| x.powf(*s)))))
        }
        (BinOpKind::Pow, Value::Tensor(a), Value::Int(n)) => {
            let exp = *n as f64;
            Ok(Value::Tensor(Rc::new(a.mapv(|x| x.powf(exp)))))
        }

        // テンソル × Int スカラー（Int を f64 に昇格）
        (BinOpKind::Add, Value::Tensor(a), Value::Int(n)) => {
            Ok(Value::Tensor(Rc::new((**a).clone() + *n as f64)))
        }
        (BinOpKind::Add, Value::Int(n), Value::Tensor(a)) => {
            Ok(Value::Tensor(Rc::new(*n as f64 + (**a).clone())))
        }
        (BinOpKind::Sub, Value::Tensor(a), Value::Int(n)) => {
            Ok(Value::Tensor(Rc::new((**a).clone() - *n as f64)))
        }
        (BinOpKind::Mul, Value::Tensor(a), Value::Int(n)) => {
            Ok(Value::Tensor(Rc::new((**a).clone() * *n as f64)))
        }
        (BinOpKind::Mul, Value::Int(n), Value::Tensor(a)) => {
            Ok(Value::Tensor(Rc::new(*n as f64 * (**a).clone())))
        }
        (BinOpKind::Div, Value::Tensor(a), Value::Int(n)) => {
            Ok(Value::Tensor(Rc::new((**a).clone() / *n as f64)))
        }

        // Float 算術（Int を昇格）
        (
            BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul | BinOpKind::Div | BinOpKind::Pow,
            _,
            _,
        ) => {
            let a = coerce_to_float(l)?;
            let b = coerce_to_float(r)?;
            let result = match op {
                BinOpKind::Add => a + b,
                BinOpKind::Sub => a - b,
                BinOpKind::Mul => a * b,
                BinOpKind::Div => {
                    if b == 0.0 {
                        return Err(EvalError::DivisionByZero);
                    }
                    a / b
                }
                BinOpKind::Pow => a.powf(b),
                _ => unreachable!(),
            };
            Ok(Value::Float(result))
        }

        // 比較
        (BinOpKind::Eq, _, _) => Ok(Value::Bool(values_equal(&l, &r)?)),
        (BinOpKind::Ne, _, _) => Ok(Value::Bool(!values_equal(&l, &r)?)),
        (BinOpKind::Lt, _, _) => Ok(Value::Bool(compare_numeric(&l, &r)? < 0.0)),
        (BinOpKind::Le, _, _) => Ok(Value::Bool(compare_numeric(&l, &r)? <= 0.0)),
        (BinOpKind::Gt, _, _) => Ok(Value::Bool(compare_numeric(&l, &r)? > 0.0)),
        (BinOpKind::Ge, _, _) => Ok(Value::Bool(compare_numeric(&l, &r)? >= 0.0)),

        _ => Err(EvalError::TypeMismatch {
            expected: "互換性のある型",
            got: "非互換な型の組み合わせ",
        }),
    }
}

// テンソル同士の要素ごと演算の前に shape が一致しているか確認する。
// 不一致の場合は演算子名と両辺の shape を含むエラーを返す。
// ndarray 自体も shape 不一致でパニックするが、事前チェックで分かりやすいメッセージを出す。
fn check_shape_match(
    op: &'static str,
    a: &ArrayD<f64>,
    b: &ArrayD<f64>,
) -> Result<(), EvalError> {
    if a.shape() != b.shape() {
        Err(EvalError::TensorShapeMismatch {
            op,
            a: a.shape().to_vec(),
            b: b.shape().to_vec(),
        })
    } else {
        Ok(())
    }
}

fn values_equal(a: &Value, b: &Value) -> Result<bool, EvalError> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(x == y),
        (Value::Float(x), Value::Float(y)) => Ok(x == y),
        (Value::Int(x), Value::Float(y)) => Ok((*x as f64) == *y),
        (Value::Float(x), Value::Int(y)) => Ok(*x == (*y as f64)),
        (Value::Bool(x), Value::Bool(y)) => Ok(x == y),
        _ => Err(EvalError::TypeMismatch {
            expected: "比較可能な型",
            got: "比較不可能な型",
        }),
    }
}

// 2つの数値の大小を「差（a - b）」として返す。
// 正ならa>b、負ならa<b、0ならa==b。
// 比較演算子（<, <=, >, >=）を一つの関数で処理できるため、
// 同じ型チェックロジックを4回書かずに済む。
fn compare_numeric(a: &Value, b: &Value) -> Result<f64, EvalError> {
    let x = match a {
        Value::Int(n) => *n as f64,
        Value::Float(x) => *x,
        _ => {
            return Err(EvalError::TypeMismatch {
                expected: "数値",
                got: value_type_name(a),
            })
        }
    };
    let y = match b {
        Value::Int(n) => *n as f64,
        Value::Float(y) => *y,
        _ => {
            return Err(EvalError::TypeMismatch {
                expected: "数値",
                got: value_type_name(b),
            })
        }
    };
    Ok(x - y)
}

// テンソルリテラルの AST ノード（Vec<Vec<Expr>>）を評価して Tensor に変換する。
// 処理の流れ:
//   1. 全要素を eval() → coerce_to_float() で f64 に変換
//   2. 全行の長さが同じか確認（非均一ならエラー）
//   3. 行数が1なら Array1（1D）、複数行なら Array2（2D）を作成
//   4. into_dyn() で ArrayD に変換し、Rc でくるんで返す
//
// 行数で 1D/2D を切り替える理由:
// [1.0, 2.0] は 1D ベクトル（shape [2]）であってほしい。
// もし常に Array2 を作ると shape [1, 2] になり、
// ベクトルとして使う演算（matmul の右辺など）でランク不一致エラーになる。
fn eval_tensor_lit(rows: &[Vec<Expr>], env: &Env) -> Result<Value, EvalError> {
    if rows.is_empty() || (rows.len() == 1 && rows[0].is_empty()) {
        // 空テンソル
        let arr = ArrayD::<f64>::zeros(IxDyn(&[0]));
        return Ok(Value::Tensor(Rc::new(arr)));
    }

    let evaluated: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|e| coerce_to_float(eval(e, env)?))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let ncols = evaluated[0].len();
    if evaluated.iter().any(|r| r.len() != ncols) {
        return Err(EvalError::TensorNonUniform);
    }
    let nrows = evaluated.len();

    if nrows == 1 {
        // 1D テンソル
        let flat: Vec<f64> = evaluated.into_iter().flatten().collect();
        let arr = ndarray::Array1::from_vec(flat).into_dyn();
        Ok(Value::Tensor(Rc::new(arr)))
    } else {
        // 2D テンソル
        let flat: Vec<f64> = evaluated.into_iter().flatten().collect();
        let arr = ndarray::Array2::from_shape_vec((nrows, ncols), flat)
            .map_err(|_| EvalError::TensorNonUniform)?
            .into_dyn();
        Ok(Value::Tensor(Rc::new(arr)))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    fn run(src: &str) -> Value {
        let tokens = lex(src).expect("lex error");
        let program = parse(&tokens).expect("parse error");
        eval_program(&program).expect("eval error")
    }

    fn run_f(src: &str) -> f64 {
        match run(src) {
            Value::Float(x) => x,
            Value::Int(n) => n as f64,
            other => panic!("expected float, got {:?}", other),
        }
    }

    // ----- G0: スカラー電卓 -----

    #[test]
    fn g0_literal_int() {
        assert!(matches!(run("main = 42"), Value::Int(42)));
    }

    #[test]
    fn g0_literal_float() {
        let v = run("main = 3.14");
        match v {
            Value::Float(x) => assert!((x - 3.14).abs() < 1e-9),
            _ => panic!(),
        }
    }

    #[test]
    fn g0_arith() {
        assert!(matches!(run("main = 2 + 3 * 4"), Value::Int(14)));
    }

    #[test]
    fn g0_sub() {
        assert!(matches!(run("main = 10 - 3"), Value::Int(7)));
    }

    #[test]
    fn g0_div() {
        assert!(matches!(run("main = 10 / 2"), Value::Int(5)));
    }

    #[test]
    fn g0_pow() {
        assert!(matches!(run("main = 2 ^ 10"), Value::Int(1024)));
    }

    #[test]
    fn g0_bool() {
        assert!(matches!(run("main = true"), Value::Bool(true)));
    }

    #[test]
    fn g0_if_true() {
        assert!(matches!(run("main = if true then 1 else 0"), Value::Int(1)));
    }

    #[test]
    fn g0_if_false() {
        assert!(matches!(
            run("main = if false then 1 else 0"),
            Value::Int(0)
        ));
    }

    #[test]
    fn g0_let() {
        assert!(matches!(
            run("main = let x = 3 in x * x"),
            Value::Int(9)
        ));
    }

    #[test]
    fn g0_fn() {
        assert!(matches!(run("f x = x + 1\nmain = f 3"), Value::Int(4)));
    }

    #[test]
    fn g0_multiarg_fn() {
        assert!(matches!(
            run("add x y = x + y\nmain = add 2 3"),
            Value::Int(5)
        ));
    }

    #[test]
    fn g0_comparison() {
        assert!(matches!(run("main = 3 > 2"), Value::Bool(true)));
        assert!(matches!(run("main = 1 == 1"), Value::Bool(true)));
        assert!(matches!(run("main = 1 != 2"), Value::Bool(true)));
    }

    #[test]
    fn g0_nested_fn() {
        let src = "
double x = x * 2
add1 x = x + 1
main = double (add1 3)
";
        assert!(matches!(run(src), Value::Int(8)));
    }

    #[test]
    fn g0_unary_minus() {
        assert!(matches!(run("main = -5"), Value::Int(-5)));
    }

    // ----- G1: テンソル電卓 -----

    #[test]
    fn g1_tensor_1d() {
        let v = run("main = [1.0, 2.0, 3.0]");
        match v {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[3]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn g1_tensor_2d() {
        let v = run("main = [1.0, 2.0; 3.0, 4.0]");
        match v {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn g1_tensor_add() {
        let v = run("main = [1.0, 2.0] + [3.0, 4.0]");
        match v {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2]);
                assert!((t[[0]] - 4.0).abs() < 1e-9);
                assert!((t[[1]] - 6.0).abs() < 1e-9);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn g1_matmul_identity() {
        let src = "
a = [1.0, 0.0; 0.0, 1.0]
main = a @ a
";
        match run(src) {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                assert!((t[[0, 0]] - 1.0).abs() < 1e-9);
                assert!((t[[0, 1]]).abs() < 1e-9);
                assert!((t[[1, 0]]).abs() < 1e-9);
                assert!((t[[1, 1]] - 1.0).abs() < 1e-9);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn g1_sum() {
        let x = run_f("main = sum [1.0, 2.0, 3.0]");
        assert!((x - 6.0).abs() < 1e-9);
    }

    #[test]
    fn g1_mean() {
        let x = run_f("main = mean [2.0, 4.0]");
        assert!((x - 3.0).abs() < 1e-9);
    }

    #[test]
    fn g1_pipe_sum() {
        let x = run_f("main = [1.0, 2.0, 3.0] |> sum");
        assert!((x - 6.0).abs() < 1e-9);
    }

    #[test]
    fn g1_scalar_broadcast() {
        // テンソル + スカラー
        let v = run("main = [1.0, 2.0, 3.0] + 1.0");
        match v {
            Value::Tensor(t) => {
                assert!((t[[0]] - 2.0).abs() < 1e-9);
                assert!((t[[2]] - 4.0).abs() < 1e-9);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn g1_tensor_pow_scalar() {
        let v = run("main = [1.0, 2.0, 3.0] ^ 2");
        match v {
            Value::Tensor(t) => {
                assert!((t[[0]] - 1.0).abs() < 1e-9);
                assert!((t[[1]] - 4.0).abs() < 1e-9);
                assert!((t[[2]] - 9.0).abs() < 1e-9);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn g1_zeros() {
        let v = run("main = zeros [3]");
        match v {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[3]);
                assert!(t.iter().all(|&x| x == 0.0));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn g1_exp() {
        let x = run_f("main = exp 0.0");
        assert!((x - 1.0).abs() < 1e-9);
    }
}
