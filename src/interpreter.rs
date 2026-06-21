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
use crate::autodiff;
use crate::value::*;

// プログラム全体を評価するエントリポイント。
// トップレベル定義からグローバル環境を構築し、main を評価して返す。
// main がなければ EvalError::NoMain を返す。
pub fn eval_program(program: &Program) -> Result<Value, EvalError> {
    let env = build_global_env(program)?;
    env.lookup("main").ok_or(EvalError::NoMain)
}

// トップレベル定義からグローバル環境を two-pass で構築する。
//
// pass1: 関数定義（params あり）をクロージャ化して globals に登録する。
//   クロージャは共有 globals マップをキャプチャするので、後から登録される定義も
//   呼び出し時に解決できる。これにより前方参照と相互再帰が動く
//   （例: f が後で定義される g を呼ぶ、f と g が互いを呼ぶ）。
// pass2: 値定義（params なし）をソース順に評価して globals に登録する。
//   値の評価は即時なので、後続の値定義は参照できない（前方参照は関数のみ）。
//   ただし pass1 で全関数が登録済みなので、値定義から任意の関数は参照できる。
//
// TypeAnnotation はここで読み飛ばす（P0/P1 では型検査をしないため）。
fn build_global_env(program: &Program) -> Result<Env, EvalError> {
    let env = load_builtins(Env::empty());

    // pass1: 関数定義をクロージャ化（本体はまだ実行しない）。
    for item in program {
        if let TopLevel::Binding { name, params, body } = item {
            if !params.is_empty() {
                let lambda = desugar_lambda(params, body);
                let cl = eval(&lambda, &env)?;
                env.define_global(name.clone(), cl);
            }
        }
    }

    // pass2: 値定義をソース順に評価。
    for item in program {
        if let TopLevel::Binding { name, params, body } = item {
            if params.is_empty() {
                let val = eval(body, &env)?;
                env.define_global(name.clone(), val);
            }
        }
    }

    Ok(env)
}

// 組み込み関数を Value::Builtin として共有 globals マップに注入する。
// ユーザ定義関数と同じ仕組み（環境内の名前束縛）で提供することで、
// eval() の App ブランチで特別扱いせずに済む。
// 組み込みもユーザ定義関数と同じように |> で使えるのはこの設計のおかげ。
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
        ("grad", BuiltinFn::Grad),
        ("iterate", BuiltinFn::Iterate),
    ];
    for (name, f) in builtins {
        env.define_global(name.to_string(), Value::Builtin(*f));
    }
    env
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
// shape_stage（P2）も同じカリー化規則を使うため pub(crate) で共有する
// （規則がずれると「eval は通るが shape_eval は別物」になりかねないため一本化）。
pub(crate) fn desugar_lambda(params: &[String], body: &Expr) -> Expr {
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
            // grad 評価中の微分追跡値はテープへ符号反転を記録する。
            Value::Tracked(id) => Ok(Value::Tracked(autodiff::neg(id))),
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

// 組み込み関数の引数個数（arity）。
// 引数がそろうまで PartialBuiltin で部分適用を貯める。
// shape_stage（P2）も同じ arity で部分適用を処理するため pub(crate) で共有する。
pub(crate) fn builtin_arity(b: BuiltinFn) -> usize {
    match b {
        BuiltinFn::Reshape | BuiltinFn::Grad => 2,
        BuiltinFn::Iterate => 3,
        _ => 1,
    }
}

// 関数値 f を引数 arg に適用する。
// Closure の場合: param を arg に束縛した新しい環境で body を評価する。
// Builtin の場合: arity に達していれば実行、未達なら PartialBuiltin に貯める。
// それ以外の値（Int, Float 等）を適用しようとした場合はエラー。
fn apply(f: Value, arg: Value) -> Result<Value, EvalError> {
    match f {
        Value::Closure(c) => {
            let new_env = c.env.extend(c.param.clone(), arg);
            eval(&c.body, &new_env)
        }
        Value::Builtin(b) => {
            if builtin_arity(b) == 1 {
                apply_builtin(b, vec![arg])
            } else {
                Ok(Value::PartialBuiltin(b, vec![arg]))
            }
        }
        Value::PartialBuiltin(b, mut args) => {
            args.push(arg);
            if args.len() == builtin_arity(b) {
                apply_builtin(b, args)
            } else {
                Ok(Value::PartialBuiltin(b, args))
            }
        }
        other => Err(EvalError::TypeMismatch {
            expected: "関数",
            got: value_type_name(&other),
        }),
    }
}

// 組み込み関数の実装。引数は arity 個そろった状態で渡される。
// テンソルに対する mapv() は要素ごとに関数を適用して新しいテンソルを返す。
// Rc を使っているため、元のテンソルはコピーされる（不変性を保つ）。
//
// grad の評価中、単項組み込み（exp/log/tanh/sqrt/sum/mean）に Tracked 値が渡されると
// テープへ演算を記録する必要がある。先頭でその分岐を処理する。
fn apply_builtin(b: BuiltinFn, args: Vec<Value>) -> Result<Value, EvalError> {
    // Tracked 引数の単項組み込みはテープへ記録する。
    if args.len() == 1 {
        if let Value::Tracked(id) = args[0] {
            let nid = match b {
                BuiltinFn::Exp => autodiff::exp(id),
                BuiltinFn::Log => autodiff::log(id),
                BuiltinFn::Tanh => autodiff::tanh(id),
                BuiltinFn::Sqrt => autodiff::sqrt(id),
                BuiltinFn::Sum => autodiff::sum(id),
                BuiltinFn::Mean => autodiff::mean(id),
                _ => {
                    return Err(EvalError::InvalidArgument(format!(
                        "組み込み {:?} は微分追跡値に適用できません",
                        b
                    )))
                }
            };
            return Ok(Value::Tracked(nid));
        }
    }

    match b {
        BuiltinFn::Sum => {
            let t = coerce_to_tensor(args[0].clone())?;
            Ok(Value::Float(t.sum()))
        }
        BuiltinFn::Mean => {
            let t = coerce_to_tensor(args[0].clone())?;
            let n = t.len() as f64;
            if n == 0.0 {
                return Err(EvalError::InvalidArgument(
                    "mean: 空のテンソル".to_string(),
                ));
            }
            Ok(Value::Float(t.sum() / n))
        }
        BuiltinFn::Exp => match args[0].clone() {
            Value::Float(x) => Ok(Value::Float(x.exp())),
            Value::Int(n) => Ok(Value::Float((n as f64).exp())),
            Value::Tensor(t) => Ok(Value::Tensor(Rc::new(t.mapv(f64::exp)))),
            other => Err(EvalError::TypeMismatch {
                expected: "数値またはテンソル",
                got: value_type_name(&other),
            }),
        },
        BuiltinFn::Log => match args[0].clone() {
            Value::Float(x) => Ok(Value::Float(x.ln())),
            Value::Int(n) => Ok(Value::Float((n as f64).ln())),
            Value::Tensor(t) => Ok(Value::Tensor(Rc::new(t.mapv(f64::ln)))),
            other => Err(EvalError::TypeMismatch {
                expected: "数値またはテンソル",
                got: value_type_name(&other),
            }),
        },
        BuiltinFn::Tanh => match args[0].clone() {
            Value::Float(x) => Ok(Value::Float(x.tanh())),
            Value::Int(n) => Ok(Value::Float((n as f64).tanh())),
            Value::Tensor(t) => Ok(Value::Tensor(Rc::new(t.mapv(f64::tanh)))),
            other => Err(EvalError::TypeMismatch {
                expected: "数値またはテンソル",
                got: value_type_name(&other),
            }),
        },
        BuiltinFn::Sqrt => match args[0].clone() {
            Value::Float(x) => Ok(Value::Float(x.sqrt())),
            Value::Int(n) => Ok(Value::Float((n as f64).sqrt())),
            Value::Tensor(t) => Ok(Value::Tensor(Rc::new(t.mapv(f64::sqrt)))),
            other => Err(EvalError::TypeMismatch {
                expected: "数値またはテンソル",
                got: value_type_name(&other),
            }),
        },
        BuiltinFn::Zeros => {
            let shape = extract_shape(args[0].clone())?;
            let arr = ArrayD::<f64>::zeros(IxDyn(&shape));
            Ok(Value::Tensor(Rc::new(arr)))
        }
        BuiltinFn::Ones => {
            let shape = extract_shape(args[0].clone())?;
            let arr = ArrayD::<f64>::ones(IxDyn(&shape));
            Ok(Value::Tensor(Rc::new(arr)))
        }
        BuiltinFn::Transpose => {
            let t = coerce_to_tensor(args[0].clone())?;
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
        // reshape tensor shape: テンソルの要素を新しい shape へ並べ替える。
        // 要素数が一致しないとエラー。shape は Int またはテンソル（整数値の列）で指定する。
        BuiltinFn::Reshape => {
            let t = coerce_to_tensor(args[0].clone())?;
            let shape = extract_shape(args[1].clone())?;
            let total: usize = shape.iter().product();
            if total != t.len() {
                return Err(EvalError::InvalidArgument(format!(
                    "reshape: 要素数 {} を shape {:?}（要素数 {}）に変形できません",
                    t.len(),
                    shape,
                    total
                )));
            }
            let data: Vec<f64> = t.iter().copied().collect();
            let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
                .map_err(|_| EvalError::InvalidArgument("reshape: 変形に失敗".to_string()))?;
            Ok(Value::Tensor(Rc::new(arr)))
        }
        BuiltinFn::Grad => builtin_grad(args),
        BuiltinFn::Iterate => builtin_iterate(args),
    }
}

// grad f x: スカラー値関数 f の x における勾配（x と同 shape のテンソル）を返す。
// 手順:
//   1. テープを開始し、x を葉ノードとして記録する。
//   2. f を Tracked 入力に適用して loss を評価（演算がテープに積まれる）。
//   3. 出力ノードから backward して入力ノードの随伴を取り出す。
// f が入力に依存しない（結果が Tracked でない）場合は勾配 0 を返す。
fn builtin_grad(args: Vec<Value>) -> Result<Value, EvalError> {
    let f = args[0].clone();
    let x = coerce_to_tensor(args[1].clone())?;

    autodiff::tape_begin();
    let input_id = autodiff::leaf((*x).clone());
    let result = apply(f, Value::Tracked(input_id));

    let grad_arr = match result {
        Ok(Value::Tracked(out_id)) => {
            let grads = autodiff::backward(out_id);
            grads[input_id].clone()
        }
        // f が入力 x を使わなかった（定数）→ 勾配は 0。
        Ok(_) => ArrayD::<f64>::zeros(x.raw_dim()),
        Err(e) => {
            autodiff::tape_end();
            return Err(e);
        }
    };
    autodiff::tape_end();

    Ok(Value::Tensor(Rc::new(grad_arr)))
}

// iterate init n f: init に f を n 回適用した結果を返す。
//   iterate a 0 f = a
//   iterate a n f = iterate (f a) (n-1) f
// 反復は再帰ではなくループで実装する（深い再帰によるスタック消費を避ける）。
fn builtin_iterate(args: Vec<Value>) -> Result<Value, EvalError> {
    let init = args[0].clone();
    let n = match &args[1] {
        Value::Int(k) => *k,
        other => {
            return Err(EvalError::TypeMismatch {
                expected: "Int（反復回数）",
                got: value_type_name(other),
            })
        }
    };
    let f = args[2].clone();

    let mut acc = init;
    for _ in 0..n.max(0) {
        acc = apply(f.clone(), acc)?;
    }
    Ok(acc)
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
    // grad の評価中、いずれかのオペランドが微分追跡値ならテープへ記録する。
    // 定数側は leaf ノードに持ち上げてから演算ノードを積む。
    if matches!(l, Value::Tracked(_)) || matches!(r, Value::Tracked(_)) {
        return eval_binop_tracked(op, l, r);
    }
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

        // スカラーが左の Div / Pow ブロードキャスト（s / t, s ^ t）。
        // ndarray は scalar / Array を直接サポートしないため、要素ごとに計算する。
        // Add/Sub/Mul はスカラー左右どちらも対応済みだが、Div/Pow はここで補う
        // （sigmoid の 1.0 / (1.0 + exp(-x)) などを SPEC どおり書けるようにするため）。
        (BinOpKind::Div, Value::Float(s), Value::Tensor(a)) => {
            Ok(Value::Tensor(Rc::new(a.mapv(|x| *s / x))))
        }
        (BinOpKind::Div, Value::Int(n), Value::Tensor(a)) => {
            let s = *n as f64;
            Ok(Value::Tensor(Rc::new(a.mapv(|x| s / x))))
        }
        (BinOpKind::Pow, Value::Float(s), Value::Tensor(a)) => {
            Ok(Value::Tensor(Rc::new(a.mapv(|x| s.powf(x)))))
        }
        (BinOpKind::Pow, Value::Int(n), Value::Tensor(a)) => {
            let s = *n as f64;
            Ok(Value::Tensor(Rc::new(a.mapv(|x| s.powf(x)))))
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

// 微分追跡値を含む二項演算をテープへ記録する。
// 算術・行列積のみ対応（比較は微分不可）。定数オペランドは to_node で leaf に
// 持ち上げてから autodiff::binop で演算ノードを積み、結果を Tracked で返す。
fn eval_binop_tracked(op: &BinOpKind, l: Value, r: Value) -> Result<Value, EvalError> {
    match op {
        BinOpKind::Add
        | BinOpKind::Sub
        | BinOpKind::Mul
        | BinOpKind::Div
        | BinOpKind::Pow
        | BinOpKind::MatMul => {
            let a = to_node(&l)?;
            let b = to_node(&r)?;
            let id = autodiff::binop(op, a, b)?;
            Ok(Value::Tracked(id))
        }
        _ => Err(EvalError::InvalidArgument(
            "微分追跡値に比較演算子は使えません".to_string(),
        )),
    }
}

// Value をテープの葉ノード index に変換する。
// すでに Tracked ならその index を、定数（テンソル/スカラー）なら leaf を積んで返す。
// スカラーは autodiff::scalar で 0 次元 ArrayD にする。
fn to_node(v: &Value) -> Result<usize, EvalError> {
    match v {
        Value::Tracked(id) => Ok(*id),
        Value::Tensor(t) => Ok(autodiff::leaf((**t).clone())),
        Value::Float(x) => Ok(autodiff::leaf(autodiff::scalar(*x))),
        Value::Int(n) => Ok(autodiff::leaf(autodiff::scalar(*n as f64))),
        other => Err(EvalError::TypeMismatch {
            expected: "数値またはテンソル",
            got: value_type_name(other),
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
        Value::PartialBuiltin(_, _) => "Builtin",
        Value::Tracked(_) => "Tracked",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    // Haplo ソース文字列を lex → parse → eval まで通し、main の評価結果を返すヘルパ。
    // テストでは「言語として動くか」を end-to-end で見たいので、パイプライン全段を通す。
    // 各段のエラーは expect で即パニックさせる（テストなので失敗箇所が分かれば十分）。
    fn run(src: &str) -> Value {
        let tokens = lex(src).expect("lex error");
        let program = parse(&tokens).expect("parse error");
        eval_program(&program).expect("eval error")
    }

    // run の結果がスカラーであることを前提に f64 を取り出すヘルパ。
    // Int も Float に昇格して返すので、`6`（Int）と `6.0`（Float）の両方を許容できる。
    // テンソルやクロージャが返ってきたらテストの想定違いなのでパニックさせる。
    fn run_f(src: &str) -> f64 {
        match run(src) {
            Value::Float(x) => x,
            Value::Int(n) => n as f64,
            other => panic!("expected float, got {:?}", other),
        }
    }

    // run の結果がテンソルであることを前提に ArrayD<f64> を取り出すヘルパ（P1 で追加）。
    // grad/iterate/reshape の戻り値はテンソルなので、要素を添字でアサートしたいときに使う。
    // Rc の中身を clone して所有権を取り、t[[i]] のような添字アクセスを書きやすくする。
    fn run_t(src: &str) -> ArrayD<f64> {
        match run(src) {
            Value::Tensor(t) => (*t).clone(),
            other => panic!("expected tensor, got {:?}", other),
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

    // ----- P1: 前方参照・相互再帰（two-pass スコープ） -----
    // P0 の env はソース順の線形スコープで前方参照ができなかった。P1 で Env に共有
    // globals マップを足し、build_global_env を「関数→値」の two-pass にしたことで、
    // 関数は順不同に互いを参照できるようになった。以下はその回帰テスト。

    #[test]
    fn p1_forward_reference() {
        // main が f より「前」に書かれていても f を解決できることを確認する。
        // pass2 で main を評価する時点では、pass1 で f が既に globals に登録済み。
        // P0 ではここで UnboundVariable になっていた。
        let src = "main = f 3\nf x = x + 1";
        assert!(matches!(run(src), Value::Int(4)));
    }

    #[test]
    fn p1_mutual_recursion() {
        // isEven と isOdd が互いを呼ぶ相互再帰。両方とも関数なので pass1 で
        // 同じ globals にクロージャ登録され、呼び出し時に相手を解決できる（knot-tying）。
        // isEven 10 → isOdd 9 → … → isEven 0 → true、と 10 段たどって true になる。
        let src = "
isEven n = if n == 0 then true else isOdd (n - 1)
isOdd n = if n == 0 then false else isEven (n - 1)
main = isEven 10
";
        assert!(matches!(run(src), Value::Bool(true)));
    }

    #[test]
    fn p1_value_can_reference_later_function() {
        // 値定義 a（`a = g 4`）が、後で定義される関数 g を参照できることを確認する。
        // pass1 で全関数を登録してから pass2 で値を評価するので、値→関数の前方参照は OK。
        // 逆に「値→後続の値」は pass2 が即時評価なので不可（これは仕様として許容）。
        // g 4 = 4*4 = 16。
        let src = "a = g 4\ng x = x * x\nmain = a";
        assert!(matches!(run(src), Value::Int(16)));
    }

    // ----- P1: iterate -----
    // iterate init n f = f を init に n 回適用した結果（builtin_iterate、ループ実装）。

    #[test]
    fn p1_iterate_scalar() {
        // 0 に inc(=+1) を 5 回 → 5。多引数組み込み（arity 3）の部分適用も経由する:
        // `iterate 0` → PartialBuiltin、`... 5` → PartialBuiltin、`... inc` で arity 到達 → 実行。
        let src = "inc x = x + 1\nmain = iterate 0 5 inc";
        assert!(matches!(run(src), Value::Int(5)));
    }

    #[test]
    fn p1_iterate_zero_times() {
        // n=0 のときは f を一度も適用せず init をそのまま返す（境界条件）。
        let src = "inc x = x + 1\nmain = iterate 42 0 inc";
        assert!(matches!(run(src), Value::Int(42)));
    }

    #[test]
    fn p1_iterate_tensor() {
        // init がテンソルでも動く（線形回帰の重み更新と同じ形）。
        // zeros [2]=[0,0] に「+1.0」を 3 回 → [3,3]。テンソル+スカラーのブロードキャストも経由。
        let src = "bump v = v + 1.0\nmain = iterate (zeros [2]) 3 bump";
        let t = run_t(src);
        assert_eq!(t.shape(), &[2]);
        assert!((t[[0]] - 3.0).abs() < 1e-9);
    }

    // ----- P1: reshape -----
    // P0 ではダミー実装（呼ぶと未定義変数エラー）だったものを本実装に置換した。

    #[test]
    fn p1_reshape() {
        // 長さ4の1Dを 2x2 に変形。要素は row-major（行優先）で詰められる:
        // [1,2,3,4] → [[1,2],[3,4]]。先頭 [0,0]=1、末尾 [1,1]=4 を確認。
        let t = run_t("main = reshape [1.0, 2.0, 3.0, 4.0] [2, 2]");
        assert_eq!(t.shape(), &[2, 2]);
        assert!((t[[0, 0]] - 1.0).abs() < 1e-9);
        assert!((t[[1, 1]] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn p1_reshape_size_mismatch_errors() {
        // 要素数3を 2x2(=4) に変形しようとするとエラーになることを確認する。
        // run() は expect でパニックするため、ここでは run() を使わず eval_program の
        // Result を直接受け取り is_err() を見る（エラーパスのテスト）。
        let tokens = crate::lexer::lex("main = reshape [1.0, 2.0, 3.0] [2, 2]").unwrap();
        let program = crate::parser::parse(&tokens).unwrap();
        assert!(eval_program(&program).is_err());
    }

    // ----- P1: スカラー左 Div / Pow ブロードキャスト -----
    // P0 では Div/Pow はテンソルが左のときだけ対応していた（`t / s` は可、`s / t` は不可）。
    // sigmoid の `1.0 / (1.0 + exp(-x))` のようにスカラーが左に来る式を書けるよう補った。

    #[test]
    fn p1_scalar_div_tensor() {
        // 1.0 / [1,2,4] = [1, 0.5, 0.25]（要素ごとに s/x を計算）。
        let t = run_t("main = 1.0 / [1.0, 2.0, 4.0]");
        assert!((t[[0]] - 1.0).abs() < 1e-9);
        assert!((t[[1]] - 0.5).abs() < 1e-9);
        assert!((t[[2]] - 0.25).abs() < 1e-9);
    }

    #[test]
    fn p1_scalar_pow_tensor() {
        // 2.0 ^ [1,2,3] = [2, 4, 8]（要素ごとに s^x を計算）。
        let t = run_t("main = 2.0 ^ [1.0, 2.0, 3.0]");
        assert!((t[[0]] - 2.0).abs() < 1e-9);
        assert!((t[[1]] - 4.0).abs() < 1e-9);
        assert!((t[[2]] - 8.0).abs() < 1e-9);
    }

    // ----- P1: grad（reverse-mode autodiff） -----
    // grad f x は f の x における勾配（x と同 shape のテンソル）を返す。
    // 各テストは手計算できる関数を使い、autodiff の結果が解析解と一致するか検証する。
    // これが合えば forward 計算とテープ backward の両方が正しいことの強い証拠になる。

    #[test]
    fn p1_grad_sum_square() {
        // f(w) = Σ w_i^2。各成分の偏微分は ∂f/∂w_i = 2 w_i なので、勾配は 2w。
        // w=[3,-2,5] → [6,-4,10]。Mul（w*w）と Sum の backward を通る。
        let t = run_t("f w = sum (w * w)\nmain = grad f [3.0, -2.0, 5.0]");
        assert!((t[[0]] - 6.0).abs() < 1e-9);
        assert!((t[[1]] + 4.0).abs() < 1e-9); // -(-4)=+4 で 0 に近いことを確認
        assert!((t[[2]] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn p1_grad_mean() {
        // f(w) = mean(w) = (Σ w_i)/n。∂f/∂w_i = 1/n（全成分一定）。
        // n=4 なので勾配は全要素 0.25。Mean の backward（随伴を 1/n で配る）を検証。
        let t = run_t("f w = mean w\nmain = grad f [1.0, 2.0, 3.0, 4.0]");
        assert_eq!(t.shape(), &[4]);
        for i in 0..4 {
            assert!((t[[i]] - 0.25).abs() < 1e-9);
        }
    }

    #[test]
    fn p1_grad_pow() {
        // f(w) = Σ w_i^2 を `w ^ 2`（Pow、定数指数）で書いた版。勾配は 2w。
        // w=[1,2,3] → [2,4,6]。Pow の backward（g * b * a^(b-1)）を検証する。
        let t = run_t("f w = sum (w ^ 2)\nmain = grad f [1.0, 2.0, 3.0]");
        assert!((t[[0]] - 2.0).abs() < 1e-9);
        assert!((t[[1]] - 4.0).abs() < 1e-9);
        assert!((t[[2]] - 6.0).abs() < 1e-9);
    }

    #[test]
    fn p1_grad_matmul() {
        // f(w) = Σ (A @ w)。A@w の i 成分は Σ_j A[i,j] w_j なので、
        // ∂f/∂w_j = Σ_i A[i,j] = A の第 j 列の和（= A^T @ ones）。
        // A=[[1,2],[3,4]] → 列和 [1+3, 2+4] = [4,6]。MatMul(2D×1D) の backward を検証。
        let src = "a = [1.0, 2.0; 3.0, 4.0]\nf w = sum (a @ w)\nmain = grad f [1.0, 1.0]";
        let t = run_t(src);
        assert!((t[[0]] - 4.0).abs() < 1e-9); // 1+3
        assert!((t[[1]] - 6.0).abs() < 1e-9); // 2+4
    }

    #[test]
    fn p1_grad_constant_is_zero() {
        // f(w) = 5.0 は入力 w に依存しない。このとき loss は Tracked にならないので、
        // builtin_grad は「x と同 shape のゼロ」を返す（勾配 0）。その経路を検証する。
        let t = run_t("f w = 5.0\nmain = grad f [1.0, 2.0]");
        assert!(t.iter().all(|&x| x == 0.0));
    }

    // ----- G3: 北極星プログラム（線形回帰の学習ループ） -----
    // P1 の総仕上げ。lexer/parser/テンソル/autodiff/反復が全てつながったことの証明。

    #[test]
    fn g3_linreg_converges() {
        // grad（勾配）+ iterate（反復）+ テンソル演算が連動して学習が進むことを確認する。
        // main は [学習前の損失, 学習後の損失] を返す。勾配降下が効いていれば
        // 学習後の損失は学習前より桁違いに小さくなるはず。
        //   predict: feats @ w + b（線形予測）
        //   mse:     mean((pred - target)^2)（平均二乗誤差）
        //   step:    w <- w - lr * ∇loss(w)（1ステップの勾配降下）
        //   trained: zeros[3] から step を 1000 回反復した重み
        let src = "
x = [1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0; 1.0, 0.0, 1.0]
y = [1.0, 2.0, 3.0, 0.5]
lr = 0.01
predict feats w b = feats @ w + b
mse pred target = mean ((pred - target) ^ 2)
loss w = mse (predict x w 0.0) y
step w = w - lr * grad loss w
trained = iterate (zeros [3]) 1000 step
main = [loss (zeros [3]), loss trained]
";
        let t = run_t(src);
        let before = t[[0]]; // 重み 0 での損失（大きい）
        let after = t[[1]]; // 1000 ステップ学習後の損失（小さいはず）
        assert!(before > 1.0, "初期損失が大きいはず: {}", before);
        // 損失が 1/100 未満まで落ちていれば、勾配の符号・大きさが正しい強い証拠。
        assert!(after < before * 0.01, "損失が大きく減少するはず: {} -> {}", before, after);
        assert!(after.is_finite()); // 発散して NaN/Inf になっていないこと
    }

    #[test]
    fn g3_northern_star_shape() {
        // SPEC §3.8 の北極星プログラムを、main が iterate の結果（学習後の重み）を
        // そのまま返す形（仕様どおり）で走らせる。converges 版と違い損失ではなく
        // main の戻り型に注目し、重みが Tensor[3] で有限値であることを確認する。
        // 「SPEC のサンプルがそのまま動く」= G3 達成の証明。
        let src = "
x = [1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0; 1.0, 0.0, 1.0]
y = [1.0, 2.0, 3.0, 0.5]
lr = 0.01
predict feats w b = feats @ w + b
mse pred target = mean ((pred - target) ^ 2)
loss w = mse (predict x w 0.0) y
step w = w - lr * grad loss w
main = iterate (zeros [3]) 1000 step
";
        let t = run_t(src);
        assert_eq!(t.shape(), &[3]); // 重みは3次元（特徴数）
        assert!(t.iter().all(|x| x.is_finite())); // 発散していないこと
    }
}
