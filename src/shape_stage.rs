// Haplo の shape staging パス（P2 / 形状抽象評価）。
//
// interpreter.rs（実際の評価器）と「対称」な構造を持つ第2の評価器。
// 評価ドメインを `Value` から `ShapeType` に替えただけで、同じ AST を同じ再帰構造で歩く。
// 目的は「実行前」に shape 不整合（行列積の内次元不一致・要素ごと演算の shape 不一致）を
// 静的に検出すること（ロードマップ G4）。run() で eval_program の前段ゲートとして走らせる。
//
// 設計上の最重要原則は「偽陽性ゼロ」:
//   staging は「通るが実行時に shape エラー」を許容する（ロードマップ §8 リスク6）。
//   しかし正しいプログラムを誤って却下してはならない。
//   よって推論できない箇所は `Unknown` を伝播させ、
//   「両辺の次元がすべて具体値（Concrete）で確定しており、かつ矛盾している」場合だけを
//   エラーにする。これにより既存の examples / 既存テストが偽陽性で壊れない。
//
// なぜ interpreter とは別ファイル・別の Env/Closure を新規に定義するのか:
//   既存の `Value`/`Env`/`Closure` をジェネリック化して共用する案もあったが、
//   変更が interpreter 全体に波及して P1 のコードを壊しやすい。
//   shape 専用の型を別に持つ方が変更が局所化でき、両評価器を独立に検証できる。
//   AST 非依存の純粋ロジック（カリー化 desugar・組み込み arity）だけは
//   interpreter から pub(crate) で借りて規則のずれを防ぐ。

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::*;
use crate::interpreter::{builtin_arity, desugar_lambda};
use crate::value::BuiltinFn;

// 抽象ドメインの値。Value と1対1ではなく、shape の検査に必要な情報だけを持つ。
#[derive(Debug, Clone)]
pub enum ShapeType {
    // スカラー（Int / Float / Bool）。shape を持たない値。
    Scalar,
    // テンソル。次元の列を保持する（例: 行列は [m, n]）。
    Tensor(Vec<DimVal>),
    // ユーザ定義関数。body と定義時の環境を保持し、引数 shape で再評価できるようにする。
    // 評価器の Closure と同型だが、env が ShapeEnv である点だけが違う。
    Closure(ShapeClosure),
    // 組み込み関数（value.rs の BuiltinFn を再利用）。
    Builtin(BuiltinFn),
    // 多引数組み込みの部分適用（reshape/grad/iterate）。arity に達したら実行する。
    PartialBuiltin(BuiltinFn, Vec<ShapeType>),
    // 推論不能。エラーにせず伝播させるための「分からない」を表す底値。
    // zeros/reshape の出力（引数の値に依存）や、Unknown を含む演算結果がこれになる。
    Unknown,
}

// 1次元ぶんの大きさ。
// Var / Unknown は P2 では生成しないが、ドメインの形を先に確定させ P4（次元変数の
// 単一化）で追加実装するだけで済むように定義だけ置く。未使用警告はその意図を込めて許可する。
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum DimVal {
    // 具体的な次元（例: 3）。P2 で検査対象にするのはこれだけ。
    Concrete(usize),
    // 次元変数（例: "m"）。P4 で単一化を導入する際に使う。
    Var(String),
    // 推論不能な次元。比較しても一致判定せず、エラーの根拠にしない。
    Unknown,
}

// shape ドメインのクロージャ。評価器の Closure の鏡像。
#[derive(Debug, Clone)]
pub struct ShapeClosure {
    pub param: String,
    pub body: Expr,
    pub env: ShapeEnv,
}

// shape 検査で報告するエラー。shape に固有の不整合だけを扱う。
// 未定義変数・main 不在・型不一致といった「実行時にも eval が検出する」エラーは
// ここでは出さず Unknown を返して通過させる（shape パスは shape 専門に徹する）。
#[derive(Debug)]
pub enum ShapeError {
    // 行列積 @ の内次元不一致（例: [m,k] @ [k2,n] で k != k2）。
    MatMulMismatch { a: Vec<DimVal>, b: Vec<DimVal> },
    // 要素ごと演算（+ - * / ^）の shape 不一致。
    ElementwiseMismatch {
        op: &'static str,
        a: Vec<DimVal>,
        b: Vec<DimVal>,
    },
    // テンソルリテラルの行の長さが揃っていない（eval と同じく早期に弾く）。
    NonUniformTensor,
}

impl std::fmt::Display for DimVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DimVal::Concrete(n) => write!(f, "{}", n),
            DimVal::Var(s) => write!(f, "{}", s),
            DimVal::Unknown => write!(f, "?"),
        }
    }
}

// 次元の列を "[2, 3]" のように整形する（エラーメッセージ用）。
fn fmt_dims(dims: &[DimVal]) -> String {
    let parts: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::MatMulMismatch { a, b } => write!(
                f,
                "行列積 `@` の内次元が一致しません: {} @ {}",
                fmt_dims(a),
                fmt_dims(b)
            ),
            ShapeError::ElementwiseMismatch { op, a, b } => write!(
                f,
                "演算子 `{}` の shape が一致しません: {} と {}",
                op,
                fmt_dims(a),
                fmt_dims(b)
            ),
            ShapeError::NonUniformTensor => {
                write!(f, "テンソルリテラルの行の長さが揃っていません")
            }
        }
    }
}

// shape ドメインの変数環境。value.rs の Env の鏡像（2層構造）。
//   locals:  関数引数・let の永続連結リスト（extend が O(1)、クロージャ間で安全に共有）
//   globals: トップレベル定義を保持する共有可変マップ（前方参照・相互再帰のため）
// 保持する値が Value ではなく ShapeType である点だけが Env と異なる。
#[derive(Debug, Clone)]
pub struct ShapeEnv {
    locals: Option<Rc<ShapeEnvNode>>,
    globals: Rc<RefCell<HashMap<String, ShapeType>>>,
}

#[derive(Debug)]
struct ShapeEnvNode {
    name: String,
    value: ShapeType,
    parent: Option<Rc<ShapeEnvNode>>,
}

impl ShapeEnv {
    fn empty() -> Self {
        ShapeEnv {
            locals: None,
            globals: Rc::new(RefCell::new(HashMap::new())),
        }
    }

    fn define_global(&self, name: String, value: ShapeType) {
        self.globals.borrow_mut().insert(name, value);
    }

    fn extend(&self, name: String, value: ShapeType) -> Self {
        ShapeEnv {
            locals: Some(Rc::new(ShapeEnvNode {
                name,
                value,
                parent: self.locals.clone(),
            })),
            globals: self.globals.clone(),
        }
    }

    // locals を先頭から辿り、なければ globals を引く。Env::lookup と同じ規則（シャドーイング）。
    fn lookup(&self, name: &str) -> Option<ShapeType> {
        let mut cur = &self.locals;
        while let Some(node) = cur {
            if node.name == name {
                return Some(node.value.clone());
            }
            cur = &node.parent;
        }
        self.globals.borrow().get(name).cloned()
    }
}

// プログラム全体の shape を評価するエントリポイント。
// build_shape_env で two-pass にグローバル環境を構築し、main の shape を返す。
// main が無ければ Unknown を返す（NoMain は eval_program が実行時に報告するので、
// shape パスはここでは黙って通す。エラー責務を二重化しない）。
pub fn shape_eval_program(program: &Program) -> Result<ShapeType, ShapeError> {
    let env = build_shape_env(program)?;
    Ok(env.lookup("main").unwrap_or(ShapeType::Unknown))
}

// interpreter::build_global_env と同じ two-pass で ShapeEnv を構築する。
//   pass1: 関数定義（params あり）を Closure 化して globals に登録（前方参照・相互再帰可）。
//   pass2: 値定義（params なし）をソース順に shape 評価して登録。
//          ここで値本体の shape 不整合（例: グローバル `bad = [1,2,3] + [1,2]`）も検出される。
// TypeAnnotation は読み飛ばす（shape の静的解釈は P3 以降。P2 は式から shape を抽象評価する）。
fn build_shape_env(program: &Program) -> Result<ShapeEnv, ShapeError> {
    let env = load_builtins(ShapeEnv::empty());

    // pass1: 関数定義を Closure 化（本体はまだ評価しない）。
    for item in program {
        if let TopLevel::Binding { name, params, body } = item {
            if !params.is_empty() {
                let lambda = desugar_lambda(params, body);
                let cl = shape_eval(&lambda, &env)?;
                env.define_global(name.clone(), cl);
            }
        }
    }

    // pass2: 値定義をソース順に shape 評価。
    for item in program {
        if let TopLevel::Binding { name, params, body } = item {
            if params.is_empty() {
                let val = shape_eval(body, &env)?;
                env.define_global(name.clone(), val);
            }
        }
    }

    Ok(env)
}

// 組み込み関数名を ShapeType::Builtin として globals に注入する。
// interpreter::load_builtins と同じ名前集合（BuiltinFn を共有しているのでずれない）。
fn load_builtins(env: ShapeEnv) -> ShapeEnv {
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
        env.define_global(name.to_string(), ShapeType::Builtin(*f));
    }
    env
}

// 式の shape を抽象評価する。interpreter::eval の鏡像。
pub fn shape_eval(expr: &Expr, env: &ShapeEnv) -> Result<ShapeType, ShapeError> {
    match expr {
        // リテラルはすべてスカラー（数値・真偽値は shape を持たない）。
        Expr::Lit(_) => Ok(ShapeType::Scalar),

        // 変数。未束縛なら Unknown を返す（未定義変数エラーは eval に委ねる）。
        Expr::Var(name) => Ok(env.lookup(name).unwrap_or(ShapeType::Unknown)),

        // 単項マイナスは shape を保存する（符号反転は要素ごと）。
        Expr::UnaryMinus(e) => shape_eval(e, env),

        Expr::BinOp { op, left, right } => {
            let l = shape_eval(left, env)?;
            let r = shape_eval(right, env)?;
            shape_eval_binop(op, l, r)
        }

        Expr::App(func, arg) => {
            let f = shape_eval(func, env)?;
            let a = shape_eval(arg, env)?;
            apply_shape(f, a)
        }

        // ラムダは現在の shape 環境をキャプチャして Closure にする（eval と同じく実行はしない）。
        Expr::Lambda { param, body } => Ok(ShapeType::Closure(ShapeClosure {
            param: param.clone(),
            body: *body.clone(),
            env: env.clone(),
        })),

        Expr::Let {
            name,
            params,
            value,
            body,
        } => {
            let val = if params.is_empty() {
                shape_eval(value, env)?
            } else {
                let lambda = desugar_lambda(params, value);
                shape_eval(&lambda, env)?
            };
            let new_env = env.extend(name.clone(), val);
            shape_eval(body, &new_env)
        }

        // if は両枝の shape を評価する。条件式も評価して内部の shape エラーを surface する。
        // 両枝の shape が一致すればそれを、違えば Unknown を返す（分岐で shape が変わる
        // 可能性を排除できないため、ここでエラーにはしない＝偽陽性を避ける）。
        Expr::If { cond, then, else_ } => {
            shape_eval(cond, env)?;
            let t = shape_eval(then, env)?;
            let e = shape_eval(else_, env)?;
            Ok(if shape_eq(&t, &e) { t } else { ShapeType::Unknown })
        }

        Expr::TensorLit(rows) => shape_eval_tensor_lit(rows, env),

        // a |> f ≡ f a。eval と同じく apply に流す。
        Expr::Pipe(left, right) => {
            let a = shape_eval(left, env)?;
            let f = shape_eval(right, env)?;
            apply_shape(f, a)
        }
    }
}

// 二項演算の shape 規則。interpreter::eval_binop の shape 版（値計算はしない）。
fn shape_eval_binop(op: &BinOpKind, l: ShapeType, r: ShapeType) -> Result<ShapeType, ShapeError> {
    match op {
        // 比較演算はスカラー（Bool）を返す。オペランドの shape は問わない。
        BinOpKind::Eq
        | BinOpKind::Ne
        | BinOpKind::Lt
        | BinOpKind::Le
        | BinOpKind::Gt
        | BinOpKind::Ge => Ok(ShapeType::Scalar),

        // 行列積。両辺が Tensor のときだけ内次元を検査する。
        // どちらかが Unknown（rank すら不明）なら結果も Unknown にして通過させる。
        BinOpKind::MatMul => match (l, r) {
            (ShapeType::Tensor(a), ShapeType::Tensor(b)) => matmul_shape(a, b),
            _ => Ok(ShapeType::Unknown),
        },

        // 要素ごと演算（+ - * / ^）。
        BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul | BinOpKind::Div | BinOpKind::Pow => {
            elementwise_shape(binop_symbol(op), l, r)
        }
    }
}

// 行列積の shape 規則。interpreter の matmul 分岐（2D×2D / 2D×1D のみ対応）を鏡写しにする。
//   2D[m,k] @ 2D[k2,n] → [m,n]（内次元 k==k2 が必要）
//   2D[m,k] @ 1D[k2]   → [m]  （内次元 k==k2 が必要）
// 内次元が両方 Concrete で異なるときだけ MatMulMismatch を出す。
// 片方でも Unknown 次元なら矛盾を断定できないので結果 Unknown（偽陽性回避）。
// 未対応の rank 組み合わせ（1D×1D など）は eval が実行時に弾くので、ここは Unknown で通す。
fn matmul_shape(a: Vec<DimVal>, b: Vec<DimVal>) -> Result<ShapeType, ShapeError> {
    match (a.len(), b.len()) {
        (2, 2) => {
            if dims_conflict(&a[1], &b[0]) {
                return Err(ShapeError::MatMulMismatch { a, b });
            }
            Ok(ShapeType::Tensor(vec![a[0].clone(), b[1].clone()]))
        }
        (2, 1) => {
            if dims_conflict(&a[1], &b[0]) {
                return Err(ShapeError::MatMulMismatch { a, b });
            }
            Ok(ShapeType::Tensor(vec![a[0].clone()]))
        }
        _ => Ok(ShapeType::Unknown),
    }
}

// 2つの次元が「確実に矛盾する」か。両方 Concrete で値が異なるときだけ true。
// 片方でも Unknown/Var なら（P2 では）矛盾と断定しない。
fn dims_conflict(x: &DimVal, y: &DimVal) -> bool {
    matches!((x, y), (DimVal::Concrete(a), DimVal::Concrete(b)) if a != b)
}

// 要素ごと演算の shape 規則。
//   Tensor × Tensor : 両方が完全に Concrete で不一致なら ElementwiseMismatch。
//                     一方でも Unknown 次元を含むなら Unknown（断定しない）。
//   Tensor × Scalar / Scalar × Tensor : スカラーをブロードキャストして Tensor 側の shape。
//   Scalar × Scalar : Scalar。
//   それ以外（Closure/Unknown が絡む）: Unknown。
fn elementwise_shape(op: &'static str, l: ShapeType, r: ShapeType) -> Result<ShapeType, ShapeError> {
    match (l, r) {
        (ShapeType::Tensor(a), ShapeType::Tensor(b)) => {
            match (all_concrete(&a), all_concrete(&b)) {
                (Some(av), Some(bv)) => {
                    if av != bv {
                        Err(ShapeError::ElementwiseMismatch { op, a, b })
                    } else {
                        Ok(ShapeType::Tensor(a))
                    }
                }
                // どちらかに Unknown 次元があれば一致/不一致を断定できない。
                _ => Ok(ShapeType::Unknown),
            }
        }
        (ShapeType::Tensor(d), ShapeType::Scalar) => Ok(ShapeType::Tensor(d)),
        (ShapeType::Scalar, ShapeType::Tensor(d)) => Ok(ShapeType::Tensor(d)),
        (ShapeType::Scalar, ShapeType::Scalar) => Ok(ShapeType::Scalar),
        _ => Ok(ShapeType::Unknown),
    }
}

// すべての次元が Concrete なら usize の列を返す（完全に確定した shape）。
// 一つでも Var/Unknown があれば None。比較できるのは完全 Concrete 同士だけ。
fn all_concrete(dims: &[DimVal]) -> Option<Vec<usize>> {
    dims.iter()
        .map(|d| match d {
            DimVal::Concrete(n) => Some(*n),
            _ => None,
        })
        .collect()
}

// 関数適用の shape 規則。interpreter::apply の鏡像。
fn apply_shape(f: ShapeType, arg: ShapeType) -> Result<ShapeType, ShapeError> {
    match f {
        // ユーザ関数: param を arg shape に束縛して body を再評価（カリー化対応）。
        ShapeType::Closure(c) => {
            let new_env = c.env.extend(c.param.clone(), arg);
            shape_eval(&c.body, &new_env)
        }
        // 組み込み: arity に達したら実行、未達なら部分適用を貯める。
        ShapeType::Builtin(b) => {
            if builtin_arity(b) == 1 {
                apply_shape_builtin(b, vec![arg])
            } else {
                Ok(ShapeType::PartialBuiltin(b, vec![arg]))
            }
        }
        ShapeType::PartialBuiltin(b, mut args) => {
            args.push(arg);
            if args.len() == builtin_arity(b) {
                apply_shape_builtin(b, args)
            } else {
                Ok(ShapeType::PartialBuiltin(b, args))
            }
        }
        // 関数でない値（Scalar/Tensor）の適用、または Unknown の適用は Unknown を返す。
        // 前者は eval が型エラーを報告するので shape パスは黙って通す。
        _ => Ok(ShapeType::Unknown),
    }
}

// 組み込み関数の shape 規則。引数は arity 個そろった状態で渡される。
fn apply_shape_builtin(b: BuiltinFn, args: Vec<ShapeType>) -> Result<ShapeType, ShapeError> {
    match b {
        // 集約は入力 shape によらずスカラーを返す（要素を1つにまとめる）。
        BuiltinFn::Sum | BuiltinFn::Mean => Ok(ShapeType::Scalar),

        // 要素ごとの単項関数は入力と同じ shape（Scalar→Scalar, Tensor→同 shape, Unknown→Unknown）。
        BuiltinFn::Exp | BuiltinFn::Log | BuiltinFn::Tanh | BuiltinFn::Sqrt => Ok(args[0].clone()),

        // 転置は 2D の行と列を入れ替える。2D 以外は Unknown。
        BuiltinFn::Transpose => match &args[0] {
            ShapeType::Tensor(d) if d.len() == 2 => {
                Ok(ShapeType::Tensor(vec![d[1].clone(), d[0].clone()]))
            }
            _ => Ok(ShapeType::Unknown),
        },

        // zeros/ones の出力 shape は「引数の値」に依存する（例: zeros [3] の中身 3 が次元）。
        // shape 情報だけからは決まらないため Unknown を返す（健全性優先で precision を捨てる）。
        // 将来、引数がリテラルのときに const-fold して具体 shape を出す改善が可能（P3+）。
        BuiltinFn::Zeros | BuiltinFn::Ones => Ok(ShapeType::Unknown),

        // reshape も第2引数の値に依存するため Unknown。
        BuiltinFn::Reshape => Ok(ShapeType::Unknown),

        // grad f x の勾配は入力 x と同じ shape。
        // f を x に1回 apply して本体内部の shape 不整合も surface する（結果は捨てる）。
        BuiltinFn::Grad => {
            let f = args[0].clone();
            let x = args[1].clone();
            apply_shape(f, x.clone())?;
            Ok(x)
        }

        // iterate init n f は f: a -> a を init に繰り返すので結果の shape は init と同じ。
        // f を init に1回 apply して本体の shape 不整合を surface する（init が Unknown なら
        // 自然に Unknown が伝播して何も誤検出しない）。
        BuiltinFn::Iterate => {
            let init = args[0].clone();
            let f = args[2].clone();
            apply_shape(f, init.clone())?;
            Ok(init)
        }
    }
}

// テンソルリテラルの shape を求める。interpreter::eval_tensor_lit の shape 版。
//   - 空リテラル → Tensor([0])（eval と同じ）
//   - 行の長さが揃わない → NonUniformTensor（eval と同じく早期検出）
//   - 全要素が Scalar のとき: 1行 → [cols]、複数行 → [rows, cols]
//   - 要素に非スカラー/Unknown が混じる場合は shape を断定せず Unknown（偽陽性回避）
// 要素は順に shape 評価するので、要素式の内部にある shape エラーもここで surface される。
fn shape_eval_tensor_lit(rows: &[Vec<Expr>], env: &ShapeEnv) -> Result<ShapeType, ShapeError> {
    if rows.is_empty() || (rows.len() == 1 && rows[0].is_empty()) {
        return Ok(ShapeType::Tensor(vec![DimVal::Concrete(0)]));
    }

    let ncols = rows[0].len();
    let mut all_scalar = true;
    for row in rows {
        if row.len() != ncols {
            return Err(ShapeError::NonUniformTensor);
        }
        for e in row {
            let s = shape_eval(e, env)?;
            if !matches!(s, ShapeType::Scalar) {
                all_scalar = false;
            }
        }
    }

    if !all_scalar {
        // 要素がスカラーでない（テンソルのネスト等）場合は具体 shape を断定しない。
        return Ok(ShapeType::Unknown);
    }

    let nrows = rows.len();
    if nrows == 1 {
        Ok(ShapeType::Tensor(vec![DimVal::Concrete(ncols)]))
    } else {
        Ok(ShapeType::Tensor(vec![
            DimVal::Concrete(nrows),
            DimVal::Concrete(ncols),
        ]))
    }
}

// shape の構造的等価判定（if の両枝比較に使う）。
// Scalar 同士・完全一致する Tensor のみ true。Closure/Builtin/Unknown は常に false。
// （等しいと言い切れないものは false にして If が Unknown を返すようにする）
fn shape_eq(a: &ShapeType, b: &ShapeType) -> bool {
    match (a, b) {
        (ShapeType::Scalar, ShapeType::Scalar) => true,
        (ShapeType::Tensor(x), ShapeType::Tensor(y)) => x == y,
        _ => false,
    }
}

// BinOpKind を演算子記号（エラーメッセージ用）に変換する。
fn binop_symbol(op: &BinOpKind) -> &'static str {
    match op {
        BinOpKind::Add => "+",
        BinOpKind::Sub => "-",
        BinOpKind::Mul => "*",
        BinOpKind::Div => "/",
        BinOpKind::Pow => "^",
        BinOpKind::MatMul => "@",
        BinOpKind::Eq => "==",
        BinOpKind::Ne => "!=",
        BinOpKind::Lt => "<",
        BinOpKind::Le => "<=",
        BinOpKind::Gt => ">",
        BinOpKind::Ge => ">=",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    // Haplo ソースを lex → parse → shape_eval_program まで通すヘルパ。
    // shape パスの Result をそのまま返し、エラー/成功の両方をテストできるようにする。
    fn check(src: &str) -> Result<ShapeType, ShapeError> {
        let tokens = lex(src).expect("lex error");
        let program = parse(&tokens).expect("parse error");
        shape_eval_program(&program)
    }

    // ----- G4: shape staging（行列積の不整合検出） -----

    #[test]
    fn g4_matmul_mismatch_detected() {
        // a は 2×3、b は 2×2。a @ b の内次元は 3（a の列）と 2（b の行）で一致しない。
        // eval を待たずに MatMulMismatch を報告できることが G4 の核心。
        let src = "
a = [1.0, 2.0, 3.0; 4.0, 5.0, 6.0]
b = [1.0, 2.0; 3.0, 4.0]
main = a @ b
";
        assert!(matches!(check(src), Err(ShapeError::MatMulMismatch { .. })));
    }

    #[test]
    fn g4_matmul_ok() {
        // a は 2×3、b は 3×2。内次元 3==3 で一致するので通過し、結果は [2,2]。
        // 正しい行列積を誤って弾かない（偽陽性なし）ことの確認も兼ねる。
        let src = "
a = [1.0, 2.0, 3.0; 4.0, 5.0, 6.0]
b = [1.0, 2.0; 3.0, 4.0; 5.0, 6.0]
main = a @ b
";
        match check(src) {
            Ok(ShapeType::Tensor(d)) => {
                assert_eq!(d, vec![DimVal::Concrete(2), DimVal::Concrete(2)]);
            }
            other => panic!("Tensor[2,2] を期待: {:?}", other),
        }
    }

    #[test]
    fn g4_elementwise_mismatch() {
        // 長さ3と長さ2のベクトルの加算は要素ごとに対応が取れず shape 不一致。
        let src = "main = [1.0, 2.0, 3.0] + [1.0, 2.0]";
        assert!(matches!(
            check(src),
            Err(ShapeError::ElementwiseMismatch { op: "+", .. })
        ));
    }

    #[test]
    fn g4_elementwise_ok() {
        // 同じ shape [3] 同士の加算は通り、結果も [3]。
        let src = "main = [1.0, 2.0, 3.0] + [4.0, 5.0, 6.0]";
        match check(src) {
            Ok(ShapeType::Tensor(d)) => assert_eq!(d, vec![DimVal::Concrete(3)]),
            other => panic!("Tensor[3] を期待: {:?}", other),
        }
    }

    #[test]
    fn g4_scalar_broadcast() {
        // テンソル + スカラーはブロードキャストでテンソル shape を保つ（[3] のまま）。
        let src = "main = [1.0, 2.0, 3.0] + 1.0";
        match check(src) {
            Ok(ShapeType::Tensor(d)) => assert_eq!(d, vec![DimVal::Concrete(3)]),
            other => panic!("Tensor[3] を期待: {:?}", other),
        }
    }

    #[test]
    fn g4_tensor_lit_shapes() {
        // 1行のリテラルは 1D（[3]）、複数行は 2D（[2,3]）と推論されることを確認する。
        match check("main = [1.0, 2.0, 3.0]") {
            Ok(ShapeType::Tensor(d)) => assert_eq!(d, vec![DimVal::Concrete(3)]),
            other => panic!("Tensor[3] を期待: {:?}", other),
        }
        match check("main = [1.0, 2.0, 3.0; 4.0, 5.0, 6.0]") {
            Ok(ShapeType::Tensor(d)) => {
                assert_eq!(d, vec![DimVal::Concrete(2), DimVal::Concrete(3)])
            }
            other => panic!("Tensor[2,3] を期待: {:?}", other),
        }
    }

    #[test]
    fn g4_sum_is_scalar() {
        // sum はテンソルを1つの値に畳むのでスカラーを返す。
        assert!(matches!(
            check("main = sum [1.0, 2.0, 3.0]"),
            Ok(ShapeType::Scalar)
        ));
    }

    #[test]
    fn g4_grad_same_shape() {
        // grad f x の勾配は入力 x と同じ shape。x=[3] なら結果も [3]。
        // f 内部（sum (w*w)）の要素ごと演算も shape 検査を通る。
        let src = "f w = sum (w * w)\nmain = grad f [1.0, 2.0, 3.0]";
        match check(src) {
            Ok(ShapeType::Tensor(d)) => assert_eq!(d, vec![DimVal::Concrete(3)]),
            other => panic!("Tensor[3] を期待: {:?}", other),
        }
    }

    #[test]
    fn g4_function_application() {
        // カリー化したユーザ関数に引数 shape を渡すと body を通して shape が伝播する。
        // double v = v + v に [2] を渡すと結果は [2]。
        let src = "double v = v + v\nmain = double [1.0, 2.0]";
        match check(src) {
            Ok(ShapeType::Tensor(d)) => assert_eq!(d, vec![DimVal::Concrete(2)]),
            other => panic!("Tensor[2] を期待: {:?}", other),
        }
    }

    #[test]
    fn g4_mismatch_inside_function_body() {
        // 関数本体に潜む shape 不整合も、適用時に body を再評価することで検出できる。
        // bad v = v + [1.0, 2.0] に長さ3を渡すと [3] + [2] で不一致になる。
        let src = "bad v = v + [1.0, 2.0]\nmain = bad [1.0, 2.0, 3.0]";
        assert!(matches!(
            check(src),
            Err(ShapeError::ElementwiseMismatch { .. })
        ));
    }

    #[test]
    fn g4_unknown_no_false_positive() {
        // zeros/reshape の出力は Unknown 扱いなので、それを含む式を誤ってエラーにしない
        // ことを確認する（健全性＝偽陽性ゼロの回帰テスト）。
        // zeros [3] は shape 不明 → a @ unknown も unknown → エラーにならず通過する。
        let src = "
a = [1.0, 2.0; 3.0, 4.0]
main = a @ zeros [2]
";
        assert!(check(src).is_ok());

        // reshape の結果（Unknown）に対する加算もエラーにしない。
        let src2 = "main = reshape [1.0, 2.0, 3.0, 4.0] [2, 2] + 1.0";
        assert!(check(src2).is_ok());
    }

    #[test]
    fn g4_linreg_passes_shape_check() {
        // 北極星プログラム（線形回帰の学習）が shape チェックを偽陽性なく通過することを確認する。
        // zeros 由来の Unknown が随所に伝播するが、確定した矛盾は無いのでエラーにならない。
        // これが P2 の最重要回帰（staging が正しいプログラムを通すこと）。
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
        assert!(check(src).is_ok());
    }

    #[test]
    fn g4_global_value_mismatch_detected() {
        // main 以外のグローバル値定義に潜む不整合も pass2 の評価で検出される。
        // bad は [2] + [3] で不一致。main が bad を使っていなくても、定義の評価で弾く。
        let src = "
bad = [1.0, 2.0] + [1.0, 2.0, 3.0]
main = 1
";
        assert!(matches!(
            check(src),
            Err(ShapeError::ElementwiseMismatch { .. })
        ));
    }
}
