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

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::*;
use crate::interpreter::{builtin_arity, desugar_lambda};
use crate::value::BuiltinFn;

// 再帰関数に対する shape 評価の停止保証。
//
// 問題: shape ドメインには「実値」が無いため、再帰関数の基底ケースを実値で判定して
// 打ち切れない。さらに If は両枝を評価する（片方が再帰呼び出しでも辿ってしまう）。
// このため再帰関数の shape 評価は自然には停止せず、無限再帰でスタックを溢れさせる。
//
// 対策は2系統の予算で「stack 深さ」と「総仕事量」の両方を縛る:
//   - APPLY_DEPTH（復元あり）: クロージャ適用のネスト深度を制限し Rust の
//     スタックオーバーフローを防ぐ。線形再帰（f n = ... f (n-1)）対策。
//   - FUEL（消費のみ・復元なし）: クロージャ適用の総回数を制限し、分岐再帰
//     （f n = ... f(n-1) + f(n-2) のように1呼び出しが複数の再帰を生む）で
//     深さは浅くてもノード数が指数的に爆発するのを防ぐ。
// どちらの上限に達しても、その適用は ShapeType::Unknown を返して打ち切る。
// Unknown は健全に伝播するので、打ち切りによる偽陽性（誤ったエラー報告）は生じない。
//
// thread_local を使う理由: shape_eval/apply_shape は AST を深く相互再帰するため、
// 予算カウンタを全段の引数に通すとシグネチャ変更が広範囲に及ぶ。autodiff のテープと
// 同じく、thread_local なら関数の形を変えずに横断的な状態を差し込める。
thread_local! {
    static APPLY_DEPTH: Cell<usize> = const { Cell::new(0) };
    static FUEL: Cell<u64> = const { Cell::new(0) };
}

// クロージャ適用のネスト深度上限。これを超えたら適用を打ち切る。
// 深度が積み上がるのは「クロージャの本体を評価している最中にさらにクロージャを適用する」
// 真の呼び出しネスト＝実質的に再帰のときだけ。`f (g (h x))` のような逐次合成は各適用が
// 戻ってから次を適用するので深度は積み上がらない（適用時点では深度 1〜2）。よって正常な
// 非再帰コードがこの上限に達することはまずない。一方この shape 評価は Rust の再帰なので、
// テストスレッド（デフォルト約2MBスタック）でも安全に辿れるよう控えめな値にする。
// 64 段あれば現実の非再帰ネストには十分で、暴走再帰は確実に打ち切れる。
const MAX_APPLY_DEPTH: usize = 64;

// クロージャ適用の総回数上限（パス全体で消費する燃料）。分岐再帰の指数爆発を止める。
// 正常プログラムの shape 評価はこれより桁違いに少ない適用回数で終わる。
const MAX_APPLY_FUEL: u64 = 100_000;

// クロージャ適用の深度を管理する RAII ガード。
// enter() で深度を1増やし（上限超過なら None）、Drop で必ず1減らす。
// `?` によるエラー伝播でも Drop が走るので、深度カウンタが片側だけずれることはない。
struct DepthGuard;

impl DepthGuard {
    fn enter() -> Option<DepthGuard> {
        APPLY_DEPTH.with(|d| {
            let cur = d.get();
            if cur >= MAX_APPLY_DEPTH {
                None
            } else {
                d.set(cur + 1);
                Some(DepthGuard)
            }
        })
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        APPLY_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

// 燃料を1消費する。残量があれば true、尽きていれば false。
// 深度ガードと違い消費したら戻さない（総仕事量を縛るため）。
fn consume_fuel() -> bool {
    FUEL.with(|f| {
        let remaining = f.get();
        if remaining == 0 {
            false
        } else {
            f.set(remaining - 1);
            true
        }
    })
}

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
    // 型注釈で宣言した shape と、本体（関数の戻り値・グローバル値）から推論した
    // shape が矛盾する（例: `f : Tensor[3] -> Tensor[2]` で `f w = w` は [3] を返す）。
    // P3 で導入。P4 でランク不一致・次元変数名不一致も対象に拡張。
    AnnotationMismatch {
        name: String,
        declared: Vec<DimVal>,
        inferred: Vec<DimVal>,
    },
    // P4 新規: 次元変数名が衝突する位置で使われた（単一化できない）。
    // 例: `Tensor[n] + Tensor[m]`（n と m は独立した型変数なので等しい保証がない）。
    // Concrete の不一致は ElementwiseMismatch/MatMulMismatch が担い、
    // VarConflict は「両方が Var だが名前が異なる」場合専用。
    VarConflict {
        op: &'static str,
        var_a: String,
        var_b: String,
    },
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
            ShapeError::AnnotationMismatch {
                name,
                declared,
                inferred,
            } => write!(
                f,
                "`{}` の型注釈 {} と本体の shape {} が一致しません",
                name,
                fmt_dims(declared),
                fmt_dims(inferred)
            ),
            ShapeError::VarConflict { op, var_a, var_b } => write!(
                f,
                "演算子 `{}` の次元変数 `{}` と `{}` が衝突しています（独立した変数のため等しい保証がありません）",
                op, var_a, var_b
            ),
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
    // 再帰停止予算をパス開始時にリセットする。深度ガードは Drop で必ず戻るので
    // 通常は 0 に復元されるが、念のため毎回明示リセットしてパスを自己完結にする。
    APPLY_DEPTH.with(|d| d.set(0));
    FUEL.with(|f| f.set(MAX_APPLY_FUEL));

    let env = build_shape_env(program)?;
    Ok(env.lookup("main").unwrap_or(ShapeType::Unknown))
}

// interpreter::build_global_env と同じ two-pass で ShapeEnv を構築し、さらに P3 で
// 型注釈駆動の検査パス（pass3）を足す。
//   pass1: 関数定義（params あり）を Closure 化して globals に登録（前方参照・相互再帰可）。
//   pass2: 値定義（params なし）をソース順に shape 評価して登録。
//          注釈があり推論値と宣言が両方 Concrete で矛盾すれば AnnotationMismatch。
//          矛盾しなければ「より具体的な方」を登録して下流の検査精度を上げる。
//          ここで値本体の shape 不整合（例: グローバル `bad = [1,2,3] + [1,2]`）も検出される。
//   pass3: 型注釈付き関数の本体を「引数=宣言 shape」で検査する（P3 の中核）。
//          リテラルからのボトムアップ推論では引数が Unknown になり本体が検査されない穴を、
//          注釈の引数 shape を束縛することで塞ぐ。固定次元（Concrete）の矛盾のみ報告し、
//          次元変数（Var）は伝播のみで単一化しない（偽陽性ゼロ。単一化は P4）。
fn build_shape_env(program: &Program) -> Result<ShapeEnv, ShapeError> {
    let env = load_builtins(ShapeEnv::empty());

    // 型注釈を name -> TypeExpr に集める。pass2/pass3 で参照する。
    // 同名が複数あれば最後の注釈を採る（eval 側の定義規則に合わせ、ここでは厳密化しない）。
    let mut annotations: HashMap<&str, &TypeExpr> = HashMap::new();
    for item in program {
        if let TopLevel::TypeAnnotation { name, ty } = item {
            annotations.insert(name.as_str(), ty);
        }
    }

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

    // pass2: 値定義をソース順に shape 評価。注釈があれば突き合わせる。
    for item in program {
        if let TopLevel::Binding { name, params, body } = item {
            if params.is_empty() {
                let inferred = shape_eval(body, &env)?;
                // 注釈ありなら宣言 shape と推論 shape を突き合わせる。
                // 宣言が Concrete Tensor で推論も Concrete Tensor かつ食い違うならエラー。
                // 矛盾しないときは宣言 shape の方が具体（推論が Unknown でも宣言で確定できる）
                // ことが多いので、宣言が Tensor なら宣言を、それ以外は推論値を登録する。
                let to_register = match annotations.get(name.as_str()) {
                    Some(ty) => {
                        let declared = shape_of_type(ty);
                        check_annotation(name, &declared, &inferred)?;
                        if matches!(declared, ShapeType::Tensor(_)) {
                            declared
                        } else {
                            inferred
                        }
                    }
                    None => inferred,
                };
                env.define_global(name.clone(), to_register);
            }
        }
    }

    // pass3: 型注釈付き関数の本体を、引数を宣言 shape に束縛して検査する。
    for item in program {
        if let TopLevel::Binding { name, params, body } = item {
            if !params.is_empty() {
                if let Some(ty) = annotations.get(name.as_str()) {
                    check_annotated_fn(name, params, body, ty, &env)?;
                }
            }
        }
    }

    Ok(env)
}

// 型注釈付き関数の本体を検査する。注釈の Arrow を引数の数だけ剥がして各引数の
// 宣言 shape を得て、globals 環境にそれらを束縛してから本体を shape 評価する。
//   - 本体内の `@`/要素ごと演算の固定次元矛盾は shape_eval がそのまま surface する。
//   - 本体から推論した戻り shape と、注釈の戻り shape が両方 Concrete で食い違えば
//     AnnotationMismatch を報告する。
// 注釈が引数より短い・戻りが関数型のまま等の「決め切れない」場合は Unknown 扱いにして
// 黙って通す（偽陽性ゼロ）。
fn check_annotated_fn(
    name: &str,
    params: &[String],
    body: &Expr,
    ty: &TypeExpr,
    globals: &ShapeEnv,
) -> Result<(), ShapeError> {
    let (param_shapes, return_shape) = decompose_arrow(ty, params.len());

    // 各引数を宣言 shape（足りない分は Unknown）に束縛した環境を作る。
    let mut env = globals.clone();
    for (i, p) in params.iter().enumerate() {
        let s = param_shapes.get(i).cloned().unwrap_or(ShapeType::Unknown);
        env = env.extend(p.clone(), s);
    }

    // 本体を評価。ここで本体内部の shape 不整合が検出される（pass3 の主目的）。
    let inferred = shape_eval(body, &env)?;
    // 戻り型の突き合わせ（両方 Concrete で矛盾するときだけエラー）。
    check_annotation(name, &return_shape, &inferred)
}

// 宣言 shape と推論 shape の矛盾検査。P4 で検査範囲を拡張:
//   P3: 両方が完全に Concrete で次元列が食い違うときだけ AnnotationMismatch。
//   P4: 以下も対象に追加（偽陽性ゼロ原則の下で断定できる範囲を広げた）。
//     - ランク不一致（両方 Tensor で次元数が異なる）
//     - 同一位置の次元変数名が異なる（宣言 Var(m) vs 推論 Var(n)：独立型変数なので不一致）
//   Unknown/Scalar/関数型が絡む場合は「矛盾と断定できない」ので黙って通す（偽陽性回避）。
fn check_annotation(
    name: &str,
    declared: &ShapeType,
    inferred: &ShapeType,
) -> Result<(), ShapeError> {
    if let (ShapeType::Tensor(d), ShapeType::Tensor(i)) = (declared, inferred) {
        // ランク不一致（P4 新規）。
        if d.len() != i.len() {
            return Err(ShapeError::AnnotationMismatch {
                name: name.to_string(),
                declared: d.clone(),
                inferred: i.clone(),
            });
        }
        // 次元ごとに衝突を検査する。
        for (dd, di) in d.iter().zip(i.iter()) {
            let mismatch = match (dd, di) {
                // Concrete 同士で値が違う（P3 から）。
                (DimVal::Concrete(a), DimVal::Concrete(b)) => a != b,
                // Var 同士で名前が違う（P4 新規）。
                // 宣言が Var(m)、推論が Var(n) で m≠n → 独立した型変数として不一致。
                // 例: `f : Tensor[n] -> Tensor[m]` で本体が Tensor[n] を返す → AnnotationMismatch。
                (DimVal::Var(a), DimVal::Var(b)) => a != b,
                // Unknown・Concrete-Var 混在は断定不可 → 通す（偽陽性ゼロ）。
                _ => false,
            };
            if mismatch {
                return Err(ShapeError::AnnotationMismatch {
                    name: name.to_string(),
                    declared: d.clone(),
                    inferred: i.clone(),
                });
            }
        }
    }
    Ok(())
}

// 型式を抽象 shape ドメインに変換する。固定次元は Concrete、次元変数は Var に対応づける。
//   f32/f64/Int/Bool 等の基本型 → Scalar
//   Tensor[..]                  → Tensor（Fixed→Concrete, Var→Var）
//   Arrow/App/未知の Named      → Unknown（高階引数や型別名は P3 では shape を決めない）
// Unknown を返す箇所は健全に伝播し偽陽性を生まない。
fn shape_of_type(ty: &TypeExpr) -> ShapeType {
    match ty {
        TypeExpr::Named(n) => match n.as_str() {
            "f32" | "f64" | "Int" | "Bool" | "Float" => ShapeType::Scalar,
            _ => ShapeType::Unknown,
        },
        TypeExpr::Tensor(dims) => ShapeType::Tensor(
            dims.iter()
                .map(|d| match d {
                    TypeDim::Fixed(n) => DimVal::Concrete(*n),
                    TypeDim::Var(s) => DimVal::Var(s.clone()),
                    // P4: shape 算術式（m+n, m*n 等）。concat/flatten 等のプリミティブが
                    // 揃う P6 以降で評価対応予定。現時点では Unknown にフォールバックして
                    // 偽陽性ゼロを保つ。
                    TypeDim::Expr(_) => DimVal::Unknown,
                })
                .collect(),
        ),
        // 関数型・型適用は shape を1つに決められないので Unknown（安全側）。
        TypeExpr::Arrow(_, _) | TypeExpr::App(_, _) => ShapeType::Unknown,
    }
}

// 関数型注釈を「引数 shape の列」と「戻り shape」に分解する。
// arity（仮引数の個数）だけ Arrow を左から剥がす。注釈が arity より短ければ
// 取れた分だけ返し（残りの引数は呼び出し側で Unknown 束縛）、戻りが Arrow のまま
// なら戻り shape は Unknown（高階返り＝tensor 形に決められない）にする。
fn decompose_arrow(ty: &TypeExpr, arity: usize) -> (Vec<ShapeType>, ShapeType) {
    let mut params = Vec::new();
    let mut cur = ty;
    for _ in 0..arity {
        match cur {
            TypeExpr::Arrow(lhs, rhs) => {
                params.push(shape_of_type(lhs));
                cur = rhs;
            }
            // これ以上 Arrow が無い（注釈が引数より短い）なら打ち切る。
            _ => break,
        }
    }
    (params, shape_of_type(cur))
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
// P4 でエラー報告の種類を細分化:
//   - 内次元が両方 Concrete で異なる → MatMulMismatch（P2 から）
//   - 内次元が両方 Var で名前が異なる → VarConflict（P4 新規）
//   - 片方が Unknown の場合は矛盾を断定できないので Unknown で通す（偽陽性回避）
// 未対応の rank 組み合わせ（1D×1D など）は eval が実行時に弾くので Unknown で通す。
fn matmul_shape(a: Vec<DimVal>, b: Vec<DimVal>) -> Result<ShapeType, ShapeError> {
    match (a.len(), b.len()) {
        (2, 2) => {
            match dim_pair_conflict(&a[1], &b[0]) {
                DimConflictKind::ConcreteMismatch => return Err(ShapeError::MatMulMismatch { a, b }),
                DimConflictKind::VarMismatch(va, vb) => {
                    return Err(ShapeError::VarConflict { op: "@", var_a: va, var_b: vb })
                }
                DimConflictKind::Ok => {}
            }
            Ok(ShapeType::Tensor(vec![a[0].clone(), b[1].clone()]))
        }
        (2, 1) => {
            match dim_pair_conflict(&a[1], &b[0]) {
                DimConflictKind::ConcreteMismatch => return Err(ShapeError::MatMulMismatch { a, b }),
                DimConflictKind::VarMismatch(va, vb) => {
                    return Err(ShapeError::VarConflict { op: "@", var_a: va, var_b: vb })
                }
                DimConflictKind::Ok => {}
            }
            Ok(ShapeType::Tensor(vec![a[0].clone()]))
        }
        _ => Ok(ShapeType::Unknown),
    }
}

// 次元ペアの衝突種別。P4 で dims_conflict（bool）から細分化。
// 衝突なし・Concrete 不一致・Var 名不一致の3種を区別することで
// 下流が適切なエラー型（MatMulMismatch vs VarConflict）を報告できる。
enum DimConflictKind {
    // 衝突なし（同じ値 / 同名変数 / Unknown が絡む）。
    Ok,
    // 両方 Concrete で値が異なる。
    ConcreteMismatch,
    // 両方 Var だが名前が異なる（独立した型変数なので等しい保証がない）。
    VarMismatch(String, String),
}

// 2次元の衝突種別を判定する。
//   (Concrete(a), Concrete(b)): a != b → ConcreteMismatch
//   (Var(x),      Var(y)):      x != y → VarMismatch
//   それ以外（同値 / Unknown 混在）: Ok
// Unknown が絡む場合は偽陽性回避のため常に Ok を返す。
fn dim_pair_conflict(x: &DimVal, y: &DimVal) -> DimConflictKind {
    match (x, y) {
        (DimVal::Concrete(a), DimVal::Concrete(b)) if a != b => DimConflictKind::ConcreteMismatch,
        (DimVal::Var(a), DimVal::Var(b)) if a != b => {
            DimConflictKind::VarMismatch(a.clone(), b.clone())
        }
        _ => DimConflictKind::Ok,
    }
}

// 要素ごと演算の shape 規則。P4 で拡張:
//   Tensor × Tensor:
//     - ランク（次元数）が異なる → ElementwiseMismatch（Var を含む場合も常にエラー）
//     - 同ランクで次元ごとにチェック:
//         Concrete 不一致 → ElementwiseMismatch（P2 から）
//         Var 名不一致   → VarConflict（P4 新規）
//         それ以外（同値/同名/Unknown混在）→ OK、結果は lhs の shape
//     - 全次元が OK なら Tensor(a) を返す（P4 改善: 以前は Unknown だった Var 同士の一致も
//       正しく Tensor を返すようになった。例: Tensor[n]+Tensor[n] → Tensor[n]）
//   Tensor × Scalar / Scalar × Tensor : ブロードキャスト（変わらず）
//   Scalar × Scalar : Scalar
//   それ以外: Unknown
fn elementwise_shape(op: &'static str, l: ShapeType, r: ShapeType) -> Result<ShapeType, ShapeError> {
    match (l, r) {
        (ShapeType::Tensor(a), ShapeType::Tensor(b)) => {
            // ランク不一致は Var を含む場合でも常にエラー。
            // 理由: ランクは型注釈で明示的に宣言されるため、ランクが違えば確実に不整合。
            if a.len() != b.len() {
                return Err(ShapeError::ElementwiseMismatch { op, a, b });
            }
            // 次元ごとに種別チェック。最初の衝突でエラーを返す。
            for (da, db) in a.iter().zip(b.iter()) {
                match dim_pair_conflict(da, db) {
                    DimConflictKind::ConcreteMismatch => {
                        return Err(ShapeError::ElementwiseMismatch { op, a, b })
                    }
                    DimConflictKind::VarMismatch(va, vb) => {
                        return Err(ShapeError::VarConflict { op, var_a: va, var_b: vb })
                    }
                    DimConflictKind::Ok => {}
                }
            }
            // 全次元が OK → lhs の shape を返す。
            // P4 改善点: 以前は all_concrete が None なら Unknown を返していたが、
            // Var 同士で名前が一致している場合（例: [n] と [n]）も Tensor を返すようになった。
            Ok(ShapeType::Tensor(a))
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
        // 再帰関数で停止しないため、深度・燃料の両予算で打ち切る（詳細はファイル冒頭の
        // thread_local ブロックのコメント参照）。どちらか尽きたら Unknown を返す。
        ShapeType::Closure(c) => {
            // 深度ガード（Drop で必ず戻す）。上限超過なら適用せず Unknown で打ち切り。
            let _guard = match DepthGuard::enter() {
                Some(g) => g,
                None => return Ok(ShapeType::Unknown),
            };
            // 総適用回数（燃料）も消費。尽きていれば Unknown で打ち切り。
            if !consume_fuel() {
                return Ok(ShapeType::Unknown);
            }
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

    // ----- G4: 再帰関数で shape パスが停止すること（無限再帰クラッシュの回帰） -----
    // shape ドメインには実値が無いため再帰の基底ケースを実値で判定できず、さらに If が
    // 両枝を評価するため、再帰関数の shape 評価は自然には停止しない。深度・燃料の予算で
    // 打ち切ることで「停止して Ok を返す」ことを確認する。打ち切り箇所は Unknown を
    // 伝播するので、誤ったエラー（偽陽性）にはならず Ok になる。
    // これらのテストが「ハングせず・スタックを溢れさせず」完了すること自体が回帰の核心。

    #[test]
    fn g4_self_recursion_terminates() {
        // 線形自己再帰。shape 評価は深度上限で打ち切られ、クラッシュせず Ok を返す。
        // 結果は then(Scalar) と else(打ち切りで Unknown) が食い違うため Unknown になる。
        let src = "f n = if n == 0 then 1 else f (n - 1)\nmain = f 5";
        assert!(check(src).is_ok());
    }

    #[test]
    fn g4_mutual_recursion_terminates() {
        // 相互再帰（P1 の isEven/isOdd）。shape パスが無限再帰せず停止することを確認する。
        // P2 導入前はこの形を CLI 実行すると shape パスがスタックを溢れさせていた。
        let src = "
isEven n = if n == 0 then true else isOdd (n - 1)
isOdd n = if n == 0 then false else isEven (n - 1)
main = isEven 10
";
        assert!(check(src).is_ok());
    }

    #[test]
    fn g4_branching_recursion_terminates() {
        // 分岐再帰（1回の呼び出しが複数の再帰を生む）。深度は浅くてもノード数が指数的に
        // 増えうるため、深度ガードだけでなく燃料（総適用回数）の上限でも打ち切る必要がある。
        // このテストが現実的な時間で完了することが、燃料による指数爆発抑止の回帰になる。
        let src = "f n = if n <= 1 then n else f (n - 1) + f (n - 2)\nmain = f 20";
        assert!(check(src).is_ok());
    }

    // ----- P3: 型注釈駆動の shape 検査（固定次元） -----
    // P2 までは式（リテラル）からのボトムアップ推論のみで、関数引数は Unknown だった。
    // P3 は型注釈を読み、引数を宣言 shape に束縛して本体を検査する（pass3）。
    // 固定次元（Concrete）の矛盾だけを報告し、次元変数（Var）は伝播のみで単一化しない。

    #[test]
    fn p3_annotation_checks_body_mismatch() {
        // 注釈付き関数の本体に潜む固定次元の不整合を検出する（P3 の中核）。
        // main は f を呼ばないので P2 のボトムアップ推論では f の本体は評価されず見逃す。
        // 注釈 `Tensor[3] -> Tensor[3]` で w を [3] に束縛すると、w + [1,2] が [3]+[2] で矛盾する。
        let src = "
f : Tensor[3] -> Tensor[3]
f w = w + [1.0, 2.0]
main = 1
";
        assert!(matches!(
            check(src),
            Err(ShapeError::ElementwiseMismatch { .. })
        ));
    }

    #[test]
    fn p3_return_type_mismatch() {
        // 本体から推論した戻り shape と宣言戻り型が両方 Concrete で食い違えば AnnotationMismatch。
        // f w = w は w（=[3]）をそのまま返すが、宣言戻り型は [2] なので矛盾する。
        let src = "
f : Tensor[3] -> Tensor[2]
f w = w
main = 1
";
        assert!(matches!(
            check(src),
            Err(ShapeError::AnnotationMismatch { .. })
        ));
    }

    #[test]
    fn p3_annotation_ok_no_false_positive() {
        // 正しい注釈付き関数は通る。w + w は [3]、宣言戻り型 [3] と一致。
        let src = "
f : Tensor[3] -> Tensor[3]
f w = w + w
main = 1
";
        assert!(check(src).is_ok());
    }

    #[test]
    fn p3_var_dims_no_false_positive() {
        // 次元変数を含む注釈は P3 では単一化しないので、矛盾を断定せず通す（偽陽性ゼロ）。
        // predict: [n,d] @ [d] は内次元が両方 Var(d) で dims_conflict=false → [n]。
        // 宣言戻り型 [n] と推論 [Var(n)] はどちらも完全 Concrete でないので照合は通過する。
        let src = "
predict : Tensor[n, d] -> Tensor[d] -> Tensor[n]
predict feats w = feats @ w
main = 1
";
        assert!(check(src).is_ok());
    }

    #[test]
    fn p3_global_value_annotation_mismatch() {
        // グローバル値の注釈と本体の固定次元矛盾も pass2 で検出する。
        // x の本体は [3] だが宣言は [2]。両方 Concrete で食い違うので AnnotationMismatch。
        let src = "
x : Tensor[2]
x = [1.0, 2.0, 3.0]
main = 1
";
        assert!(matches!(
            check(src),
            Err(ShapeError::AnnotationMismatch { .. })
        ));
    }

    #[test]
    fn p3_global_annotation_improves_precision() {
        // 注釈付きグローバルは宣言 shape で登録されるため、推論不能（Unknown）な値でも
        // 下流の検査が効くようになる（静的型が精度を上げる例）。
        // w = zeros [3] は本来 Unknown だが、注釈 Tensor[3] で確定。下流の w + [1,2] が
        // [3]+[2] で矛盾し検出される（注釈が無ければ Unknown 伝播で見逃していた）。
        let src = "
w : Tensor[3]
w = zeros [3]
main = w + [1.0, 2.0]
";
        assert!(matches!(
            check(src),
            Err(ShapeError::ElementwiseMismatch { .. })
        ));
    }

    #[test]
    fn p3_underspecified_annotation_is_lenient() {
        // 注釈が引数より短い（部分注釈）場合、足りない引数は Unknown 束縛にして黙って通す。
        // ここでは注釈が引数1個ぶんしか無いが f は2引数。2番目 b は Unknown となり、
        // a + b は Unknown 側があるので矛盾を断定しない（偽陽性ゼロ）。
        let src = "
f : Tensor[3]
f a b = a + b
main = 1
";
        assert!(check(src).is_ok());
    }

    // ----- P4: 次元変数の単一化・shape 算術 -----
    // P3 では Var 同士の比較は常に Unknown または「断定せず通す」だった。
    // P4 では:
    //   - 同名 Var 同士 → 一致（Tensor[n]+Tensor[n] → Tensor[n]）
    //   - 異名 Var 同士 → VarConflict（Tensor[n]+Tensor[m] は潜在的不一致）
    //   - 異なるランク → ElementwiseMismatch（Var を含む場合も常に）
    //   - AnnotationMismatch: ランク不一致・Var 名不一致も対象に追加

    #[test]
    fn p4_var_conflict_elementwise() {
        // n と m は独立な型変数（異なる名前 → 等しい保証なし）。
        // Tensor[n] + Tensor[m] は型レベルで不整合なので VarConflict を報告する。
        let src = "
f : Tensor[n] -> Tensor[m] -> Tensor[n]
f a b = a + b
main = 1
";
        assert!(
            matches!(check(src), Err(ShapeError::VarConflict { .. })),
            "異名変数の elementwise は VarConflict を期待"
        );
    }

    #[test]
    fn p4_var_same_name_elementwise_ok() {
        // 同名変数（n==n）→ 等しい次元が保証されるので通過し、結果は Tensor[n]。
        // P3 では Unknown を返していた（all_concrete が None だったため）。
        // P4 では dim_pair_conflict が Ok を返し、Tensor[n] を正しく伝播する。
        let src = "
f : Tensor[n] -> Tensor[n] -> Tensor[n]
f a b = a + b
main = 1
";
        assert!(check(src).is_ok(), "同名変数の elementwise は通過を期待");
    }

    #[test]
    fn p4_var_same_name_result_shape() {
        // 同名変数の elementwise が正しく Tensor[n] を返すことを確認する（P4 精度改善）。
        // P3 では Unknown になっていた部分。
        let src = "
f : Tensor[n] -> Tensor[n] -> Tensor[n]
f a b = a + b
g : Tensor[3] -> Tensor[3]
g v = f v v
main = 1
";
        // g の本体: f v v = (Tensor[3]) + (Tensor[3]) → Tensor[3]
        // 宣言戻り型 Tensor[3] と一致 → Ok
        assert!(check(src).is_ok(), "Tensor[3] + Tensor[3] → Tensor[3] であるべき");
    }

    #[test]
    fn p4_var_conflict_matmul_inner() {
        // 行列積の内次元が異名変数（k と j）→ 独立型変数なので等しい保証がない → VarConflict。
        // 正しい注釈は `Tensor[m,k] -> Tensor[k,n]`（共通名 k）にすべきである。
        let src = "
f : Tensor[m, k] -> Tensor[j, n] -> Tensor[m, n]
f a b = a @ b
main = 1
";
        assert!(
            matches!(check(src), Err(ShapeError::VarConflict { op: "@", .. })),
            "matmul 内次元の異名変数は VarConflict を期待"
        );
    }

    #[test]
    fn p4_var_matmul_inner_same_ok() {
        // 内次元が同名変数（k==k）→ 一致が保証されるので通過。結果は [m,n]。
        // P3 でも通っていたが、P4 で VarConflict の対称性を確認するためテストを追加。
        let src = "
f : Tensor[m, k] -> Tensor[k, n] -> Tensor[m, n]
f a b = a @ b
main = 1
";
        assert!(check(src).is_ok(), "内次元同名変数の matmul は通過を期待");
    }

    #[test]
    fn p4_rank_mismatch_always_errors() {
        // ランクが異なるテンソルの加算は、次元変数を含む場合でも常にエラー。
        // P3 では Var が絡むと Unknown を返して見逃していた。
        let src = "
f : Tensor[n] -> Tensor[n, m] -> Tensor[n]
f a b = a + b
main = 1
";
        assert!(
            matches!(check(src), Err(ShapeError::ElementwiseMismatch { .. })),
            "ランク不一致は常に ElementwiseMismatch を期待"
        );
    }

    #[test]
    fn p4_annotation_rank_mismatch() {
        // 宣言戻り型のランクと本体推論のランクが異なる → AnnotationMismatch（P4 拡張）。
        // f : Tensor[n] -> Tensor[n,n] と宣言しているが本体は Tensor[n]（1D）を返す。
        let src = "
f : Tensor[n] -> Tensor[n, n]
f a = a
main = 1
";
        assert!(
            matches!(check(src), Err(ShapeError::AnnotationMismatch { .. })),
            "ランク不一致の戻り型は AnnotationMismatch を期待"
        );
    }

    #[test]
    fn p4_annotation_var_name_mismatch() {
        // 宣言戻り型 Tensor[m] vs 本体推論 Tensor[n]（変数名が異なる）→ AnnotationMismatch。
        // n と m は独立変数なので、`f : Tensor[n] -> Tensor[m]` で `f a = a` は矛盾する。
        let src = "
f : Tensor[n] -> Tensor[m]
f a = a
main = 1
";
        assert!(
            matches!(check(src), Err(ShapeError::AnnotationMismatch { .. })),
            "変数名不一致の戻り型は AnnotationMismatch を期待"
        );
    }

    #[test]
    fn p4_annotation_var_name_same_ok() {
        // 宣言戻り型 Tensor[n] vs 本体推論 Tensor[n]（同名）→ Ok（P4 精度改善）。
        // P3 では all_concrete が None なので通っていたが、理由が曖昧だった。
        // P4 では dim_pair_conflict が Ok (同名) を返すので明示的に通る。
        let src = "
f : Tensor[n] -> Tensor[n]
f a = a
main = 1
";
        assert!(check(src).is_ok(), "同名変数戻り型は通過を期待");
    }

    #[test]
    fn p4_shape_arithmetic_parses_and_no_false_positive() {
        // shape 算術（m+n, m*n）が型注釈に書けて、偽陽性なく通ることを確認する。
        // concat/flatten 等の算術プリミティブは未実装なので、算術次元は Unknown として
        // 扱われる。これにより戻り型の突き合わせで矛盾が断定されず通過する。
        //
        // 本体は Unknown/Scalar を返す例を使い、ランク不一致の誤検出が起きないことを示す:
        //   concat a b = a : 引数 Tensor[m] と戻り Tensor[Unknown] は同ランク（rank 1）
        //   sliding a = a  : 引数 Tensor[Unknown]（m+n を 1 次元と見る）と戻り Tensor[m] は同ランク
        // なお flatten : Tensor[m,n] -> Tensor[m*n] は体が rank-2 を返し宣言戻り rank-1 なので
        // ランク不一致（AnnotationMismatch）になる―これは正しい挙動（本体が間違っている）。
        let src = "
concat : Tensor[m] -> Tensor[n] -> Tensor[m+n]
concat a b = a
sliding : Tensor[m+n] -> Tensor[m+n]
sliding a = a
main = 1
";
        assert!(check(src).is_ok(), "shape 算術注釈は偽陽性なく通過を期待");
    }

    #[test]
    fn p4_shape_arithmetic_flatten_rank_mismatch() {
        // flatten : Tensor[m,n] -> Tensor[m*n] は rank-2 引数を受け取り rank-1 を返す。
        // 本体 `flatten a = a` は rank-2 の a をそのまま返すため rank 不一致 → AnnotationMismatch。
        // これは P4 のランク検査が正当に型エラーを検出する例（偽陰性ゼロの確認）。
        let src = "
flatten : Tensor[m, n] -> Tensor[m*n]
flatten a = a
main = 1
";
        assert!(
            matches!(check(src), Err(ShapeError::AnnotationMismatch { .. })),
            "rank-2 本体と rank-1 宣言戻り型は AnnotationMismatch を期待"
        );
    }

    #[test]
    fn p4_shape_arithmetic_complex_expr() {
        // 複合算術式（m+1, m+n-1）がパースエラーなく通ることを確認する。
        // AST に DimExpr として保持されることが目的。
        // 本体と宣言のランクを揃えて偽陽性が出ないことも確認する:
        //   shift : Tensor[m] -> Tensor[m+1] : rank-1 in, rank-1 out (Unknown). body 返り Unknown.
        //   clip  : Tensor[m+n-1] -> Tensor[m+n-1] : rank-1 in/out. body そのまま返す。
        let src = "
shift : Tensor[m] -> Tensor[m+1]
shift a = zeros [1]
clip : Tensor[m+n-1] -> Tensor[m+n-1]
clip a = a
main = 1
";
        assert!(check(src).is_ok(), "複合 shape 算術式は通過を期待");
    }

    #[test]
    fn p3_linreg_annotations_pass() {
        // 北極星プログラムの完全な型注釈付き版が、pass3 の本体検査でも偽陽性なく通ること。
        // loss/step は Tensor[3] 束縛で本体が評価され、predict/mse は次元変数で素通しになる。
        let src = "
x = [1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0; 1.0, 0.0, 1.0]
y = [1.0, 2.0, 3.0, 0.5]
lr = 0.01
predict : Tensor[n, d] -> Tensor[d] -> f32 -> Tensor[n]
predict feats w b = feats @ w + b
mse : Tensor[n] -> Tensor[n] -> f32
mse pred target = mean ((pred - target) ^ 2)
loss : Tensor[3] -> f32
loss w = mse (predict x w 0.0) y
step : Tensor[3] -> Tensor[3]
step w = w - lr * grad loss w
main : Tensor[3]
main = iterate (zeros [3]) 1000 step
";
        assert!(check(src).is_ok());
    }
}
