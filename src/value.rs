// 実行時の値表現と、変数束縛の環境（スコープ）を定義する。
// インタプリタは式を評価して Value を返し、次の評価の入力として使う。

use ndarray::ArrayD;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::Expr;

// Haplo の実行時値。評価器が生成し、環境に保存し、関数に渡す。
// Int と Float を分けて保持する理由:
//   - Int × Int の演算は整数のまま返す（3/2 = 1 のような整数除算を維持）
//   - Float が混ざった時点で自動的に Float に昇格する
//   - 型を捨てて全部 f64 にする方法もあるが、整数リテラルの出力が "2.0" になって
//     ユーザが違和感を覚えるため分けた
#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    // テンソルを Rc（参照カウントポインタ）でくるんで保持する。
    // Rc を使う理由: テンソルは大きくなりうるため、クロージャがキャプチャするたびに
    // ディープコピーするのは高コスト。Rc なら参照カウントを増やすだけ（O(1)）。
    // Arc（原子的参照カウント）ではなく Rc を使う理由: P0 はシングルスレッドなので
    // スレッドセーフティは不要。Rc の方が Arc より軽量。
    //
    // ArrayD（動的ランク）を選んだ理由:
    // Array2（2D に固定）の方がコンパイル時保証が強いが、
    // 将来 3D 以上のテンソルに対応するとき Value の型を変える必要が生じる。
    // ArrayD にしておけば後からランクを増やしても Value の定義は変わらない。
    // 要素型を f64 に固定しているのは P0 の簡略化; dtype の静的検査は P2 以降の課題。
    Tensor(Rc<ArrayD<f64>>),
    Closure(Closure),
    // 組み込み関数を Value として扱う専用バリアント。
    // Closure（ユーザ定義関数）と区別することで、組み込みの処理を Rust 関数として
    // 直接書ける（AST や Expr を経由しなくて済む）。
    // 代替案: 組み込みもダミーの Expr を body に持つ Closure にする方法があるが、
    // それだと apply() で Expr を評価しようとして無限ループやパニックのリスクがある。
    Builtin(BuiltinFn),
    // 多引数組み込みの部分適用を表す。Vec に引数を貯め、arity に達したら実行する。
    // reshape（2引数）・grad（2引数）・iterate（3引数）のように、カリー化された
    // 組み込みを「引数を1つずつ受け取る」形で扱うために必要。
    // 1引数組み込み（sum 等）は Builtin のまま即座に評価されるのでここには現れない。
    PartialBuiltin(BuiltinFn, Vec<Value>),
    // autodiff（自動微分）のテープノードを指す微分追跡値。
    // grad の評価中だけ出現し、内部の usize は autodiff::Tape のノード index。
    // テンソルとスカラーのどちらも表せる（ノードの value が 0 次元なら スカラー）。
    // 通常の評価では生成されず、grad のスコープ外に漏れることもない。
    Tracked(usize),
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
    Grad,
    Iterate,
    // P6 新規: 標準ライブラリ
    // 要素ごとの絶対値（スカラー・テンソル両対応）
    Abs,
    // テンソル全要素の最大値（スカラー返し）
    MaxVal,
    // テンソル全要素の最小値（スカラー返し）
    MinVal,
    // 1D テンソルの連結。concat a b : Tensor[m] -> Tensor[n] -> Tensor[m+n]
    Concat,
    // 2D → 1D 展開。flatten t : Tensor[m,n] -> Tensor[m*n]
    Flatten,
    // L2 ノルム（norm t = sqrt (sum (t ^ 2))）のフュージョンされた版
    Norm,
    // 要素を [lo, hi] でクリップ。clip lo hi t
    Clip,
}

// ユーザ定義関数（クロージャ）の実行時表現。
// param: 受け取るパラメータ名（1つだけ; 多引数はカリー化でネスト）
// body:  関数本体の式（AST ノード）
// env:   クロージャが定義された時点の環境（変数スコープのスナップショット）
//
// env を Closure に持たせることで語彙的スコープ（lexical scope）を実現する。
// 呼び出し時の環境ではなく、定義時の環境を使うため、
// 関数を引数として渡したり、戻り値として返したりしても正しく動く。
#[derive(Debug, Clone)]
pub struct Closure {
    pub param: String,
    pub body: Expr,
    pub env: Env,
}

// 変数名から値へのマッピング（スコープ / 変数環境）。
// 2層構造を持つ:
//   - locals: ローカル束縛（関数引数・let）の永続連結リスト
//   - globals: トップレベル定義を保持する共有可変マップ
//
// locals を永続連結リストにする理由:
// クロージャは env をキャプチャするので、extend() のたびに HashMap をクローンすると
// O(n) のコピーが走る。Rc 連結リストなら extend() は O(1) で、
// 複数のクロージャが同じ親 env を安全に共有できる。
//
// globals を Rc<RefCell<HashMap>> の共有マップにする理由:
// 前方参照・相互再帰を可能にするため。全クロージャが同じ globals を共有して
// キャプチャするので、定義後に globals へ追加された束縛も呼び出し時に解決できる
// （knot-tying）。これにより f が後で定義される g を呼ぶ、といった相互再帰が動く。
#[derive(Debug, Clone)]
pub struct Env {
    locals: Option<Rc<EnvNode>>,
    globals: Rc<RefCell<HashMap<String, Value>>>,
}

#[derive(Debug)]
pub struct EnvNode {
    name: String,
    value: Value,
    parent: Option<Rc<EnvNode>>,
}

impl Env {
    // 空のグローバルマップを持つ環境を作る（テストや単独のクロージャ用）。
    pub fn empty() -> Self {
        Env {
            locals: None,
            globals: Rc::new(RefCell::new(HashMap::new())),
        }
    }

    // 既存の共有グローバルマップから環境を作る。
    // build_global_env が全トップレベル定義で共有する globals を渡すために使う。
    pub fn with_globals(globals: Rc<RefCell<HashMap<String, Value>>>) -> Self {
        Env {
            locals: None,
            globals,
        }
    }

    // 共有グローバルマップへの参照を返す（クローンは Rc の参照カウント増加のみ）。
    pub fn globals(&self) -> Rc<RefCell<HashMap<String, Value>>> {
        self.globals.clone()
    }

    // グローバルマップに束縛を1つ登録する。build_global_env の2パスで使う。
    pub fn define_global(&self, name: String, value: Value) {
        self.globals.borrow_mut().insert(name, value);
    }

    // 現在の環境を親として、name → value のローカル束縛を先頭に追加した新しい環境を返す。
    // globals は共有したまま（Rc クローン）。元の環境は変更されないため、
    // クロージャが Rc でキャプチャしている場合も安全。
    pub fn extend(&self, name: String, value: Value) -> Self {
        Env {
            locals: Some(Rc::new(EnvNode {
                name,
                value,
                parent: self.locals.clone(),
            })),
            globals: self.globals.clone(),
        }
    }

    // 名前で変数を検索する。まず locals 連鎖を先頭（最も新しい束縛）から辿り、
    // 見つからなければ globals マップを引く。
    // 同名が複数あれば最も近い（内側の）束縛が返される（シャドーイング）。
    // ローカルがグローバルを覆い隠せるよう、locals を先に探す。
    pub fn lookup(&self, name: &str) -> Option<Value> {
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

// Value から型名文字列を返す内部ユーティリティ。
// エラーメッセージで「Int が必要ですが Tensor でした」のように表示するために使う。
pub fn value_type_name(v: &Value) -> &'static str {
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

// Int または Float を f64 に統一して返す。
// Haplo では Int と Float が混在した演算（例: 1 + 2.0）を許可するため、
// 演算子の評価時に両辺を f64 に揃えるために使う。
// Tensor を受け取った場合はエラーにする（テンソルとスカラーの混在は
// 演算子のパターンマッチで先に処理されるため、ここに到達することは想定しない）。
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

// Value が Tensor であることを確認して中身を返す。
// sum/mean などのテンソル専用組み込み関数が引数の型を検証するために使う。
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
    // main の戻り値を println! で表示するときに使われる。
    // Float の表示: 整数値の Float（例: 6.0）を "6.0" と表示する（".1" フォーマット）。
    // 理由: Rust のデフォルトは "6" と表示してしまい、Int の 6 と区別がつかない。
    // Tensor の表示: ndarray が実装する Display をそのまま利用する。
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
            Value::PartialBuiltin(b, args) => {
                write!(f, "<builtin:{:?} (部分適用 {} 引数)>", b, args.len())
            }
            Value::Tracked(_) => write!(f, "<tracked>"),
        }
    }
}
