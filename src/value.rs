// 実行時の値表現と、変数束縛の環境（スコープ）を定義する。
// インタプリタは式を評価して Value を返し、次の評価の入力として使う。

use ndarray::ArrayD;
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
// 永続連結リストとして実装する。
//
// extend() は新しいノードを先頭に追加して新しい Env を返す（元の Env は変更しない）。
// lookup() はリストを先頭から辿り、最初にヒットした束縛を返す。
// これによってシャドーイング（同名の再束縛）が自動的に実現される。
//
// 永続データ構造を選んだ理由:
// クロージャは env をキャプチャするので、extend() のたびに HashMap をクローンすると
// O(n) のコピーが走る。Rc 連結リストなら extend() は O(1) で、
// 複数のクロージャが同じ親 env を安全に共有できる。
// 代替: HashMap<String, Value> をクローンする方式はシンプルだが、
// スコープが深くなるほど（クロージャのネスト等）コストが線形に増える。
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

    // 現在の環境を親として、name → value の束縛を先頭に追加した新しい環境を返す。
    // 元の環境は変更されないため、クロージャが Rc でキャプチャしている場合も安全。
    pub fn extend(&self, name: String, value: Value) -> Self {
        Env(Some(Rc::new(EnvNode {
            name,
            value,
            parent: self.clone(),
        })))
    }

    // 名前で変数を検索する。リストを先頭（最も新しい束縛）から辿る。
    // 同名が複数あれば最も近い（内側の）束縛が返される（シャドーイング）。
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
        }
    }
}
