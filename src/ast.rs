// AST（抽象構文木）の型定義。ロジックを一切持たず純粋なデータ構造のみ。
// パーサが出力し、インタプリタが消費する中間表現として機能する。

// ソースファイル全体をトップレベル定義の列として表す。
// 評価順はこの列の順序に依存するため、前方参照は P0 では不可（P1 以降で対応予定）。
pub type Program = Vec<TopLevel>;

// トップレベルに置けるのは「束縛」と「型注釈」の2種類だけ。
// 型注釈は P0 評価器では無視されるが、パーサで読み捨てずに保持しておく。
// 将来の型検査フェーズで再利用できるようにするため。
#[derive(Debug, Clone)]
pub enum TopLevel {
    Binding {
        name: String,
        params: Vec<String>,
        body: Expr,
    },
    TypeAnnotation {
        name: String,
        ty: TypeExpr,
    },
}

#[derive(Debug, Clone)]
pub enum Expr {
    Lit(Literal),
    Var(String),

    // 関数適用を二項木（left=関数, right=引数）で表す。
    // `f a b` は App(App(Var"f", Var"a"), Var"b") と左再帰でネストする。
    // これはカリー化（多引数関数を1引数関数の連鎖として扱う）の帰結で、
    // Haskell/ML 系言語で広く使われる表現方法。
    // 代替: 引数リストを持つ App(func, Vec<Expr>) にすることも可能だが、
    // 部分適用（`f a` で止める）を自然に扱えなくなるためこの形を選んだ。
    App(Box<Expr>, Box<Expr>),

    BinOp {
        op: BinOpKind,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    UnaryMinus(Box<Expr>),

    // |> を AST ノードとして保持する（`f a` に脱糖しない）。
    // 理由: 将来の pretty-printer やデバッガでソースの構造を再現したいため。
    // 評価時には apply(eval(f), eval(a)) と同等に処理される。
    Pipe(Box<Expr>, Box<Expr>),

    If {
        cond: Box<Expr>,
        then: Box<Expr>,
        else_: Box<Expr>,
    },

    Let {
        name: String,
        params: Vec<String>,
        value: Box<Expr>,
        body: Box<Expr>,
    },

    // テンソルリテラルを「行のリスト」として表す。
    // 外側の Vec が行（;区切り）、内側の Vec が列（,区切り）に対応する。
    // [1,2; 3,4] → vec![vec![1,2], vec![3,4]]
    // 代替: フラットな Vec<Expr> + shape を持たせることもできるが、
    // 行の長さが揃っているかの検証がパーサ or 評価器どちらでやるか曖昧になる。
    // 行単位の構造を保持することで「行の長さ不一致」を評価器で明確にチェックできる。
    TensorLit(Vec<Vec<Expr>>),

    // パーサは Lambda を直接生成しない。
    // `f x y = body` という束縛をパーサは Binding{params:["x","y"]} のまま保持し、
    // インタプリタの desugar_lambda が Lambda{x, Lambda{y, body}} に変換する。
    // パーサでの脱糖も可能だが、AST に元の引数リストを残した方が
    // エラーメッセージや将来の型推論で役立つため、遅延脱糖にした。
    Lambda {
        param: String,
        body: Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    MatMul,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

// 型式の AST。P0 では評価器に渡されず、パースして捨てるだけ。
// 型注釈をパースエラーにせずに受け入れることで、要件定義書の文法例を
// そのまま入力できる（型検査フェーズは P2 以降で実装する）。
#[derive(Debug, Clone)]
pub enum TypeExpr {
    Named(String),
    Arrow(Box<TypeExpr>, Box<TypeExpr>),
    Tensor(Vec<Option<usize>>),
    App(Box<TypeExpr>, Box<TypeExpr>),
}
